#!/usr/bin/env python
# coding=utf-8

"""
Fine-tune a pretrained BERT on the QQP (Quora Question Pairs) dataset
to obtain a teacher model and save it.

文件命名和路径中包含数据集名字：dataset = "qqp"
默认 checkpoint 路径: ./checkpoints_qqp/teacher_qqp_bert-base-uncased.pth
"""

import argparse
import math
import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    DataCollatorWithPadding,
)

DATASET_NAME = "qqp"


# ======================
# Utils
# ======================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保持行为确定（可能稍慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================
# Data
# ======================

def get_dataloaders_qqp(
    tokenizer: BertTokenizerFast,
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    使用 HuggingFace datasets 加载 GLUE/QQP，并构造 DataLoader。
    """

    raw_datasets = load_dataset("glue", DATASET_NAME)

    def preprocess_function(examples):
        # QQP 的字段是 question1 / question2
        return tokenizer(
            examples["question1"],
            examples["question2"],
            truncation=True,
            max_length=max_length,
        )

    # map 会自动按批处理
    encoded_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=["question1", "question2", "idx"],
    )

    # 将 label 重命名为 labels，方便 model(**batch) 直接使用
    encoded_datasets = encoded_datasets.rename_column("label", "labels")

    # 转成 PyTorch tensor 格式
    encoded_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
    )

    # Data collator 负责 padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = encoded_datasets["train"]
    val_dataset = encoded_datasets["validation"]

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# ======================
# Model
# ======================

def build_teacher(
    pretrained_model_name: str = "bert-base-uncased",
    num_labels: int = 2,
) -> BertForSequenceClassification:
    """
    构造 BERT teacher，使用 transformers 的 BertForSequenceClassification。
    """
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=num_labels,
    )
    return model


# ======================
# Train / Eval
# ======================

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

        total_loss += loss.item() * batch["labels"].size(0)

        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)

    return avg_loss, avg_acc


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    global_step_start: int = 0,
    writer: SummaryWriter = None,
):
    model.train()
    total_loss = 0.0
    total_samples = 0
    global_step = global_step_start

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if writer is not None:
            writer.add_scalar("train/step_loss", loss.item(), global_step)

        global_step += 1

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, global_step


# ======================
# Main
# ======================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT teacher on QQP dataset."
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace pretrained model name.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length for tokenization.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Dataloader num_workers.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"./checkpoints_{DATASET_NAME}",
        help="Directory to save teacher checkpoints.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=f"./runs_{DATASET_NAME}_teacher",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter(log_dir=args.log_dir)

    # Tokenizer + Data
    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_model_name)
    train_loader, val_loader = get_dataloaders_qqp(
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    # Model
    model = build_teacher(
        pretrained_model_name=args.pretrained_model_name,
        num_labels=2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    # 模仿 imagenet_tiny_teacher: 简单的 epoch 级 cosine lr
    def adjust_lr(optimizer, epoch, num_epochs, base_lr):
        if num_epochs <= 1:
            lr = base_lr
        else:
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / (num_epochs - 1)))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        lr = adjust_lr(optimizer, epoch, args.epochs, args.lr)
        print(f"\nEpoch [{epoch + 1}/{args.epochs}] - LR: {lr:.6f}")

        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            global_step_start=global_step,
            writer=writer,
        )

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4%}"
        )

        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("train/lr", lr, epoch)

        # 保存最优 teacher
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_name = f"teacher_{DATASET_NAME}_{args.pretrained_model_name.replace('/', '_')}.pth"
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            print(f"New best val_acc={best_val_acc:.4%}, saved to: {ckpt_path}")

    print(f"\nTraining finished. Best val_acc = {best_val_acc:.4%}")
    writer.close()


if __name__ == "__main__":
    main()

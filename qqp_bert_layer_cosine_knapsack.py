#!/usr/bin/env python
# coding=utf-8

"""
Cosine-similarity based layer-wise (attention / FFN) pruning for BERT on QQP.

- Phase 0: 加载已训练好的 QQP BERT teacher
- Phase 1: 基于 CLS 的 cosine similarity 计算每个 (layer, attn/ffn) 的冗余度：
          importance = 1 - cos(CLS_in, CLS_out)
- Phase 2: 对若干 target FLOPs ratio：
    - 用静态 importance + knapsack 得到 binary mask（带宽容上界）
    - 打印每个 unit 的 score & binary mask
    - 可选：构建物理剪枝后的模型并做 1 epoch KD finetune
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizerFast

from bert_pruning_utils import (
    set_seed,
    estimate_bert_layer_flops,
    BertCLSCollector,
    LayerUnitInfo,
    knapsack_select_mask,
)  # type: ignore


# ============================================================
# Global config
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_NAME = "bert-base-uncased"
DATASET_NAME = "glue"
DATASET_CONFIG = "qqp"

# Data
MAX_SEQ_LEN = 128
BATCH_SIZE = 64
NUM_WORKERS = 4

# KD / Finetune
TEMP_KD = 4.0
FT_EPOCHS = 1          # 你要求 finetune 1 epoch
FT_LR = 3e-5

# Cosine importance 统计多少个 batch
COSINE_NUM_ITERS = 50  # 可以通过 CLI 覆盖

# Ratios to evaluate (on prunable part)
# 12 层 * 2 units(attn+ffn) = 24 units
EVAL_RATIOS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]


# ============================================================
# QQP dataloaders
# ============================================================

def build_qqp_loaders(
    model_name: str = BERT_NAME,
    batch_size: int = BATCH_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, BertTokenizerFast]:
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    raw_datasets = load_dataset(DATASET_NAME, DATASET_CONFIG)

    def preprocess(batch):
        return tokenizer(
            batch["question1"],
            batch["question2"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
        )

    encoded = raw_datasets.map(
        preprocess,
        batched=True,
        remove_columns=["question1", "question2", "idx"],
    )

    encoded.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"],
    )

    def collate_fn(features):
        input_ids = torch.stack([f["input_ids"] for f in features], dim=0)
        attention_mask = torch.stack([f["attention_mask"] for f in features], dim=0)
        token_type_ids = torch.stack([f["token_type_ids"] for f in features], dim=0)
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        return batch

    train_loader = DataLoader(
        encoded["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        encoded["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, tokenizer


# ============================================================
# Model / loss helpers
# ============================================================

def build_bert_classifier(model_name: str = BERT_NAME, num_labels: int = 2) -> nn.Module:
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return model


def forward_logits(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    只取 logits，避免传 labels 触发内部 CE。
    """
    inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "token_type_ids": batch["token_type_ids"],
    }
    outputs = model(**inputs)
    if hasattr(outputs, "logits"):
        return outputs.logits
    else:
        return outputs[0]


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> float:
    model.eval()
    correct, total = 0, 0
    pbar = tqdm(loader, desc="[Eval] QQP", ncols=100)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = forward_logits(model, batch)
        preds = logits.argmax(dim=-1)
        labels = batch["labels"]
        correct += (preds == labels).sum().item()
        total += labels.numel()
        if total > 0:
            pbar.set_postfix(acc=f"{correct / total * 100:.2f}%")
    return correct / max(total, 1)


def kd_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    T: float = TEMP_KD,
) -> torch.Tensor:
    log_p_s = F.log_softmax(logits_s / T, dim=-1)
    p_t = F.softmax(logits_t / T, dim=-1)
    loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    return loss


# ============================================================
# Physical pruning helpers: identity sublayers
# （直接搬你现有脚本里的实现，以保证兼容性）
# ============================================================

class IdentityBertAttention(nn.Module):
    """
    物理剪枝后的 attention 子层：
      行为等价于“完全跳过 self-attention + LN”，但返回结构要和 BertAttention 对齐。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        **kwargs,
    ):
        if output_attentions:
            return hidden_states, None
        else:
            return (hidden_states,)


class IdentityBertOutput(nn.Module):
    """
    物理剪枝后的 FFN 子层（BertOutput）：
      y = input_tensor（完全跳过 FFN + LN）
    """

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, input_tensor):
        return input_tensor


class IdentityBertIntermediate(nn.Module):
    """
    物理剪枝后的 FFN 中间层（BertIntermediate）：
      直接返回输入，不再做 dense + 激活。
    """

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        return hidden_states


def apply_physical_bert_pruning(
    model: nn.Module,
    units: List[LayerUnitInfo],
    mask: torch.Tensor,
) -> None:
    """
    根据二值 mask 物理剪枝 BERT：
      - mask[i] = 0 → 对应 unit 替换成 Identity 子层
      - mask[i] = 1 → 保留原子层
    """
    encoder_layers = model.bert.encoder.layer  # type: ignore[attr-defined]
    mask_cpu = mask.detach().cpu().view(-1).tolist()

    assert len(units) == len(mask_cpu), "units 和 mask 长度不一致"

    for u, m in zip(units, mask_cpu):
        if m >= 0.5:
            continue

        layer = encoder_layers[u.layer_idx]
        if u.unit_type == "attn":
            layer.attention = IdentityBertAttention()
        elif u.unit_type == "ffn":
            layer.intermediate = IdentityBertIntermediate()
            layer.output = IdentityBertOutput()
        else:
            raise ValueError(f"Unknown unit_type: {u.unit_type}")


# ============================================================
# Cosine similarity based importance
# ============================================================

@torch.no_grad()
def compute_unit_cosine_importance(
    model: nn.Module,
    units: List[LayerUnitInfo],
    train_loader: DataLoader,
    num_iters: int = COSINE_NUM_ITERS,
    device: torch.device = DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对每个 (layer_idx, unit_type) 计算:
        cos = mean_batch( cos_sim(CLS_in, CLS_out) )
        importance = 1 - cos
    返回:
        avg_cos:      [N]
        importance:   [N] （作为 knapsack 的 values，越大越重要）
    """
    model.eval()
    collector = BertCLSCollector(model)

    N = len(units)
    cos_accum = torch.zeros(N, dtype=torch.float32, device=device)
    counts = torch.zeros(N, dtype=torch.float32, device=device)

    loader_iter = iter(train_loader)

    pbar = tqdm(
        range(num_iters),
        desc=f"[Cosine] Estimating layer importance ({num_iters} iters)",
        ncols=120,
    )

    for _ in pbar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        collector.clear()
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }

        _ = model(**inputs)

        records = collector.get_records()
        attn_records = records["attn"]
        ffn_records = records["ffn"]

        attn_map = {(r.layer_idx, "attn"): r for r in attn_records}
        ffn_map = {(r.layer_idx, "ffn"): r for r in ffn_records}

        for idx, u in enumerate(units):
            if u.unit_type == "attn":
                r = attn_map.get((u.layer_idx, "attn"), None)
            else:
                r = ffn_map.get((u.layer_idx, "ffn"), None)

            if r is None:
                continue

            cls_in = r.cls_in.to(device)   # [B, H]
            cls_out = r.cls_out.to(device) # [B, H]

            cos_vec = F.cosine_similarity(cls_in, cls_out, dim=-1)  # [B]
            cos_mean = cos_vec.mean()

            cos_accum[idx] += cos_mean
            counts[idx] += 1.0

    collector.close()

    counts = counts.clamp_min(1.0)
    avg_cos = cos_accum / counts

    # importance 越大表示 CLS_in / CLS_out 差异越大，层越“重要”
    importance = 1.0 - avg_cos
    return avg_cos, importance


# ============================================================
# Finetune pruned model
# ============================================================

def finetune_pruned_model(
    teacher: nn.Module,
    pruned_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    writer: SummaryWriter,
    ratio_tag: str,
) -> Tuple[nn.Module, float]:
    device = DEVICE
    teacher.eval()
    pruned_model.train()

    opt = torch.optim.AdamW(pruned_model.parameters(), lr=FT_LR)
    global_step = 0

    best_acc = 0.0
    best_state = {k: v.cpu() for k, v in pruned_model.state_dict().items()}

    for epoch in range(FT_EPOCHS):
        pruned_model.train()
        epoch_loss = 0.0
        epoch_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"[Finetune {ratio_tag}] Epoch {epoch+1}/{FT_EPOCHS}",
            ncols=120,
        )
        running_loss = 0.0
        running_batches = 0

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                logits_T = forward_logits(teacher, batch)

            logits_S = forward_logits(pruned_model, batch)
            loss = kd_loss(logits_S, logits_T, T=TEMP_KD)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            epoch_batches += 1
            global_step += 1

            running_loss += loss.item()
            running_batches += 1

            if running_batches % 10 == 0:
                pbar.set_postfix(loss=f"{running_loss / running_batches:.4f}")

            if global_step % 100 == 0:
                writer.add_scalar(
                    f"finetune_{ratio_tag}/train_loss",
                    loss.item(),
                    global_step,
                )

        avg_loss = epoch_loss / max(epoch_batches, 1)
        val_acc = evaluate_accuracy(pruned_model, val_loader, device=device)

        writer.add_scalar(f"finetune_{ratio_tag}/epoch_loss", avg_loss, epoch)
        writer.add_scalar(f"finetune_{ratio_tag}/val_acc", val_acc, epoch)

        print(
            f"[Finetune {ratio_tag}] Epoch {epoch+1}/{FT_EPOCHS} | "
            f"Loss: {avg_loss:.4f} | ValAcc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in pruned_model.state_dict().items()}

    pruned_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return pruned_model, best_acc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        default="./checkpoints_qqp/teacher_qqp_bert-base-uncased.pth",
        help="Path to QQP BERT teacher checkpoint.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints_qqp_bert_layer_cosine",
        help="Output directory for cosine-based pruning results.",
    )
    parser.add_argument(
        "--do_finetune",
        action="store_true",
        help="If set, finetune pruned models after building masks.",
    )
    parser.add_argument(
        "--cosine_iters",
        type=int,
        default=COSINE_NUM_ITERS,
        help="How many batches to use when estimating cosine similarity.",
    )
    args = parser.parse_args()

    OUT_DIR = args.out_dir
    RESULTS_FILE = os.path.join(OUT_DIR, "qqp_bert_layer_cosine_results.json")
    TB_LOG_DIR = os.path.join(OUT_DIR, "tb_logs")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    description = (
        "QQP BERT layer-wise pruning with cosine-similarity importance and knapsack.\n"
        f"EVAL_RATIOS={EVAL_RATIOS}, FT_EPOCHS={FT_EPOCHS}, "
        f"COSINE_NUM_ITERS={args.cosine_iters}."
    )
    with open(os.path.join(OUT_DIR, "description.txt"), "w", encoding="utf-8") as f:
        f.write(description)

    set_seed(42)
    writer = SummaryWriter(log_dir=TB_LOG_DIR)

    # Data
    train_loader, val_loader, _ = build_qqp_loaders(
        model_name=BERT_NAME,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        num_workers=NUM_WORKERS,
    )

    # Teacher
    teacher = build_bert_classifier(model_name=BERT_NAME, num_labels=2).to(DEVICE)
    if os.path.isfile(args.teacher_ckpt):
        ckpt = torch.load(args.teacher_ckpt, map_location=DEVICE)
        if isinstance(ckpt, dict) and "model" in ckpt:
            teacher.load_state_dict(ckpt["model"])
        else:
            teacher.load_state_dict(ckpt)
        print(f"Loaded teacher from {args.teacher_ckpt}")
    else:
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_ckpt}")

    teacher_acc = evaluate_accuracy(teacher, val_loader, device=DEVICE)
    print(f"[Teacher] Val ACC: {teacher_acc * 100:.2f}%")
    writer.add_scalar("teacher/val_acc", teacher_acc, 0)

    # FLOPs estimation
    units, fixed_flops, total_flops_full = estimate_bert_layer_flops(
        teacher, seq_len=MAX_SEQ_LEN
    )
    prunable_costs = torch.tensor(
        [u.cost for u in units],
        dtype=torch.float32,
        device=DEVICE,
    )
    print(f"Number of prunable units: {len(units)}")
    print(f"Fixed FLOPs (approx): {fixed_flops:.2e}")
    print(f"Total FLOPs (full): {total_flops_full:.2e}")

    # Cosine-based importance（只做一次，所有 ratio 共用）
    avg_cos, importance = compute_unit_cosine_importance(
        model=teacher,
        units=units,
        train_loader=train_loader,
        num_iters=args.cosine_iters,
        device=DEVICE,
    )

    # 保存一下原始 cosine 和 importance 便于后处理
    torch.save(
        {
            "units": [(u.layer_idx, u.unit_type, float(u.cost)) for u in units],
            "avg_cos": avg_cos.cpu(),
            "importance": importance.cpu(),
            "prunable_costs": prunable_costs.cpu(),
            "fixed_flops": fixed_flops,
            "total_flops_full": total_flops_full,
        },
        os.path.join(OUT_DIR, "cosine_importance.pt"),
    )
    print(f"Saved cosine importance to {os.path.join(OUT_DIR, 'cosine_importance.pt')}")

    # Phase 2: evaluate several pruning ratios
    results = {
        "prunable_costs_per_unit": prunable_costs.detach().cpu().tolist(),
        "fixed_flops": fixed_flops,
        "total_flops_full": total_flops_full,
        "avg_cos": avg_cos.detach().cpu().tolist(),
        "importance": importance.detach().cpu().tolist(),
        "ratios": [],
    }

    for idx, ratio in enumerate(EVAL_RATIOS):
        print(f"\n=== Evaluating target prunable ratio {ratio:.3f} ===")

        # 为当前 ratio 计算一个更“宽松”的上界（保持你原来的逻辑）
        if idx < len(EVAL_RATIOS) - 1:
            flex_upper_ratio = 0.5 * (ratio + EVAL_RATIOS[idx + 1])
        else:
            flex_upper_ratio = 0.5 * (ratio + 1.0)

        # importance 直接作为 knapsack 的 values
        mask, actual_ratio_knap = knapsack_select_mask(
            values=importance,
            costs=prunable_costs,
            target_ratio=ratio,
            max_ratio=flex_upper_ratio,
        )
        mask = mask.to(DEVICE)

        kept_prunable = float((mask * prunable_costs).sum().item())
        total_prunable = float(prunable_costs.sum().item())
        prunable_ratio = kept_prunable / max(total_prunable, 1e-8)
        total_kept = fixed_flops + kept_prunable
        total_ratio = total_kept / max(total_flops_full, 1e-8)

        # 打印 per-unit score & mask（你要求的“同款详细打印”）
        print(
            f"\n--- Per-unit importance & binary mask "
            f"(ratio={ratio:.2f}, layer_idx, unit_type) ---"
        )
        scores_list = importance.detach().cpu().tolist()
        mask_list = mask.detach().cpu().tolist()
        for u, s, m in zip(units, scores_list, mask_list):
            print(
                f"  layer={u.layer_idx:02d}, type={u.unit_type:4s} | "
                f"score={s:.4f} | mask={int(m)}"
            )

        print("\n--- FLOPs summary ---")
        print(f"Target prunable ratio: {ratio:.4f}")
        print(f"Knapsack actual_ratio (by costs): {actual_ratio_knap:.4f}")
        print(f"Prunable ratio (mask * costs / total_prunable): {prunable_ratio:.4f}")
        print(
            f"Total FLOPs kept / full: {total_kept:.2e} / {total_flops_full:.2e} "
            f"({total_ratio * 100:.2f}%)"
        )

        result_entry = {
            "target_ratio_prunable": ratio,
            "actual_ratio_prunable_knapsack": actual_ratio_knap,
            "actual_ratio_prunable_mask": prunable_ratio,
            "mask": mask_list,
            "importance": scores_list,
            "kept_prunable_flops": kept_prunable,
            "total_prunable_flops": total_prunable,
            "total_kept_flops": total_kept,
            "total_full_flops": total_flops_full,
            "total_ratio": total_ratio,
        }

        if args.do_finetune:
            # 1) 从 teacher 拷贝一份模型
            pruned_model = build_bert_classifier(model_name=BERT_NAME, num_labels=2).to(
                DEVICE
            )
            pruned_model.load_state_dict(teacher.state_dict())

            # 2) 物理剪枝
            apply_physical_bert_pruning(pruned_model, units, mask)

            # 3) 1 epoch KD finetune
            ratio_tag = f"ratio_{ratio:.2f}"
            pruned_model, best_acc = finetune_pruned_model(
                teacher=teacher,
                pruned_model=pruned_model,
                train_loader=train_loader,
                val_loader=val_loader,
                writer=writer,
                ratio_tag=ratio_tag,
            )
            print(f"[Result {ratio_tag}] Best Val Acc: {best_acc * 100:.2f}%")

            # 4) 保存 pruned 模型和 mask
            ckpt_path = os.path.join(OUT_DIR, f"qqp_bert_layer_cosine_pruned_{ratio:.2f}.pth")
            torch.save(
                {
                    "model": pruned_model.state_dict(),
                    "mask": mask.cpu(),
                    "importance": importance.cpu(),
                    "target_ratio_prunable": ratio,
                    "actual_ratio_prunable_mask": prunable_ratio,
                    "total_ratio": total_ratio,
                    "kept_prunable_flops": kept_prunable,
                    "total_kept_flops": total_kept,
                },
                ckpt_path,
            )
            print(f"Saved pruned model to {ckpt_path}")

            result_entry["best_val_acc"] = best_acc

        results["ratios"].append(result_entry)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()

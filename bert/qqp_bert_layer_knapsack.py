#!/usr/bin/env python
# coding=utf-8

"""
Layer-wise (attention / FFN) pruning for BERT on QQP with knapsack.

- Phase 0: 加载已训练好的 QQP BERT teacher
- Phase 1: 训练 policy encoder（基于 CLS summarizer + static features）
- Phase 2: 对若干 target FLOPs ratio：
    - 用 encoder + knapsack 得到 binary mask
    - 打印每个 unit 的 soft score & binary mask
    - 可选：构建“逻辑上的 pruned 模型”（通过 gating 固定 mask）并做 1 epoch KD finetune
"""
from __future__ import annotations
import argparse
import json
import os
import random
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
    compute_flops_with_mask,
    BertCLSCollector,
    CLSSummarizer,
    TokenProjector,
    CompressionPolicyEncoder,
    build_static_features,
    LayerUnitInfo,
    knapsack_select_mask,
)  # type: ignore


# ============================================================
# Global config（大部分可以直接改常量，有些可通过 CLI 参数覆盖）
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_NAME = "bert-base-uncased"
DATASET_NAME = "glue"
DATASET_CONFIG = "qqp"

# Data
MAX_SEQ_LEN = 128
BATCH_SIZE = 64
NUM_WORKERS = 4

# Encoder training
POLICY_WARMUP_EPOCHS = 1
POLICY_TRAIN_EPOCHS = 5
ENC_LR = 1e-4
TEMP_KD = 4.0
MIN_RATIO = 0.2
MAX_RATIO = 0.9
RATIO_WEIGHT = 25.0
KD_WEIGHT = 0.05
GATE_TEMP_START = 5.0
GATE_TEMP_END = 0.3

# Summarizer / tokens
SUMMARY_DIM = 128
STATIC_DIM = 4
TOKEN_DIM = 128

# Finetune
FT_EPOCHS = 1  # 按你的要求：finetune 1 epoch
FT_LR = 3e-5

# Ratios to evaluate (prunable 部分)
# BERT-base 有 12 层，每层 2 个 unit (attn + ffn) → 共 24 个 unit，
# 这里给得稍微密一点。
EVAL_RATIOS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]


# ============================================================
# Helper: QQP dataloaders
# ============================================================

def build_qqp_loaders(
    model_name: str = BERT_NAME,
    batch_size: int = BATCH_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, BertTokenizerFast]:
    """
    使用 HuggingFace datasets + tokenizer 构建 QQP 的 train / validation dataloader。
    """
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

    encoded = raw_datasets.map(preprocess, batched=True, remove_columns=[
        "question1",
        "question2",
        "idx",
    ])

    encoded.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"],
    )

    def collate_fn(features):
        # 这里 features 是一个 list[dict]
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
        # 兼容 tuple 返回
        return outputs[0]


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> float:
    """
    简单的 ACC 计算（不用 F1，避免额外依赖）。
    """
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
    """
    标准 KL-based KD loss。
    """
    log_p_s = F.log_softmax(logits_s / T, dim=-1)
    p_t = F.softmax(logits_t / T, dim=-1)
    loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    return loss


# ============================================================
# Gating controller（不放到 utils 里，专门给 BERT 用）
# ============================================================

class BertUnitGating:
    """
    在 BERT 的 attention / FFN 子层上做 gating。
    - attention: output' = hidden_states + g * (output - hidden_states)
    - FFN:       output' = input_tensor + g * (output - input_tensor)
    g:
      - 训练 encoder 时：来自 encoder 输出的 soft gate（0~1）
      - 做“逻辑 prune”时：来自固定的 0/1 mask
    """

    def __init__(
        self,
        model: nn.Module,
        units: List[LayerUnitInfo],
    ) -> None:
        self.model = model
        self.units = units

        # (layer_idx, unit_type) → global unit index
        self.unit_index_map: Dict[Tuple[int, str], int] = {
            (u.layer_idx, u.unit_type): u.index for u in units
        }

        self.train_gates: torch.Tensor | None = None
        self.fixed_mask: torch.Tensor | None = None

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _get_gate_scalar(
        self,
        layer_idx: int,
        unit_type: str,
        ref_tensor: torch.Tensor,
    ) -> torch.Tensor | None:
        key = (layer_idx, unit_type)
        idx = self.unit_index_map.get(key, None)
        if idx is None:
            return None

        if self.train_gates is not None:
            g_all = self.train_gates
        elif self.fixed_mask is not None:
            g_all = self.fixed_mask
        else:
            return None

        if idx >= g_all.numel():
            return None

        g = g_all[idx]
        if not torch.is_tensor(g):
            g = torch.tensor(float(g), dtype=torch.float32, device=ref_tensor.device)
        else:
            g = g.to(ref_tensor.device, dtype=torch.float32)
        return g

    def _register_hooks(self) -> None:
        encoder_layers = self.model.bert.encoder.layer  # type: ignore[attr-defined]

        for layer_idx, layer in enumerate(encoder_layers):
            # attention hook
            def attn_hook(module, inputs, outputs, layer_idx=layer_idx):
                """
                inputs: (hidden_states, attention_mask, ...)
                outputs: attention_output 或 (attention_output, ...)
                """
                out = outputs[0] if isinstance(outputs, tuple) else outputs
                hidden_states = inputs[0]  # [B, L, H]

                g = self._get_gate_scalar(layer_idx, "attn", out)
                if g is None:
                    return outputs

                g_b = g.view(1, 1, 1)
                new_out = hidden_states + g_b * (out - hidden_states)

                if isinstance(outputs, tuple):
                    return (new_out,) + outputs[1:]
                else:
                    return new_out

            h1 = layer.attention.register_forward_hook(attn_hook)

            # FFN hook (BertOutput)
            def ffn_hook(module, inputs, outputs, layer_idx=layer_idx):
                """
                inputs: (hidden_states, input_tensor)
                  - hidden_states: intermediate_output
                  - input_tensor: attention_output
                outputs: layer_output tensor
                """
                out = outputs[0] if isinstance(outputs, tuple) else outputs
                input_tensor = inputs[1]  # attention_output

                g = self._get_gate_scalar(layer_idx, "ffn", out)
                if g is None:
                    return outputs

                g_b = g.view(1, 1, 1)
                new_out = input_tensor + g_b * (out - input_tensor)

                if isinstance(outputs, tuple):
                    return (new_out,) + outputs[1:]
                else:
                    return new_out

            h2 = layer.output.register_forward_hook(ffn_hook)

            self._handles.extend([h1, h2])

    def set_train_gates(self, gates: torch.Tensor) -> None:
        """
        gates: [N]，0~1 的软 gate。
        """
        self.train_gates = gates

    def clear_train_gates(self) -> None:
        self.train_gates = None

    def set_fixed_mask(self, mask: torch.Tensor) -> None:
        """
        mask: [N]，0/1 的 tensor。用于“逻辑 prune”。
        """
        self.fixed_mask = mask.detach().float()

    def clear_fixed_mask(self) -> None:
        self.fixed_mask = None

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

# ============================================================
# Physical pruning helpers: identity sublayers
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
        past_key_value=None,      # 保留这个没问题
        output_attentions=False,
        **kwargs,                 # 新增：吃掉 past_key_values 等额外参数
    ):
        if output_attentions:
            # BertAttention 通常返回 (attention_output, attn_probs)
            return hidden_states, None
        else:
            # BertLayer 会做 self_attention_outputs[0]，所以必须返回 tuple
            return (hidden_states,)




class IdentityBertOutput(nn.Module):
    """
    物理剪枝后的 FFN 子层（BertOutput）：
    行为等价于 gate=0 时的 BertUnitGating.ffn_hook：
        y = input_tensor （完全跳过 FFN + LN）
    """
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, input_tensor):
        # hidden_states 是 intermediate_output，直接丢弃
        return input_tensor

class IdentityBertIntermediate(nn.Module):
    """
    物理剪枝后的 FFN 中间层（BertIntermediate）：
    直接返回输入，不再做 dense + 激活。
    """
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        # 原本是 dense + 激活，这里直接跳过
        return hidden_states



def apply_physical_bert_pruning(
    model: nn.Module,
    units: List[LayerUnitInfo],
    mask: torch.Tensor,
) -> None:
    """
    根据二值 mask 物理剪枝 BERT 模型：
    - mask[i] = 0 → 把对应 unit 替换成 Identity 子层
    - mask[i] = 1 → 保留原子层

    注意：这个操作是 in-place 的，会直接修改传入的 model。
    """
    encoder_layers = model.bert.encoder.layer  # type: ignore[attr-defined]
    mask_cpu = mask.detach().cpu().view(-1).tolist()

    assert len(units) == len(mask_cpu), "units 和 mask 长度不一致"

    for u, m in zip(units, mask_cpu):
        if m >= 0.5:  # 保留
            continue

        layer = encoder_layers[u.layer_idx]
        if u.unit_type == "attn":
            layer.attention = IdentityBertAttention()
        elif u.unit_type == "ffn":
            # 完全跳过 FFN：intermediate 不再算，output 直接回传 residual
            layer.intermediate = IdentityBertIntermediate()
            layer.output = IdentityBertOutput()

        else:
            raise ValueError(f"Unknown unit_type: {u.unit_type}")




# ============================================================
# Token building for BERT (基于 CLS)
# ============================================================

def build_bert_unit_tokens_for_batch(
    student: nn.Module,
    collector: BertCLSCollector,
    summarizer: CLSSummarizer,
    token_proj: TokenProjector,
    static_feats: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    对一个 batch：
        - 用 CLSCollector 收集每一层 attention / FFN 的 CLS_in / CLS_out；
        - 用 CLSSummarizer 得到每个 unit 的 summary；
        - 再用 TokenProjector 拼上 static features 得到 [1, N, token_dim] 的 tokens。
    """
    student.eval()
    collector.clear()

    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "token_type_ids": batch["token_type_ids"].to(device),
    }

    with torch.no_grad():
        _ = student(**inputs)

    records = collector.get_records()
    attn_records: List = records["attn"]
    ffn_records: List = records["ffn"]

    # 这里假设 utils 里 estimate_bert_layer_flops 构造 units 的顺序是：
    # layer0-attn, layer0-ffn, layer1-attn, layer1-ffn, ...
    # static_feats 与 units 是一一对应的。
    N = static_feats.size(0)
    hidden_size = attn_records[0].cls_in.size(-1)

    summaries: List[torch.Tensor] = []

    # 建表： (layer_idx, unit_type) → CLSPairRecord
    attn_map = {(r.layer_idx, "attn"): r for r in attn_records}
    ffn_map = {(r.layer_idx, "ffn"): r for r in ffn_records}

    for idx in range(N):
        # 根据 static_feats 的构造方式，这里需要知道 units 对应的 layer 和 type。
        # 为了避免传 units 进来，这里简单用 index // 2 / index % 2 来解码：
        # 不过更安全的方式是：直接根据 static_feats 还原不方便，还是传 units。
        # 所以我们改为：static_feats 之外再传一个 units 进来。
        raise RuntimeError(
            "build_bert_unit_tokens_for_batch 需要 units 信息（layer_idx, unit_type），"
            "请调用 build_bert_unit_tokens_for_batch_with_units 替代。"
        )


def build_bert_unit_tokens_for_batch_with_units(
    student: nn.Module,
    collector: BertCLSCollector,
    summarizer: CLSSummarizer,
    token_proj: TokenProjector,
    static_feats: torch.Tensor,
    units: List[LayerUnitInfo],
    batch: Dict[str, torch.Tensor],
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    带 units 信息的版本，真正使用的函数。

    Returns:
        tokens: [1, N, token_dim]
    """
    student.eval()
    collector.clear()

    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "token_type_ids": batch["token_type_ids"].to(device),
    }

    with torch.no_grad():
        _ = student(**inputs)

    records = collector.get_records()
    attn_records: List = records["attn"]
    ffn_records: List = records["ffn"]

    attn_map = {(r.layer_idx, "attn"): r for r in attn_records}
    ffn_map = {(r.layer_idx, "ffn"): r for r in ffn_records}

    summaries: List[torch.Tensor] = []

    for u in units:
        if u.unit_type == "attn":
            r = attn_map[(u.layer_idx, "attn")]
        else:
            r = ffn_map[(u.layer_idx, "ffn")]

        cls_in = r.cls_in.to(device)   # [B, H]
        cls_out = r.cls_out.to(device) # [B, H]

        # summarizer 对 batch 内每个样本做一遍，然后再做 mean pooling，得到 1 x summary_dim
        summary_batch = summarizer(cls_in, cls_out)  # [B, summary_dim]
        summary_vec = summary_batch.mean(dim=0, keepdim=True)  # [1, summary_dim]

        summaries.append(summary_vec)

    summary_all = torch.cat(summaries, dim=0)  # [N, summary_dim]
    tokens = token_proj(summary_all, static_feats)  # [1, N, token_dim]
    return tokens


# ============================================================
# Phase 1: Train policy encoder
# ============================================================

def train_layer_policy_encoder(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    units: List[LayerUnitInfo],
    prunable_flops: torch.Tensor,
    writer: SummaryWriter,
) -> Tuple[CLSSummarizer, TokenProjector, CompressionPolicyEncoder, torch.Tensor]:
    """
    训练基于 CLS 的 layer-wise policy encoder。
    只更新 summarizer / token_proj / encoder，teacher / student 冻结。
    """
    device = DEVICE
    teacher.eval()
    student.eval()

    # 冻结 teacher / student 参数
    for p in teacher.parameters():
        p.requires_grad_(False)
    for p in student.parameters():
        p.requires_grad_(False)

    hidden_size = int(student.config.hidden_size)

    summarizer = CLSSummarizer(hidden_size=hidden_size, summary_dim=SUMMARY_DIM).to(device)
    token_proj = TokenProjector(summary_dim=SUMMARY_DIM, static_dim=STATIC_DIM, token_dim=TOKEN_DIM).to(device)
    encoder = CompressionPolicyEncoder(token_dim=TOKEN_DIM, num_layers=2, num_heads=4, ff_dim=256, dropout=0.1).to(device)

    static_feats = build_static_features(units, device=device)  # [N, 4]
    collector = BertCLSCollector(student)
    gating = BertUnitGating(student, units)

    params = list(summarizer.parameters()) + list(token_proj.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=ENC_LR)

    total_prunable = prunable_flops.sum().item()

    total_epochs = POLICY_WARMUP_EPOCHS + POLICY_TRAIN_EPOCHS
    global_step = 0

    for epoch in range(total_epochs):
        summarizer.train()
        token_proj.train()
        encoder.train()

        # gate temperature schedule
        t_frac = epoch / max(total_epochs - 1, 1)
        gate_temp = GATE_TEMP_START + (GATE_TEMP_END - GATE_TEMP_START) * t_frac

        pbar = tqdm(
            train_loader,
            desc=f"[PolicyEnc] Epoch {epoch+1}/{total_epochs}",
            ncols=120,
        )

        running_loss = 0.0
        running_kd = 0.0
        running_ratio = 0.0
        running_flops_ratio = 0.0
        running_batches = 0

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            # teacher logits
            with torch.no_grad():
                logits_T = forward_logits(teacher, batch)

            # 随机目标 FLOPs ratio
            target_ratio = random.uniform(MIN_RATIO, MAX_RATIO)

            # 构建 tokens (不带梯度通过 student，但有梯度通过 summarizer / token_proj)
            tokens = build_bert_unit_tokens_for_batch_with_units(
                student=student,
                collector=collector,
                summarizer=summarizer,
                token_proj=token_proj,
                static_feats=static_feats,
                units=units,
                batch=batch,
                device=device,
            )  # [1, N, token_dim]

            logits_units = encoder(tokens, target_ratio=torch.tensor(target_ratio, device=device))  # [N]
            gates = torch.sigmoid(logits_units / gate_temp)  # [N]

            # 计算当前 FLOPs ratio（仅用 prunable 部分）
            curr_flops = (gates * prunable_flops).sum()
            flops_ratio = curr_flops / max(total_prunable, 1e-8)

            # gating + KD
            gating.set_train_gates(gates)
            logits_S = forward_logits(student, batch)
            gating.clear_train_gates()

            loss_kd = kd_loss(logits_S, logits_T, T=TEMP_KD)
            loss_ratio = (flops_ratio - target_ratio) ** 2

            loss = KD_WEIGHT * loss_kd + RATIO_WEIGHT * loss_ratio

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            running_kd += loss_kd.item()
            running_ratio += loss_ratio.item()
            running_flops_ratio += float(flops_ratio.item())
            running_batches += 1
            global_step += 1

            if running_batches > 0:
                pbar.set_postfix(
                    loss=f"{running_loss / running_batches:.4f}",
                    kd=f"{running_kd / running_batches:.4f}",
                    ratio_loss=f"{running_ratio / running_batches:.4f}",
                    # flops=f"{running_flops_ratio / running_batches:.3f}",
                    # gate_T=f"{gate_temp:.2f}",
                )

            if global_step % 50 == 0:
                writer.add_scalar("encoder/train_loss", loss.item(), global_step)
                writer.add_scalar("encoder/kd_loss", loss_kd.item(), global_step)
                writer.add_scalar("encoder/ratio_loss", loss_ratio.item(), global_step)
                writer.add_scalar("encoder/flops_ratio", flops_ratio.item(), global_step)
                writer.add_scalar("encoder/gate_temp", gate_temp, global_step)

        print(
            f"[PolicyEnc] Epoch {epoch+1}/{total_epochs} "
            f"loss={running_loss / max(running_batches,1):.4f} "
            f"flops_ratio={running_flops_ratio / max(running_batches,1):.4f}"
        )

    collector.close()
    gating.remove()

    return summarizer, token_proj, encoder, static_feats


# ============================================================
# Phase 2: materialize mask (with knapsack)
# ============================================================

@torch.no_grad()
def materialize_unit_mask_for_ratio(
    teacher: nn.Module,
    encoder: CompressionPolicyEncoder,
    summarizer: CLSSummarizer,
    token_proj: TokenProjector,
    static_feats: torch.Tensor,
    units: List[LayerUnitInfo],
    train_loader: DataLoader,
    prunable_flops: torch.Tensor,
    target_ratio: float,
    num_iters: int = 20,
    max_ratio: float | None = None,
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """
    类似 imagenet 版本的 materialize_block_mask_for_ratio：
    - 用 teacher 的权重复制一个 student（或直接用 teacher 本身）收集 CLS；
    - 重复 num_iters 次，累积 encoder 输出的 soft scores；
    - 平均得到 per-unit score；
    - 用 knapsack_select_mask 得到 0/1 mask（上界用 max_ratio）。
    """
    device = DEVICE
    teacher.eval()
    encoder.eval()
    summarizer.eval()
    token_proj.eval()

    # 这里直接用 teacher 自己作为“student”，反正只是收 CLS
    student = teacher
    for p in student.parameters():
        p.requires_grad_(False)
    student.eval()
    collector = BertCLSCollector(student)

    scores_accum = torch.zeros(len(units), dtype=torch.float32, device=device)
    count = 0

    loader_iter = iter(train_loader)
    for it in range(num_iters):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        tokens = build_bert_unit_tokens_for_batch_with_units(
            student=student,
            collector=collector,
            summarizer=summarizer,
            token_proj=token_proj,
            static_feats=static_feats,
            units=units,
            batch=batch,
            device=device,
        )

        logits_units = encoder(tokens, target_ratio=torch.tensor(target_ratio, device=device))  # [N]
        probs = torch.sigmoid(logits_units)  # soft scores

        scores_accum += probs
        count += 1

    collector.close()

    avg_scores = scores_accum / max(count, 1)
    mask, used_ratio = knapsack_select_mask(
        values=avg_scores,
        costs=prunable_flops,
        target_ratio=target_ratio,
        max_ratio=max_ratio,
    )
    return mask.to(device), used_ratio, avg_scores


# ============================================================
# Phase 2: Finetune pruned model (optional)
# ============================================================

def finetune_pruned_model(
    teacher: nn.Module,
    pruned_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    writer: SummaryWriter,
    ratio_tag: str,
) -> Tuple[nn.Module, float]:
    """
    对“逻辑 pruned”的 BERT 做 1 epoch KD finetune。
    """
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
        help="Path to QQP BERT teacher checkpoint (state_dict or {'model': ...}).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints_qqp_bert_layer_knapsack",
        help="Output directory for encoder & pruned models.",
    )
    parser.add_argument(
        "--do_finetune",
        action="store_true",
        help="If set, finetune pruned models after building masks.",
    )
    parser.add_argument(
        "--retrain_encoder",
        action="store_true",
        help="If set, ignore existing encoder ckpt and retrain from scratch.",
    )
    args = parser.parse_args()

    OUT_DIR = args.out_dir
    ENCODER_CKPT = os.path.join(OUT_DIR, "qqp_bert_layer_encoder.pth")
    RESULTS_FILE = os.path.join(OUT_DIR, "qqp_bert_layer_results.json")
    TB_LOG_DIR = os.path.join(OUT_DIR, "tb_logs")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    # 写一份简单的 description
    description = (
        "QQP BERT layer-wise pruning with CLS-based encoder and knapsack.\n"
        f"KD_WEIGHT={KD_WEIGHT}, RATIO_WEIGHT={RATIO_WEIGHT}, ENC_LR={ENC_LR}, "
        f"gate_T: {GATE_TEMP_START}->{GATE_TEMP_END}, "
        f"EVAL_RATIOS={EVAL_RATIOS}, FT_EPOCHS={FT_EPOCHS}."
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

    # Student (for policy training)
    student = build_bert_classifier(model_name=BERT_NAME, num_labels=2).to(DEVICE)
    student.load_state_dict(teacher.state_dict())
    print("[Student] Initialized from teacher weights.")

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

    # Phase 1: Encoder training / loading
    if os.path.isfile(ENCODER_CKPT) and not args.retrain_encoder:
        ckpt = torch.load(ENCODER_CKPT, map_location=DEVICE)
        hidden_size = int(teacher.config.hidden_size)
        summarizer = CLSSummarizer(hidden_size=hidden_size, summary_dim=SUMMARY_DIM).to(DEVICE)
        token_proj = TokenProjector(summary_dim=SUMMARY_DIM, static_dim=STATIC_DIM, token_dim=TOKEN_DIM).to(DEVICE)
        encoder = CompressionPolicyEncoder(token_dim=TOKEN_DIM, num_layers=2, num_heads=4, ff_dim=256, dropout=0.1).to(DEVICE)

        summarizer.load_state_dict(ckpt["summarizer"])
        token_proj.load_state_dict(ckpt["token_proj"])
        encoder.load_state_dict(ckpt["encoder"])

        static_feats = ckpt.get("static_feats", build_static_features(units, device=DEVICE))
        print(f"Loaded encoder from {ENCODER_CKPT}")
    else:
        print("Training encoder from scratch.")
        summarizer, token_proj, encoder, static_feats = train_layer_policy_encoder(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            units=units,
            prunable_flops=prunable_costs,
            writer=writer,
        )

        torch.save(
            {
                "summarizer": summarizer.state_dict(),
                "token_proj": token_proj.state_dict(),
                "encoder": encoder.state_dict(),
                "static_feats": static_feats.cpu(),
                "prunable_costs": prunable_costs.cpu(),
                "fixed_flops": fixed_flops,
                "total_flops_full": total_flops_full,
            },
            ENCODER_CKPT,
        )
        print(f"Saved encoder checkpoint to {ENCODER_CKPT}")

    # Phase 2: evaluate several pruning ratios
    results = {
        "prunable_costs_per_unit": prunable_costs.detach().cpu().tolist(),
        "fixed_flops": fixed_flops,
        "total_flops_full": total_flops_full,
        "ratios": [],
    }

    for idx, ratio in enumerate(EVAL_RATIOS):
        print(f"\n=== Evaluating target prunable ratio {ratio:.3f} ===")

        # 为当前 ratio 计算一个更“宽松”的上界（保持你原来的逻辑）
        if idx < len(EVAL_RATIOS) - 1:
            flex_upper_ratio = 0.5 * (ratio + EVAL_RATIOS[idx + 1])
        else:
            flex_upper_ratio = 0.5 * (ratio + 1.0)

        mask, actual_ratio_knap, avg_scores = materialize_unit_mask_for_ratio(
            teacher=teacher,
            encoder=encoder,
            summarizer=summarizer,
            token_proj=token_proj,
            static_feats=static_feats.to(DEVICE),
            units=units,
            train_loader=train_loader,
            prunable_flops=prunable_costs,
            target_ratio=ratio,
            num_iters=20,
            max_ratio=flex_upper_ratio,
        )

        kept_prunable = float((mask * prunable_costs).sum().item())
        total_prunable = float(prunable_costs.sum().item())
        prunable_ratio = kept_prunable / max(total_prunable, 1e-8)
        total_kept = fixed_flops + kept_prunable
        total_ratio = total_kept / max(total_flops_full, 1e-8)

        # print("\n--- Per-unit soft scores & binary mask (layer_idx, unit_type) ---")
        print(f"\n--- Per-unit soft scores & binary mask (ratio={ratio:.2f}, layer_idx, unit_type) ---")
        scores_list = avg_scores.detach().cpu().tolist()
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
            "avg_scores": scores_list,
            "kept_prunable_flops": kept_prunable,
            "total_prunable_flops": total_prunable,
            "total_kept_flops": total_kept,
            "total_full_flops": total_flops_full,
            "total_ratio": total_ratio,
        }

        if args.do_finetune:
            # 1) 从 teacher 拷贝一份模型
            pruned_model = build_bert_classifier(model_name=BERT_NAME, num_labels=2).to(DEVICE)
            pruned_model.load_state_dict(teacher.state_dict())

            # 2) 根据 mask 做物理剪枝（替换掉被剪 unit）
            apply_physical_bert_pruning(pruned_model, units, mask)

            # 3) 对这个物理剪枝后的模型做 1 epoch KD finetune
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
            ckpt_path = os.path.join(OUT_DIR, f"qqp_bert_layer_pruned_ratio_{ratio:.2f}.pth")
            torch.save(
                {
                    "model": pruned_model.state_dict(),
                    "mask": mask.cpu(),
                    "avg_scores": avg_scores.cpu(),
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

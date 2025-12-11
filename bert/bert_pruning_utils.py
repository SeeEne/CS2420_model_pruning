#!/usr/bin/env python
# coding=utf-8

"""
Utilities for layer-wise (attention / FFN) pruning of BERT on sequence classification tasks.

设计目标：
- 适配 HuggingFace Transformers 的 BERT 结构；
- 每个“layer unit”指代一个 attention 子层 或 一个 FFN 子层；
- 提供：
    - 随机种子控制；
    - FLOPs 估算（按 unit 拆分）；
    - CLS 向量收集（每个 unit 的输入 / 输出 CLS）；
    - CLS summarizer (MLP)；
    - token 投影器；
    - 压缩策略编码器（Transformer encoder）；
    - 0/1 背包求 mask。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Randomness utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for Python, NumPy and PyTorch (CPU & CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保持行为确定（可能稍慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# FLOPs estimation for BERT layer-wise pruning
# ============================================================

@dataclass
class LayerUnitInfo:
    """
    描述一个可剪枝单元（unit），这里的 unit 要么是 attention 子层，要么是 FFN 子层。

    Attributes:
        index:      unit 在所有 prunable units 列表中的全局索引（0..N-1）
        layer_idx:  所在的 Transformer 层的索引（0..num_hidden_layers-1）
        unit_type:  "attn" 或 "ffn"
        cost:       该 unit 的 FLOPs 估计值（浮点数，后续用于 knapsack）
    """
    index: int
    layer_idx: int
    unit_type: str  # "attn" or "ffn"
    cost: float


def _approx_bert_attention_flops(
    seq_len: int,
    hidden_size: int,
    num_heads: int,
) -> float:
    """
    更精细的 BERT self-attention FLOPs 估计（1 MAC = 2 FLOPs）：

    记:
        L = seq_len
        H = hidden_size
        A = num_heads
        d = H / A  (每个 head 的维度)

    组成部分：
    1) Q/K/V 投影: 3 * (L * H * H) * 2
    2) attention scores: Q K^T
         每个 head: L * L * d  次 MAC
         所有 head: A * L * L * d = L * L * H
         FLOPs = 2 * L * L * H
    3) softmax (exp + sum + div): 约 4 * L * L * A FLOPs（粗略，但数量级合理）
    4) attn_weights @ V:
         同 2), 也是 2 * L * L * H
    5) output projection: (L * H * H) * 2
    """
    L = float(seq_len)
    H = float(hidden_size)
    A = float(max(num_heads, 1))

    # 1) Q/K/V projections
    qkv_flops = 3.0 * (L * H * H) * 2.0

    # 2) attention scores Q K^T
    attn_scores_flops = 2.0 * L * L * H

    # 3) softmax
    softmax_flops = 4.0 * L * L * A

    # 4) attn_weights @ V
    attn_value_flops = 2.0 * L * L * H

    # 5) output projection
    out_proj_flops = (L * H * H) * 2.0

    # 6) LayerNorm: ~6 * L * H
    ln_flops = 6.0 * L * H

    total = (
            qkv_flops
            + attn_scores_flops
            + softmax_flops
            + attn_value_flops
            + out_proj_flops
            + ln_flops
    )
    return total



def _approx_bert_ffn_flops(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
) -> float:
    """
    更精细的 BERT FFN FLOPs 估计（1 MAC = 2 FLOPs）：

    记:
        L = seq_len
        H = hidden_size
        I = intermediate_size

    组成部分：
    1) hidden -> intermediate: L * H * I 次 MAC => 2 * L * H * I
    2) GELU/激活: 约 4 * L * I FLOPs（近似）
    3) intermediate -> hidden: L * I * H 次 MAC => 2 * L * I * H
    """
    L = float(seq_len)
    H = float(hidden_size)
    I = float(intermediate_size)

    fc1_flops = 2.0 * L * H * I
    act_flops = 4.0 * L * I
    fc2_flops = 2.0 * L * I * H

    # LayerNorm: ~6 * L * H
    ln_flops = 6.0 * L * H

    total = fc1_flops + act_flops + fc2_flops + ln_flops
    return total



def estimate_bert_layer_flops(
    model: nn.Module,
    seq_len: int,
) -> Tuple[List[LayerUnitInfo], float, float]:
    """
    基于模型 config，估算在给定序列长度下 BERT 的 FLOPs，并拆出每个 attention / FFN 的成本。
    """
    cfg = model.config
    num_layers = int(cfg.num_hidden_layers)
    hidden_size = int(cfg.hidden_size)
    intermediate_size = int(cfg.intermediate_size)
    num_heads = int(cfg.num_attention_heads)
    num_labels = int(getattr(cfg, "num_labels", 2))

    # --- per-layer costs (精细版本) ---
    attn_cost = _approx_bert_attention_flops(seq_len, hidden_size, num_heads)
    ffn_cost = _approx_bert_ffn_flops(seq_len, hidden_size, intermediate_size)

    units: List[LayerUnitInfo] = []
    idx = 0
    for layer_idx in range(num_layers):
        units.append(
            LayerUnitInfo(
                index=idx,
                layer_idx=layer_idx,
                unit_type="attn",
                cost=attn_cost,
            )
        )
        idx += 1

        units.append(
            LayerUnitInfo(
                index=idx,
                layer_idx=layer_idx,
                unit_type="ffn",
                cost=ffn_cost,
            )
        )
        idx += 1

    # --- fixed FLOPs 更合理的近似 ---
    L = float(seq_len)
    H = float(hidden_size)

    # (1) Embedding 部分: 简单认为只有若干加法 + LayerNorm
    # token embedding + position + token_type: 2 次加法 => 大约 2 * L * H
    emb_add_flops = 2.0 * L * H
    # LayerNorm: mean + var + scale_shift ~ 6 * L * H
    emb_ln_flops = 6.0 * L * H
    embedding_flops = emb_add_flops + emb_ln_flops

    # (2) Pooler: 一个 FC + Tanh
    # FC: H * H 次 MAC => 2 * H * H
    # Tanh: ~4 * H FLOPs
    pooler_flops = 2.0 * H * H + 4.0 * H

    # (3) Classifier: CLS -> num_labels 的 FC
    classifier_flops = 2.0 * H * num_labels

    fixed_flops = embedding_flops + pooler_flops + classifier_flops

    prunable_flops = sum(u.cost for u in units)
    total_flops = fixed_flops + prunable_flops

    return units, fixed_flops, total_flops




def compute_flops_with_mask(
    units: List[LayerUnitInfo],
    fixed_flops: float,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    """
    根据 unit 列表和 0/1 mask 计算当前模型总 FLOPs 及其相对比例。

    Args:
        units:       与 mask 对应的 LayerUnitInfo 列表，长度为 N。
        fixed_flops: 不可剪枝部分的 FLOPs。
        mask:        shape [N]，元素为 {0.0, 1.0} 的 tensor。

    Returns:
        total_flops_pruned: 当前 mask 下的总 FLOPs。
        ratio:               总 FLOPs / full FLOPs（全保留） 的比例。
    """
    assert len(units) == mask.numel(), "units 和 mask 长度必须一致"

    device = mask.device
    costs = torch.tensor([u.cost for u in units], dtype=torch.float32, device=device)
    full_prunable = costs.sum().item()
    used_prunable = (costs * mask.float()).sum().item()

    total_flops_full = fixed_flops + full_prunable
    total_flops_pruned = fixed_flops + used_prunable
    ratio = total_flops_pruned / max(total_flops_full, 1e-8)

    return total_flops_pruned, ratio


# ============================================================
# CLS collector for attention / FFN
# ============================================================

@dataclass
class CLSPairRecord:
    """
    一次 forward 中，某个 unit 的输入 / 输出 CLS 记录。
    注意：cls_in / cls_out 的形状是 [batch_size, hidden_size]。
    """
    layer_idx: int
    unit_type: str  # "attn" or "ffn"
    cls_in: torch.Tensor
    cls_out: torch.Tensor


class BertCLSCollector:
    """
    在 HuggingFace BERT 模型上注册 forward hooks，用于收集：
    - 每一层 attention 子层的输入 / 输出 CLS 向量；
    - 每一层 FFN 子层的输入 / 输出 CLS 向量。

    使用方式：
        model = BertForSequenceClassification(...)
        collector = BertCLSCollector(model)

        for batch in dataloader:
            collector.clear()
            outputs = model(**batch)
            records = collector.get_records()
            # records["attn"] / records["ffn"] 是 CLSPairRecord 列表
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.attn_records: List[CLSPairRecord] = []
        self.ffn_records: List[CLSPairRecord] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        在 model.bert.encoder.layer[i].attention 和 .output 上注册 forward hook。
        """
        # 假定是标准 BERT 结构：model.bert.encoder.layer 是 ModuleList[BertLayer]
        encoder_layers = self.model.bert.encoder.layer  # type: ignore[attr-defined]

        for layer_idx, layer in enumerate(encoder_layers):
            # attention 子层 hook
            def attn_hook(module, inputs, outputs, layer_idx=layer_idx):
                # inputs: (hidden_states, attention_mask, ...)
                # outputs: attention_output (tensor 或 tuple[0])
                hidden_states = inputs[0]  # [B, L, H]
                out = outputs[0] if isinstance(outputs, tuple) else outputs  # [B, L, H]

                cls_in = hidden_states[:, 0, :].detach()
                cls_out = out[:, 0, :].detach()

                self.attn_records.append(
                    CLSPairRecord(
                        layer_idx=layer_idx,
                        unit_type="attn",
                        cls_in=cls_in,
                        cls_out=cls_out,
                    )
                )

            h1 = layer.attention.register_forward_hook(attn_hook)

            # FFN 子层 hook（在 BertLayer.output 上）
            def ffn_hook(module, inputs, outputs, layer_idx=layer_idx):
                """
                BertOutput.forward(self, hidden_states, input_tensor):
                    hidden_states: intermediate_output
                    input_tensor: attention_output
                    output: LayerNorm(hidden_states + input_tensor)

                我们把 input_tensor 当作 FFN 子层的“输入”，output 当作“输出”。
                """
                # inputs: (hidden_states, input_tensor)
                # outputs: layer_output tensor
                input_tensor = inputs[1]  # attention_output: [B, L, H]
                out = outputs[0] if isinstance(outputs, tuple) else outputs  # [B, L, H]

                cls_in = input_tensor[:, 0, :].detach()
                cls_out = out[:, 0, :].detach()

                self.ffn_records.append(
                    CLSPairRecord(
                        layer_idx=layer_idx,
                        unit_type="ffn",
                        cls_in=cls_in,
                        cls_out=cls_out,
                    )
                )

            h2 = layer.output.register_forward_hook(ffn_hook)

            self._handles.extend([h1, h2])

    def clear(self) -> None:
        """
        清除当前已收集的记录（用于新一批数据）。
        """
        self.attn_records.clear()
        self.ffn_records.clear()

    def get_records(self) -> Dict[str, List[CLSPairRecord]]:
        """
        返回当前收集到的所有记录。
        """
        return {
            "attn": list(self.attn_records),
            "ffn": list(self.ffn_records),
        }

    def close(self) -> None:
        """
        移除所有 hooks，释放资源。
        """
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ============================================================
# Summarizer / Token projector / Policy encoder
# ============================================================

class CLSSummarizer(nn.Module):
    """
    使用简单 MLP 将 (CLS_in, CLS_out) 映射到一个 summary 向量。

    输入:
        cls_in:  [N, H]
        cls_out: [N, H]
    输出:
        summary: [N, summary_dim]
    """

    def __init__(
        self,
        hidden_size: int,
        summary_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        input_dim = 2 * hidden_size

        if num_layers == 1:
            self.mlp = nn.Linear(input_dim, summary_dim)
        else:
            mid_dim = max(summary_dim * 2, summary_dim + 16)
            layers: List[nn.Module] = [
                nn.Linear(input_dim, mid_dim),
                nn.GELU(),
                nn.Linear(mid_dim, summary_dim),
            ]
            self.mlp = nn.Sequential(*layers)

    def forward(self, cls_in: torch.Tensor, cls_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_in:  [N, H]
            cls_out: [N, H]

        Returns:
            summary: [N, summary_dim]
        """
        assert cls_in.shape == cls_out.shape, "cls_in / cls_out 形状必须一致"
        x = torch.cat([cls_in, cls_out], dim=-1)
        return self.mlp(x)


def build_static_features(
    units: List[LayerUnitInfo],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    为每个 unit 生成简单的静态特征向量。

    设计的 features：
        - layer_idx_norm: layer_idx / (num_layers - 1)
        - is_attn: 1.0 if unit_type == "attn" else 0.0
        - is_ffn: 1.0 if unit_type == "ffn" else 0.0
        - cost_log_norm: log(cost) / log(max_cost)

    Args:
        units:  LayerUnitInfo 列表。
        device: 可选，将特征 tensor 放到指定 device。

    Returns:
        static_feats: [N, 4] 的 tensor。
    """
    N = len(units)
    num_layers = max(u.layer_idx for u in units) + 1 if units else 1
    max_cost = max(u.cost for u in units) if units else 1.0

    layer_idx_norm = []
    is_attn = []
    is_ffn = []
    cost_log_norm = []

    for u in units:
        layer_idx_norm.append(u.layer_idx / max(num_layers - 1, 1))
        is_attn.append(1.0 if u.unit_type == "attn" else 0.0)
        is_ffn.append(1.0 if u.unit_type == "ffn" else 0.0)
        log_c = math.log(max(u.cost, 1e-6))
        log_max = math.log(max(max_cost, 1e-6))
        cost_log_norm.append(log_c / log_max if log_max > 0 else 0.0)

    feats = torch.tensor(
        list(zip(layer_idx_norm, is_attn, is_ffn, cost_log_norm)),
        dtype=torch.float32,
        device=device,
    )
    return feats


class TokenProjector(nn.Module):
    """
    将 [summary, static_features] 映射到 token 表示，用于喂给策略 encoder。

    输入:
        summary:        [N, summary_dim]
        static_feats:   [N, static_dim]
    输出:
        tokens:         [1, N, token_dim]   # 1 是 batch 维（统一用在 Transformer encoder 上）
    """

    def __init__(
        self,
        summary_dim: int,
        static_dim: int,
        token_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = summary_dim + static_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(
        self,
        summary: torch.Tensor,
        static_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            summary:      [N, summary_dim]
            static_feats: [N, static_dim]

        Returns:
            tokens: [1, N, token_dim]
        """
        assert summary.shape[0] == static_feats.shape[0], "summary 和 static_feats 的 N 必须一致"
        x = torch.cat([summary, static_feats], dim=-1)
        tokens = self.proj(x)  # [N, token_dim]
        # transformer encoder 习惯形状 [S, B, C]，这里我们用 B=1，在外面再 permute。
        tokens = tokens.unsqueeze(0)  # [1, N, token_dim]
        return tokens


class CompressionPolicyEncoder(nn.Module):
    """
    使用 TransformerEncoder 对 “budget token + layer tokens” 编码，
    输出每个 unit 的 score（logits），可用于后续的 gating / knapsack。

    输入:
        tokens:       [1, N, token_dim]，来自 TokenProjector。
        target_ratio: [batch] 或标量张量，表示目标 FLOPs ratio（0~1）。

    输出:
        logits: [N] 的 score。
    """

    def __init__(
        self,
        token_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # 我们使用 [S, B, C]
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # budget token embedding，从 scalar ratio 映射到 token_dim
        self.budget_mlp = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        # head 把每个 unit 的 token 映射到一个标量 score
        self.head = nn.Linear(token_dim, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        target_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens:       [1, N, token_dim]
            target_ratio: 标量或 shape [*]，在此会被平均后视作一个标量。

        Returns:
            logits: [N]，每个 unit 的 score。
        """
        assert tokens.dim() == 3 and tokens.size(0) == 1, "tokens 形状应为 [1, N, token_dim]"
        _, N, C = tokens.shape
        device = tokens.device

        # 归一化 ratio 到 [0,1]，简单 clip
        if not torch.is_tensor(target_ratio):
            target_ratio = torch.tensor(float(target_ratio), dtype=torch.float32, device=device)
        else:
            target_ratio = target_ratio.to(device=device, dtype=torch.float32)
        ratio_scalar = torch.clamp(target_ratio.mean(), 0.0, 1.0).view(1, 1)  # [1,1]

        budget_token = self.budget_mlp(ratio_scalar)  # [1, token_dim]
        budget_token = budget_token.unsqueeze(0)      # [1, 1, token_dim]

        # 拼接成 [S, B, C]，S = 1 + N, B = 1
        tokens_all = torch.cat([budget_token, tokens], dim=1)  # [1, 1+N, C]
        tokens_all = tokens_all.permute(1, 0, 2)               # [1+N, 1, C]

        encoded = self.encoder(tokens_all)  # [1+N, 1, C]

        # 去掉第一个 budget token，只取后面 N 个 unit tokens
        encoded_units = encoded[1:, 0, :]  # [N, C]

        logits = self.head(encoded_units).squeeze(-1)  # [N]
        return logits


# ============================================================
# 0/1 Knapsack for mask selection
# ============================================================

def knapsack_select_mask(
    values: torch.Tensor,
    costs: torch.Tensor,
    target_ratio: float,
    max_ratio: Optional[float] = None,
) -> Tuple[torch.Tensor, float]:
    """
    0/1 背包：在 FLOPs 预算约束下，从 units 中挑选子集，使得 sum(values) 最大。

    Args:
        values:       [N]，每个 unit 的“价值”（score），越大越重要。
        costs:        [N]，每个 unit 的 FLOPs。
        target_ratio: 目标总 FLOPs 占比（0~1），以 “保留的总 FLOPs / 全保留 FLOPs” 计算。
        max_ratio:    可选，允许的最大 FLOPs 占比（若不指定，则 == target_ratio）。

    Returns:
        mask:         [N]，0/1 浮点 tensor，表示是否保留该 unit。
        used_ratio:   实际使用的 FLOPs 占比（0~1）。
    """
    assert values.shape == costs.shape, "values / costs 形状必须一致"
    device = values.device

    # 转 CPU 做 DP 会更方便
    v_cpu = values.detach().cpu().float().numpy()
    c_cpu = costs.detach().cpu().float().numpy()

    N = v_cpu.shape[0]
    total_cost = float(c_cpu.sum())
    if total_cost <= 0:
        # degenerate：直接全 0
        return torch.zeros_like(values, dtype=torch.float32), 0.0

    # 预算：总 FLOPs 不能超过 max_ratio * total_cost
    if max_ratio is None:
        max_ratio = target_ratio
    max_ratio = max(0.0, min(1.0, float(max_ratio)))
    capacity = max_ratio * total_cost

    # 将 costs 量化为整数，方便 DP
    min_cost = float(c_cpu.min())
    if min_cost <= 0:
        # 防止异常，shift 一点点
        min_cost = max(min_cost, 1e-6)
    scaled_costs = np.ceil(c_cpu / min_cost).astype(np.int32)
    capacity_int = int(capacity / min_cost)

    # capacity_int 太大时做个上限，保证 DP 复杂度可控
    capacity_int = max(1, min(capacity_int, 5000))

    # DP 表：dp[i][w] = 前 i 个物品中，在容量 w 内的最大 value
    dp = np.zeros((N + 1, capacity_int + 1), dtype=np.float32)
    keep = np.zeros((N + 1, capacity_int + 1), dtype=np.int8)

    for i in range(1, N + 1):
        cost_i = int(scaled_costs[i - 1])
        val_i = float(v_cpu[i - 1])
        for w in range(capacity_int + 1):
            # 不选第 i 个
            best_without = dp[i - 1, w]
            best_with = -1e9
            if cost_i <= w:
                cand = dp[i - 1, w - cost_i] + val_i
                best_with = cand

            if best_with > best_without:
                dp[i, w] = best_with
                keep[i, w] = 1
            else:
                dp[i, w] = best_without
                keep[i, w] = 0

    # 回溯找出选择的物品
    w = int(capacity_int)
    chosen = np.zeros(N, dtype=np.int8)
    for i in range(N, 0, -1):
        if keep[i, w] == 1:
            chosen[i - 1] = 1
            w -= int(scaled_costs[i - 1])

    mask = torch.from_numpy(chosen).to(device=device, dtype=torch.float32)

    used_cost = float((c_cpu * chosen).sum())
    used_ratio = used_cost / total_cost

    # 如果 used_ratio 仍然明显小于 target_ratio，可以视情况在主逻辑中做调整；
    # 这里仅返回当前选择结果。
    return mask, used_ratio

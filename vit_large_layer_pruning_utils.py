#!/usr/bin/env python
# coding=utf-8

"""
Utilities for layer-wise (attention / MLP) pruning of ViT-Large on image classification tasks.

Design goals:
- Adapt HuggingFace BERT layer-wise pruning approach to timm ViT-Large
- Each "layer unit" refers to either an attention sublayer or an MLP sublayer
- ViT-Large has 24 blocks -> 48 prunable units (24 attn + 24 mlp)
- Provides:
    - Random seed control
    - FLOPs estimation (per unit: attn vs mlp)
    - CLS vector collection (input/output CLS for each unit)
    - CLS summarizer (MLP)
    - Token projector
    - Compression policy encoder (Transformer encoder)
    - 0/1 knapsack for mask selection
    - Gating controller for soft/hard pruning
    - Physical pruning (replace with Identity modules)
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
    """Set random seed for Python, NumPy and PyTorch (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# FLOPs estimation for ViT layer-wise pruning
# ============================================================

@dataclass
class LayerUnitInfo:
    """
    Describes a prunable unit (either attention or MLP sublayer).

    Attributes:
        index:      Global index in all prunable units (0..N-1)
        block_idx:  Which transformer block (0..num_blocks-1)
        unit_type:  "attn" or "mlp"
        cost:       FLOPs estimate for this unit
    """
    index: int
    block_idx: int
    unit_type: str  # "attn" or "mlp"
    cost: float


def _approx_vit_attention_flops(
    seq_len: int,
    embed_dim: int,
    num_heads: int,
) -> float:
    """
    Approximate ViT self-attention FLOPs (1 MAC = 2 FLOPs).

    Components:
    1) QKV projection: 3 * (L * D * D) * 2
    2) Attention scores Q @ K^T: 2 * L * L * D
    3) Softmax: ~4 * L * L * num_heads
    4) Attention @ V: 2 * L * L * D
    5) Output projection: (L * D * D) * 2
    6) LayerNorm (pre-norm): ~6 * L * D
    """
    L = float(seq_len)
    D = float(embed_dim)
    H = float(max(num_heads, 1))

    qkv_flops = 3.0 * (L * D * D) * 2.0
    attn_scores_flops = 2.0 * L * L * D
    softmax_flops = 4.0 * L * L * H
    attn_value_flops = 2.0 * L * L * D
    out_proj_flops = (L * D * D) * 2.0
    ln_flops = 6.0 * L * D

    total = (
        qkv_flops
        + attn_scores_flops
        + softmax_flops
        + attn_value_flops
        + out_proj_flops
        + ln_flops
    )
    return total


def _approx_vit_mlp_flops(
    seq_len: int,
    embed_dim: int,
    mlp_ratio: float = 4.0,
) -> float:
    """
    Approximate ViT MLP (FFN) FLOPs (1 MAC = 2 FLOPs).

    Components:
    1) FC1: embed_dim -> mlp_dim: L * D * (D * mlp_ratio) * 2
    2) GELU activation: ~4 * L * mlp_dim
    3) FC2: mlp_dim -> embed_dim: L * (D * mlp_ratio) * D * 2
    4) LayerNorm (pre-norm): ~6 * L * D
    """
    L = float(seq_len)
    D = float(embed_dim)
    mlp_dim = D * mlp_ratio

    fc1_flops = 2.0 * L * D * mlp_dim
    act_flops = 4.0 * L * mlp_dim
    fc2_flops = 2.0 * L * mlp_dim * D
    ln_flops = 6.0 * L * D

    total = fc1_flops + act_flops + fc2_flops + ln_flops
    return total


def estimate_vit_layer_flops(
    model: nn.Module,
    img_size: int = 224,
    patch_size: int = 16,
) -> Tuple[List[LayerUnitInfo], float, float]:
    """
    Estimate FLOPs for ViT model, split by attention/MLP units.

    Args:
        model: timm ViT model
        img_size: Input image size
        patch_size: Patch size

    Returns:
        units: List of LayerUnitInfo for all prunable units
        fixed_flops: FLOPs for non-prunable parts (patch embed, norm, head)
        total_flops: Total FLOPs when all units are kept
    """
    num_blocks = len(model.blocks)
    embed_dim = model.embed_dim
    num_heads = model.blocks[0].attn.num_heads

    # Sequence length = num_patches + 1 (CLS token)
    num_patches = (img_size // patch_size) ** 2
    seq_len = num_patches + 1  # +1 for CLS token

    # MLP ratio (typically 4.0 for ViT)
    mlp_hidden = model.blocks[0].mlp.fc1.out_features
    mlp_ratio = mlp_hidden / embed_dim

    # Per-unit costs
    attn_cost = _approx_vit_attention_flops(seq_len, embed_dim, num_heads)
    mlp_cost = _approx_vit_mlp_flops(seq_len, embed_dim, mlp_ratio)

    units: List[LayerUnitInfo] = []
    idx = 0
    for block_idx in range(num_blocks):
        # Attention unit
        units.append(LayerUnitInfo(
            index=idx,
            block_idx=block_idx,
            unit_type="attn",
            cost=attn_cost,
        ))
        idx += 1

        # MLP unit
        units.append(LayerUnitInfo(
            index=idx,
            block_idx=block_idx,
            unit_type="mlp",
            cost=mlp_cost,
        ))
        idx += 1

    # Fixed FLOPs (patch embedding, final norm, classification head)
    L = float(seq_len)
    D = float(embed_dim)
    num_classes = model.head.out_features

    # Patch embedding: conv projection
    # Input: (B, 3, H, W) -> (B, num_patches, embed_dim)
    patch_embed_flops = 2.0 * 3 * (patch_size ** 2) * embed_dim * num_patches

    # Final LayerNorm
    final_ln_flops = 6.0 * L * D

    # Classification head: embed_dim -> num_classes
    head_flops = 2.0 * D * num_classes

    fixed_flops = patch_embed_flops + final_ln_flops + head_flops
    prunable_flops = sum(u.cost for u in units)
    total_flops = fixed_flops + prunable_flops

    return units, fixed_flops, total_flops


def compute_flops_with_mask(
    units: List[LayerUnitInfo],
    fixed_flops: float,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute total FLOPs given a 0/1 mask.

    Args:
        units: LayerUnitInfo list (length N)
        fixed_flops: Non-prunable FLOPs
        mask: Shape [N], elements {0.0, 1.0}

    Returns:
        total_flops_pruned: Current FLOPs with mask applied
        ratio: Pruned FLOPs / Full FLOPs
    """
    assert len(units) == mask.numel(), "units and mask length must match"

    device = mask.device
    costs = torch.tensor([u.cost for u in units], dtype=torch.float32, device=device)
    full_prunable = costs.sum().item()
    used_prunable = (costs * mask.float()).sum().item()

    total_flops_full = fixed_flops + full_prunable
    total_flops_pruned = fixed_flops + used_prunable
    ratio = total_flops_pruned / max(total_flops_full, 1e-8)

    return total_flops_pruned, ratio


# ============================================================
# CLS collector for attention / MLP in ViT
# ============================================================

@dataclass
class CLSPairRecord:
    """
    Records input/output CLS vectors for a unit during forward pass.
    cls_in/cls_out shape: [batch_size, embed_dim]
    """
    block_idx: int
    unit_type: str  # "attn" or "mlp"
    cls_in: torch.Tensor
    cls_out: torch.Tensor


class ViTCLSCollector:
    """
    Registers forward hooks on ViT to collect CLS vectors for each unit.

    Usage:
        model = timm.create_model('vit_large_patch16_224', ...)
        collector = ViTCLSCollector(model)

        for batch in dataloader:
            collector.clear()
            outputs = model(batch)
            records = collector.get_records()
            # records["attn"] / records["mlp"] are CLSPairRecord lists
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.attn_records: List[CLSPairRecord] = []
        self.mlp_records: List[CLSPairRecord] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on each block's attn and mlp."""
        for block_idx, block in enumerate(self.model.blocks):
            # Attention hook
            def attn_hook(module, inputs, outputs, block_idx=block_idx):
                # timm ViT: block.attn takes x, returns attn_output
                # Input x has shape [B, L, D]
                x_in = inputs[0]  # [B, L, D]
                x_out = outputs   # [B, L, D]

                cls_in = x_in[:, 0, :].detach()   # [B, D]
                cls_out = x_out[:, 0, :].detach() # [B, D]

                self.attn_records.append(CLSPairRecord(
                    block_idx=block_idx,
                    unit_type="attn",
                    cls_in=cls_in,
                    cls_out=cls_out,
                ))

            h1 = block.attn.register_forward_hook(attn_hook)

            # MLP hook
            def mlp_hook(module, inputs, outputs, block_idx=block_idx):
                # timm ViT: block.mlp takes x, returns mlp_output
                x_in = inputs[0]  # [B, L, D]
                x_out = outputs   # [B, L, D]

                cls_in = x_in[:, 0, :].detach()   # [B, D]
                cls_out = x_out[:, 0, :].detach() # [B, D]

                self.mlp_records.append(CLSPairRecord(
                    block_idx=block_idx,
                    unit_type="mlp",
                    cls_in=cls_in,
                    cls_out=cls_out,
                ))

            h2 = block.mlp.register_forward_hook(mlp_hook)

            self._handles.extend([h1, h2])

    def clear(self) -> None:
        """Clear collected records for new batch."""
        self.attn_records.clear()
        self.mlp_records.clear()

    def get_records(self) -> Dict[str, List[CLSPairRecord]]:
        """Return all collected records."""
        return {
            "attn": list(self.attn_records),
            "mlp": list(self.mlp_records),
        }

    def close(self) -> None:
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ============================================================
# Summarizer / Token projector / Policy encoder
# ============================================================

class CLSSummarizer(nn.Module):
    """
    MLP to map (CLS_in, CLS_out) to a summary vector.

    Input:
        cls_in:  [N, D]
        cls_out: [N, D]
    Output:
        summary: [N, summary_dim]
    """

    def __init__(
        self,
        embed_dim: int,
        summary_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        input_dim = 2 * embed_dim

        if num_layers == 1:
            self.mlp = nn.Linear(input_dim, summary_dim)
        else:
            mid_dim = max(summary_dim * 2, summary_dim + 16)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, mid_dim),
                nn.GELU(),
                nn.Linear(mid_dim, summary_dim),
            )

    def forward(self, cls_in: torch.Tensor, cls_out: torch.Tensor) -> torch.Tensor:
        assert cls_in.shape == cls_out.shape
        x = torch.cat([cls_in, cls_out], dim=-1)
        return self.mlp(x)


def build_static_features(
    units: List[LayerUnitInfo],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate static features for each unit.

    Features (4-dim):
        - block_idx_norm: block_idx / (num_blocks - 1)
        - is_attn: 1.0 if "attn" else 0.0
        - is_mlp: 1.0 if "mlp" else 0.0
        - cost_log_norm: log(cost) / log(max_cost)

    Returns:
        static_feats: [N, 4]
    """
    N = len(units)
    num_blocks = max(u.block_idx for u in units) + 1 if units else 1
    max_cost = max(u.cost for u in units) if units else 1.0

    block_idx_norm = []
    is_attn = []
    is_mlp = []
    cost_log_norm = []

    for u in units:
        block_idx_norm.append(u.block_idx / max(num_blocks - 1, 1))
        is_attn.append(1.0 if u.unit_type == "attn" else 0.0)
        is_mlp.append(1.0 if u.unit_type == "mlp" else 0.0)
        log_c = math.log(max(u.cost, 1e-6))
        log_max = math.log(max(max_cost, 1e-6))
        cost_log_norm.append(log_c / log_max if log_max > 0 else 0.0)

    feats = torch.tensor(
        list(zip(block_idx_norm, is_attn, is_mlp, cost_log_norm)),
        dtype=torch.float32,
        device=device,
    )
    return feats


class TokenProjector(nn.Module):
    """
    Project [summary, static_features] to token representation.

    Input:
        summary:      [N, summary_dim]
        static_feats: [N, static_dim]
    Output:
        tokens:       [1, N, token_dim]
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
        assert summary.shape[0] == static_feats.shape[0]
        x = torch.cat([summary, static_feats], dim=-1)
        tokens = self.proj(x)  # [N, token_dim]
        tokens = tokens.unsqueeze(0)  # [1, N, token_dim]
        return tokens


class CompressionPolicyEncoder(nn.Module):
    """
    Transformer encoder for "budget token + layer tokens".
    Outputs per-unit scores (logits) for gating/knapsack.

    Input:
        tokens:       [1, N, token_dim]
        target_ratio: Scalar or tensor, target FLOPs ratio (0~1)

    Output:
        logits: [N] scores for each unit
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
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Budget token embedding
        self.budget_mlp = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        # Output head: per-unit score
        self.head = nn.Linear(token_dim, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        target_ratio: torch.Tensor,
    ) -> torch.Tensor:
        assert tokens.dim() == 3 and tokens.size(0) == 1
        _, N, C = tokens.shape
        device = tokens.device

        if not torch.is_tensor(target_ratio):
            target_ratio = torch.tensor(float(target_ratio), dtype=torch.float32, device=device)
        else:
            target_ratio = target_ratio.to(device=device, dtype=torch.float32)
        ratio_scalar = torch.clamp(target_ratio.mean(), 0.0, 1.0).view(1, 1)

        budget_token = self.budget_mlp(ratio_scalar)  # [1, token_dim]
        budget_token = budget_token.unsqueeze(0)      # [1, 1, token_dim]

        # Concatenate: [1, 1+N, C]
        tokens_all = torch.cat([budget_token, tokens], dim=1)
        tokens_all = tokens_all.permute(1, 0, 2)  # [1+N, 1, C]

        encoded = self.encoder(tokens_all)  # [1+N, 1, C]

        # Remove budget token, take unit tokens
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
    min_attn_layers: int = 1,
    min_mlp_layers: int = 1,
    num_blocks: int = 24,
) -> Tuple[torch.Tensor, float]:
    """
    0/1 knapsack: Select subset of units to maximize sum(values) under FLOPs budget.

    Args:
        values:         [N] per-unit importance scores
        costs:          [N] per-unit FLOPs
        target_ratio:   Target FLOPs ratio (0~1)
        max_ratio:      Maximum allowed ratio (default: target_ratio)
        min_attn_layers: Minimum attention layers to keep (default: 1).
                         ViT requires attention for CLS token to aggregate patch info.
                         Set to 0 to disable this constraint.
        min_mlp_layers: Minimum MLP layers to keep (default: 1).
                        MLP provides transformation capability.
                        Set to 0 to disable this constraint.
        num_blocks:     Number of transformer blocks (default: 24 for ViT-Large)

    Returns:
        mask:       [N] 0/1 tensor
        used_ratio: Actual FLOPs ratio achieved
    """
    assert values.shape == costs.shape
    device = values.device

    v_cpu = values.detach().cpu().float().numpy()
    c_cpu = costs.detach().cpu().float().numpy()

    N = v_cpu.shape[0]
    total_cost = float(c_cpu.sum())
    if total_cost <= 0:
        return torch.zeros_like(values, dtype=torch.float32), 0.0

    if max_ratio is None:
        max_ratio = target_ratio
    max_ratio = max(0.0, min(1.0, float(max_ratio)))
    capacity = max_ratio * total_cost

    # Identify attention and MLP layer indices
    # Layout: [attn0, mlp0, attn1, mlp1, ..., attn23, mlp23]
    attn_indices = list(range(0, min(N, num_blocks * 2), 2))  # 0, 2, 4, ...
    mlp_indices = list(range(1, min(N, num_blocks * 2), 2))   # 1, 3, 5, ...

    # Quantize costs for DP
    min_cost = float(c_cpu.min())
    if min_cost <= 0:
        min_cost = max(min_cost, 1e-6)
    scaled_costs = np.ceil(c_cpu / min_cost).astype(np.int32)
    capacity_int = int(capacity / min_cost)
    capacity_int = max(1, min(capacity_int, 5000))

    # DP
    dp = np.zeros((N + 1, capacity_int + 1), dtype=np.float32)
    keep = np.zeros((N + 1, capacity_int + 1), dtype=np.int8)

    for i in range(1, N + 1):
        cost_i = int(scaled_costs[i - 1])
        val_i = float(v_cpu[i - 1])
        for w in range(capacity_int + 1):
            best_without = dp[i - 1, w]
            best_with = -1e9
            if cost_i <= w:
                best_with = dp[i - 1, w - cost_i] + val_i

            if best_with > best_without:
                dp[i, w] = best_with
                keep[i, w] = 1
            else:
                dp[i, w] = best_without
                keep[i, w] = 0

    # Backtrack
    w = int(capacity_int)
    chosen = np.zeros(N, dtype=np.int8)
    for i in range(N, 0, -1):
        if keep[i, w] == 1:
            chosen[i - 1] = 1
            w -= int(scaled_costs[i - 1])

    # Enforce minimum attention constraint
    # ViT requires at least some attention layers for CLS token to aggregate info
    # IMPORTANT: The first attention layer (index 0) is critical - it's the first
    # opportunity for CLS to gather information from patch tokens
    if min_attn_layers > 0:
        # Always force-keep the first attention layer (block 0 attention)
        # Without it, CLS token never gets information from patches
        if len(attn_indices) > 0 and chosen[attn_indices[0]] == 0:
            chosen[attn_indices[0]] = 1

        attn_chosen = sum(chosen[i] for i in attn_indices)
        if attn_chosen < min_attn_layers:
            # Find additional attention layers not chosen, sorted by score (descending)
            attn_not_chosen = [(i, v_cpu[i]) for i in attn_indices if chosen[i] == 0]
            attn_not_chosen.sort(key=lambda x: -x[1])  # Highest score first

            # Force-add attention layers until we have min_attn_layers
            need = min_attn_layers - attn_chosen
            for idx, _ in attn_not_chosen[:need]:
                chosen[idx] = 1

    # Enforce minimum MLP constraint
    # MLP provides transformation capability needed for good representations
    if min_mlp_layers > 0:
        mlp_chosen = sum(chosen[i] for i in mlp_indices)
        if mlp_chosen < min_mlp_layers:
            # Find MLP layers not chosen, sorted by score (descending)
            mlp_not_chosen = [(i, v_cpu[i]) for i in mlp_indices if chosen[i] == 0]
            mlp_not_chosen.sort(key=lambda x: -x[1])  # Highest score first

            # Force-add MLP layers until we have min_mlp_layers
            need = min_mlp_layers - mlp_chosen
            for idx, _ in mlp_not_chosen[:need]:
                chosen[idx] = 1

    mask = torch.from_numpy(chosen).to(device=device, dtype=torch.float32)
    used_cost = float((c_cpu * chosen).sum())
    used_ratio = used_cost / total_cost

    return mask, used_ratio


# ============================================================
# Gating controller for ViT
# ============================================================

class ViTUnitGating:
    """
    Apply gating to ViT attention/MLP sublayers via hooks.

    Gate behavior:
    - Training: soft gates (0~1) from encoder
    - Inference: fixed 0/1 mask for "logical pruning"

    For ViT (pre-norm architecture):
    - Attention: x = x + gate * attn(norm1(x))
    - MLP: x = x + gate * mlp(norm2(x))

    When gate=0: sublayer is skipped (identity)
    When gate=1: sublayer is fully used
    """

    def __init__(
        self,
        model: nn.Module,
        units: List[LayerUnitInfo],
    ) -> None:
        self.model = model
        self.units = units

        # (block_idx, unit_type) -> global unit index
        self.unit_index_map: Dict[Tuple[int, str], int] = {
            (u.block_idx, u.unit_type): u.index for u in units
        }

        self.train_gates: torch.Tensor | None = None
        self.fixed_mask: torch.Tensor | None = None

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _get_gate_scalar(
        self,
        block_idx: int,
        unit_type: str,
        ref_tensor: torch.Tensor,
    ) -> torch.Tensor | None:
        key = (block_idx, unit_type)
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
        """Register hooks on each block's attn and mlp."""
        for block_idx, block in enumerate(self.model.blocks):
            # Attention hook
            # timm ViT block: x = x + self.attn(self.norm1(x))
            # We hook on attn to modify its output
            def attn_hook(module, inputs, outputs, block_idx=block_idx):
                g = self._get_gate_scalar(block_idx, "attn", outputs)
                if g is None:
                    return outputs
                # Scale the attention output by gate
                # The residual connection is handled outside (in block.forward)
                # So we just scale the output here
                return g * outputs

            h1 = block.attn.register_forward_hook(attn_hook)

            # MLP hook
            def mlp_hook(module, inputs, outputs, block_idx=block_idx):
                g = self._get_gate_scalar(block_idx, "mlp", outputs)
                if g is None:
                    return outputs
                return g * outputs

            h2 = block.mlp.register_forward_hook(mlp_hook)

            self._handles.extend([h1, h2])

    def set_train_gates(self, gates: torch.Tensor) -> None:
        """Set soft gates [N] for training."""
        self.train_gates = gates

    def clear_train_gates(self) -> None:
        self.train_gates = None

    def set_fixed_mask(self, mask: torch.Tensor) -> None:
        """Set 0/1 mask [N] for logical pruning."""
        self.fixed_mask = mask.detach().float()

    def clear_fixed_mask(self) -> None:
        self.fixed_mask = None

    def remove(self) -> None:
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ============================================================
# Physical pruning: Identity sublayers
# ============================================================

class IdentityAttention(nn.Module):
    """Identity attention: returns zeros (will be added to residual, so no change)."""
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        # Accept any kwargs (like attn_mask) but ignore them
        # Return zeros so that x + attn(x) = x + 0 = x
        return torch.zeros_like(x)


class IdentityMLP(nn.Module):
    """Identity MLP: returns zeros (will be added to residual, so no change)."""
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        # Accept any kwargs but ignore them
        return torch.zeros_like(x)


def apply_physical_vit_pruning(
    model: nn.Module,
    units: List[LayerUnitInfo],
    mask: torch.Tensor,
) -> None:
    """
    Physically prune ViT by replacing sublayers with Identity modules.

    Args:
        model: timm ViT model (modified in-place)
        units: LayerUnitInfo list
        mask: [N] 0/1 tensor, 0 = prune, 1 = keep

    Note: This is in-place modification!
    """
    mask_cpu = mask.detach().cpu().view(-1).tolist()
    assert len(units) == len(mask_cpu)

    for u, m in zip(units, mask_cpu):
        if m >= 0.5:  # Keep
            continue

        block = model.blocks[u.block_idx]
        if u.unit_type == "attn":
            block.attn = IdentityAttention()
        elif u.unit_type == "mlp":
            block.mlp = IdentityMLP()
        else:
            raise ValueError(f"Unknown unit_type: {u.unit_type}")


# ============================================================
# Token building for ViT (based on CLS)
# ============================================================

def build_vit_unit_tokens(
    model: nn.Module,
    collector: ViTCLSCollector,
    summarizer: CLSSummarizer,
    token_proj: TokenProjector,
    static_feats: torch.Tensor,
    units: List[LayerUnitInfo],
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Build tokens for policy encoder from a batch.

    Args:
        model: ViT model (for forward pass to collect CLS)
        collector: ViTCLSCollector with hooks registered
        summarizer: CLSSummarizer
        token_proj: TokenProjector
        static_feats: [N, 4] static features
        units: LayerUnitInfo list
        x: Input images [B, 3, H, W]
        device: Device

    Returns:
        tokens: [1, N, token_dim]
    """
    model.eval()
    collector.clear()

    with torch.no_grad():
        _ = model(x.to(device))

    records = collector.get_records()
    attn_records = records["attn"]
    mlp_records = records["mlp"]

    # Build lookup
    attn_map = {(r.block_idx, "attn"): r for r in attn_records}
    mlp_map = {(r.block_idx, "mlp"): r for r in mlp_records}

    summaries: List[torch.Tensor] = []

    for u in units:
        if u.unit_type == "attn":
            r = attn_map[(u.block_idx, "attn")]
        else:
            r = mlp_map[(u.block_idx, "mlp")]

        cls_in = r.cls_in.to(device)   # [B, D]
        cls_out = r.cls_out.to(device) # [B, D]

        # Mean pool across batch
        summary_batch = summarizer(cls_in, cls_out)  # [B, summary_dim]
        summary_vec = summary_batch.mean(dim=0, keepdim=True)  # [1, summary_dim]
        summaries.append(summary_vec)

    summary_all = torch.cat(summaries, dim=0)  # [N, summary_dim]
    tokens = token_proj(summary_all, static_feats)  # [1, N, token_dim]
    return tokens


# ============================================================
# Loss functions
# ============================================================

def kd_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    T: float = 4.0,
) -> torch.Tensor:
    """Standard KL-based KD loss."""
    log_p_s = F.log_softmax(logits_s / T, dim=-1)
    p_t = F.softmax(logits_t / T, dim=-1)
    loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    return loss


def distillation_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    labels: torch.Tensor,
    T: float = 4.0,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard KD loss: alpha * task_loss + (1 - alpha) * kd_loss

    Returns:
        total_loss, kd_loss, task_loss
    """
    loss_kd = kd_loss(logits_s, logits_t, T)
    loss_task = F.cross_entropy(logits_s, labels)
    total_loss = alpha * loss_task + (1 - alpha) * loss_kd
    return total_loss, loss_kd, loss_task

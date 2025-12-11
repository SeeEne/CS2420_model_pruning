#!/usr/bin/env python
# coding=utf-8

"""
L2-norm based layer-wise (attention / FFN) pruning for BERT on QQP.

Important:
- layer 的定义与 qqp_bert_layer_knapsack / qqp_bert_layer_cosine_knapsack 完全一致：
    一个 unit = (layer_idx, unit_type in {attn, ffn})
- importance 定义为该 unit 参数的平均 L2 norm（sqrt(mean(w^2))），越大越重要；
- 其它流程（FLOPs 估计、knapsack + 宽容上界、finetune epoch 数、ratio 列表、
  日志与进度条、mask 打印）全部和 cosine 版本对齐。
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizerFast

from bert_pruning_utils import (
    set_seed,
    estimate_bert_layer_flops,
    LayerUnitInfo,
    knapsack_select_mask,
)  # type: ignore

# 直接复用 cosine 版本里的通用工具和 finetune / 物理剪枝实现
from qqp_bert_layer_cosine_knapsack import (  # type: ignore
    build_qqp_loaders,
    build_bert_classifier,
    forward_logits,
    evaluate_accuracy,
    kd_loss,
    IdentityBertAttention,
    IdentityBertOutput,
    IdentityBertIntermediate,
    apply_physical_bert_pruning,
    finetune_pruned_model,
)


# ============================================================
# Global config（与 cosine 版本保持一致）
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
FT_EPOCHS = 1          # 同 cosine：finetune 1 epoch
FT_LR = 3e-5

# Ratios to evaluate (on prunable part)
# 12 层 * 2 units(attn+ffn) = 24 units
EVAL_RATIOS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]


# ============================================================
# L2-norm based importance
# ============================================================

@torch.no_grad()
def compute_unit_l2_importance(
    model: nn.Module,
    units: List[LayerUnitInfo],
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    对每个 (layer_idx, unit_type) 计算参数平均 L2 norm：

        importance_i = sqrt( mean_j( w_{i,j}^2 ) )

    其中：
        - attn unit:  layer.encoder.layer[layer_idx].attention 的所有参数；
        - ffn unit:   layer.intermediate + layer.output 的所有参数。

    返回:
        importance: [N] tensor，越大表示参数幅值越大、越重要。
    """
    model.eval()
    encoder_layers = model.bert.encoder.layer  # type: ignore[attr-defined]

    scores = torch.zeros(len(units), dtype=torch.float32, device=device)

    pbar = tqdm(
        range(len(units)),
        desc="[L2] Computing per-unit parameter L2 importance",
        ncols=120,
    )

    for i, u in zip(pbar, units):
        layer = encoder_layers[u.layer_idx]

        if u.unit_type == "attn":
            params_iter = layer.attention.parameters()
        elif u.unit_type == "ffn":
            # FFN = intermediate + output
            params_iter = list(layer.intermediate.parameters()) + list(layer.output.parameters())
        else:
            raise ValueError(f"Unknown unit_type: {u.unit_type}")

        total_sq = torch.tensor(0.0, device=device)
        total_count = 0

        for p in params_iter:
            if p is None:
                continue
            w = p.detach().to(device=device, dtype=torch.float32)
            total_sq += (w * w).sum()
            total_count += w.numel()

        if total_count == 0:
            score = torch.tensor(0.0, device=device)
        else:
            mean_sq = total_sq / float(total_count)
            score = torch.sqrt(mean_sq)  # sqrt(mean(w^2))

        scores[i] = score

        if (i + 1) % 4 == 0:
            pbar.set_postfix(last_score=f"{score.item():.4e}")

    return scores


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
        default="checkpoints_qqp_bert_layer_l2",
        help="Output directory for L2-based pruning results.",
    )
    parser.add_argument(
        "--do_finetune",
        action="store_true",
        help="If set, finetune pruned models after building masks.",
    )
    args = parser.parse_args()

    OUT_DIR = args.out_dir
    RESULTS_FILE = os.path.join(OUT_DIR, "qqp_bert_layer_l2_results.json")
    TB_LOG_DIR = os.path.join(OUT_DIR, "tb_logs")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    description = (
        "QQP BERT layer-wise pruning with per-layer L2-norm importance and knapsack.\n"
        f"EVAL_RATIOS={EVAL_RATIOS}, FT_EPOCHS={FT_EPOCHS}."
    )
    with open(os.path.join(OUT_DIR, "description.txt"), "w", encoding="utf-8") as f:
        f.write(description)

    set_seed(42)
    writer = SummaryWriter(log_dir=TB_LOG_DIR)

    # Data（主要用来算 teacher acc + finetune）
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

    # FLOPs estimation（与前两个脚本一致）
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

    # L2-based importance（只做一次，所有 ratio 共用）
    importance = compute_unit_l2_importance(
        model=teacher,
        units=units,
        device=DEVICE,
    )

    # 保存一下原始 L2 importance 便于后处理
    torch.save(
        {
            "units": [(u.layer_idx, u.unit_type, float(u.cost)) for u in units],
            "importance_l2": importance.cpu(),
            "prunable_costs": prunable_costs.cpu(),
            "fixed_flops": fixed_flops,
            "total_flops_full": total_flops_full,
        },
        os.path.join(OUT_DIR, "l2_importance.pt"),
    )
    print(f"Saved L2 importance to {os.path.join(OUT_DIR, 'l2_importance.pt')}")

    # Phase 2: evaluate several pruning ratios
    results = {
        "prunable_costs_per_unit": prunable_costs.detach().cpu().tolist(),
        "fixed_flops": fixed_flops,
        "total_flops_full": total_flops_full,
        "importance_l2": importance.detach().cpu().tolist(),
        "ratios": [],
    }

    for idx, ratio in enumerate(EVAL_RATIOS):
        print(f"\n=== Evaluating target prunable ratio {ratio:.3f} ===")

        # 为当前 ratio 计算一个更“宽松”的上界（和 cosine / encoder 版本完全一致）
        if idx < len(EVAL_RATIOS) - 1:
            flex_upper_ratio = 0.5 * (ratio + EVAL_RATIOS[idx + 1])
        else:
            flex_upper_ratio = 0.5 * (ratio + 1.0)

        # importance 直接作为 knapsack 的 values（越大越重要）
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

        # 同款打印：per-unit score & mask
        print(
            f"\n--- Per-unit L2 importance & binary mask "
            f"(ratio={ratio:.2f}, layer_idx, unit_type) ---"
        )
        scores_list = importance.detach().cpu().tolist()
        mask_list = mask.detach().cpu().tolist()
        for u, s, m in zip(units, scores_list, mask_list):
            print(
                f"  layer={u.layer_idx:02d}, type={u.unit_type:4s} | "
                f"score={s:.4e} | mask={int(m)}"
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
            "importance_l2": scores_list,
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

            # 2) 物理剪枝（与 cosine 完全复用）
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
            ckpt_path = os.path.join(OUT_DIR, f"qqp_bert_layer_l2_pruned_{ratio:.2f}.pth")
            torch.save(
                {
                    "model": pruned_model.state_dict(),
                    "mask": mask.cpu(),
                    "importance_l2": importance.cpu(),
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

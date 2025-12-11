#!/usr/bin/env python
# coding=utf-8

"""
Random baseline for layer-wise (attention / FFN) pruning on QQP.

For each target prunable FLOPs ratio r in EVAL_RATIOS:
    - Sample K (default 5) different random masks around r:
        * Draw random values for each unit ~ U(0,1)
        * Use the same knapsack_select_mask + flexible upper ratio as other baselines
        * Ensure masks are pairwise different (up to a max number of trials)
    - If --do_finetune:
        * For each mask, build a physically pruned model and finetune 1 epoch with KD
        * Record per-model details and mean validation accuracy across the K models
    - If not --do_finetune:
        * Print per-unit mask & FLOPs summary for all K masks
        * Print whether the K masks are all different for this ratio
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from bert_pruning_utils import (
    set_seed,
    estimate_bert_layer_flops,
    LayerUnitInfo,
    knapsack_select_mask,
)  # type: ignore :contentReference[oaicite:4]{index=4}

# 直接复用 cosine 版本里的公共配置 / 工具函数 / 物理剪枝和 finetune 逻辑
from qqp_bert_layer_cosine_knapsack import (  # type: ignore :contentReference[oaicite:5]{index=5}
    DEVICE,
    BERT_NAME,
    DATASET_NAME,
    DATASET_CONFIG,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    NUM_WORKERS,
    TEMP_KD,
    FT_EPOCHS,
    FT_LR,
    EVAL_RATIOS,
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


# =====================================================================
# Random mask generation
# =====================================================================

def generate_unique_random_masks_for_ratio(
    num_samples: int,
    ratio: float,
    next_ratio: float | None,
    prunable_costs: torch.Tensor,
    max_trials: int = 200,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:
    """
    对给定 target prunable ratio，生成若干不相同的随机 mask。

    每次：
        - values ~ Uniform(0,1) 作为“随机 importance”；
        - 用 knapsack_select_mask(values, costs, target_ratio, max_ratio) 选子集。

    Args:
        num_samples:   期望的 mask 个数（默认 5）。
        ratio:         当前 target prunable ratio。
        next_ratio:    EVAL_RATIOS 里的下一个 ratio，用于构造宽容上界；
                       若为 None，则使用 (ratio + 1.0) / 2 作为上界。
        prunable_costs: [N]，每个 unit 的 FLOPs。
        max_trials:    最多尝试多少次采样以凑够 num_samples 个不相同的 mask。

    Returns:
        masks:         List[mask tensor]，每个 shape [N]，元素为 {0.0, 1.0}
        values_list:   List[values tensor]，与 masks 对应的随机 importance。
        flex_upper_ratio: 对应这个 ratio 使用的 max_ratio（方便记录）。
    """
    device = prunable_costs.device
    masks: List[torch.Tensor] = []
    values_list: List[torch.Tensor] = []
    mask_strings = set()

    # 与其它脚本保持一致的 max_ratio 逻辑
    if next_ratio is not None:
        flex_upper_ratio = 0.5 * (ratio + next_ratio)
    else:
        flex_upper_ratio = 0.5 * (ratio + 1.0)

    N = prunable_costs.numel()
    trials = 0

    while len(masks) < num_samples and trials < max_trials:
        trials += 1
        # 随机 importance
        values = torch.rand(N, device=device, dtype=torch.float32)

        mask, _ = knapsack_select_mask(
            values=values,
            costs=prunable_costs,
            target_ratio=ratio,
            max_ratio=flex_upper_ratio,
        )
        mask = mask.to(device=device, dtype=torch.float32)

        # 把 mask 序列化成字符串用于判重
        mask_str = "".join(str(int(x)) for x in mask.detach().cpu().tolist())

        if mask_str in mask_strings:
            continue  # 重复了，重新采样

        mask_strings.add(mask_str)
        masks.append(mask)
        values_list.append(values)

    if len(masks) < num_samples:
        print(
            f"[Random] Warning: ratio={ratio:.3f} 只生成了 {len(masks)} "
            f"个 unique masks (目标 {num_samples})，已达到 max_trials={max_trials}。"
        )

    return masks, values_list, flex_upper_ratio


# =====================================================================
# Main
# =====================================================================

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
        default="checkpoints_qqp_bert_layer_random",
        help="Output directory for random baseline pruning results.",
    )
    parser.add_argument(
        "--do_finetune",
        action="store_true",
        help="If set, finetune ALL random pruned models (5 per ratio).",
    )
    parser.add_argument(
        "--num_samples_per_ratio",
        type=int,
        default=5,
        help="How many random masks to sample per target ratio.",
    )
    parser.add_argument(
        "--max_trials_per_ratio",
        type=int,
        default=200,
        help="Maximum random sampling trials per ratio to find unique masks.",
    )
    args = parser.parse_args()

    OUT_DIR = args.out_dir
    RESULTS_FILE = os.path.join(OUT_DIR, "qqp_bert_layer_random_results.json")
    TB_LOG_DIR = os.path.join(OUT_DIR, "tb_logs")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    description = (
        "QQP BERT layer-wise random baseline with knapsack and flexible upper ratio.\n"
        f"EVAL_RATIOS={EVAL_RATIOS}, FT_EPOCHS={FT_EPOCHS}, "
        f"num_samples_per_ratio={args.num_samples_per_ratio}.\n"
        "For each ratio, sample different random importance vectors, run knapsack, "
        "and (optionally) finetune each pruned model."
    )
    with open(os.path.join(OUT_DIR, "description.txt"), "w", encoding="utf-8") as f:
        f.write(description)

    set_seed(42)

    writer = SummaryWriter(log_dir=TB_LOG_DIR)

    # ---------------- Data ----------------
    train_loader, val_loader, _ = build_qqp_loaders(
        model_name=BERT_NAME,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        num_workers=NUM_WORKERS,
    )

    # ---------------- Teacher ----------------
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

    # ---------------- FLOPs estimation ----------------
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

    # ---------------- Results container ----------------
    results: Dict = {
        "num_samples_per_ratio": args.num_samples_per_ratio,
        "prunable_costs_per_unit": prunable_costs.detach().cpu().tolist(),
        "fixed_flops": fixed_flops,
        "total_flops_full": total_flops_full,
        "ratios": [],
    }

    # ---------------- Loop over ratios ----------------
    num_ratios = len(EVAL_RATIOS)

    for idx, ratio in enumerate(EVAL_RATIOS):
        print(f"\n=== [Random] Evaluating target prunable ratio {ratio:.3f} ===")

        next_ratio = EVAL_RATIOS[idx + 1] if idx + 1 < num_ratios else None

        masks, values_list, flex_upper_ratio = generate_unique_random_masks_for_ratio(
            num_samples=args.num_samples_per_ratio,
            ratio=ratio,
            next_ratio=next_ratio,
            prunable_costs=prunable_costs,
            max_trials=args.max_trials_per_ratio,
        )

        num_found = len(masks)
        # 判定 unique 情况
        mask_strs = [
            "".join(str(int(x)) for x in m.detach().cpu().tolist()) for m in masks
        ]
        unique_strs = sorted(set(mask_strs))
        all_unique = len(unique_strs) == num_found

        print(
            f"[Random] ratio={ratio:.3f} 找到 {num_found} 个随机 mask，"
            f"其中 unique={len(unique_strs)}，all_unique={all_unique}"
        )

        ratio_entry = {
            "target_ratio_prunable": ratio,
            "flex_upper_ratio": flex_upper_ratio,
            "num_samples_found": num_found,
            "all_masks_unique": all_unique,
            "num_unique_masks": len(unique_strs),
            "samples": [],
        }

        # 记录每个 sample 的结果
        sample_val_accs: List[float] = []

        for s_idx, (mask, values) in enumerate(zip(masks, values_list)):
            print(
                f"\n--- [Random] ratio={ratio:.2f}, sample={s_idx} "
                f"(layer_idx, unit_type) ---"
            )

            scores_list = values.detach().cpu().tolist()
            mask_list = mask.detach().cpu().tolist()

            # 计算 FLOPs 信息
            kept_prunable = float((mask * prunable_costs).sum().item())
            total_prunable = float(prunable_costs.sum().item())
            prunable_ratio = kept_prunable / max(total_prunable, 1e-8)
            total_kept = fixed_flops + kept_prunable
            total_ratio = total_kept / max(total_flops_full, 1e-8)

            # 同款打印每个 unit 的 score & mask
            for u, s, m in zip(units, scores_list, mask_list):
                print(
                    f"  layer={u.layer_idx:02d}, type={u.unit_type:4s} | "
                    f"rand_score={s:.4f} | mask={int(m)}"
                )

            print("\n--- FLOPs summary (this random sample) ---")
            print(f"Target prunable ratio: {ratio:.4f}")
            print(f"Flexible upper ratio (max_ratio): {flex_upper_ratio:.4f}")
            print(f"Prunable ratio (mask * costs / total_prunable): {prunable_ratio:.4f}")
            print(
                f"Total FLOPs kept / full: {total_kept:.2e} / {total_flops_full:.2e} "
                f"({total_ratio * 100:.2f}%)"
            )

            sample_entry = {
                "sample_idx": s_idx,
                "rand_scores": scores_list,
                "mask": mask_list,
                "prunable_ratio_mask": prunable_ratio,
                "kept_prunable_flops": kept_prunable,
                "total_prunable_flops": total_prunable,
                "total_kept_flops": total_kept,
                "total_full_flops": total_flops_full,
                "total_ratio": total_ratio,
            }

            if args.do_finetune:
                # 1) 从 teacher 拷贝一份模型
                pruned_model = build_bert_classifier(
                    model_name=BERT_NAME, num_labels=2
                ).to(DEVICE)
                pruned_model.load_state_dict(teacher.state_dict())

                # 2) 物理剪枝
                apply_physical_bert_pruning(pruned_model, units, mask)

                # 3) 1 epoch KD finetune
                ratio_tag = f"ratio_{ratio:.2f}_sample_{s_idx}"
                pruned_model, best_acc = finetune_pruned_model(
                    teacher=teacher,
                    pruned_model=pruned_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    writer=writer,
                    ratio_tag=ratio_tag,
                )
                print(
                    f"[Result {ratio_tag}] Best Val Acc: {best_acc * 100:.2f}%"
                )

                sample_val_accs.append(float(best_acc))
                sample_entry["best_val_acc"] = float(best_acc)

                # 4) 保存 pruned 模型
                ckpt_path = os.path.join(
                    OUT_DIR,
                    f"qqp_bert_layer_random_pruned_ratio_{ratio:.2f}_sample_{s_idx}.pth",
                )
                torch.save(
                    {
                        "model": pruned_model.state_dict(),
                        "mask": mask.cpu(),
                        "rand_scores": values.cpu(),
                        "target_ratio_prunable": ratio,
                        "flex_upper_ratio": flex_upper_ratio,
                        "prunable_ratio_mask": prunable_ratio,
                        "total_ratio": total_ratio,
                        "kept_prunable_flops": kept_prunable,
                        "total_kept_flops": total_kept,
                    },
                    ckpt_path,
                )
                print(f"Saved pruned model to {ckpt_path}")

            ratio_entry["samples"].append(sample_entry)

        if args.do_finetune and len(sample_val_accs) > 0:
            mean_acc = sum(sample_val_accs) / len(sample_val_accs)
            ratio_entry["mean_best_val_acc"] = mean_acc
            print(
                f"[Random] ratio={ratio:.3f} 平均 best Val Acc "
                f"over {len(sample_val_accs)} samples: {mean_acc * 100:.2f}%"
            )

        results["ratios"].append(ratio_entry)

    # 保存总结果 JSON
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved random baseline results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()

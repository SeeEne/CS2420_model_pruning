#!/usr/bin/env python
# coding=utf-8

"""
Layer-wise (attention / MLP) pruning for ViT-Large on Tiny-ImageNet with knapsack.

Adapted from BERT layer-wise pruning approach:
- Phase 0: Load trained ViT-Large teacher
- Phase 1: Train policy encoder (CLS summarizer + static features + Transformer encoder)
- Phase 2: For each target FLOPs ratio:
    - Use encoder + knapsack to get binary mask
    - Print per-unit soft scores & binary mask
    - Optional: Build physically pruned model and do KD finetune

Key differences from ViT-Small:
- 48 prunable units (24 attn + 24 mlp) instead of 24 units
- ViT-Large: 24 blocks, embed_dim=1024, 16 heads
- Finer granularity: can prune attention but keep MLP (or vice versa)
- Uses 0/1 knapsack for optimal mask selection
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import os
import json
import copy
import gc
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import timm
import numpy as np

from vit_large_layer_pruning_utils import (
    set_seed,
    estimate_vit_layer_flops,
    compute_flops_with_mask,
    ViTCLSCollector,
    CLSSummarizer,
    TokenProjector,
    CompressionPolicyEncoder,
    build_static_features,
    LayerUnitInfo,
    knapsack_select_mask,
    ViTUnitGating,
    apply_physical_vit_pruning,
    build_vit_unit_tokens,
    kd_loss,
    distillation_loss,
)

# ============================================================
# Config
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/tiny-imagenet-200"
CKPT_TEACHER = "checkpoints/teacher_vit_large.pth"
OUT_DIR = "checkpoints"
ENCODER_CKPT = "vit_large_layer_encoder.pth"
RESULTS_FILE = "vit_large_layer_wise_results.json"

# Data
BATCH_SIZE = 64  # Reduced for ViT-Large due to memory
NUM_WORKERS = 2
IMG_SIZE = 224

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

# Teacher training
TEACHER_EPOCHS = 30
TEACHER_LR = 1e-4
TEACHER_WEIGHT_DECAY = 0.05

# Finetune
FT_EPOCHS = 7
FT_LR = 1e-4
FT_ALPHA = 0.5  # Weight for task loss in KD

# ViT-Large: 24 blocks -> 48 units
NUM_BLOCKS = 24
NUM_UNITS = 48

# Minimum attention/MLP layers constraint
# Set to 0 to disable constraint (let knapsack decide freely)
MIN_ATTN_LAYERS = 0
MIN_MLP_LAYERS = 0

# Ratios to evaluate
EVAL_RATIOS = [0.25, 0.35, 0.55, 0.65, 0.75, 0.85]

SEED = 42


# ============================================================
# Dataset
# ============================================================

class TinyImageNetVal(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        annotations_file = os.path.join(root, 'val_annotations.txt')
        self.images = []
        self.labels = []
        train_dir = os.path.join(os.path.dirname(root), 'train')
        if os.path.exists(train_dir):
            self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(train_dir)))}
        else:
            self.class_to_idx = {}
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    self.images.append(os.path.join(root, 'images', parts[0]))
                    self.labels.append(self.class_to_idx.get(parts[1], 0))

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def get_loaders():
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if not os.path.exists(os.path.join(DATA_DIR, 'train')):
        print(f"Warning: Dataset not found at {DATA_DIR}. Using FakeData.")
        dummy_ds = datasets.FakeData(size=1000, image_size=(3, IMG_SIZE, IMG_SIZE),
                                     num_classes=200, transform=transforms.ToTensor())
        return DataLoader(dummy_ds, 32), DataLoader(dummy_ds, 32), 200

    train_ds = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_tf)
    val_ds = TinyImageNetVal(root=os.path.join(DATA_DIR, 'val'), transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes)


# ============================================================
# Model
# ============================================================

def build_vit_large(num_classes=200, pretrained=False):
    model = timm.create_model('vit_large_patch16_224', pretrained=pretrained, num_classes=num_classes)
    return model


# ============================================================
# Teacher Training
# ============================================================

def train_teacher(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
) -> nn.Module:
    """
    Train ViT-Large teacher model on Tiny-ImageNet.
    Uses ImageNet pretrained weights and fine-tunes on Tiny-ImageNet.
    """
    device = DEVICE
    print("="*60)
    print("Phase 0: Training ViT-Large Teacher Model")
    print("="*60)

    # Build model with pretrained weights
    model = build_vit_large(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TEACHER_LR,
        weight_decay=TEACHER_WEIGHT_DECAY,
    )

    # Cosine annealing scheduler with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (TEACHER_EPOCHS - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    best_state = None

    for epoch in range(TEACHER_EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Mixed precision forward
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            # Backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            preds = logits.argmax(dim=-1)
            running_correct += (preds == y).sum().item()
            running_total += y.numel()

            if batch_idx % 50 == 0 and batch_idx > 0:
                avg_loss = running_loss / (batch_idx + 1)
                avg_acc = running_correct / running_total
                print(f"  [Epoch {epoch+1}/{TEACHER_EPOCHS}][{batch_idx}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} | Train Acc: {avg_acc*100:.2f}%")

        scheduler.step()

        # Evaluate
        val_acc = evaluate_accuracy(model, val_loader, device)
        train_acc = running_correct / running_total
        avg_loss = running_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        print(f"[Epoch {epoch+1}/{TEACHER_EPOCHS}] "
              f"Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Acc: {val_acc*100:.2f}% | LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"  -> New best model! Val Acc: {best_acc*100:.2f}%")

    # Load best state
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Cleanup training artifacts
    del optimizer, scheduler, scaler, best_state
    gc.collect()
    torch.cuda.empty_cache()

    # Save checkpoint
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'val_acc': best_acc,
        'num_classes': num_classes,
    }, CKPT_TEACHER)
    print(f"\nTeacher model saved to {CKPT_TEACHER}")
    print(f"Best validation accuracy: {best_acc*100:.2f}%")

    return model


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device=DEVICE) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_with_gating(
    model: nn.Module,
    gating: ViTUnitGating,
    mask: torch.Tensor,
    loader: DataLoader,
    device=DEVICE,
) -> float:
    """Evaluate with logical pruning (gating hooks)."""
    model.eval()
    gating.set_fixed_mask(mask)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()
    gating.clear_fixed_mask()
    return correct / max(total, 1)


# ============================================================
# Helper: Log mask for a specific ratio during training
# ============================================================

@torch.no_grad()
def _log_mask_for_ratio(
    teacher: nn.Module,
    encoder: CompressionPolicyEncoder,
    summarizer: CLSSummarizer,
    token_proj: TokenProjector,
    static_feats: torch.Tensor,
    units: List[LayerUnitInfo],
    train_loader: DataLoader,
    prunable_flops: torch.Tensor,
    collector: "ViTCLSCollector",
    target_ratio: float,
    epoch: int,
):
    """Log mask result for a specific ratio during training."""
    device = DEVICE
    encoder.eval()
    summarizer.eval()
    token_proj.eval()

    # Get one batch for evaluation
    x, _ = next(iter(train_loader))
    x = x.to(device)

    tokens = build_vit_unit_tokens(
        model=teacher,
        collector=collector,
        summarizer=summarizer,
        token_proj=token_proj,
        static_feats=static_feats,
        units=units,
        x=x,
        device=device,
    )

    logits_units = encoder(tokens, target_ratio=torch.tensor(target_ratio, device=device))
    scores = torch.sigmoid(logits_units)

    # Use knapsack to get mask
    mask, used_ratio = knapsack_select_mask(
        values=scores,
        costs=prunable_flops,
        target_ratio=target_ratio,
        max_ratio=target_ratio,
        min_attn_layers=MIN_ATTN_LAYERS,
        min_mlp_layers=MIN_MLP_LAYERS,
        num_blocks=NUM_BLOCKS,
    )

    # Count attention and MLP kept
    mask_list = mask.cpu().tolist()
    kept_attn = sum(1 for i in range(0, NUM_UNITS, 2) if mask_list[i] >= 0.5)
    kept_mlp = sum(1 for i in range(1, NUM_UNITS, 2) if mask_list[i] >= 0.5)

    # Build mask string: [AM] = both kept, [A.] = attn only, [.M] = mlp only, [..] = both pruned
    mask_str = ""
    for block_idx in range(NUM_BLOCKS):
        attn_kept = mask_list[block_idx * 2] >= 0.5
        mlp_kept = mask_list[block_idx * 2 + 1] >= 0.5
        a = "A" if attn_kept else "."
        m = "M" if mlp_kept else "."
        mask_str += f"[{a}{m}]"

    print(f"  [Epoch {epoch}] Ratio {target_ratio} mask: {mask_str} "
          f"(attn: {kept_attn}/{NUM_BLOCKS}, mlp: {kept_mlp}/{NUM_BLOCKS}, actual: {used_ratio:.3f})")

    # Back to train mode
    encoder.train()
    summarizer.train()
    token_proj.train()


# ============================================================
# Phase 1: Train policy encoder
# ============================================================

def train_layer_policy_encoder(
    teacher: nn.Module,
    train_loader: DataLoader,
    units: List[LayerUnitInfo],
    prunable_flops: torch.Tensor,
) -> Tuple[CLSSummarizer, TokenProjector, CompressionPolicyEncoder, torch.Tensor]:
    """Train CLS-based layer-wise policy encoder."""
    device = DEVICE
    teacher.eval()

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Use teacher as student for CLS collection (same weights)
    student = teacher

    embed_dim = teacher.embed_dim

    summarizer = CLSSummarizer(embed_dim=embed_dim, summary_dim=SUMMARY_DIM).to(device)
    token_proj = TokenProjector(summary_dim=SUMMARY_DIM, static_dim=STATIC_DIM, token_dim=TOKEN_DIM).to(device)
    encoder = CompressionPolicyEncoder(token_dim=TOKEN_DIM, num_layers=2, num_heads=4,
                                        ff_dim=256, dropout=0.1).to(device)

    static_feats = build_static_features(units, device=device)  # [N, 4]
    collector = ViTCLSCollector(student)
    gating = ViTUnitGating(student, units)

    params = list(summarizer.parameters()) + list(token_proj.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=ENC_LR)

    total_prunable = prunable_flops.sum().item()
    total_epochs = POLICY_WARMUP_EPOCHS + POLICY_TRAIN_EPOCHS
    global_step = 0

    print(f"\n{'='*60}")
    print(f"Phase 1: Training Layer-wise Policy Encoder")
    print(f"{'='*60}")
    print(f"Units: {len(units)} ({NUM_BLOCKS} attn + {NUM_BLOCKS} mlp)")
    print(f"Epochs: {total_epochs} ({POLICY_WARMUP_EPOCHS} warmup + {POLICY_TRAIN_EPOCHS} train)")
    print(f"Gate temp: {GATE_TEMP_START} -> {GATE_TEMP_END}")

    for epoch in range(total_epochs):
        summarizer.train()
        token_proj.train()
        encoder.train()

        # Gate temperature schedule
        t_frac = epoch / max(total_epochs - 1, 1)
        gate_temp = GATE_TEMP_START + (GATE_TEMP_END - GATE_TEMP_START) * t_frac

        running_loss = 0.0
        running_kd = 0.0
        running_ratio = 0.0
        running_flops_ratio = 0.0
        running_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Teacher logits
            with torch.no_grad():
                logits_T = teacher(x)

            # Random target ratio
            target_ratio = random.uniform(MIN_RATIO, MAX_RATIO)

            # Build tokens (CLS-based)
            tokens = build_vit_unit_tokens(
                model=student,
                collector=collector,
                summarizer=summarizer,
                token_proj=token_proj,
                static_feats=static_feats,
                units=units,
                x=x,
                device=device,
            )  # [1, N, token_dim]

            logits_units = encoder(tokens, target_ratio=torch.tensor(target_ratio, device=device))  # [N]
            gates = torch.sigmoid(logits_units / gate_temp)  # [N]

            # Compute current FLOPs ratio
            curr_flops = (gates * prunable_flops).sum()
            flops_ratio = curr_flops / max(total_prunable, 1e-8)

            # Gating + KD
            gating.set_train_gates(gates)
            student.train()
            logits_S = student(x)
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

            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"  [Ep {epoch+1}/{total_epochs}][{batch_idx}] "
                      f"loss={running_loss/running_batches:.4f} "
                      f"kd={running_kd/running_batches:.4f} "
                      f"ratio_loss={running_ratio/running_batches:.4f} "
                      f"flops={running_flops_ratio/running_batches:.3f}")

        print(f"[Epoch {epoch+1}/{total_epochs}] "
              f"loss={running_loss/max(running_batches,1):.4f} "
              f"flops_ratio={running_flops_ratio/max(running_batches,1):.3f} "
              f"gate_temp={gate_temp:.2f}")

        # Log mask for ratio 0.1 at each epoch to verify attention constraint
        _log_mask_for_ratio(
            teacher=teacher,
            encoder=encoder,
            summarizer=summarizer,
            token_proj=token_proj,
            static_feats=static_feats,
            units=units,
            train_loader=train_loader,
            prunable_flops=prunable_flops,
            collector=collector,
            target_ratio=0.1,
            epoch=epoch+1,
        )

    collector.close()
    gating.remove()

    # Cleanup after encoder training
    del opt
    gc.collect()
    torch.cuda.empty_cache()

    return summarizer, token_proj, encoder, static_feats


# ============================================================
# Phase 2: Materialize mask with knapsack
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
    Materialize binary mask using encoder scores + knapsack.

    Returns:
        mask: [N] 0/1 tensor
        used_ratio: Actual FLOPs ratio
        avg_scores: [N] average soft scores
    """
    device = DEVICE
    teacher.eval()
    encoder.eval()
    summarizer.eval()
    token_proj.eval()

    collector = ViTCLSCollector(teacher)

    scores_accum = torch.zeros(len(units), dtype=torch.float32, device=device)
    count = 0

    loader_iter = iter(train_loader)
    for it in range(num_iters):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x = x.to(device)

        tokens = build_vit_unit_tokens(
            model=teacher,
            collector=collector,
            summarizer=summarizer,
            token_proj=token_proj,
            static_feats=static_feats,
            units=units,
            x=x,
            device=device,
        )

        logits_units = encoder(tokens, target_ratio=torch.tensor(target_ratio, device=device))
        probs = torch.sigmoid(logits_units)

        scores_accum += probs
        count += 1

    collector.close()

    avg_scores = scores_accum / max(count, 1)

    # Use knapsack for optimal selection
    # min_attn_layers ensures at least some attention is kept (CLS needs to aggregate)
    # min_mlp_layers ensures at least some MLP is kept (transformation capability)
    mask, used_ratio = knapsack_select_mask(
        values=avg_scores,
        costs=prunable_flops,
        target_ratio=target_ratio,
        max_ratio=max_ratio,
        min_attn_layers=MIN_ATTN_LAYERS,
        min_mlp_layers=MIN_MLP_LAYERS,
        num_blocks=NUM_BLOCKS,
    )

    return mask.to(device), used_ratio, avg_scores


# ============================================================
# Phase 2: Finetune pruned model
# ============================================================

def finetune_pruned_model(
    teacher: nn.Module,
    units: List[LayerUnitInfo],
    mask: torch.Tensor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    ratio_tag: str,
) -> Tuple[nn.Module, float, float]:
    """
    Finetune pruned model with KD.

    Returns:
        pruned_model: Finetuned model
        best_acc: Best validation accuracy
        acc_before: Accuracy before finetuning
    """
    device = DEVICE
    teacher.eval()

    # Create pruned model (physical pruning)
    pruned_model = copy.deepcopy(teacher)
    apply_physical_vit_pruning(pruned_model, units, mask)
    pruned_model = pruned_model.to(device)

    # Evaluate before finetuning
    acc_before = evaluate_accuracy(pruned_model, val_loader, device)
    print(f"  Accuracy before FT: {acc_before*100:.2f}%")

    opt = torch.optim.AdamW(pruned_model.parameters(), lr=FT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)

    best_acc = acc_before
    best_state = {k: v.cpu() for k, v in pruned_model.state_dict().items()}

    for epoch in range(FT_EPOCHS):
        pruned_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                logits_T = teacher(x)

            logits_S = pruned_model(x)
            loss, loss_kd, loss_task = distillation_loss(logits_S, logits_T, y, T=TEMP_KD, alpha=FT_ALPHA)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        val_acc = evaluate_accuracy(pruned_model, val_loader, device)

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  [FT {ratio_tag}] Ep {epoch+1}/{FT_EPOCHS} | Loss: {avg_loss:.4f} | Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in pruned_model.state_dict().items()}

    pruned_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Cleanup
    del opt, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return pruned_model, best_acc, acc_before


# ============================================================
# Visualization helpers
# ============================================================

def print_mask_details(
    units: List[LayerUnitInfo],
    mask: torch.Tensor,
    scores: torch.Tensor,
    target_ratio: float,
    actual_ratio: float,
):
    """Print detailed mask information."""
    mask_list = mask.cpu().tolist()
    scores_list = scores.cpu().tolist()

    print(f"\n  --- Ratio {target_ratio:.1f} (actual: {actual_ratio:.3f}) ---")

    # Group by block
    print(f"  {'Block':<6} {'Attn':<8} {'MLP':<8} {'Attn Score':<12} {'MLP Score':<12}")
    print(f"  {'-'*50}")

    for block_idx in range(NUM_BLOCKS):
        attn_idx = block_idx * 2
        mlp_idx = block_idx * 2 + 1

        attn_mask = int(mask_list[attn_idx])
        mlp_mask = int(mask_list[mlp_idx])
        attn_score = scores_list[attn_idx]
        mlp_score = scores_list[mlp_idx]

        attn_str = "KEEP" if attn_mask else "PRUNE"
        mlp_str = "KEEP" if mlp_mask else "PRUNE"

        print(f"  {block_idx:<6} {attn_str:<8} {mlp_str:<8} {attn_score:<12.4f} {mlp_score:<12.4f}")

    # Summary
    kept_attn = sum(1 for i in range(0, NUM_UNITS, 2) if mask_list[i] >= 0.5)
    kept_mlp = sum(1 for i in range(1, NUM_UNITS, 2) if mask_list[i] >= 0.5)
    print(f"  {'-'*50}")
    print(f"  Kept: {kept_attn}/{NUM_BLOCKS} attn, {kept_mlp}/{NUM_BLOCKS} mlp, {kept_attn + kept_mlp}/{NUM_UNITS} total")


def mask_to_string(mask: torch.Tensor) -> str:
    """Convert mask to readable string: A=attn, M=mlp, .=pruned."""
    result = []
    mask_list = mask.cpu().tolist()
    for i in range(NUM_BLOCKS):
        attn = 'A' if mask_list[i*2] >= 0.5 else '.'
        mlp = 'M' if mask_list[i*2+1] >= 0.5 else '.'
        result.append(f"[{attn}{mlp}]")
    return ''.join(result)


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("="*60)
    print("ViT-Large Layer-wise Pruning (Attention + MLP)")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Units: {NUM_UNITS} ({NUM_BLOCKS} attn + {NUM_BLOCKS} mlp)")
    print(f"Eval ratios: {EVAL_RATIOS}")

    # Load data
    train_loader, val_loader, num_classes = get_loaders()

    # Load or train teacher
    if not os.path.exists(CKPT_TEACHER):
        print(f"\nTeacher checkpoint not found: {CKPT_TEACHER}")
        print("Training teacher model from scratch...")
        teacher = train_teacher(train_loader, val_loader, num_classes)
        teacher_acc = evaluate_accuracy(teacher, val_loader, DEVICE)
        # Cleanup after training
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[VRAM Cleanup] After teacher training")
    else:
        print(f"\nLoading teacher from {CKPT_TEACHER}")
        teacher = build_vit_large(num_classes).to(DEVICE)
        ckpt = torch.load(CKPT_TEACHER, map_location='cpu')
        teacher.load_state_dict(ckpt['state_dict'])
        del ckpt  # Free checkpoint memory
        gc.collect()
        torch.cuda.empty_cache()
        teacher.eval()
        teacher_acc = evaluate_accuracy(teacher, val_loader, DEVICE)
    print(f"Teacher Accuracy: {teacher_acc*100:.2f}%")

    # Estimate FLOPs
    units, fixed_flops, total_flops = estimate_vit_layer_flops(teacher, IMG_SIZE)
    prunable_flops = torch.tensor([u.cost for u in units], dtype=torch.float32, device=DEVICE)

    print(f"\nFLOPs breakdown:")
    print(f"  Fixed (embed + norm + head): {fixed_flops/1e9:.2f} GFLOPs")
    print(f"  Prunable ({NUM_UNITS} units): {prunable_flops.sum().item()/1e9:.2f} GFLOPs")
    print(f"  Total: {total_flops/1e9:.2f} GFLOPs")
    print(f"  Per-unit attn: {units[0].cost/1e9:.3f} GFLOPs")
    print(f"  Per-unit mlp: {units[1].cost/1e9:.3f} GFLOPs")

    # Check for existing encoder
    encoder_path = os.path.join(OUT_DIR, ENCODER_CKPT)
    if os.path.exists(encoder_path):
        print(f"\nLoading existing encoder from {encoder_path}")
        ckpt_enc = torch.load(encoder_path, map_location='cpu')
        embed_dim = teacher.embed_dim
        summarizer = CLSSummarizer(embed_dim=embed_dim, summary_dim=SUMMARY_DIM).to(DEVICE)
        token_proj = TokenProjector(summary_dim=SUMMARY_DIM, static_dim=STATIC_DIM, token_dim=TOKEN_DIM).to(DEVICE)
        encoder = CompressionPolicyEncoder(token_dim=TOKEN_DIM, num_layers=2, num_heads=4,
                                           ff_dim=256, dropout=0.1).to(DEVICE)
        summarizer.load_state_dict(ckpt_enc['summarizer'])
        token_proj.load_state_dict(ckpt_enc['token_proj'])
        encoder.load_state_dict(ckpt_enc['encoder'])
        static_feats = build_static_features(units, device=DEVICE)
        del ckpt_enc  # Free checkpoint memory
        gc.collect()
        torch.cuda.empty_cache()
    else:
        # Phase 1: Train encoder
        summarizer, token_proj, encoder, static_feats = train_layer_policy_encoder(
            teacher, train_loader, units, prunable_flops
        )

        # Save encoder
        torch.save({
            'summarizer': summarizer.state_dict(),
            'token_proj': token_proj.state_dict(),
            'encoder': encoder.state_dict(),
        }, encoder_path)
        print(f"\nEncoder saved to {encoder_path}")

        # Cleanup after encoder training
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[VRAM Cleanup] After encoder training")

    # Phase 2: Evaluate at different ratios
    print(f"\n{'='*60}")
    print("Phase 2: Evaluating at different FLOPs ratios")
    print("="*60)

    all_results = {
        "teacher_accuracy": teacher_acc,
        "total_flops": total_flops,
        "fixed_flops": fixed_flops,
        "num_units": NUM_UNITS,
        "num_blocks": NUM_BLOCKS,
        "results": []
    }

    for target_ratio in EVAL_RATIOS:
        print(f"\n[Ratio {target_ratio}]")

        # Materialize mask
        mask, actual_ratio, scores = materialize_unit_mask_for_ratio(
            teacher=teacher,
            encoder=encoder,
            summarizer=summarizer,
            token_proj=token_proj,
            static_feats=static_feats,
            units=units,
            train_loader=train_loader,
            prunable_flops=prunable_flops,
            target_ratio=target_ratio,
            max_ratio=target_ratio + 0.1,  # Allow slightly over
        )

        # Print details
        print_mask_details(units, mask, scores, target_ratio, actual_ratio)
        print(f"  Mask: {mask_to_string(mask)}")

        # Finetune
        pruned_model, best_acc, acc_before = finetune_pruned_model(
            teacher=teacher,
            units=units,
            mask=mask,
            train_loader=train_loader,
            val_loader=val_loader,
            ratio_tag=f"r{target_ratio}",
        )

        # Record results
        result = {
            "target_ratio": target_ratio,
            "actual_ratio": actual_ratio,
            "mask": mask.cpu().tolist(),
            "scores": scores.cpu().tolist(),
            "accuracy_before_ft": acc_before,
            "accuracy_after_ft": best_acc,
            "accuracy_drop": teacher_acc - best_acc,
            "kept_attn": sum(1 for i in range(0, NUM_UNITS, 2) if mask[i] >= 0.5),
            "kept_mlp": sum(1 for i in range(1, NUM_UNITS, 2) if mask[i] >= 0.5),
        }
        all_results["results"].append(result)

        print(f"  Final Accuracy: {best_acc*100:.2f}% (drop: {(teacher_acc-best_acc)*100:.2f}%)")

        # Cleanup
        del pruned_model
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    results_path = os.path.join(OUT_DIR, RESULTS_FILE)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"{'Ratio':<8} {'Actual':<8} {'Attn':<6} {'MLP':<6} {'Before':<10} {'After':<10} {'Drop':<8}")
    print("-"*60)
    for r in all_results["results"]:
        print(f"{r['target_ratio']:<8.2f} {r['actual_ratio']:<8.3f} "
              f"{r['kept_attn']:<6d} {r['kept_mlp']:<6d} "
              f"{r['accuracy_before_ft']*100:<10.2f} "
              f"{r['accuracy_after_ft']*100:<10.2f} "
              f"{r['accuracy_drop']*100:<8.2f}")
    print("="*60)


if __name__ == "__main__":
    main()

# Model Pruning Experiments

This repository contains experiments for neural network pruning using Knowledge Distillation (KD) and various policy networks.

## Requirements

```bash
pip install torch torchvision timm wandb
```

## Dataset Setup

Download Tiny-ImageNet-200:
```bash
# Download and extract to data/tiny-imagenet-200/
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d data/
```

---

## Part 1: ResNet Experiments (Files 1-5)

### 1. Train Teacher Model

**File:** `1_build_teacher.py`

Trains a ResNet-18 teacher model on Tiny-ImageNet-200.

```bash
python 1_build_teacher.py --data data/tiny-imagenet-200 --epochs 10 --batch_size 256
```

**Output:** `checkpoints/teacher.pth`

---

### 2. L2 Baseline Pruning

**File:** `2_build_baseline.py` (16-gate, unit-wise)
**File:** `2_build_baseline_block.py` (8-gate, block-wise)

Baseline pruning using L2 norm of weights to score importance.

**Unit-wise (16 gates - conv1/conv2 separately):**
```bash
python 2_build_baseline.py
```
- Prunes individual conv layers within blocks
- Output: `checkpoints/baseline_l2_16gate_results.json`

**Block-wise (8 gates - entire blocks):**
```bash
python 2_build_baseline_block.py
```
- Prunes entire BasicBlocks, replaces with skip connection
- Output: `checkpoints/baseline_l2_block_results.json`

---

### 3. Transformer Encoder Policy

**File:** `3_encoder_block.py` (8-gate, block-wise)
**File:** `3_encoder_pruning.py` (16-gate, unit-wise)

Uses a Transformer encoder to predict pruning decisions based on:
- Block features (summarized via pooling)
- Taylor importance scores (gradient * activation)
- Target FLOPs ratio as conditioning

**Block-wise pruning:**
```bash
python 3_encoder_block.py
```

**Unit-wise pruning:**
```bash
python 3_encoder_pruning.py
```

**Key components:**
- `Summarizer`: Extracts block features (GAP, GMP of input/residual)
- `CompressionAwareEncoder`: Transformer that takes block tokens + budget token
- Soft gates with temperature annealing during training
- Greedy mask materialization at inference

**Output:** `checkpoints/encoder_block_results_tinyimagenet.json`

---

### 4. MLP Policy Baseline

**File:** `4_mlp.py`

Compares MLP-based policy (no self-attention) vs Transformer encoder.

```bash
python 4_mlp.py
```

**Architecture:**
- Flattens all block tokens into single vector
- Concatenates budget embedding
- 2-layer MLP predicts all gates at once

**Output:** `checkpoints/mlp_policy_results.json`

---

### 5. ResNet-50 Block Pruning

**File:** `5_resnet50_block.py`

Extends experiments to ResNet-50 (16 Bottleneck blocks).
Compares three methods: L2 Baseline, MLP Policy, Transformer Encoder.

```bash
python 5_resnet50_block.py
```

**Features:**
- Handles Bottleneck blocks (conv1→conv2→conv3)
- Checkpoint/resume support for long experiments
- Automatic teacher training if checkpoint not found

**Outputs:**
- `checkpoints/resnet50_l2_baseline_results.json`
- `checkpoints/resnet50_mlp_block_results.json`
- `checkpoints/resnet50_encoder_block_results.json`
- `checkpoints/resnet50_comparison_results.json`

---

## Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 256-512 | Training batch size |
| `TEMP_KD` | 2.0 | Knowledge distillation temperature |
| `RATIO_WEIGHT` | 25.0 | Weight for FLOPs ratio loss |
| `GATE_TEMP_START/END` | 5.0→0.3 | Gate temperature annealing |
| `FT_EPOCHS` | 10 | Fine-tuning epochs after pruning |
| `EVAL_RATIOS` | [0.1-0.9] | Target FLOPs ratios to evaluate |

---

## Pipeline Overview

1. **Train Teacher** → Full accuracy baseline
2. **Train Policy** → Learn to predict pruning masks
   - Sample random target ratios
   - Soft gating with KD loss + ratio loss
3. **Materialize Masks** → Convert soft gates to binary masks
   - Greedy selection by efficiency (score/FLOPs)
4. **BN Recalibration** → Update BatchNorm statistics
5. **Fine-tune** → KD from teacher to pruned student
6. **Evaluate** → Report accuracy vs FLOPs trade-off

---

## Part 2: Overfitting/Generalization Experiment (File 6)

### 6. Cross-Dataset Transfer Experiment

**File:** `6_overfit_exp.py`

Tests whether pruning policies overfit to the training dataset (Tiny-ImageNet) or generalize to new datasets (CIFAR-100).

```bash
python 6_overfit_exp.py
```

**Experiment Design:**
- **Old Policy**: Use masks from policies trained on Tiny-ImageNet
- **New Policy**: Train fresh policies on CIFAR-100
- Compare accuracy after fine-tuning on CIFAR-100

**Pipeline (6 Phases):**
1. Train CIFAR-100 teacher (ResNet-50)
2. Old Encoder Student: Apply Tiny-ImageNet encoder masks → Fine-tune
3. Train New Encoder on CIFAR-100
4. New Encoder Student: Apply CIFAR-100 encoder masks → Fine-tune
5. Old MLP Student: Apply Tiny-ImageNet MLP masks → Fine-tune
6. Train New MLP + New MLP Student

**Prerequisites:**
- Run `5_resnet50_block.py` first to generate old policy masks

**Output:** `checkpoints/cifar100_overfit_results.json`

**Hypothesis:** If old policies overfit to Tiny-ImageNet structure, new policies trained on CIFAR-100 should produce better pruning masks.

---

## Part 3: AZ-NAS Regularization (Files 7-8)

These experiments explore using AZ-NAS (Zero-Shot NAS) metrics as training signals instead of or alongside Knowledge Distillation.

### AZ-NAS Metrics

- **Expressivity**: Entropy of feature covariance eigenvalues (higher = more diverse representations)
- **Progressivity**: Minimum increase in expressivity across layers (ensures gradual feature refinement)
- **Trainability**: Gradient flow quality
- **Complexity**: Model complexity penalty

### 7. AZ-NAS Encoder (Pure AZ-NAS Loss)

**File:** `7_aznas_encoder_exp.py`

Replaces KD loss entirely with AZ-NAS loss during encoder training.

```bash
python 7_aznas_encoder_exp.py
```

**Key Difference from KD-based training:**
```python
# AZ-NAS Loss (replaces KD)
loss_aznas = -expressivity - 0.1 * progressivity
loss = loss_aznas + RATIO_WEIGHT * loss_ratio + L1_M_WEIGHT * loss_l1
```

**Pipeline:**
1. Train AZ-NAS encoder on Tiny-ImageNet (using AZ-NAS loss, not KD)
2. Fine-tune students at various ratios on Tiny-ImageNet
3. Train CIFAR-100 teacher
4. Test old policy (from Tiny-ImageNet) on CIFAR-100
5. Train new policy on CIFAR-100 and compare

**Output:** `checkpoints/aznas_encoder_results.json`

---

### 8. Hybrid KD + AZ-NAS Training

**File:** `8_aznas_encoder_exp_hybird.py`

Uses a two-phase training strategy:
1. **KD Warmup**: Train with KD loss only to establish semantic understanding
2. **Co-training**: Combine KD + AZ-NAS regularization

```bash
python 8_aznas_encoder_exp_hybird.py
```

**Loss Function (Co-training phase):**
```python
loss = loss_kd + RATIO_WEIGHT * loss_ratio + L1_M_WEIGHT * loss_l1
if not is_warmup:
    loss += AZNAS_WEIGHT * loss_aznas  # Add AZ-NAS regularization
```

**Key Parameters:**
- `KD_WARMUP_EPOCHS = 1`: Initial epochs with KD only
- `AZNAS_WEIGHT = 0.5`: Weight for AZ-NAS regularization

**Hypothesis:** Combining KD (semantic) with AZ-NAS (structural) signals produces policies that generalize better across datasets.

**Output:** `checkpoints/aznas_encoder_results_hybird.json`

---

## Part 4: Vision Transformer (ViT) Experiments (Files 9-13)

These experiments extend pruning to Vision Transformers using the `timm` library.

### 9. ViT-Small Block-Wise Pruning

**File:** `9_vit_small_block.py`

Block-wise pruning for ViT-Small (12 transformer blocks).
Compares 4 methods: Transformer Encoder, MLP Policy, L2 Baseline, Cosine Similarity Baseline.

```bash
python 9_vit_small_block.py
```

**Model:** `timm.vit_small_patch16_224` (12 blocks, 384 hidden dim)

**Methods Compared:**
- **Encoder**: Transformer-based policy with budget conditioning
- **MLP**: Simple feedforward policy
- **L2 Baseline**: Weight magnitude scoring
- **Cosine Baseline**: Input-output similarity scoring

**Features:**
- Wandb integration for logging (optional)
- Checkpoint/resume support

**Outputs:**
- `checkpoints/vit_small_encoder_block_results.json`
- `checkpoints/vit_small_mlp_block_results.json`
- `checkpoints/vit_small_l2_baseline_results.json`
- `checkpoints/vit_small_cosine_baseline_results.json`

---

### 10. ViT with AZ-NAS Residual Learning

**File:** `10_vit_small_block_az_regularization.py`

Advanced encoder training with:
1. **Residual Learning**: Use AZ-NAS scores as input prior, learn deviations
2. **Diversity Loss**: Cross-ratio contrastive regularization

```bash
python 10_vit_small_block_az_regularization.py
```

**Key Innovation:**
```python
# Residual Learning
logits = encoder(x, ratio) + lambda * aznas_scores

# Diversity Loss (penalize similar outputs for different ratios)
if cosine_sim(output_r1, output_r2) > threshold:
    loss += diversity_penalty
```

**Parameters:**
- `DIVERSITY_WEIGHT = 2.0`
- `DIVERSITY_SIM_THRESHOLD = 0.5`

**Output:** `checkpoints/vit_small_encoder_az_policy.pth`

---

### 11. Fine-tune ViT with Pre-computed Masks

**File:** `11_finetune_vit_with_masks.py`

Fine-tunes ViT-Small using pre-computed masks from encoder validation.
Physically removes blocks (real FLOPs savings).

```bash
python 11_finetune_vit_with_masks.py
```

**Pre-defined Masks:**
```python
MASKS = {
    0.3: [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    0.5: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    0.7: [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1],
    # ...
}
```

**Key Feature:** `PrunedViT` class physically removes blocks for real speedup.

**Output:** `checkpoints/vit_small_az_finetune_results.json`

---

### 12. ViT-Small Layer-Wise Pruning

**File:** `12_vit_layer_wise_pruning.py`

Finer-grained pruning: prune attention and MLP layers separately.

```bash
python 12_vit_layer_wise_pruning.py
```

**Key Difference from Block-Wise:**
- 24 prunable units (12 attention + 12 MLP) instead of 12 blocks
- Can keep attention but prune MLP (or vice versa)
- Uses 0/1 knapsack algorithm for optimal mask selection

**Requires:** `vit_layer_pruning_utils.py` (utility functions)

**Parameters:**
- `EVAL_RATIOS = [0.25, 0.35, 0.55, 0.65, 0.75, 0.85]`

**Output:** `checkpoints/vit_layer_wise_results.json`

---

### 13. ViT-Large Layer-Wise Pruning

**File:** `13_vit_large_layer_wise_pruning.py`

Extends layer-wise pruning to ViT-Large.

```bash
python 13_vit_large_layer_wise_pruning.py
```

**Model:** `timm.vit_large_patch16_224` (24 blocks, 1024 hidden dim, 16 heads)

**Key Differences from ViT-Small:**
- 48 prunable units (24 attention + 24 MLP)
- Reduced batch size (64) for memory
- Includes teacher training code (30 epochs with warmup)

**Requires:** `vit_large_layer_pruning_utils.py`

**Output:** `checkpoints/vit_large_layer_wise_results.json`

---

## Summary: File Quick Reference

| File | Model | Pruning Type | Method |
|------|-------|--------------|--------|
| 1 | ResNet-18 | - | Train teacher |
| 2 | ResNet-18 | Unit/Block | L2 baseline |
| 3 | ResNet-18 | Unit/Block | Encoder policy |
| 4 | ResNet-18 | Block | MLP policy |
| 5 | ResNet-50 | Block | Encoder + MLP + L2 |
| 6 | ResNet-50 | Block | Cross-dataset transfer |
| 7 | ResNet-18 | Block | AZ-NAS loss |
| 8 | ResNet-18 | Block | Hybrid KD + AZ-NAS |
| 9 | ViT-Small | Block | Encoder + MLP + L2 + Cosine |
| 10 | ViT-Small | Block | AZ-NAS residual + diversity |
| 11 | ViT-Small | Block | Fine-tune with masks |
| 12 | ViT-Small | Layer | Attn/MLP separate |
| 13 | ViT-Large | Layer | Attn/MLP separate |


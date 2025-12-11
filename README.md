# CS2420 Model Pruning

A comprehensive neural network pruning framework exploring **Knowledge Distillation (KD)**, **AZ-NAS metrics**, and **learned policy networks** for structured pruning across CNNs and Vision Transformers. Refer to the individual subdirectory READMEs for detailed instructions.

## Overview

This project implements and compares multiple pruning strategies:
- **Baseline Methods**: L2-norm, Random, Cosine Similarity
- **Learned Policies**: Transformer Encoder, MLP Policy
- **AZ-NAS Integration**: Zero-shot architecture metrics as training signals
- **Multi-granularity**: Block-wise, Unit-wise, and Layer-wise pruning

### Supported Models
| Model | Architecture | Prunable Units |
|-------|--------------|----------------|
| ResNet-18 | 8 BasicBlocks | 8 (block) / 16 (unit) |
| ResNet-50 | 16 Bottlenecks | 16 (block) |
| ViT-Small | 12 Transformer Blocks | 12 (block) / 24 (layer) |
| ViT-Large | 24 Transformer Blocks | 24 (block) / 48 (layer) |
| BERT | Transformer Encoder | Layer-wise |

### Datasets
- **CIFAR-10/100**: 32x32, 10/100 classes
- **Tiny-ImageNet-200**: 64x64 (resized to 224x224), 200 classes

---

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd CS2420_model_pruning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install GPU-accelerated PyTorch (choose your CUDA version)
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install timm wandb numpy Pillow tqdm transformers datasets
```

### Verify GPU
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

---

## Project Structure

```
CS2420_model_pruning/
├── README.md                    # This file
├── bert/                        # BERT/NLP pruning experiments
│   ├── 1_build_teacher.py
│   ├── 2_build_baseline.py
│   ├── 3_encoder_pruning.py
│   └── README.md
├── resnet+vit/                  # ResNet & ViT experiments
│   ├── 1_build_teacher.py       # Train ResNet teacher
│   ├── 2_build_baseline.py      # L2 baseline (unit-wise)
│   ├── 2_build_baseline_block.py
│   ├── 3_encoder_block.py       # Transformer encoder policy
│   ├── 3_encoder_pruning.py
│   ├── 4_mlp.py                 # MLP policy baseline
│   ├── 5_resnet50_block.py      # ResNet-50 experiments
│   ├── 6_overfit_exp.py         # Cross-dataset transfer
│   ├── 7_aznas_encoder_exp.py   # Pure AZ-NAS loss
│   ├── 8_aznas_encoder_exp_hybird.py  # Hybrid KD + AZ-NAS
│   ├── 9_vit_small_block.py     # ViT-Small block pruning
│   ├── 10_vit_small_block_az_regularization.py
│   ├── 11_finetune_vit_with_masks.py
│   ├── 12_vit_layer_wise_pruning.py
│   ├── 13_vit_large_layer_wise_pruning.py
│   ├── vit_layer_pruning_utils.py
│   ├── vit_large_layer_pruning_utils.py
│   └── README.md
├── AZ-NAS/                      # AZ-NAS focused experiments
│   ├── 9_aznas_loss_c100.py     # AZ-NAS vs baselines
│   ├── 9_aznas_component_ablation.py  # Component ablation
│   ├── 9_aznas_overfitting_test.py    # Generalization test
│   ├── AZNAS_LOSS_TEST.ipynb    # Colab notebook
│   └── README.md
├── checkpoints/                 # Model checkpoints (auto-created)
└── data/                        # Datasets (auto-downloaded)
```

---

## Quick Start

### 1. BERT Experiments (bert/)

Train teacher → L2 baseline → Encoder pruning:

```bash
cd bert
python 1_build_teacher.py --epochs 10
python 2_build_baseline.py
python 3_encoder_pruning.py
```

### 2. ResNet/ViT Experiments (resnet+vit/)

```bash
cd resnet+vit

# Download Tiny-ImageNet
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d data/

# ResNet-18 experiments
python 1_build_teacher.py
python 2_build_baseline.py
python 3_encoder_block.py
python 4_mlp.py

# ResNet-50 with all methods
python 5_resnet50_block.py

# ViT-Small block-wise
python 9_vit_small_block.py

# ViT layer-wise (finer granularity)
python 12_vit_layer_wise_pruning.py
```

### 3. AZ-NAS Experiments (AZ-NAS/)

```bash
cd AZ-NAS

# Fast full experiments (~20 min each)
python 9_aznas_loss_c100.py --fast_full
python 9_aznas_component_ablation.py --fast_full
python 9_aznas_overfitting_test.py --fast_full

# Quick tests (~5-10 min each)
python 9_aznas_loss_c100.py --quick_test
```

---

## Core Concepts

### 1. Knowledge Distillation (KD)

Teacher-student framework where a pruned student learns to mimic the full teacher:

```python
# KD Loss
L_KD = KL_div(student_logits / T, teacher_logits / T) * T²
```
- Temperature `T` softens probability distributions
- Richer training signal than hard labels

### 2. Compression-Aware Policy

Transformer encoder predicts pruning gates conditioned on target FLOPs ratio:

```
Input: [budget_token, block_token_1, ..., block_token_N]
         ↓
   Transformer Encoder (2 layers, 4 heads)
         ↓
Output: gate_1, gate_2, ..., gate_N  (soft gates via sigmoid)
```

### 3. AZ-NAS Metrics

Zero-shot architecture quality indicators (no training required):

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Expressivity** | Entropy of covariance eigenvalues | Feature diversity |
| **Progressivity** | Min expressivity increase across layers | Gradual refinement |
| **Trainability** | Gradient flow quality | Learning capacity |
| **Complexity** | Parameter/FLOPs penalty | Efficiency |

```python
# AZ-NAS Loss (replaces or augments KD)
loss_aznas = -expressivity - 0.1 * progressivity + 0.001 * complexity
```

### 4. Pruning Granularities

| Granularity | Description | Example |
|-------------|-------------|---------|
| **Block-wise** | Entire residual blocks | ResNet BasicBlock, ViT Block |
| **Unit-wise** | Individual conv layers | conv1, conv2 within block |
| **Layer-wise** | Attention & MLP separately | 12 attn + 12 MLP = 24 units |

---

## Experiment Categories

### A. Baseline Comparisons (Files 1-5)

Compare L2 baseline, MLP policy, and Transformer encoder on ResNet-18/50.

### B. Cross-Dataset Transfer (File 6)

Test if pruning policies overfit to training dataset:
- Train policy on Tiny-ImageNet
- Apply to CIFAR-100
- Compare with fresh CIFAR-100 policy

### C. AZ-NAS Integration (Files 7-8)

Replace or augment KD loss with AZ-NAS metrics:
- **Pure AZ-NAS**: Only expressivity + progressivity
- **Hybrid**: KD warmup + AZ-NAS co-training

### D. ViT Experiments (Files 9-13)

Extend to Vision Transformers:
- Block-wise: 12 gates for 12 transformer blocks
- Layer-wise: 24 gates (attention + MLP separate)
- AZ-NAS residual learning + diversity loss

### E. AZ-NAS Ablation (AZ-NAS/)

Systematic study of AZ-NAS components:
- Individual component effects
- Component combinations
- Overfitting/generalization analysis

---

## Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 256 | Training batch size |
| `TEMP_KD` | 2.0-4.0 | KD temperature |
| `RATIO_WEIGHT` | 25.0 | FLOPs ratio loss weight |
| `GATE_TEMP_START/END` | 5.0→0.3 | Gate temperature annealing |
| `FT_EPOCHS` | 10 | Fine-tuning epochs |
| `EVAL_RATIOS` | [0.1-0.9] | Target FLOPs ratios |

---

## Pipeline Overview

```
1. Train Teacher        → Full accuracy baseline
2. Train Policy         → Learn compression-aware gates
   - Sample random ratios
   - Soft gating + KD loss + ratio loss
3. Materialize Masks    → Greedy selection (score/FLOPs)
4. BN Recalibration     → Update statistics
5. Fine-tune            → KD from teacher to pruned student
6. Evaluate             → Accuracy vs FLOPs trade-off
```

---

## Output Format

Results saved as JSON:

```json
{
  "teacher_accuracy": 93.5,
  "results": [
    {
      "target_ratio": 0.5,
      "actual_ratio": 0.48,
      "kept": 8,
      "accuracy_before_ft": 65.2,
      "accuracy_after_ft": 89.1,
      "accuracy_drop": 4.4,
      "mask": [1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
    }
  ]
}
```

---

## File Quick Reference

| File | Model | Pruning | Method |
|------|-------|---------|--------|
| `bert/1-3` | BERT | Layer | Encoder policy |
| `resnet+vit/1` | ResNet-18 | - | Train teacher |
| `resnet+vit/2` | ResNet-18 | Unit/Block | L2 baseline |
| `resnet+vit/3` | ResNet-18 | Unit/Block | Encoder policy |
| `resnet+vit/4` | ResNet-18 | Block | MLP policy |
| `resnet+vit/5` | ResNet-50 | Block | All methods |
| `resnet+vit/6` | ResNet-50 | Block | Transfer test |
| `resnet+vit/7` | ResNet-18 | Block | Pure AZ-NAS |
| `resnet+vit/8` | ResNet-18 | Block | Hybrid KD+AZ-NAS |
| `resnet+vit/9` | ViT-Small | Block | 4 methods |
| `resnet+vit/10` | ViT-Small | Block | AZ-NAS residual |
| `resnet+vit/11` | ViT-Small | Block | Mask fine-tune |
| `resnet+vit/12` | ViT-Small | Layer | Attn/MLP separate |
| `resnet+vit/13` | ViT-Large | Layer | Attn/MLP separate |
| `AZ-NAS/9_loss` | ResNet-18 | Block | AZ-NAS vs baselines |
| `AZ-NAS/9_ablation` | ResNet-18 | Block | Component ablation |
| `AZ-NAS/9_overfit` | ResNet-18 | Block | Generalization test |

---

## Hardware Requirements

- **Minimum**: 8GB GPU VRAM (reduce batch size if needed)
- **Recommended**: 16GB+ GPU VRAM
- **ViT-Large**: 24GB+ recommended (or use batch_size=64)

---

## Citation

If using AZ-NAS metrics, please cite:
```
@article{aznas2023,
  title={AZ-NAS: Assembling Zero-Cost Proxies for Network Architecture Search},
  author={...},
  year={2023}
}
```

---

## License

MIT License - See LICENSE file for details.

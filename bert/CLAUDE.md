# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neural network pruning research project implementing a **knowledge distillation pipeline with compression-aware encoder training** for ResNet-18 on CIFAR-10. The project explores structured pruning at the unit level (per-convolution layer) using a learned policy network.

## Pipeline Architecture

The project follows a **3-stage sequential pipeline**:

### Stage 1: Teacher Model Training ([1_build_teacher.py](1_build_teacher.py))
- Trains a ResNet-18 teacher model on CIFAR-10
- Uses ImageNet pretrained weights, fine-tuned on CIFAR-10 (224×224 resized)
- Outputs: `checkpoints/teacher.pth`

### Stage 2: L2 Baseline ([2_build_baseline_l2.py](2_build_baseline_l2.py))
- Establishes a baseline using L2-norm pruning (no learned policy)
- Operates on **16 units** (2 per block: conv1 and conv2 for each of 8 BasicBlocks)
- Computes L2 scores per convolution: `||W||_F / sqrt(#params)`
- Selects top-k units by L2 score to match target FLOPs ratios [0.1-0.8]
- Fine-tunes with KD-only (no task loss) for 10 epochs
- Outputs: `checkpoints/baseline_l2_16gate_results.json`

### Stage 3: Encoder-Based Pruning ([3_encoder_pruning.py](3_encoder_pruning.py))
- **Main contribution**: Trains a transformer-based policy encoder to learn compression-aware pruning
- **Phase 1**: Train policy encoder (12 epochs total: 2 warmup + 10 training)
  - Policy network predicts 16-d soft gates given target FLOPs ratio
  - Uses **Taylor expansion** for sensitivity: `∇L_KD(r) * r` where r is conv output
  - Token features combine: summarizer(h_in, r) + static features (stage, idx, sub_id, BN gamma, spatial dims) + Taylor score
  - Losses: KD loss + ratio matching (×25) + L1 sparsity on gates (×1e-3)
- **Phase 2**: Materialize binary masks using greedy cost-aware selection based on learned scores
- **Phase 3**: Fine-tune pruned model with KD for 10 epochs per ratio
- Outputs: `checkpoints/compression_aware_encoder_16gate.pth`, `checkpoints/encoder_16gate_results.json`

## Key Architectural Details

### 16-Gate Pruning Granularity
All methods operate on **16 units** (not 8 blocks):
- Each BasicBlock has 2 units: conv1 (3×3, possibly stride=2) and conv2 (3×3, stride=1)
- Gates multiply the output of each conv (after BN+ReLU for conv1, after BN for conv2)
- Skip connections are never pruned

### FLOPs Computation
Per-unit FLOPs include:
- conv1 unit: 3×3 conv + 1×1 downsample (if stride=2)
- conv2 unit: 3×3 conv only
- Formula: `H × W × C_in × C_out × k² / stride²`

### Knowledge Distillation
- Temperature: T=2.0
- KL divergence loss between teacher and student soft logits
- **KD-only training** (no task loss) for all fine-tuning stages

### Compression-Aware Token Features (Stage 3)
Each of 16 tokens contains:
- Summarizer output (64-d): dual-stream pooling of (h_in, conv_output)
- Static features (6-d): stage/3, idx, sub_id (0=conv1, 1=conv2), BN_gamma/(C_in+C_out), H/224, W/224
- Taylor sensitivity (1-d): log1p(|∇L_KD * r|.mean())
- Total: 64+6+1=71-d, projected to 128-d via TokenProj

### Policy Encoder Architecture
- Budget embedding: 1-d ratio → MLP → 128-d budget token
- Transformer encoder: 2 layers, 4 heads, 128-d width, 256-d FFN
- Positional encoding for 17 tokens (1 budget + 16 units)
- Output: 16-d logits → sigmoid(logits/temp) for soft gates

## Running the Pipeline

### Prerequisites
```bash
# Install PyTorch with CUDA support
pip install torch torchvision
```

### Sequential Execution

```bash
# Step 1: Train teacher model (required first)
python 1_build_teacher.py --epochs 10 --batch_size 256 --lr 1e-3

# Step 2: Run L2 baseline (requires teacher.pth)
python 2_build_baseline.py

# Step 3: Train encoder-based pruning (requires teacher.pth)
python 3_encoder_pruning.py
```

### Key Hyperparameters

**Stage 1 (Teacher):**
- `--epochs 10`: Training epochs
- `--batch_size 256`: Batch size
- `--lr 1e-3`: Learning rate with cosine annealing
- `--no_pretrained`: Train from scratch (default: uses ImageNet pretrained)

**Stage 2 (L2 Baseline):**
- Hardcoded in script: `RATIO_LIST = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]`
- `FT_EPOCHS = 10`: Fine-tuning epochs per ratio
- `CALIB_ITERS = 30`: BN recalibration iterations

**Stage 3 (Encoder):**
- `POLICY_TRAIN_EPOCHS = 10`: Policy training epochs (+ 2 warmup)
- `FT_EPOCHS = 10`: Fine-tuning epochs per ratio
- `RATIO_WEIGHT = 25.0`: Weight for ratio matching loss
- `GATE_TEMP_START/END = 5.0 → 0.3`: Temperature annealing for soft gates
- `MIN_RATIO/MAX_RATIO = 0.1/0.8`: Budget sampling range during training

## Important Implementation Notes

### Data Handling
- CIFAR-10 images resized to 224×224 (ResNet-18 expects ImageNet size)
- Normalization: CIFAR-10 stats `(mean=(0.491,0.482,0.446), std=(0.247,0.243,0.261))`
- Auto-download enabled for CIFAR-10 dataset

### Device & Performance
- Mixed precision training (AMP) used in Stage 1
- `torch.backends.cudnn.benchmark = True` for performance
- Stage 3 uses `torch.set_float32_matmul_precision("high")`

### Checkpoints
- Teacher checkpoint stores: `{"state_dict": ..., "val_acc": ...}`
- Encoder checkpoint stores: `{"encoder": ..., "summarizer": ..., "token_proj": ...}`
- Results saved as JSON with full mask details

### Jupyter Compatibility
Stage 1 includes special handling for Jupyter notebooks (detects `ipykernel` in `sys.argv`)

## Modifying the Pipeline

### Changing Pruning Ratios
Edit target ratios in scripts:
- Stage 2: `RATIO_LIST` in [2_build_baseline.py](2_build_baseline_l2.py:28)
- Stage 3: `EVAL_RATIOS` in [3_encoder_pruning.py](3_encoder_pruning.py:51)

### Changing Encoder Architecture
Modify constants in [3_encoder_pruning.py](3_encoder_pruning.py):
- `ENC_WIDTH = 128`: Transformer hidden dimension
- `ENC_LAYERS = 2`: Number of transformer layers
- `ENC_HEADS = 4`: Number of attention heads
- `TOKEN_DIM = 128`: Token embedding dimension
- `SUM_DIM = 64`: Summarizer output dimension

### Loss Weights
Critical hyperparameters in Stage 3:
- `RATIO_WEIGHT = 25.0`: Controls FLOPs budget adherence
- `L1_M_WEIGHT = 1e-3`: Encourages sparsity in soft gates
- `TEMP_KD = 2.0`: KD temperature (shared across all stages)

## Expected Outputs

All checkpoints and results saved to `checkpoints/`:
- `teacher.pth`: Trained teacher model
- `baseline_l2_16gate_results.json`: L2 baseline results for 8 ratios
- `compression_aware_encoder_16gate.pth`: Trained policy encoder
- `encoder_16gate_results.json`: Encoder-based pruning results for 8 ratios

Each JSON contains per-ratio metrics:
- `target_ratio`, `actual_ratio`: Desired vs. achieved FLOPs retention
- `kept`: Number of units kept (out of 16)
- `accuracy_before_ft`, `accuracy_after_ft`: Accuracy metrics
- `accuracy_drop`: Drop from teacher accuracy
- `mask`: Binary mask (16-d list)
- `scores`: Per-unit importance scores (encoder only)

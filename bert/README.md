# ResNet-18 Model Pruning with Compression-Aware Encoder

This repository implements a neural network pruning pipeline using knowledge distillation and a learned compression-aware policy encoder for ResNet-18 on CIFAR-10.

## Overview

The project explores **structured pruning at the unit level** (per-convolution layer) using a transformer-based policy network that learns to predict which layers to keep given a target computational budget (FLOPs ratio).

### Key Features
- **16-gate pruning**: Prunes at convolution granularity (2 units per BasicBlock: conv1 and conv2)
- **Taylor sensitivity**: Uses gradient-based importance scores for each layer
- **Compression-aware tokens**: Rich feature representation combining activation statistics, structural info, and sensitivity
- **Budget-conditioned policy**: Single policy network handles multiple compression ratios
- **Knowledge distillation**: KD-only fine-tuning (no task loss) for better compression

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd CS2420_model_pruning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

For CUDA support, install PyTorch with the appropriate CUDA version:
```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

The pipeline consists of **3 sequential stages** that must be run in order:

### Stage 1: Train Teacher Model

Train a ResNet-18 teacher model on CIFAR-10 (fine-tuned from ImageNet pretrained weights):

```bash
python 1_build_teacher.py --epochs 10 --batch_size 256 --lr 1e-3
```

**Output**: `checkpoints/teacher.pth` (required for subsequent stages)

**Options**:
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 256)
- `--lr`: Learning rate with cosine annealing (default: 1e-3)
- `--wd`: Weight decay (default: 1e-4)
- `--no_pretrained`: Train from scratch instead of using ImageNet weights
- `--data`: Data directory (default: "data")
- `--out`: Output directory for checkpoints (default: "checkpoints")

**Expected result**: ~93-94% accuracy on CIFAR-10 test set

---

### Stage 2: Run L2 Baseline

Establish a baseline using simple L2-norm pruning (no learned policy):

```bash
python 2_build_baseline.py
```

**Requires**: `checkpoints/teacher.pth`

**Output**: `checkpoints/baseline_l2_16gate_results.json`

**What it does**:
- Computes L2 norm for each convolution: `||W||_F / sqrt(#params)`
- Selects top-k units by L2 score to match target FLOPs ratios [0.1, 0.2, ..., 0.8]
- Performs BN recalibration (30 iterations)
- Fine-tunes with KD-only for 10 epochs per ratio
- Evaluates accuracy drop compared to unpruned teacher

**Key hyperparameters** (edit in script):
- `RATIO_LIST`: Target FLOPs retention ratios (default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
- `FT_EPOCHS`: Fine-tuning epochs per ratio (default: 10)
- `CALIB_ITERS`: BN recalibration iterations (default: 30)
- `TEMP_KD`: Knowledge distillation temperature (default: 2.0)

---

### Stage 3: Train Encoder-Based Pruning

Train the compression-aware policy encoder and evaluate pruned models:

```bash
python 3_encoder_pruning.py
```

**Requires**: `checkpoints/teacher.pth`

**Outputs**:
- `checkpoints/compression_aware_encoder_16gate.pth` (trained policy)
- `checkpoints/encoder_16gate_results.json` (evaluation results)

**What it does**: See detailed training illustration below.

**Key hyperparameters** (edit in script):
- `POLICY_TRAIN_EPOCHS`: Policy training epochs (default: 10, plus 2 warmup)
- `FT_EPOCHS`: Fine-tuning epochs per ratio (default: 10)
- `EVAL_RATIOS`: Target ratios for evaluation (default: [0.1, 0.2, ..., 0.8])
- `RATIO_WEIGHT`: Weight for FLOPs ratio matching loss (default: 25.0)
- `L1_M_WEIGHT`: Weight for L1 sparsity on gates (default: 1e-3)
- `GATE_TEMP_START/END`: Temperature annealing for soft gates (default: 5.0 → 0.3)

**Note**: If the encoder checkpoint already exists, Stage 3 will skip training and directly load the pre-trained policy for evaluation.

---

## Detailed Training Illustration: Encoder Method

### Architecture Overview

```
Input: Images (x) + Target FLOPs Ratio (r_target)
           ↓
    ┌──────────────────────────────────────┐
    │  Teacher Network (Frozen)            │
    │  - Forward pass for soft labels      │
    │  - Collect intermediate activations  │
    └──────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────┐
    │  Token Builder (16 tokens)           │
    │  For each of 16 units (conv layers): │
    │  ┌────────────────────────────────┐  │
    │  │ 1. Summarizer(h_in, r_out)     │  │
    │  │    - Dual-stream pooling       │  │
    │  │    - Output: 64-d vector       │  │
    │  │                                │  │
    │  │ 2. Static Features (6-d)       │  │
    │  │    - stage/3, idx, sub_id      │  │
    │  │    - BN_gamma/(C_in+C_out)     │  │
    │  │    - H/224, W/224              │  │
    │  │                                │  │
    │  │ 3. Taylor Sensitivity (1-d)    │  │
    │  │    - log1p(|∇L_KD * r|.mean()) │  │
    │  └────────────────────────────────┘  │
    │  Concatenate → 71-d per unit         │
    └──────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────┐
    │  Token Projection                    │
    │  71-d → 128-d (LayerNorm + Linear)   │
    └──────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────┐
    │  Budget Embedding                    │
    │  r_target (1-d) → MLP → 128-d        │
    └──────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────┐
    │  Compression-Aware Encoder           │
    │  Input: [budget_token, 16 tokens]    │
    │  - Positional encoding (17 positions)│
    │  - Transformer: 2 layers, 4 heads    │
    │  - Hidden dim: 128, FFN: 256         │
    └──────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────┐
    │  Gate Prediction Head                │
    │  128-d → 1-d per token (Linear)      │
    │  Output: 16 logits                   │
    └──────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────┐
    │  Soft Gates (differentiable)         │
    │  m = sigmoid(logits / temp)          │
    │  temp: 5.0 → 0.3 (annealed)          │
    └──────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────┐
    │  Student Network (Frozen weights)    │
    │  Forward with gated outputs:         │
    │  - r1 = conv1(h) * gate[i]           │
    │  - r2 = conv2(r1) * gate[i+1]        │
    └──────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────┐
    │  Loss Computation                    │
    │  L = L_KD + λ_ratio·L_ratio + λ_L1·L_L1 │
    │                                      │
    │  L_KD = KL(student || teacher)       │
    │  L_ratio = (actual_ratio - r_target)²│
    │  L_L1 = mean(|m|)                    │
    │                                      │
    │  λ_ratio = 25.0, λ_L1 = 1e-3         │
    └──────────────────────────────────────┘
```

### Training Process

#### **Phase 1: Policy Encoder Training (12 epochs)**

The policy encoder learns to predict soft gates for any target compression ratio.

**Training loop** (per mini-batch):

1. **Sample target ratio**: `r_target ~ Uniform(0.1, 0.8)`

2. **Build tokens** (requires backward pass for Taylor):
   ```python
   # Forward pass through frozen student to get conv outputs r1, r2
   logits_S, r1_list, r2_list = student.forward(x)

   # Backward pass to compute gradients w.r.t. KD loss
   L_KD = KL_div(logits_S / T, teacher_logits / T)
   L_KD.backward()

   # Compute Taylor sensitivity for each unit
   taylor[i] = log1p(|grad[r_i] * r_i|.mean())
   ```

3. **Forward through encoder**:
   ```python
   # Assemble 16 tokens: [summarizer_output (64-d), static (6-d), taylor (1-d)]
   tokens = [token_1, token_2, ..., token_16]  # Each 71-d
   tokens = TokenProj(tokens)  # 71-d → 128-d

   # Embed budget
   budget_token = BudgetEmbed(r_target)  # 1-d → 128-d

   # Transformer encoding
   logits = Encoder([budget_token, tokens])  # Output: 16 logits
   m = sigmoid(logits / temp)  # Soft gates
   ```

4. **Gated forward through student**:
   ```python
   # Apply gates to conv outputs
   for i, block in enumerate(student.blocks):
       r1 = block.conv1(h) * m[2*i]
       r2 = block.conv2(r1) * m[2*i+1]
       h = block.relu(skip + r2)
   ```

5. **Compute losses**:
   ```python
   # KD loss
   L_KD = KL_div(student_logits / T, teacher_logits / T)

   # Ratio matching loss
   actual_ratio = sum(m[i] * flops[i]) / sum(flops)
   L_ratio = (actual_ratio - r_target)²

   # L1 sparsity on gates
   L_L1 = mean(|m|)

   # Total loss
   L = L_KD + 25.0 * L_ratio + 1e-3 * L_L1
   ```

6. **Update encoder parameters**:
   ```python
   optimizer.zero_grad()
   L.backward()
   optimizer.step()
   ```

**Key techniques**:
- **Cosine learning rate annealing**: `lr = lr_0 * 0.5 * (1 + cos(π * epoch / total_epochs))`
- **Temperature annealing**: Makes gates sharper over time (5.0 → 0.3)
- **Budget diversity**: Random sampling ensures policy works for all ratios
- **Frozen student weights**: Only encoder parameters are trained

---

#### **Phase 2: Mask Materialization**

After training, convert soft gates to binary masks for each target ratio.

**Process** (per ratio):

1. **Collect soft scores**:
   ```python
   # Average over 20 mini-batches for stability
   scores = []
   for x in train_loader[:20]:
       tokens, flops = build_tokens(x)
       logits = encoder(tokens, target_ratio)
       scores.append(sigmoid(logits))
   avg_scores = mean(scores)  # [16]
   ```

2. **Greedy cost-aware selection**:
   ```python
   # Compute efficiency: importance / cost
   efficiency = avg_scores / (flops + ε)

   # Sort units by efficiency (descending)
   sorted_indices = argsort(efficiency, descending=True)

   # Greedily add units until budget is met
   mask = zeros(16)
   accumulated_flops = 0
   for i in sorted_indices:
       if (accumulated_flops + flops[i]) / total_flops <= target_ratio:
           mask[i] = 1.0
           accumulated_flops += flops[i]
       if accumulated_flops / total_flops >= target_ratio * 0.98:
           break

   # Ensure at least one unit is kept
   if sum(mask) == 0:
       mask[sorted_indices[0]] = 1.0
   ```

3. **Compute actual ratio**:
   ```python
   actual_ratio = sum(mask * flops) / sum(flops)
   ```

---

#### **Phase 3: Fine-Tuning (10 epochs per ratio)**

Fine-tune the pruned model with the binary mask to recover accuracy.

**Training loop**:

1. **Initialize student** from teacher weights
2. **Enable gradients** for student parameters
3. **For each epoch**:
   ```python
   for x, y in train_loader:
       # Teacher soft labels
       y_T = teacher(x)

       # Masked student forward
       y_S = student_masked_forward(x, mask)

       # KD-only loss (no task loss)
       L = KL_div(y_S / T, y_T / T) * T²

       # Update student
       optimizer.zero_grad()
       L.backward()
       optimizer.step()
   ```

4. **Evaluate** after each epoch:
   ```python
   accuracy = evaluate(student, test_loader, mask)
   ```

**Why KD-only?**
- Soft labels from teacher provide richer training signal
- Helps pruned model mimic teacher behavior
- Better accuracy recovery than task loss alone

---

### Key Insights

1. **Unit-level granularity**: Pruning at conv layer level (not block level) provides finer control
2. **Taylor sensitivity**: Gradient-based importance captures layer's contribution to KD loss
3. **Compression-aware tokens**: Rich features help encoder understand layer characteristics
4. **Budget conditioning**: Single policy handles multiple compression ratios (no need to retrain)
5. **Cost-aware selection**: Considers both importance (scores) and cost (FLOPs) when selecting layers

---

## Results Format

Both baseline and encoder methods output JSON files with the following structure:

```json
{
  "teacher_accuracy": 93.5,
  "rows": [
    {
      "target_ratio": 0.1,
      "actual_ratio": 0.098,
      "kept": 2,
      "accuracy_before_ft": 20.2,
      "accuracy_after_ft": 82.1,
      "accuracy_drop": 11.4,
      "mask": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      "scores": [0.95, 0.12, 0.08, 0.87, ...]  // Encoder only
    },
    ...
  ]
}
```

**Metrics**:
- `target_ratio`: Desired FLOPs retention (e.g., 0.1 = 10% of original FLOPs)
- `actual_ratio`: Achieved FLOPs retention
- `kept`: Number of units kept out of 16
- `accuracy_before_ft`: Accuracy immediately after applying mask
- `accuracy_after_ft`: Accuracy after fine-tuning
- `accuracy_drop`: Drop from teacher accuracy (percentage points)
- `mask`: Binary mask (16-d list, 1 = keep, 0 = prune)
- `scores`: Per-unit importance scores (encoder method only)

---

## Project Structure

```
CS2420_model_pruning/
├── 1_build_teacher.py           # Stage 1: Train teacher model
├── 2_build_baseline.py          # Stage 2: L2 baseline pruning
├── 3_encoder_pruning.py         # Stage 3: Encoder-based pruning
├── checkpoints/                 # Output directory (auto-created)
│   ├── teacher.pth              # Trained teacher model
│   ├── baseline_l2_16gate_results.json
│   └── compression_aware_encoder_16gate.pth
├── data/                        # CIFAR-10 dataset (auto-downloaded)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── CLAUDE.md                    # Claude Code guidance
└── README.md                    # This file
```


"""
Compression-Aware Block-Wise Pruning for ViT-Small (12 Transformer Blocks)
Model: timm vit_small_patch16_224

Encoder Training with Residual Learning + Diversity Loss:
1. Residual Learning: AZ-NAS scores as INPUT PRIOR (not loss target)
   - Logits = Encoder(x, ratio) + λ * AZ-NAS-Scores
   - Model learns DEVIATIONS from stable AZ-NAS ranking
   - Prevents overfitting while allowing combinatorial exploration

2. Diversity Loss: Cross-ratio contrastive regularization
   - Penalizes when outputs for different ratios are too similar
   - Forces encoder to explore different architectures per budget
   - Uses cosine similarity with threshold (only penalize if sim > 0.5)

AZ-NAS metrics used: Expressivity, Progressivity, Trainability, Complexity
"""
# close warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

import os, math, random, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import json
import copy
import timm
import gc
import numpy as np

# Weights & Biases for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Run 'pip install wandb' for online logging.")

# =========================
# Config
# =========================
DATA_DIR = "data/tiny-imagenet-200"
CKPT_TEACHER = "checkpoints/teacher_vit_small.pth"
OUT_DIR = "checkpoints"
ENCODER_SAVE_PATH = "vit_small_encoder_az_policy.pth"

BATCH_SIZE = 256
NUM_WORKERS = 2
IMG_SIZE = 224
TOKEN_DIM = 128
SUM_DIM = 64
ENC_WIDTH = 128
ENC_LAYERS = 2
ENC_HEADS = 4
TEMP_KD = 2.0

# Policy training config
POLICY_WARMUP_EPOCHS = 2
POLICY_TRAIN_EPOCHS = 10
POLICY_LR = 1e-4
WEIGHT_DECAY = 0.0
SEED = 42

# Budget training config
MIN_RATIO = 0.05
MAX_RATIO = 0.9
RATIO_WEIGHT = 25.0
GATE_TEMP_START = 5.0
GATE_TEMP_END = 0.3

# AZ-NAS Residual Learning config
AZNAS_CACHE_STEPS = 100  # Recompute AZ-NAS scores every N steps
AZNAS_SUBSET_SIZE = 32   # Batch subset for AZ-NAS computation (efficiency)

# Diversity Loss config (Cross-Ratio Contrastive)
DIVERSITY_WEIGHT = 2.0       # Weight for diversity loss
DIVERSITY_RATIO_DIFF = 0.3   # Minimum ratio difference for contrastive pair
DIVERSITY_SIM_THRESHOLD = 0.5  # Only penalize if cosine similarity > threshold

# Inference config
INFERENCE_RATIO_BUFFER = 0.1  # Allow actual_ratio <= target_ratio + buffer

# Early stopping config
EARLY_STOP_PATIENCE = 3       # Stop if no improvement for N epochs
# Stop if no improvement for N epochs
EARLY_STOP_MIN_EPOCHS = 2     # Don't stop before this many epochs
EARLY_STOP_MIN_DELTA = 0.01   # Minimum improvement to reset patience

# ViT-Small has 12 transformer blocks
NUM_BLOCKS = 12

# Validation ratios
VAL_RATIOS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Wandb config
USE_WANDB = True
WANDB_PROJECT = "vit-small-pruning-aznas"
WANDB_ENTITY = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_wandb(run_name, config_dict=None):
    """Initialize wandb run if available and enabled."""
    if USE_WANDB and WANDB_AVAILABLE:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config=config_dict or {},
            reinit=True
        )
        return True
    return False

def log_wandb(metrics, step=None):
    """Log metrics to wandb if available."""
    if USE_WANDB and WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(metrics, step=step)

def finish_wandb():
    """Finish wandb run if active."""
    if USE_WANDB and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def print_vram_usage(phase_name=""):
    """Print current VRAM usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM {phase_name}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def clear_vram():
    """Clear VRAM by running garbage collection and emptying CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_vram_usage("after cleanup")

# =========================
# Dataset
# =========================
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
        if self.transform: image = self.transform(image)
        return image, self.labels[idx]

def get_loaders():
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(mean, std),
    ])

    if not os.path.exists(os.path.join(DATA_DIR, 'train')):
        print(f"Warning: Dataset not found at {DATA_DIR}. Using FakeData for testing.")
        dummy_ds = datasets.FakeData(size=1000, image_size=(3, IMG_SIZE, IMG_SIZE), num_classes=200, transform=transforms.ToTensor())
        return DataLoader(dummy_ds, 32), DataLoader(dummy_ds, 32), 200

    train_ds = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_tf)
    val_ds = TinyImageNetVal(root=os.path.join(DATA_DIR, 'val'), transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes)

# =========================
# Model Building
# =========================
def build_vit_small(num_classes=200, pretrained=False):
    """Build ViT-Small model using timm"""
    if pretrained:
        model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
    else:
        model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
    return model

def train_teacher(train_loader, val_loader, num_classes, epochs=10):
    """Train teacher ViT-Small if checkpoint doesn't exist"""
    print("\nTraining Teacher ViT-Small...")
    model = build_vit_small(num_classes, pretrained=True).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_acc = 0.0
    for ep in range(epochs):
        print(f"Beginning Epoch {ep+1}/{epochs}...")
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()

        acc = run_evaluation(model, val_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save({'state_dict': model.state_dict(), 'val_acc': acc}, CKPT_TEACHER)
        print(f"[Teacher Ep {ep+1}] Acc: {acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

    return model, best_acc

# =========================
# Gated Forward Pass for ViT
# =========================
class GatedViTWrapper(nn.Module):
    """
    Wrapper that applies gates to each transformer block.
    Gate = 0 means skip the block (identity), Gate = 1 means use full block.
    """
    def __init__(self, vit_model):
        super().__init__()
        self.model = vit_model

    def forward_with_gates(self, x, gates):
        """Forward with block-wise gates"""
        # Patch embedding
        x = self.model.patch_embed(x)

        # Add cls token
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embedding
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        # Apply blocks with gates
        for i, block in enumerate(self.model.blocks):
            if i < len(gates):
                g = gates[i]
                block_out = block(x)
                x = g * block_out + (1 - g) * x
            else:
                x = block(x)

        # Final norm and head
        x = self.model.norm(x)
        x = self.model.head(x[:, 0])
        return x

    def forward_collect_features(self, x, collect_grads=False):
        """Forward and collect input/output features for each block"""
        infos = []

        # Patch embedding
        x = self.model.patch_embed(x)

        # Add cls token
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embedding
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        # Collect features from each block
        for i, block in enumerate(self.model.blocks):
            h_in = x
            if collect_grads:
                h_in.retain_grad()

            x = block(x)
            r_out = x
            if collect_grads:
                r_out.retain_grad()

            infos.append({
                "h_in": h_in,
                "r_out": r_out,
                "block_idx": i,
                "seq_len": h_in.size(1),
                "hidden_dim": h_in.size(2)
            })

        # Final norm and head
        x = self.model.norm(x)
        logits = self.model.head(x[:, 0])

        return logits, infos

# =========================
# FLOPs Calculation for ViT Block
# =========================
def get_vit_block_flops(block, seq_len, hidden_dim):
    """
    Estimate FLOPs for a ViT transformer block.
    Each block has: Multi-Head Attention + MLP
    """
    mlp_ratio = 4.0

    # Attention FLOPs
    attn_flops = 4 * seq_len * hidden_dim * hidden_dim
    attn_flops += 2 * seq_len * seq_len * hidden_dim

    # MLP FLOPs
    mlp_hidden = int(hidden_dim * mlp_ratio)
    mlp_flops = 2 * seq_len * hidden_dim * mlp_hidden

    return float(attn_flops + mlp_flops)

def get_all_block_flops(model, input_shape):
    """Get FLOPs for each block"""
    B, C, H, W = input_shape
    patch_size = 16
    num_patches = (H // patch_size) * (W // patch_size)
    seq_len = num_patches + 1
    hidden_dim = model.embed_dim

    flops_list = []
    for block in model.blocks:
        flops = get_vit_block_flops(block, seq_len, hidden_dim)
        flops_list.append(flops)

    return flops_list

# =========================
# AZ-NAS Score Computation for ViT Blocks
# =========================
def compute_vit_block_aznas_scores(model, data_batch, targets=None, subset_size=32):
    """
    Compute AZ-NAS scores (Expressivity, Progressivity, Trainability, Complexity)
    for each ViT transformer block.

    Args:
        model: ViT model
        data_batch: Input batch
        targets: Labels (optional)
        subset_size: Number of samples to use for score computation (for efficiency)

    Returns: Dict with per-block scores and combined importance tensor
    """
    model.train()

    # Use subset for efficiency (SVD on full batch is very expensive)
    x = data_batch[:subset_size].to(device)
    if targets is not None:
        targets = targets[:subset_size].to(device)

    # Forward pass collecting features
    block_features = []

    # Patch embedding
    h = model.patch_embed(x)
    cls_token = model.cls_token.expand(h.shape[0], -1, -1)
    h = torch.cat((cls_token, h), dim=1)
    h = h + model.pos_embed
    h = model.pos_drop(h)

    # Store input to first block
    block_features.append(h)

    # Pass through each block and collect outputs
    for block in model.blocks:
        h = block(h)
        block_features.append(h)

    # Final output
    h_final = model.norm(h)
    logits = model.head(h_final[:, 0])

    # Compute loss for gradient-based scores
    if targets is not None and logits.shape[-1] >= int(targets.max().item()) + 1:
        loss = F.cross_entropy(logits, targets)
    else:
        loss = (logits ** 2).mean()

    model.zero_grad()
    loss.backward(retain_graph=True)

    # Compute scores for each block
    expressivity_scores = []
    progressivity_scores = []
    trainability_scores = []
    complexity_scores = []

    hidden_dim = model.embed_dim
    seq_len = block_features[0].size(1)

    # Expressivity: Entropy of covariance eigenvalues
    with torch.no_grad():
        for i in range(NUM_BLOCKS):
            feat = block_features[i + 1]  # Output of block i
            B, S, D = feat.shape

            # Use subset of sequence for efficiency
            max_seq = min(S, 64)  # Limit sequence length for covariance
            X = feat[:, :max_seq, :].detach().reshape(-1, D)

            mu = X.mean(dim=0, keepdim=True)
            Xc = X - mu
            sigma = (Xc.T @ Xc) / max(1, Xc.shape[0])

            # Eigenvalue-based expressivity with numerical stability
            try:
                eigvals = torch.linalg.eigvalsh(sigma)
                eigvals = torch.relu(eigvals) + 1e-6  # Safety epsilon
                total = eigvals.sum()
                if total > 1e-8:
                    p = eigvals / total
                    exp_score = (-p * torch.log(p + 1e-12)).sum().item()  # Log safety
                else:
                    exp_score = 0.0
            except Exception:
                exp_score = 0.0

            expressivity_scores.append(exp_score)

            # Complexity (FLOPs for this block)
            block_flops = get_vit_block_flops(model.blocks[i], seq_len, hidden_dim)
            complexity_scores.append(block_flops)

    # Progressivity: Change in expressivity between consecutive blocks
    expressivity_arr = np.array(expressivity_scores, dtype=np.float64)
    for i in range(NUM_BLOCKS):
        if i == 0:
            prog = expressivity_arr[i]  # First block uses its own expressivity
        else:
            prog = expressivity_arr[i] - expressivity_arr[i - 1]
        progressivity_scores.append(prog)

    # Trainability: Gradient flow analysis between blocks
    # Use smaller subset for SVD efficiency
    for i in range(NUM_BLOCKS):
        f_out = block_features[i + 1]
        f_in = block_features[i]

        # Random gradient for probing
        g_out = torch.randn_like(f_out) * 0.5
        g_out = torch.sign(g_out)

        try:
            g_in = torch.autograd.grad(
                outputs=f_out, inputs=f_in, grad_outputs=g_out,
                retain_graph=True, allow_unused=True
            )[0]

            if g_in is None:
                train_score = 0.0
            else:
                # Use subset for SVD efficiency
                B, S, D = g_out.shape
                max_samples = min(B * S, 2048)  # Limit samples for SVD

                Go = g_out.view(B * S, D)[:max_samples]
                Gi = g_in.view(B * S, D)[:max_samples]
                M = (Gi.T @ Go) / max(1, max_samples)

                if M.shape[0] < M.shape[1]:
                    M = M.T

                svals = torch.linalg.svdvals(M)
                smax = svals.max().item() if svals.numel() > 0 else 0.0
                train_score = -smax - 1.0 / (smax + 1e-6) + 2.0
        except Exception:
            train_score = 0.0

        trainability_scores.append(train_score)

    model.zero_grad()

    # Convert to tensors
    expressivity = torch.tensor(expressivity_scores, dtype=torch.float32, device=device)
    progressivity = torch.tensor(progressivity_scores, dtype=torch.float32, device=device)
    trainability = torch.tensor(trainability_scores, dtype=torch.float32, device=device)
    complexity = torch.tensor(complexity_scores, dtype=torch.float32, device=device)

    return {
        "expressivity": expressivity,
        "progressivity": progressivity,
        "trainability": trainability,
        "complexity": complexity
    }


def assemble_aznas_importance(scores_dict, complexity_weight=2.0):
    """
    Combine AZ-NAS scores into a single importance score per block.
    Uses rank-based aggregation to handle scale differences.

    Higher score = more important block
    """
    exp = scores_dict["expressivity"]
    prog = scores_dict["progressivity"]
    trn = scores_dict["trainability"]
    comp = scores_dict["complexity"]

    def _ranks(arr, descending=True):
        """Convert values to ranks (1 = best)"""
        if descending:
            order = torch.argsort(arr, descending=True)
        else:
            order = torch.argsort(arr, descending=False)
        ranks = torch.empty_like(arr)
        ranks[order] = torch.arange(1, len(arr) + 1, dtype=arr.dtype, device=arr.device)
        return ranks

    # Rank each metric (lower rank = better)
    # For expressivity, progressivity, trainability: higher is better -> descending
    # For complexity: lower is better -> ascending
    r_exp = _ranks(exp, descending=True)
    r_prog = _ranks(prog, descending=True)
    r_trn = _ranks(trn, descending=True)
    r_comp = _ranks(comp, descending=False)  # Lower complexity is better

    # Combined rank score (lower = more important)
    combined_rank = r_exp + r_prog + r_trn + complexity_weight * r_comp

    # Invert so higher = more important (for consistency with gates)
    importance = -combined_rank

    return importance


# =========================
# Policy Components
# =========================
class ViTSummarizer(nn.Module):
    """Summarizes a ViT Block: Input tokens vs Output tokens"""
    def __init__(self, sum_dim=SUM_DIM, hidden_dim=384):
        super().__init__()
        self.proj_in = nn.Linear(hidden_dim, 128)
        self.proj_out = nn.Linear(hidden_dim, 128)

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, sum_dim), nn.LayerNorm(sum_dim)
        )

    def forward(self, h_in, r_out):
        cls_in = h_in[:, 0]
        mean_in = h_in[:, 1:].mean(dim=1)

        cls_out = r_out[:, 0]
        mean_out = r_out[:, 1:].mean(dim=1)

        feat_in = self.proj_in(cls_in + mean_in)
        feat_out = self.proj_out(cls_out + mean_out)

        combined = torch.cat([feat_in, feat_out], dim=1)
        z = self.mlp(combined)

        return z.mean(dim=0)

class CompressionAwareEncoder(nn.Module):
    """
    Transformer-based policy encoder with:
    1. Residual Learning: Logits = Encoder(x, ratio) + λ * AZ-NAS-Prior
       - Model learns DEVIATIONS from the stable AZ-NAS ranking
       - Prevents overfitting while allowing combinatorial exploration
    2. Strong ratio conditioning for diverse architectures per budget
    """
    def __init__(self, num_blocks=NUM_BLOCKS, dim=ENC_WIDTH, depth=ENC_LAYERS, heads=ENC_HEADS):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim*2.0),
            batch_first=True, activation='gelu', norm_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos = nn.Parameter(torch.zeros(1, num_blocks + 1, dim))

        # Budget embedding for the budget token
        self.budget_embed = nn.Sequential(
            nn.Linear(1, dim * 2), nn.LayerNorm(dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim)
        )

        # Ratio conditioning: modulate each block token based on target ratio
        self.ratio_modulation = nn.Sequential(
            nn.Linear(1, dim), nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Output head with ratio-aware bias
        self.head = nn.Linear(dim, 1)
        self.ratio_bias = nn.Sequential(
            nn.Linear(1, num_blocks * 2), nn.GELU(),
            nn.Linear(num_blocks * 2, num_blocks)
        )

        # Learnable weight for AZ-NAS prior (initialized high for stability)
        # Model learns how much to trust the prior vs its own predictions
        self.aznas_weight = nn.Parameter(torch.tensor(1.0))

        self.num_blocks = num_blocks

    def forward(self, tokens, target_ratio, aznas_prior=None):
        if not torch.is_tensor(target_ratio):
            target_ratio = torch.tensor([[float(target_ratio)]], dtype=torch.float32, device=tokens.device)
        else:
            target_ratio = target_ratio.float().view(1, 1).to(tokens.device)

        budget_tok = self.budget_embed(target_ratio)

        # Modulate block tokens with ratio information (multiplicative + additive)
        ratio_mod = self.ratio_modulation(target_ratio)  # [1, dim]
        tokens_modulated = tokens * (1 + 0.5 * ratio_mod.unsqueeze(1))  # Scale tokens by ratio

        x = torch.cat([budget_tok.unsqueeze(1), tokens_modulated], dim=1)
        x = x + self.pos[:, :x.size(1)]
        h = self.enc(x)
        block_h = h[:, 1:, :]

        # Base logits from transformer output (the "combinatorial context")
        logits = self.head(block_h).squeeze(-1)

        # Add ratio-dependent bias
        ratio_bias = self.ratio_bias(target_ratio)  # [1, num_blocks]
        logits = logits + ratio_bias

        # Add AZ-NAS prior as residual (the "generalization anchor")
        # If encoder outputs 0 (unsure), decision falls back to AZ-NAS
        if aznas_prior is not None:
            # Normalize prior to match logit scale approximately
            prior = aznas_prior.float()
            prior = (prior - prior.mean()) / (prior.std() + 1e-6)
            logits = logits + (self.aznas_weight * prior)

        return logits

# =========================
# Token Building
# =========================
def build_vit_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device):
    """Build tokens for all ViT blocks with Taylor scores"""
    teacher_wrapper = GatedViTWrapper(teacher)
    student_wrapper = GatedViTWrapper(student)

    with torch.no_grad():
        _, teacher_infos = teacher_wrapper.forward_collect_features(x, collect_grads=False)

    logits_S, student_infos = student_wrapper.forward_collect_features(x, collect_grads=True)
    loss = F.kl_div(F.log_softmax(logits_S/TEMP_KD, dim=1),
                   F.softmax(y_T/TEMP_KD, dim=1), reduction='batchmean') * (TEMP_KD**2)

    r_outs = [info["r_out"] for info in student_infos]
    grads = torch.autograd.grad(loss, r_outs, retain_graph=False, create_graph=False, allow_unused=True)

    token_list = []
    flops_list = []
    hidden_dim = teacher.embed_dim
    seq_len = teacher_infos[0]["seq_len"]

    for i in range(NUM_BLOCKS):
        t_info = teacher_infos[i]
        s_info = student_infos[i]

        feats = summarizer(t_info["h_in"], t_info["r_out"]).to(device)

        # Taylor Score
        r_out = s_info["r_out"]
        g = grads[i]

        if g is not None:
            taylor = (g * r_out).abs().mean().detach().item()
        else:
            taylor = 0.0

        # Metadata
        meta = torch.tensor([
            i / (NUM_BLOCKS - 1),
            seq_len / 200.0,
            hidden_dim / 512.0,
        ], dtype=torch.float32, device=device)

        tay_tensor = torch.tensor([math.log1p(taylor)], dtype=torch.float32, device=device)
        tok = torch.cat([feats, meta, tay_tensor], dim=0)

        token_list.append(tok)
        flops_list.append(get_vit_block_flops(teacher.blocks[i], seq_len, hidden_dim))

    # Clean up gradients
    for info in student_infos:
        if info["r_out"].grad is not None:
            info["r_out"].grad = None

    tokens = torch.stack(token_list).unsqueeze(0)
    tokens = token_proj(tokens)
    flops = torch.tensor(flops_list, dtype=torch.float32, device=device)

    student.zero_grad()
    return tokens, flops

# =========================
# Common Utilities
# =========================
def kd_loss(student_logits, teacher_logits, T):
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction='batchmean') * (T*T)

def run_evaluation(model, loader):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

# =========================
# Validation: Test encoder masks at different ratios
# =========================
def validate_encoder_masks(teacher, encoder, summarizer, token_proj, train_loader, global_step, aznas_prior=None):
    """
    Validate encoder by testing mask generation at different ratios.
    Logs masks and statistics for each ratio.
    """
    print("\n" + "="*60)
    print("Validation: Testing encoder masks at different ratios")
    print("="*60)

    encoder.eval()
    summarizer.eval()
    token_proj.eval()

    # Get a sample batch for token building
    x_sample, _ = next(iter(train_loader))
    x_sample = x_sample[:32].to(device)

    with torch.no_grad():
        y_T = teacher(x_sample)

    temp_student = copy.deepcopy(teacher)
    temp_student.eval()

    tokens, flops = build_vit_block_tokens(teacher, temp_student, summarizer, token_proj, x_sample, y_T, device)
    total_flops = flops.sum().item()

    validation_results = {}

    for ratio in VAL_RATIOS:
        with torch.no_grad():
            logits = encoder(tokens, ratio, aznas_prior=aznas_prior).squeeze(0)
            scores = torch.sigmoid(logits)

        # Efficiency-based selection
        # Normalize FLOPs to [0, 1] range for meaningful efficiency display
        flops_normalized = flops / (flops.max() + 1e-9)
        eff = (scores / (flops_normalized + 1e-9)).cpu().numpy()
        indices = sorted(range(NUM_BLOCKS), key=lambda i: eff[i], reverse=True)

        # Find best mask within budget (target_ratio + buffer)
        # This matches training where soft gates can slightly exceed target
        max_allowed_ratio = ratio + INFERENCE_RATIO_BUFFER
        best_mask = None
        best_ratio_diff = float('inf')
        best_actual_ratio = 0

        for num_keep in range(1, NUM_BLOCKS + 1):
            trial_mask = [0] * NUM_BLOCKS
            trial_flops = 0
            for i in range(num_keep):
                idx = indices[i]
                trial_mask[idx] = 1
                trial_flops += flops[idx].item()
            trial_ratio = trial_flops / total_flops

            # Only consider if within allowed budget (with buffer)
            if trial_ratio <= max_allowed_ratio:
                ratio_diff = abs(trial_ratio - ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_mask = trial_mask
                    best_actual_ratio = trial_ratio

        mask = best_mask if best_mask else [1] + [0] * (NUM_BLOCKS - 1)
        actual_ratio = best_actual_ratio
        kept = sum(mask)

        # Detailed logging
        mask_str = ''.join(str(m) for m in mask)
        print(f"\n  --- Ratio {ratio:.1f} (max allowed: {max_allowed_ratio:.2f}) ---")
        print(f"  Mask:         {mask_str}")
        print(f"  Kept Blocks:  {kept}/{NUM_BLOCKS}")
        print(f"  Target Ratio: {ratio:.3f}")
        print(f"  Actual Ratio: {actual_ratio:.3f}")
        print(f"  Ratio Diff:   {abs(actual_ratio - ratio):.3f}")
        print(f"  Encoder Scores (sigmoid): {scores.cpu().numpy().round(3)}")
        print(f"  Encoder Logits (raw):     {logits.cpu().numpy().round(3)}")
        print(f"  Efficiency (score/flops): {eff.round(6)}")
        print(f"  Selection Order:          {indices}")

        # Store results
        validation_results[f"val/ratio_{ratio}/mask"] = mask
        validation_results[f"val/ratio_{ratio}/kept_blocks"] = kept
        validation_results[f"val/ratio_{ratio}/actual_ratio"] = actual_ratio
        validation_results[f"val/ratio_{ratio}/scores"] = scores.cpu().tolist()

        # Log to wandb with detailed metrics
        log_wandb({
            f"val/ratio_{ratio}/kept_blocks": kept,
            f"val/ratio_{ratio}/target_ratio": ratio,
            f"val/ratio_{ratio}/actual_ratio": actual_ratio,
            f"val/ratio_{ratio}/ratio_diff": abs(actual_ratio - ratio),
            f"val/ratio_{ratio}/max_allowed_ratio": max_allowed_ratio,
            f"val/ratio_{ratio}/scores_mean": scores.mean().item(),
            f"val/ratio_{ratio}/scores_std": scores.std().item(),
            f"val/ratio_{ratio}/scores_min": scores.min().item(),
            f"val/ratio_{ratio}/scores_max": scores.max().item(),
            f"val/ratio_{ratio}/logits_mean": logits.mean().item(),
            f"val/ratio_{ratio}/logits_std": logits.std().item(),
        }, step=global_step)

        # Log per-block scores to wandb
        for block_idx in range(NUM_BLOCKS):
            log_wandb({
                f"val/ratio_{ratio}/block_{block_idx}_score": scores[block_idx].item(),
                f"val/ratio_{ratio}/block_{block_idx}_logit": logits[block_idx].item(),
                f"val/ratio_{ratio}/block_{block_idx}_kept": mask[block_idx],
            }, step=global_step)

    # Summary: Compare selection orders across ratios
    print("\n  --- Selection Order Comparison Across Ratios ---")
    all_orders = []
    for ratio in VAL_RATIOS:
        with torch.no_grad():
            logits = encoder(tokens, ratio, aznas_prior=aznas_prior).squeeze(0)
            scores = torch.sigmoid(logits)
        flops_normalized = flops / (flops.max() + 1e-9)
        eff = (scores / (flops_normalized + 1e-9)).cpu().numpy()
        indices = sorted(range(NUM_BLOCKS), key=lambda i: eff[i], reverse=True)
        all_orders.append(indices)
        print(f"  Ratio {ratio:.1f}: {indices}")

    # Check diversity: count how many different orderings
    unique_orders = len(set(tuple(o) for o in all_orders))
    print(f"\n  Unique orderings: {unique_orders}/{len(VAL_RATIOS)}")
    print("="*60 + "\n")

    encoder.train()
    summarizer.train()
    token_proj.train()

    # Return both validation results and unique orderings count for early stopping
    validation_results["unique_orders"] = unique_orders
    return validation_results

# =========================
# ENCODER TRAINING with Residual Learning + Diversity Loss
# =========================
def train_encoder_policy(teacher, train_loader):
    """
    Train transformer encoder policy with:
    1. Residual Learning: AZ-NAS scores as input prior (not loss target)
    2. Diversity Loss: Cross-ratio contrastive to ensure different architectures per ratio
    """
    print("\nPhase 1: Training Block-Wise Policy Encoder for ViT-Small")
    print("Using Residual Learning (AZ-NAS Prior) + Diversity Loss")

    # Initialize wandb for encoder training
    init_wandb("encoder_aznas_residual_diversity", {
        "method": "Transformer Encoder (Residual Learning + Diversity)",
        "model": "ViT-Small",
        "num_blocks": NUM_BLOCKS,
        "policy_lr": POLICY_LR,
        "warmup_epochs": POLICY_WARMUP_EPOCHS,
        "train_epochs": POLICY_TRAIN_EPOCHS,
        "ratio_weight": RATIO_WEIGHT,
        "diversity_weight": DIVERSITY_WEIGHT,
        "diversity_ratio_diff": DIVERSITY_RATIO_DIFF,
        "diversity_sim_threshold": DIVERSITY_SIM_THRESHOLD,
        "aznas_cache_steps": AZNAS_CACHE_STEPS,
        "aznas_subset_size": AZNAS_SUBSET_SIZE,
        "gate_temp_start": GATE_TEMP_START,
        "gate_temp_end": GATE_TEMP_END,
    })

    student = copy.deepcopy(teacher)
    student.eval()

    hidden_dim = teacher.embed_dim
    summarizer = ViTSummarizer(sum_dim=SUM_DIM, hidden_dim=hidden_dim).to(device)
    token_proj = nn.Sequential(
        nn.LayerNorm(SUM_DIM + 4),
        nn.Linear(SUM_DIM + 4, TOKEN_DIM)
    ).to(device)

    encoder = CompressionAwareEncoder(num_blocks=NUM_BLOCKS, dim=ENC_WIDTH).to(device)

    params = list(summarizer.parameters()) + list(token_proj.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=POLICY_LR, weight_decay=WEIGHT_DECAY)

    total_epochs = POLICY_WARMUP_EPOCHS + POLICY_TRAIN_EPOCHS
    loss_history = []
    global_step = 0

    wrapper = GatedViTWrapper(student)

    # Cached AZ-NAS scores (used as prior, not loss target)
    cached_aznas_importance = None
    cached_aznas_scores = None

    # Early stopping tracking (based on total loss)
    best_loss = float('inf')
    epochs_without_improvement = 0
    best_encoder_state = None
    best_summarizer_state = None
    best_token_proj_state = None

    for epoch in range(total_epochs):
        encoder.train(); summarizer.train(); token_proj.train()

        t = epoch / max(1, total_epochs-1)
        temp = GATE_TEMP_START * (1-t) + GATE_TEMP_END * t

        acc_loss_kd = 0; acc_loss_ratio = 0; acc_loss_diversity = 0; acc_loss_total = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                y_T = teacher(x)

            # Recompute AZ-NAS scores every AZNAS_CACHE_STEPS
            if global_step % AZNAS_CACHE_STEPS == 0:
                print(f"\n[Step {global_step}] Recomputing AZ-NAS scores (used as prior)...")
                teacher_copy = copy.deepcopy(teacher)
                cached_aznas_scores = compute_vit_block_aznas_scores(teacher_copy, x, y)
                cached_aznas_importance = assemble_aznas_importance(cached_aznas_scores)
                del teacher_copy

                # Log AZ-NAS scores
                print(f"  Expressivity: {cached_aznas_scores['expressivity'].cpu().numpy()}")
                print(f"  Progressivity: {cached_aznas_scores['progressivity'].cpu().numpy()}")
                print(f"  Trainability: {cached_aznas_scores['trainability'].cpu().numpy()}")
                print(f"  Importance (Prior): {cached_aznas_importance.cpu().numpy()}")

                log_wandb({
                    "aznas/expressivity_mean": cached_aznas_scores['expressivity'].mean().item(),
                    "aznas/progressivity_mean": cached_aznas_scores['progressivity'].mean().item(),
                    "aznas/trainability_mean": cached_aznas_scores['trainability'].mean().item(),
                    "aznas/importance_std": cached_aznas_importance.std().item(),
                }, step=global_step)

            tokens, flops = build_vit_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device)

            # ============================================
            # 1. Primary Forward Pass (Target Ratio)
            # ============================================
            target_ratio = random.uniform(MIN_RATIO, MAX_RATIO)

            # Pass AZ-NAS as prior (residual learning) - NOT as loss target
            logits_primary = encoder(tokens, target_ratio, aznas_prior=cached_aznas_importance).squeeze(0)
            gates = torch.sigmoid(logits_primary / temp)

            full_flops = flops.sum()
            kept_flops = (gates * flops).sum()
            curr_ratio = kept_flops / (full_flops + 1e-6)

            y_S = wrapper.forward_with_gates(x, gates)

            # KD Loss and Ratio Loss
            loss_kd = kd_loss(y_S, y_T, TEMP_KD)
            loss_ratio = (curr_ratio - target_ratio) ** 2

            # ============================================
            # 2. Diversity Loss (Cross-Ratio Contrastive)
            # ============================================
            # Sample a contrast ratio that is significantly different
            if target_ratio + DIVERSITY_RATIO_DIFF <= MAX_RATIO:
                contrast_ratio = target_ratio + DIVERSITY_RATIO_DIFF
            else:
                contrast_ratio = target_ratio - DIVERSITY_RATIO_DIFF

            # Forward pass with SAME tokens but DIFFERENT ratio
            logits_contrast = encoder(tokens, contrast_ratio, aznas_prior=cached_aznas_importance).squeeze(0)

            # Compute cosine similarity between logits for different ratios
            # We want to MINIMIZE similarity (encourage different rankings)
            cos_sim = F.cosine_similarity(
                logits_primary.unsqueeze(0),
                logits_contrast.unsqueeze(0),
                dim=1
            ).mean()

            # Only penalize if similarity is above threshold
            loss_diversity = torch.relu(cos_sim - DIVERSITY_SIM_THRESHOLD)

            # ============================================
            # 3. Total Loss (No structural loss - AZ-NAS is now input prior)
            # ============================================
            loss = loss_kd + RATIO_WEIGHT * loss_ratio + DIVERSITY_WEIGHT * loss_diversity

            opt.zero_grad()
            loss.backward()
            opt.step()

            acc_loss_kd += loss_kd.item()
            acc_loss_ratio += loss_ratio.item()
            acc_loss_diversity += loss_diversity.item()
            acc_loss_total += loss.item()
            global_step += 1

            # Brief progress logging every 50 batches
            if i % 50 == 0:
                print(f"[Ep {epoch}][{i}] KD: {acc_loss_kd/(i+1):.4f} Ratio: {acc_loss_ratio/(i+1):.4f} Div: {acc_loss_diversity/(i+1):.4f} Temp: {temp:.2f}")

        n_batches = len(train_loader)
        epoch_losses = {
            "epoch": epoch,
            "loss_kd": acc_loss_kd / n_batches,
            "loss_ratio": acc_loss_ratio / n_batches,
            "loss_diversity": acc_loss_diversity / n_batches,
            "loss_total": acc_loss_total / n_batches,
            "temp": temp
        }
        loss_history.append(epoch_losses)

        # Detailed epoch-level logging
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Summary")
        print(f"{'='*60}")
        print(f"  KD Loss:         {epoch_losses['loss_kd']:.6f}")
        print(f"  Ratio Loss:      {epoch_losses['loss_ratio']:.6f} (weighted: {RATIO_WEIGHT * epoch_losses['loss_ratio']:.6f})")
        print(f"  Diversity Loss:  {epoch_losses['loss_diversity']:.6f} (weighted: {DIVERSITY_WEIGHT * epoch_losses['loss_diversity']:.6f})")
        print(f"  Total Loss:      {epoch_losses['loss_total']:.6f}")
        print(f"  Gate Temp:       {temp:.3f}")
        print(f"  AZ-NAS Weight:   {encoder.aznas_weight.item():.4f}")
        print(f"{'='*60}")

        # Log epoch metrics to wandb
        log_wandb({
            "encoder/epoch": epoch,
            "encoder/epoch_loss_kd": epoch_losses["loss_kd"],
            "encoder/epoch_loss_ratio": epoch_losses["loss_ratio"],
            "encoder/epoch_loss_diversity": epoch_losses["loss_diversity"],
            "encoder/epoch_loss_total": epoch_losses["loss_total"],
            "encoder/temp": temp,
            "encoder/aznas_weight": encoder.aznas_weight.item(),
        }, step=global_step)

        # Validation at end of each epoch
        val_results = validate_encoder_masks(teacher, encoder, summarizer, token_proj, train_loader, global_step, cached_aznas_importance)
        unique_orders = val_results.get("unique_orders", 0)

        # Log unique orders to wandb
        log_wandb({"encoder/unique_orders": unique_orders}, step=global_step)

        # Early stopping logic (based on total loss)
        current_loss = epoch_losses["loss_total"]
        if current_loss < best_loss - EARLY_STOP_MIN_DELTA:
            best_loss = current_loss
            epochs_without_improvement = 0
            # Save best model state
            best_encoder_state = copy.deepcopy(encoder.state_dict())
            best_summarizer_state = copy.deepcopy(summarizer.state_dict())
            best_token_proj_state = copy.deepcopy(token_proj.state_dict())
            print(f"  [Early Stop] New best loss: {best_loss:.4f} (unique orders: {unique_orders}/{len(VAL_RATIOS)})")
        else:
            epochs_without_improvement += 1
            print(f"  [Early Stop] No improvement (current: {current_loss:.4f}, best: {best_loss:.4f}). Patience: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}")

        # Check if we should stop
        if epoch >= EARLY_STOP_MIN_EPOCHS and epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered at epoch {epoch}!")
            print(f"Best loss: {best_loss:.4f}")
            print(f"No improvement for {epochs_without_improvement} epochs")
            print(f"{'='*60}\n")
            # Restore best model state
            if best_encoder_state is not None:
                encoder.load_state_dict(best_encoder_state)
                summarizer.load_state_dict(best_summarizer_state)
                token_proj.load_state_dict(best_token_proj_state)
            break

    finish_wandb()
    return encoder, summarizer, token_proj, loss_history, cached_aznas_importance

# =========================
# Main
# =========================
def main():
    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("="*60)
    print("ViT-Small Block-Wise Pruning - Encoder Training")
    print("Residual Learning (AZ-NAS Prior) + Diversity Loss")
    print("="*60)

    train_loader, val_loader, num_classes = get_loaders()

    # Load or train teacher
    if os.path.exists(CKPT_TEACHER):
        print(f"\nLoading teacher from {CKPT_TEACHER}")
        teacher = build_vit_small(num_classes).to(device)
        ckpt = torch.load(CKPT_TEACHER, map_location='cpu')
        teacher.load_state_dict(ckpt['state_dict'])
    else:
        print("\nTeacher checkpoint not found. Training teacher...")
        teacher, _ = train_teacher(train_loader, val_loader, num_classes, epochs=10)

    teacher.eval()
    teacher_acc = run_evaluation(teacher, val_loader)
    print(f"\nTeacher Accuracy: {teacher_acc*100:.2f}%")

    # Train encoder policy
    encoder, summarizer, token_proj, loss_history, cached_aznas = train_encoder_policy(teacher, train_loader)

    # Save encoder
    encoder_path = os.path.join(OUT_DIR, ENCODER_SAVE_PATH)
    torch.save({
        'enc': encoder.state_dict(),
        'sum': summarizer.state_dict(),
        'proj': token_proj.state_dict(),
        'loss_history': loss_history,
        'aznas_prior': cached_aznas.cpu() if cached_aznas is not None else None
    }, encoder_path)
    print(f"\nEncoder saved to {encoder_path}")

    # Save loss history as JSON
    loss_history_path = os.path.join(OUT_DIR, "encoder_aznas_loss_history.json")
    with open(loss_history_path, 'w') as f:
        json.dump({
            "method": "Transformer Encoder (Residual Learning + Diversity)",
            "model": "ViT-Small",
            "teacher_accuracy": teacher_acc,
            "num_blocks": NUM_BLOCKS,
            "diversity_weight": DIVERSITY_WEIGHT,
            "diversity_ratio_diff": DIVERSITY_RATIO_DIFF,
            "aznas_subset_size": AZNAS_SUBSET_SIZE,
            "loss_history": loss_history
        }, f, indent=2)
    print(f"Loss history saved to {loss_history_path}")

    # Final validation
    print("\n" + "="*60)
    print("Final Validation")
    print("="*60)
    validate_encoder_masks(teacher, encoder, summarizer, token_proj, train_loader, -1, cached_aznas)

    print("\n" + "="*60)
    print("Encoder Training Complete!")
    print("="*60)

if __name__ == "__main__":
    main()

"""
Compression-Aware Block-Wise Pruning for ViT-Small (12 Transformer Blocks)
Model: timm vit_small_patch16_224

Includes:
- Transformer Encoder Policy
- MLP Policy
- L2 Baseline
- Cosine Similarity Baseline (NEW)

With checkpoint/resume capability.
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
RESULTS_FILE_ENCODER = "vit_small_encoder_block_results.json"
RESULTS_FILE_MLP = "vit_small_mlp_block_results.json"
RESULTS_FILE_L2 = "vit_small_l2_baseline_results.json"
RESULTS_FILE_COSINE = "vit_small_cosine_baseline_results.json"
RESULTS_FILE_COMPARISON = "vit_small_comparison_results.json"
CHECKPOINT_FILE = "vit_small_phase2_checkpoint.json"

BATCH_SIZE = 256  # ViT needs more memory
NUM_WORKERS = 2
IMG_SIZE = 224
TOKEN_DIM = 128
SUM_DIM = 64
ENC_WIDTH = 128
ENC_LAYERS = 2
ENC_HEADS = 4
MLP_HIDDEN_DIM = 512
TEMP_KD = 2.0

# Policy training config
POLICY_WARMUP_EPOCHS = 2
POLICY_TRAIN_EPOCHS = 10
POLICY_LR = 1e-3
WEIGHT_DECAY = 0.0
MLP_WEIGHT_DECAY = 1e-4
SEED = 42

# Budget training config
MIN_RATIO = 0.1
MAX_RATIO = 0.8
RATIO_WEIGHT = 25.0
GATE_TEMP_START = 5.0
GATE_TEMP_END = 0.3

# Fine-tuning config
FT_EPOCHS = 10
FT_LR = 1e-4  # Lower LR for ViT
CALIB_ITERS = 100

EVAL_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
L1_M_WEIGHT = 1e-3

# ViT-Small has 12 transformer blocks
NUM_BLOCKS = 12

# Wandb config
USE_WANDB = True  # Set to False to disable wandb logging
WANDB_PROJECT = "vit-small-pruning"
WANDB_ENTITY = None  # Set to your wandb username/team if needed

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
# ViT Block Access Helpers
# =========================
def get_vit_blocks(model):
    """Get the transformer blocks from ViT model"""
    return model.blocks

def get_num_blocks(model):
    """Get number of transformer blocks"""
    return len(model.blocks)

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
                # Soft gating: output = g * block(x) + (1-g) * x
                # When g=0, identity; when g=1, full block
                block_out = block(x)
                x = g * block_out + (1 - g) * x
            else:
                x = block(x)

        # Final norm and head
        x = self.model.norm(x)
        x = self.model.head(x[:, 0])  # Use CLS token
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
            r_out = x  # Output of the block
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

    Attention FLOPs: 4 * seq_len * hidden_dim^2 + 2 * seq_len^2 * hidden_dim
    MLP FLOPs: 2 * seq_len * hidden_dim * mlp_ratio * hidden_dim
    """
    mlp_ratio = 4.0  # Standard ViT MLP ratio

    # Attention FLOPs
    attn_flops = 4 * seq_len * hidden_dim * hidden_dim  # Q, K, V, Out projections
    attn_flops += 2 * seq_len * seq_len * hidden_dim  # Attention computation

    # MLP FLOPs
    mlp_hidden = int(hidden_dim * mlp_ratio)
    mlp_flops = 2 * seq_len * hidden_dim * mlp_hidden  # Two linear layers

    return float(attn_flops + mlp_flops)

def get_all_block_flops(model, input_shape):
    """Get FLOPs for each block"""
    # For ViT-Small: hidden_dim=384, seq_len=197 (196 patches + 1 cls token)
    B, C, H, W = input_shape
    patch_size = 16
    num_patches = (H // patch_size) * (W // patch_size)
    seq_len = num_patches + 1  # +1 for CLS token
    hidden_dim = model.embed_dim

    flops_list = []
    for block in model.blocks:
        flops = get_vit_block_flops(block, seq_len, hidden_dim)
        flops_list.append(flops)

    return flops_list

# =========================
# Policy Components
# =========================
class ViTSummarizer(nn.Module):
    """Summarizes a ViT Block: Input tokens vs Output tokens"""
    def __init__(self, sum_dim=SUM_DIM, hidden_dim=384):
        super().__init__()
        # Project from hidden_dim to a fixed size
        self.proj_in = nn.Linear(hidden_dim, 128)
        self.proj_out = nn.Linear(hidden_dim, 128)

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, sum_dim), nn.LayerNorm(sum_dim)
        )

    def forward(self, h_in, r_out):
        # h_in, r_out: [B, seq_len, hidden_dim]
        # Use CLS token and mean pooling
        cls_in = h_in[:, 0]  # [B, hidden_dim]
        mean_in = h_in[:, 1:].mean(dim=1)  # [B, hidden_dim]

        cls_out = r_out[:, 0]
        mean_out = r_out[:, 1:].mean(dim=1)

        # Project and combine
        feat_in = self.proj_in(cls_in + mean_in)  # [B, 128]
        feat_out = self.proj_out(cls_out + mean_out)  # [B, 128]

        combined = torch.cat([feat_in, feat_out], dim=1)  # [B, 256]
        z = self.mlp(combined)  # [B, sum_dim]

        return z.mean(dim=0)  # [sum_dim]

class CompressionAwareEncoder(nn.Module):
    """Transformer-based policy encoder"""
    def __init__(self, num_blocks=NUM_BLOCKS, dim=ENC_WIDTH, depth=ENC_LAYERS, heads=ENC_HEADS):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim*2.0),
            batch_first=True, activation='gelu', norm_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos = nn.Parameter(torch.zeros(1, num_blocks + 1, dim))

        self.budget_embed = nn.Sequential(
            nn.Linear(1, dim * 2), nn.LayerNorm(dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim)
        )
        self.head = nn.Linear(dim, 1)
        self.num_blocks = num_blocks

    def forward(self, tokens, target_ratio):
        if not torch.is_tensor(target_ratio):
            target_ratio = torch.tensor([[float(target_ratio)]], dtype=torch.float32, device=tokens.device)
        else:
            target_ratio = target_ratio.float().view(1, 1).to(tokens.device)

        budget_tok = self.budget_embed(target_ratio)
        x = torch.cat([budget_tok.unsqueeze(1), tokens], dim=1)
        x = x + self.pos[:, :x.size(1)]
        h = self.enc(x)
        block_h = h[:, 1:, :]
        logits = self.head(block_h).squeeze(-1)
        return logits

class GlobalMLPPolicy(nn.Module):
    """MLP-based policy (no self-attention)"""
    def __init__(self, num_blocks=NUM_BLOCKS, token_dim=TOKEN_DIM, hidden_dim=MLP_HIDDEN_DIM):
        super().__init__()
        self.budget_embed = nn.Sequential(nn.Linear(1, 64), nn.GELU())
        input_dim = (num_blocks * token_dim) + 64

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_blocks)
        )
        self.num_blocks = num_blocks

    def forward(self, tokens, target_ratio):
        B = tokens.size(0)
        flat_tokens = tokens.view(B, -1)

        if not torch.is_tensor(target_ratio):
            target_ratio = torch.tensor([[float(target_ratio)]], dtype=torch.float32, device=tokens.device)
        else:
            target_ratio = target_ratio.float().view(1, 1).to(tokens.device)

        b_emb = self.budget_embed(target_ratio)
        if b_emb.size(0) != B:
            b_emb = b_emb.expand(B, -1)

        x = torch.cat([flat_tokens, b_emb], dim=1)
        logits = self.net(x)
        return logits

# =========================
# Token Building
# =========================
def build_vit_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device):
    """Build tokens for all ViT blocks with Taylor scores"""
    # x = x.requires_grad_(True)

    teacher_wrapper = GatedViTWrapper(teacher)
    student_wrapper = GatedViTWrapper(student)

    with torch.no_grad():
        _, teacher_infos = teacher_wrapper.forward_collect_features(x, collect_grads=False)

    logits_S, student_infos = student_wrapper.forward_collect_features(x, collect_grads=True)
    loss = F.kl_div(F.log_softmax(logits_S/TEMP_KD, dim=1),
                   F.softmax(y_T/TEMP_KD, dim=1), reduction='batchmean') * (TEMP_KD**2)
    
    # optimize for vram
    r_outs = [info["r_out"] for info in student_infos]
    grads = torch.autograd.grad(loss, r_outs, retain_graph=False, create_graph=False, allow_unused=True)
    
    # loss.backward()

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

        # Metadata: block_idx normalized, relative depth
        meta = torch.tensor([
            i / (NUM_BLOCKS - 1),  # Block position
            seq_len / 200.0,  # Sequence length normalized
            hidden_dim / 512.0,  # Hidden dim normalized
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

def run_evaluation_gated(model, loader, gates):
    """Evaluate with gated forward pass"""
    model.eval()
    wrapper = GatedViTWrapper(model)
    correct = 0; total = 0

    gates_tensor = torch.tensor(gates, dtype=torch.float32, device=device)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = wrapper.forward_with_gates(x, gates_tensor)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

# =========================
# Checkpoint Management
# =========================
def load_checkpoint():
    checkpoint_path = os.path.join(OUT_DIR, CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None

def save_checkpoint(checkpoint_data):
    checkpoint_path = os.path.join(OUT_DIR, CHECKPOINT_FILE)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"Checkpoint saved to {checkpoint_path}")

def clear_checkpoint():
    checkpoint_path = os.path.join(OUT_DIR, CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint cleared.")

# =========================
# L2 BASELINE METHOD
# =========================
def compute_l2_scores(model):
    """Compute L2-norm based importance scores for each ViT block"""
    scores = []

    for block in model.blocks:
        total_norm = 0.0
        total_params = 0

        # Sum L2 norms of all parameters in the block
        for param in block.parameters():
            total_norm += param.data.norm(p=2).item() ** 2
            total_params += param.numel()

        score = math.sqrt(total_norm) / math.sqrt(total_params + 1e-6)
        scores.append(score)

    return scores

def l2_baseline_materialize(teacher, train_loader, val_loader, ratio):
    """L2 baseline: select blocks based on L2 norm scores

    Selection picks blocks closest to target ratio (consistent with encoder/MLP)
    """
    print(f"\n[L2 Baseline] Target Ratio: {ratio}")

    l2_scores = compute_l2_scores(teacher)

    x_sample, _ = next(iter(train_loader))
    flops_list = get_all_block_flops(teacher, x_sample.shape)
    flops = torch.tensor(flops_list, dtype=torch.float32)

    scores_tensor = torch.tensor(l2_scores, dtype=torch.float32)
    eff = (scores_tensor / (flops + 1e-9)).numpy()
    indices = sorted(range(NUM_BLOCKS), key=lambda i: eff[i], reverse=True)

    mask = [0.0] * NUM_BLOCKS
    current_flops = 0
    total_flops = flops.sum().item()

    # Select blocks to get as close to target ratio as possible
    best_mask = None
    best_ratio_diff = float('inf')

    for num_keep in range(1, NUM_BLOCKS + 1):
        trial_mask = [0.0] * NUM_BLOCKS
        trial_flops = 0
        for i in range(num_keep):
            idx = indices[i]
            trial_mask[idx] = 1.0
            trial_flops += flops[idx].item()
        trial_ratio = trial_flops / total_flops
        ratio_diff = abs(trial_ratio - ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_mask = trial_mask
            current_flops = trial_flops

    mask = best_mask if best_mask else [1.0] + [0.0] * (NUM_BLOCKS - 1)
    if best_mask is None:
        current_flops = flops[indices[0]].item()

    real_ratio = current_flops / total_flops
    print(f"Mask: {[int(m) for m in mask]}")
    print(f"Actual Ratio: {real_ratio:.3f}, Kept: {int(sum(mask))}/{NUM_BLOCKS}")

    acc_pre = run_evaluation_gated(teacher, val_loader, mask)
    print(f"Accuracy Pre-FT: {acc_pre*100:.2f}%")

    # Fine-tune
    student = copy.deepcopy(teacher)
    wrapper = GatedViTWrapper(student)
    gates_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(student.parameters(), lr=FT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)

    best_acc = acc_pre
    ft_loss_history = []

    for ep in range(FT_EPOCHS):
        student.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): y_T = teacher(x)
            y_S = wrapper.forward_with_gates(x, gates_tensor)
            loss = kd_loss(y_S, y_T, TEMP_KD)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        acc = run_evaluation_gated(student, val_loader, mask)
        if acc > best_acc: best_acc = acc

        avg_loss = epoch_loss / n_batches
        ft_loss_history.append({"epoch": ep, "loss": avg_loss, "accuracy": acc})
        print(f" [FT Ep {ep+1}] Loss: {avg_loss:.4f} Acc: {acc*100:.2f}%")

    return best_acc, real_ratio, mask, l2_scores, acc_pre, ft_loss_history

# =========================
# COSINE SIMILARITY BASELINE (NEW)
# =========================
def compute_cosine_similarity_scores(model, train_loader, num_batches=20):
    """
    Compute importance scores based on cosine similarity between
    block input and output. Lower similarity = more transformation = more important.

    Score = 1 - cos_sim(input, output)
    """
    print("Computing Cosine Similarity scores...")
    model.eval()
    wrapper = GatedViTWrapper(model)

    cos_sims = [[] for _ in range(NUM_BLOCKS)]

    with torch.no_grad():
        batch_count = 0
        for x, _ in train_loader:
            x = x.to(device)
            _, infos = wrapper.forward_collect_features(x, collect_grads=False)

            for i, info in enumerate(infos):
                h_in = info["h_in"]  # [B, seq_len, hidden_dim]
                r_out = info["r_out"]  # [B, seq_len, hidden_dim]

                # Compute cosine similarity between input and output
                # Use CLS token for comparison
                cls_in = h_in[:, 0]  # [B, hidden_dim]
                cls_out = r_out[:, 0]  # [B, hidden_dim]

                cos_sim = F.cosine_similarity(cls_in, cls_out, dim=1)  # [B]
                cos_sims[i].extend(cos_sim.cpu().tolist())

            batch_count += 1
            if batch_count >= num_batches:
                break

    # Compute importance: 1 - cos_sim (lower similarity = more important)
    scores = []
    for i in range(NUM_BLOCKS):
        avg_cos_sim = sum(cos_sims[i]) / len(cos_sims[i])
        importance = 1.0 - avg_cos_sim  # Invert: lower sim = higher importance
        scores.append(importance)
        print(f"  Block {i}: cos_sim={avg_cos_sim:.4f}, importance={importance:.4f}")

    return scores

def cosine_baseline_materialize(teacher, train_loader, val_loader, ratio):
    """Cosine similarity baseline: select blocks based on transformation importance

    Selection picks blocks closest to target ratio (consistent with encoder/MLP)
    """
    print(f"\n[Cosine Baseline] Target Ratio: {ratio}")

    cosine_scores = compute_cosine_similarity_scores(teacher, train_loader)

    x_sample, _ = next(iter(train_loader))
    flops_list = get_all_block_flops(teacher, x_sample.shape)
    flops = torch.tensor(flops_list, dtype=torch.float32)

    scores_tensor = torch.tensor(cosine_scores, dtype=torch.float32)
    eff = (scores_tensor / (flops + 1e-9)).numpy()
    indices = sorted(range(NUM_BLOCKS), key=lambda i: eff[i], reverse=True)

    mask = [0.0] * NUM_BLOCKS
    current_flops = 0
    total_flops = flops.sum().item()

    # Select blocks to get as close to target ratio as possible
    best_mask = None
    best_ratio_diff = float('inf')

    for num_keep in range(1, NUM_BLOCKS + 1):
        trial_mask = [0.0] * NUM_BLOCKS
        trial_flops = 0
        for i in range(num_keep):
            idx = indices[i]
            trial_mask[idx] = 1.0
            trial_flops += flops[idx].item()
        trial_ratio = trial_flops / total_flops
        ratio_diff = abs(trial_ratio - ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_mask = trial_mask
            current_flops = trial_flops

    mask = best_mask if best_mask else [1.0] + [0.0] * (NUM_BLOCKS - 1)
    if best_mask is None:
        current_flops = flops[indices[0]].item()

    real_ratio = current_flops / total_flops
    print(f"Mask: {[int(m) for m in mask]}")
    print(f"Actual Ratio: {real_ratio:.3f}, Kept: {int(sum(mask))}/{NUM_BLOCKS}")

    acc_pre = run_evaluation_gated(teacher, val_loader, mask)
    print(f"Accuracy Pre-FT: {acc_pre*100:.2f}%")

    # Fine-tune
    student = copy.deepcopy(teacher)
    wrapper = GatedViTWrapper(student)
    gates_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(student.parameters(), lr=FT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)

    best_acc = acc_pre
    ft_loss_history = []

    for ep in range(FT_EPOCHS):
        student.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): y_T = teacher(x)
            y_S = wrapper.forward_with_gates(x, gates_tensor)
            loss = kd_loss(y_S, y_T, TEMP_KD)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        acc = run_evaluation_gated(student, val_loader, mask)
        if acc > best_acc: best_acc = acc

        avg_loss = epoch_loss / n_batches
        ft_loss_history.append({"epoch": ep, "loss": avg_loss, "accuracy": acc})
        print(f" [FT Ep {ep+1}] Loss: {avg_loss:.4f} Acc: {acc*100:.2f}%")

    return best_acc, real_ratio, mask, cosine_scores, acc_pre, ft_loss_history

# =========================
# ENCODER METHOD
# =========================
def train_encoder_policy(teacher, train_loader):
    """Train transformer encoder policy"""
    print("\nPhase 1: Training Block-Wise Policy Encoder for ViT-Small")

    # Initialize wandb for encoder training
    init_wandb("encoder_policy_training", {
        "method": "Transformer Encoder",
        "model": "ViT-Small",
        "num_blocks": NUM_BLOCKS,
        "policy_lr": POLICY_LR,
        "warmup_epochs": POLICY_WARMUP_EPOCHS,
        "train_epochs": POLICY_TRAIN_EPOCHS,
        "ratio_weight": RATIO_WEIGHT,
        "l1_weight": L1_M_WEIGHT,
        "gate_temp_start": GATE_TEMP_START,
        "gate_temp_end": GATE_TEMP_END,
    })

    student = copy.deepcopy(teacher)
    student.eval()

    hidden_dim = teacher.embed_dim
    summarizer = ViTSummarizer(sum_dim=SUM_DIM, hidden_dim=hidden_dim).to(device)
    token_proj = nn.Sequential(
        nn.LayerNorm(SUM_DIM + 4),  # sum_dim + 3 meta + 1 taylor
        nn.Linear(SUM_DIM + 4, TOKEN_DIM)
    ).to(device)

    encoder = CompressionAwareEncoder(num_blocks=NUM_BLOCKS, dim=ENC_WIDTH).to(device)

    params = list(summarizer.parameters()) + list(token_proj.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=POLICY_LR, weight_decay=WEIGHT_DECAY)

    total_epochs = POLICY_WARMUP_EPOCHS + POLICY_TRAIN_EPOCHS
    loss_history = []
    global_step = 0

    wrapper = GatedViTWrapper(student)

    for epoch in range(total_epochs):
        encoder.train(); summarizer.train(); token_proj.train()

        t = epoch / max(1, total_epochs-1)
        temp = GATE_TEMP_START * (1-t) + GATE_TEMP_END * t

        acc_loss_kd = 0; acc_loss_ratio = 0; acc_loss_l1 = 0; acc_loss_total = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                y_T = teacher(x)

            tokens, flops = build_vit_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device)

            target_ratio = random.uniform(MIN_RATIO, MAX_RATIO)
            logits = encoder(tokens, target_ratio).squeeze(0)

            gates = torch.sigmoid(logits / temp)

            full_flops = flops.sum()
            kept_flops = (gates * flops).sum()
            curr_ratio = kept_flops / (full_flops + 1e-6)

            y_S = wrapper.forward_with_gates(x, gates)

            loss_kd = kd_loss(y_S, y_T, TEMP_KD)
            loss_ratio = (curr_ratio - target_ratio) ** 2
            loss_l1 = gates.mean()

            loss = loss_kd + RATIO_WEIGHT * loss_ratio + L1_M_WEIGHT * loss_l1

            opt.zero_grad()
            loss.backward()
            opt.step()

            acc_loss_kd += loss_kd.item()
            acc_loss_ratio += loss_ratio.item()
            acc_loss_l1 += loss_l1.item()
            acc_loss_total += loss.item()
            global_step += 1

            # Log batch metrics to wandb
            if i % 10 == 0:
                log_wandb({
                    "encoder/batch_loss_kd": loss_kd.item(),
                    "encoder/batch_loss_ratio": loss_ratio.item(),
                    "encoder/batch_loss_l1": loss_l1.item(),
                    "encoder/batch_loss_total": loss.item(),
                    "encoder/curr_ratio": curr_ratio.item(),
                    "encoder/target_ratio": target_ratio,
                    "encoder/gate_temp": temp,
                    "encoder/gates_mean": gates.mean().item(),
                    "encoder/gates_std": gates.std().item(),
                }, step=global_step)

            if i % 50 == 0:
                print(f"[Ep {epoch}][{i}] KD: {acc_loss_kd/(i+1):.4f} Ratio: {acc_loss_ratio/(i+1):.4f} L1: {acc_loss_l1/(i+1):.4f} Temp: {temp:.2f}")

        n_batches = len(train_loader)
        epoch_losses = {
            "epoch": epoch,
            "loss_kd": acc_loss_kd / n_batches,
            "loss_ratio": acc_loss_ratio / n_batches,
            "loss_l1": acc_loss_l1 / n_batches,
            "loss_total": acc_loss_total / n_batches,
            "temp": temp
        }
        loss_history.append(epoch_losses)

        # Log epoch metrics to wandb
        log_wandb({
            "encoder/epoch": epoch,
            "encoder/epoch_loss_kd": epoch_losses["loss_kd"],
            "encoder/epoch_loss_ratio": epoch_losses["loss_ratio"],
            "encoder/epoch_loss_l1": epoch_losses["loss_l1"],
            "encoder/epoch_loss_total": epoch_losses["loss_total"],
            "encoder/temp": temp,
        }, step=global_step)

        print(f"Epoch {epoch} Summary - KD: {epoch_losses['loss_kd']:.4f}, Ratio: {epoch_losses['loss_ratio']:.4f}, Total: {epoch_losses['loss_total']:.4f}")

    finish_wandb()
    return encoder, summarizer, token_proj, loss_history

def encoder_materialize_and_finetune(teacher, encoder, summarizer, token_proj, train_loader, val_loader, ratio):
    """Materialize pruned model using encoder scores

    Selection logic matches training: pick blocks closest to target ratio
    (allows going slightly over, consistent with soft gate training)
    """
    print(f"\n[Encoder] Target Ratio: {ratio}")

    x_sample, _ = next(iter(train_loader))
    x_sample = x_sample[:32].to(device)
    with torch.no_grad(): y_T = teacher(x_sample)

    temp_student = copy.deepcopy(teacher)
    temp_student.eval()

    tokens, flops = build_vit_block_tokens(teacher, temp_student, summarizer, token_proj, x_sample, y_T, device)
    logits = encoder(tokens, ratio).squeeze(0)
    scores = torch.sigmoid(logits)

    # Efficiency-based ranking (same as before)
    eff = (scores / (flops + 1e-9)).detach().cpu().numpy()
    indices = sorted(range(NUM_BLOCKS), key=lambda i: eff[i], reverse=True)

    mask = [0.0] * NUM_BLOCKS
    current_flops = 0
    total_flops = flops.sum().item()

    # NEW: Select blocks to get as close to target ratio as possible
    # (allows going slightly over, matching training behavior)
    best_mask = None
    best_ratio_diff = float('inf')

    for num_keep in range(1, NUM_BLOCKS + 1):
        trial_mask = [0.0] * NUM_BLOCKS
        trial_flops = 0
        for i in range(num_keep):
            idx = indices[i]
            trial_mask[idx] = 1.0
            trial_flops += flops[idx].item()
        trial_ratio = trial_flops / total_flops
        ratio_diff = abs(trial_ratio - ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_mask = trial_mask
            current_flops = trial_flops

    mask = best_mask if best_mask else [1.0] + [0.0] * (NUM_BLOCKS - 1)
    if best_mask is None:
        current_flops = flops[indices[0]].item()

    real_ratio = current_flops / total_flops
    print(f"Mask: {[int(m) for m in mask]}")
    print(f"Actual Ratio: {real_ratio:.3f}, Kept: {int(sum(mask))}/{NUM_BLOCKS}")

    acc_pre = run_evaluation_gated(teacher, val_loader, mask)
    print(f"Accuracy Pre-FT: {acc_pre*100:.2f}%")

    # Fine-tune
    student = copy.deepcopy(teacher)
    wrapper = GatedViTWrapper(student)
    gates_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(student.parameters(), lr=FT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)

    best_acc = acc_pre
    ft_loss_history = []

    for ep in range(FT_EPOCHS):
        student.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): y_T = teacher(x)
            y_S = wrapper.forward_with_gates(x, gates_tensor)
            loss = kd_loss(y_S, y_T, TEMP_KD)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        acc = run_evaluation_gated(student, val_loader, mask)
        if acc > best_acc: best_acc = acc

        avg_loss = epoch_loss / n_batches
        ft_loss_history.append({"epoch": ep, "loss": avg_loss, "accuracy": acc})
        print(f" [FT Ep {ep+1}] Loss: {avg_loss:.4f} Acc: {acc*100:.2f}%")

    return best_acc, real_ratio, mask, scores.detach().cpu().tolist(), acc_pre, ft_loss_history

# =========================
# MLP METHOD
# =========================
def train_mlp_policy(teacher, train_loader):
    """Train MLP-based policy"""
    print("\nPhase 1: Training MLP Policy for ViT-Small")

    # Initialize wandb for MLP training
    init_wandb("mlp_policy_training", {
        "method": "MLP Policy",
        "model": "ViT-Small",
        "num_blocks": NUM_BLOCKS,
        "policy_lr": POLICY_LR,
        "warmup_epochs": POLICY_WARMUP_EPOCHS,
        "train_epochs": POLICY_TRAIN_EPOCHS,
        "ratio_weight": RATIO_WEIGHT,
        "l1_weight": L1_M_WEIGHT,
        "mlp_weight_decay": MLP_WEIGHT_DECAY,
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

    policy = GlobalMLPPolicy(num_blocks=NUM_BLOCKS, token_dim=TOKEN_DIM).to(device)

    params = list(summarizer.parameters()) + list(token_proj.parameters()) + list(policy.parameters())
    opt = torch.optim.AdamW(params, lr=POLICY_LR, weight_decay=MLP_WEIGHT_DECAY)

    total_epochs = POLICY_WARMUP_EPOCHS + POLICY_TRAIN_EPOCHS
    loss_history = []
    global_step = 0

    wrapper = GatedViTWrapper(student)

    for epoch in range(total_epochs):
        policy.train(); summarizer.train(); token_proj.train()

        t = epoch / max(1, total_epochs-1)
        temp = GATE_TEMP_START * (1-t) + GATE_TEMP_END * t

        acc_loss_kd = 0; acc_loss_ratio = 0; acc_loss_l1 = 0; acc_loss_total = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                y_T = teacher(x)

            tokens, flops = build_vit_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device)

            target_ratio = random.uniform(MIN_RATIO, MAX_RATIO)
            logits = policy(tokens, target_ratio).squeeze(0)

            gates = torch.sigmoid(logits / temp)

            full_flops = flops.sum()
            kept_flops = (gates * flops).sum()
            curr_ratio = kept_flops / (full_flops + 1e-6)

            y_S = wrapper.forward_with_gates(x, gates)

            loss_kd = kd_loss(y_S, y_T, TEMP_KD)
            loss_ratio = (curr_ratio - target_ratio) ** 2
            loss_l1 = gates.mean()

            loss = loss_kd + RATIO_WEIGHT * loss_ratio + L1_M_WEIGHT * loss_l1

            opt.zero_grad()
            loss.backward()
            opt.step()

            acc_loss_kd += loss_kd.item()
            acc_loss_ratio += loss_ratio.item()
            acc_loss_l1 += loss_l1.item()
            acc_loss_total += loss.item()
            global_step += 1

            # Log batch metrics to wandb
            if i % 10 == 0:
                log_wandb({
                    "mlp/batch_loss_kd": loss_kd.item(),
                    "mlp/batch_loss_ratio": loss_ratio.item(),
                    "mlp/batch_loss_l1": loss_l1.item(),
                    "mlp/batch_loss_total": loss.item(),
                    "mlp/curr_ratio": curr_ratio.item(),
                    "mlp/target_ratio": target_ratio,
                    "mlp/gate_temp": temp,
                    "mlp/gates_mean": gates.mean().item(),
                    "mlp/gates_std": gates.std().item(),
                }, step=global_step)

            if i % 50 == 0:
                print(f"[Ep {epoch}][{i}] KD: {acc_loss_kd/(i+1):.4f} Ratio: {acc_loss_ratio/(i+1):.4f} L1: {acc_loss_l1/(i+1):.4f} Temp: {temp:.2f}")

        n_batches = len(train_loader)
        epoch_losses = {
            "epoch": epoch,
            "loss_kd": acc_loss_kd / n_batches,
            "loss_ratio": acc_loss_ratio / n_batches,
            "loss_l1": acc_loss_l1 / n_batches,
            "loss_total": acc_loss_total / n_batches,
            "temp": temp
        }
        loss_history.append(epoch_losses)

        # Log epoch metrics to wandb
        log_wandb({
            "mlp/epoch": epoch,
            "mlp/epoch_loss_kd": epoch_losses["loss_kd"],
            "mlp/epoch_loss_ratio": epoch_losses["loss_ratio"],
            "mlp/epoch_loss_l1": epoch_losses["loss_l1"],
            "mlp/epoch_loss_total": epoch_losses["loss_total"],
            "mlp/temp": temp,
        }, step=global_step)

        print(f"Epoch {epoch} Summary - KD: {epoch_losses['loss_kd']:.4f}, Ratio: {epoch_losses['loss_ratio']:.4f}, Total: {epoch_losses['loss_total']:.4f}")

    finish_wandb()
    return policy, summarizer, token_proj, loss_history

def mlp_materialize_and_finetune(teacher, policy, summarizer, token_proj, train_loader, val_loader, ratio):
    """Materialize pruned model using MLP policy scores

    Selection logic matches training: pick blocks closest to target ratio
    (allows going slightly over, consistent with soft gate training)
    """
    print(f"\n[MLP] Target Ratio: {ratio}")

    x_sample, _ = next(iter(train_loader))
    x_sample = x_sample[:32].to(device)
    with torch.no_grad(): y_T = teacher(x_sample)

    temp_student = copy.deepcopy(teacher)
    temp_student.eval()

    tokens, flops = build_vit_block_tokens(teacher, temp_student, summarizer, token_proj, x_sample, y_T, device)
    logits = policy(tokens, ratio).squeeze(0)
    scores = torch.sigmoid(logits)

    # Efficiency-based ranking (same as before)
    eff = (scores / (flops + 1e-9)).detach().cpu().numpy()
    indices = sorted(range(NUM_BLOCKS), key=lambda i: eff[i], reverse=True)

    mask = [0.0] * NUM_BLOCKS
    current_flops = 0
    total_flops = flops.sum().item()

    # NEW: Select blocks to get as close to target ratio as possible
    # (allows going slightly over, matching training behavior)
    best_mask = None
    best_ratio_diff = float('inf')

    for num_keep in range(1, NUM_BLOCKS + 1):
        trial_mask = [0.0] * NUM_BLOCKS
        trial_flops = 0
        for i in range(num_keep):
            idx = indices[i]
            trial_mask[idx] = 1.0
            trial_flops += flops[idx].item()
        trial_ratio = trial_flops / total_flops
        ratio_diff = abs(trial_ratio - ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_mask = trial_mask
            current_flops = trial_flops

    mask = best_mask if best_mask else [1.0] + [0.0] * (NUM_BLOCKS - 1)
    if best_mask is None:
        current_flops = flops[indices[0]].item()

    real_ratio = current_flops / total_flops
    print(f"Mask: {[int(m) for m in mask]}")
    print(f"Actual Ratio: {real_ratio:.3f}, Kept: {int(sum(mask))}/{NUM_BLOCKS}")

    acc_pre = run_evaluation_gated(teacher, val_loader, mask)
    print(f"Accuracy Pre-FT: {acc_pre*100:.2f}%")

    # Fine-tune
    student = copy.deepcopy(teacher)
    wrapper = GatedViTWrapper(student)
    gates_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(student.parameters(), lr=FT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)

    best_acc = acc_pre
    ft_loss_history = []

    for ep in range(FT_EPOCHS):
        student.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): y_T = teacher(x)
            y_S = wrapper.forward_with_gates(x, gates_tensor)
            loss = kd_loss(y_S, y_T, TEMP_KD)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        acc = run_evaluation_gated(student, val_loader, mask)
        if acc > best_acc: best_acc = acc

        avg_loss = epoch_loss / n_batches
        ft_loss_history.append({"epoch": ep, "loss": avg_loss, "accuracy": acc})
        print(f" [FT Ep {ep+1}] Loss: {avg_loss:.4f} Acc: {acc*100:.2f}%")

    return best_acc, real_ratio, mask, scores.detach().cpu().tolist(), acc_pre, ft_loss_history

# =========================
# Main
# =========================
def main():
    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("="*60)
    print("ViT-Small Block-Wise Pruning (12 Transformer Blocks)")
    print("Methods: Encoder + MLP + L2 + Cosine Similarity Baseline")
    print("="*60)

    # Initialize wandb for overall experiment tracking
    init_wandb("vit_small_pruning_experiment", {
        "model": "ViT-Small",
        "num_blocks": NUM_BLOCKS,
        "eval_ratios": EVAL_RATIOS,
        "ft_epochs": FT_EPOCHS,
        "ft_lr": FT_LR,
        "batch_size": BATCH_SIZE,
    })

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

    # Log teacher accuracy
    log_wandb({"teacher_accuracy": teacher_acc})

    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"\nResuming from checkpoint...")
        print(f"  Completed methods: {checkpoint.get('completed_methods', [])}")

    # Initialize results storage
    l2_results = checkpoint.get('l2_results', []) if checkpoint else []
    cosine_results = checkpoint.get('cosine_results', []) if checkpoint else []
    mlp_results = checkpoint.get('mlp_results', []) if checkpoint else []
    encoder_results = checkpoint.get('encoder_results', []) if checkpoint else []
    completed_methods = checkpoint.get('completed_methods', []) if checkpoint else []

    # =========================
    # Run L2 Baseline
    # =========================
    if 'l2' not in completed_methods:
        print("\n" + "="*60)
        print("Running L2 Baseline Method")
        print("="*60)

        completed_l2_ratios = [r['target_ratio'] for r in l2_results]

        for r in EVAL_RATIOS:
            if r in completed_l2_ratios:
                print(f"\n[L2 Baseline] Ratio {r} already completed, skipping...")
                continue

            best_acc, actual_ratio, mask, scores, acc_before_ft, ft_losses = l2_baseline_materialize(
                teacher, train_loader, val_loader, r
            )
            l2_results.append({
                "target_ratio": r,
                "actual_ratio": actual_ratio,
                "kept": int(sum(mask)),
                "accuracy_before_ft": acc_before_ft,
                "accuracy_after_ft": best_acc,
                "accuracy_drop": teacher_acc - best_acc,
                "mask": [int(m) for m in mask],
                "scores": scores,
                "ft_loss_history": ft_losses
            })

            # Log L2 results to wandb
            log_wandb({
                f"l2/ratio_{r}/accuracy_before_ft": acc_before_ft,
                f"l2/ratio_{r}/accuracy_after_ft": best_acc,
                f"l2/ratio_{r}/accuracy_drop": teacher_acc - best_acc,
                f"l2/ratio_{r}/actual_ratio": actual_ratio,
                f"l2/ratio_{r}/kept_blocks": int(sum(mask)),
            })

            save_checkpoint({
                'teacher_accuracy': teacher_acc,
                'completed_methods': completed_methods,
                'current_method': 'l2',
                'l2_results': l2_results,
                'cosine_results': cosine_results,
                'mlp_results': mlp_results,
                'encoder_results': encoder_results
            })

        # Mark L2 as complete and save checkpoint
        completed_methods.append('l2')
        save_checkpoint({
            'teacher_accuracy': teacher_acc,
            'completed_methods': completed_methods,
            'current_method': 'l2',
            'l2_results': l2_results,
            'cosine_results': cosine_results,
            'mlp_results': mlp_results,
            'encoder_results': encoder_results
        })

        l2_results_path = os.path.join(OUT_DIR, RESULTS_FILE_L2)
        with open(l2_results_path, 'w') as f:
            json.dump({
                "method": "L2 Baseline",
                "model": "ViT-Small",
                "teacher_accuracy": teacher_acc,
                "num_blocks": NUM_BLOCKS,
                "results": l2_results
            }, f, indent=2)
        print(f"\nL2 Baseline results saved to {l2_results_path}")
        clear_vram()

    # =========================
    # Run Cosine Similarity Baseline
    # =========================
    if 'cosine' not in completed_methods:
        print("\n" + "="*60)
        print("Running Cosine Similarity Baseline Method")
        print("="*60)

        completed_cosine_ratios = [r['target_ratio'] for r in cosine_results]

        for r in EVAL_RATIOS:
            if r in completed_cosine_ratios:
                print(f"\n[Cosine Baseline] Ratio {r} already completed, skipping...")
                continue

            best_acc, actual_ratio, mask, scores, acc_before_ft, ft_losses = cosine_baseline_materialize(
                teacher, train_loader, val_loader, r
            )
            cosine_results.append({
                "target_ratio": r,
                "actual_ratio": actual_ratio,
                "kept": int(sum(mask)),
                "accuracy_before_ft": acc_before_ft,
                "accuracy_after_ft": best_acc,
                "accuracy_drop": teacher_acc - best_acc,
                "mask": [int(m) for m in mask],
                "scores": scores,
                "ft_loss_history": ft_losses
            })

            # Log Cosine results to wandb
            log_wandb({
                f"cosine/ratio_{r}/accuracy_before_ft": acc_before_ft,
                f"cosine/ratio_{r}/accuracy_after_ft": best_acc,
                f"cosine/ratio_{r}/accuracy_drop": teacher_acc - best_acc,
                f"cosine/ratio_{r}/actual_ratio": actual_ratio,
                f"cosine/ratio_{r}/kept_blocks": int(sum(mask)),
            })

            save_checkpoint({
                'teacher_accuracy': teacher_acc,
                'completed_methods': completed_methods,
                'current_method': 'cosine',
                'l2_results': l2_results,
                'cosine_results': cosine_results,
                'mlp_results': mlp_results,
                'encoder_results': encoder_results
            })

        # Mark Cosine as complete and save checkpoint
        completed_methods.append('cosine')
        save_checkpoint({
            'teacher_accuracy': teacher_acc,
            'completed_methods': completed_methods,
            'current_method': 'cosine',
            'l2_results': l2_results,
            'cosine_results': cosine_results,
            'mlp_results': mlp_results,
            'encoder_results': encoder_results
        })

        cosine_results_path = os.path.join(OUT_DIR, RESULTS_FILE_COSINE)
        with open(cosine_results_path, 'w') as f:
            json.dump({
                "method": "Cosine Similarity Baseline",
                "model": "ViT-Small",
                "teacher_accuracy": teacher_acc,
                "num_blocks": NUM_BLOCKS,
                "results": cosine_results
            }, f, indent=2)
        print(f"\nCosine Baseline results saved to {cosine_results_path}")
        clear_vram()

    # =========================
    # Run MLP Method
    # =========================
    if 'mlp' not in completed_methods:
        print("\n" + "="*60)
        print("Running MLP Policy Method")
        print("="*60)

        mlp_policy_path = os.path.join(OUT_DIR, "vit_small_mlp_policy.pth")

        if os.path.exists(mlp_policy_path):
            print("Loading MLP Policy...")
            state = torch.load(mlp_policy_path)
            hidden_dim = teacher.embed_dim
            mlp_policy = GlobalMLPPolicy(num_blocks=NUM_BLOCKS).to(device)
            mlp_policy.load_state_dict(state['pol'])
            mlp_summarizer = ViTSummarizer(sum_dim=SUM_DIM, hidden_dim=hidden_dim).to(device)
            mlp_summarizer.load_state_dict(state['sum'])
            mlp_token_proj = nn.Sequential(nn.LayerNorm(SUM_DIM+4), nn.Linear(SUM_DIM+4, TOKEN_DIM)).to(device)
            mlp_token_proj.load_state_dict(state['proj'])
            mlp_policy_loss_history = state.get('loss_history', [])
        else:
            mlp_policy, mlp_summarizer, mlp_token_proj, mlp_policy_loss_history = train_mlp_policy(teacher, train_loader)
            torch.save({
                'pol': mlp_policy.state_dict(),
                'sum': mlp_summarizer.state_dict(),
                'proj': mlp_token_proj.state_dict(),
                'loss_history': mlp_policy_loss_history
            }, mlp_policy_path)

        completed_mlp_ratios = [r['target_ratio'] for r in mlp_results]

        for r in EVAL_RATIOS:
            if r in completed_mlp_ratios:
                print(f"\n[MLP] Ratio {r} already completed, skipping...")
                continue

            best_acc, actual_ratio, mask, scores, acc_before_ft, ft_losses = mlp_materialize_and_finetune(
                teacher, mlp_policy, mlp_summarizer, mlp_token_proj, train_loader, val_loader, r
            )
            mlp_results.append({
                "target_ratio": r,
                "actual_ratio": actual_ratio,
                "kept": int(sum(mask)),
                "accuracy_before_ft": acc_before_ft,
                "accuracy_after_ft": best_acc,
                "accuracy_drop": teacher_acc - best_acc,
                "mask": [int(m) for m in mask],
                "scores": scores,
                "ft_loss_history": ft_losses
            })

            # Log MLP results to wandb
            log_wandb({
                f"mlp_eval/ratio_{r}/accuracy_before_ft": acc_before_ft,
                f"mlp_eval/ratio_{r}/accuracy_after_ft": best_acc,
                f"mlp_eval/ratio_{r}/accuracy_drop": teacher_acc - best_acc,
                f"mlp_eval/ratio_{r}/actual_ratio": actual_ratio,
                f"mlp_eval/ratio_{r}/kept_blocks": int(sum(mask)),
            })

            save_checkpoint({
                'teacher_accuracy': teacher_acc,
                'completed_methods': completed_methods,
                'current_method': 'mlp',
                'l2_results': l2_results,
                'cosine_results': cosine_results,
                'mlp_results': mlp_results,
                'encoder_results': encoder_results
            })

        # Mark MLP as complete and save checkpoint
        completed_methods.append('mlp')
        save_checkpoint({
            'teacher_accuracy': teacher_acc,
            'completed_methods': completed_methods,
            'current_method': 'mlp',
            'l2_results': l2_results,
            'cosine_results': cosine_results,
            'mlp_results': mlp_results,
            'encoder_results': encoder_results
        })

        mlp_results_path = os.path.join(OUT_DIR, RESULTS_FILE_MLP)
        with open(mlp_results_path, 'w') as f:
            json.dump({
                "method": "MLP Policy",
                "model": "ViT-Small",
                "teacher_accuracy": teacher_acc,
                "num_blocks": NUM_BLOCKS,
                "policy_loss_history": mlp_policy_loss_history,
                "results": mlp_results
            }, f, indent=2)
        print(f"\nMLP results saved to {mlp_results_path}")
        clear_vram()

    # =========================
    # Run Encoder Method
    # =========================
    if 'encoder' not in completed_methods:
        print("\n" + "="*60)
        print("Running Transformer Encoder Method")
        print("="*60)

        encoder_path = os.path.join(OUT_DIR, "vit_small_encoder_policy.pth")

        if os.path.exists(encoder_path):
            print("Loading Encoder Policy...")
            state = torch.load(encoder_path)
            hidden_dim = teacher.embed_dim
            encoder = CompressionAwareEncoder(num_blocks=NUM_BLOCKS).to(device)
            encoder.load_state_dict(state['enc'])
            summarizer = ViTSummarizer(sum_dim=SUM_DIM, hidden_dim=hidden_dim).to(device)
            summarizer.load_state_dict(state['sum'])
            token_proj = nn.Sequential(nn.LayerNorm(SUM_DIM+4), nn.Linear(SUM_DIM+4, TOKEN_DIM)).to(device)
            token_proj.load_state_dict(state['proj'])
            policy_loss_history = state.get('loss_history', [])
        else:
            encoder, summarizer, token_proj, policy_loss_history = train_encoder_policy(teacher, train_loader)
            torch.save({
                'enc': encoder.state_dict(),
                'sum': summarizer.state_dict(),
                'proj': token_proj.state_dict(),
                'loss_history': policy_loss_history
            }, encoder_path)

        completed_encoder_ratios = [r['target_ratio'] for r in encoder_results]

        for r in EVAL_RATIOS:
            if r in completed_encoder_ratios:
                print(f"\n[Encoder] Ratio {r} already completed, skipping...")
                continue

            best_acc, actual_ratio, mask, scores, acc_before_ft, ft_losses = encoder_materialize_and_finetune(
                teacher, encoder, summarizer, token_proj, train_loader, val_loader, r
            )
            encoder_results.append({
                "target_ratio": r,
                "actual_ratio": actual_ratio,
                "kept": int(sum(mask)),
                "accuracy_before_ft": acc_before_ft,
                "accuracy_after_ft": best_acc,
                "accuracy_drop": teacher_acc - best_acc,
                "mask": [int(m) for m in mask],
                "scores": scores,
                "ft_loss_history": ft_losses
            })

            # Log Encoder results to wandb
            log_wandb({
                f"encoder_eval/ratio_{r}/accuracy_before_ft": acc_before_ft,
                f"encoder_eval/ratio_{r}/accuracy_after_ft": best_acc,
                f"encoder_eval/ratio_{r}/accuracy_drop": teacher_acc - best_acc,
                f"encoder_eval/ratio_{r}/actual_ratio": actual_ratio,
                f"encoder_eval/ratio_{r}/kept_blocks": int(sum(mask)),
            })

            save_checkpoint({
                'teacher_accuracy': teacher_acc,
                'completed_methods': completed_methods,
                'current_method': 'encoder',
                'l2_results': l2_results,
                'cosine_results': cosine_results,
                'mlp_results': mlp_results,
                'encoder_results': encoder_results
            })

        # Mark Encoder as complete and save checkpoint
        completed_methods.append('encoder')
        save_checkpoint({
            'teacher_accuracy': teacher_acc,
            'completed_methods': completed_methods,
            'current_method': 'encoder',
            'l2_results': l2_results,
            'cosine_results': cosine_results,
            'mlp_results': mlp_results,
            'encoder_results': encoder_results
        })

        encoder_results_path = os.path.join(OUT_DIR, RESULTS_FILE_ENCODER)
        with open(encoder_results_path, 'w') as f:
            json.dump({
                "method": "Transformer Encoder",
                "model": "ViT-Small",
                "teacher_accuracy": teacher_acc,
                "num_blocks": NUM_BLOCKS,
                "policy_loss_history": policy_loss_history,
                "results": encoder_results
            }, f, indent=2)
        print(f"\nEncoder results saved to {encoder_results_path}")
        clear_vram()

    # =========================
    # Save Comparison
    # =========================
    comparison_path = os.path.join(OUT_DIR, RESULTS_FILE_COMPARISON)
    with open(comparison_path, 'w') as f:
        json.dump({
            "model": "ViT-Small",
            "teacher_accuracy": teacher_acc,
            "num_blocks": NUM_BLOCKS,
            "eval_ratios": EVAL_RATIOS,
            "l2_baseline": l2_results,
            "cosine_baseline": cosine_results,
            "mlp": mlp_results,
            "encoder": encoder_results
        }, f, indent=2)
    print(f"\nComparison results saved to {comparison_path}")

    clear_checkpoint()

    # =========================
    # Print Summary
    # =========================
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Ratio':<8} {'L2 Acc':<12} {'Cosine Acc':<12} {'MLP Acc':<12} {'Encoder Acc':<12} {'Best':<10}")
    print("-"*66)
    for i, r in enumerate(EVAL_RATIOS):
        l2_acc = l2_results[i]["accuracy_after_ft"] * 100 if i < len(l2_results) else 0
        cos_acc = cosine_results[i]["accuracy_after_ft"] * 100 if i < len(cosine_results) else 0
        mlp_acc = mlp_results[i]["accuracy_after_ft"] * 100 if i < len(mlp_results) else 0
        enc_acc = encoder_results[i]["accuracy_after_ft"] * 100 if i < len(encoder_results) else 0
        best = max(l2_acc, cos_acc, mlp_acc, enc_acc)
        best_method = "L2" if best == l2_acc else ("Cos" if best == cos_acc else ("MLP" if best == mlp_acc else "Enc"))
        print(f"{r:<8.1f} {l2_acc:<12.2f} {cos_acc:<12.2f} {mlp_acc:<12.2f} {enc_acc:<12.2f} {best_method:<10}")

        # Log comparison summary to wandb
        log_wandb({
            f"comparison/ratio_{r}/l2_acc": l2_acc,
            f"comparison/ratio_{r}/cosine_acc": cos_acc,
            f"comparison/ratio_{r}/mlp_acc": mlp_acc,
            f"comparison/ratio_{r}/encoder_acc": enc_acc,
            f"comparison/ratio_{r}/best_acc": best,
        })

    print("\n" + "="*80)

    # Finish wandb run
    finish_wandb()

if __name__ == "__main__":
    main()

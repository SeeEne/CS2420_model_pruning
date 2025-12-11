"""
Compression-Aware Block-Wise Pruning for ResNet-50 (16 Bottleneck Blocks)
Includes: Transformer Encoder Policy + MLP Policy + L2 Baseline for comparison
With checkpoint/resume capability for Phase 2
"""

import os, math, random, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
import json
import copy

# =========================
# Config
# =========================
DATA_DIR = "data/tiny-imagenet-200"
CKPT_TEACHER = "checkpoints/teacher_resnet50.pth"
OUT_DIR = "checkpoints"
RESULTS_FILE_ENCODER = "resnet50_encoder_block_results.json"
RESULTS_FILE_MLP = "resnet50_mlp_block_results.json"
RESULTS_FILE_L2 = "resnet50_l2_baseline_results.json"
RESULTS_FILE_COMPARISON = "resnet50_comparison_results.json"
CHECKPOINT_FILE = "resnet50_phase2_checkpoint.json"

BATCH_SIZE = 256  # Smaller batch for ResNet-50 (more memory)
NUM_WORKERS = 0
IMG_SIZE = 224
TOKEN_DIM = 128
SUM_DIM = 64
ENC_WIDTH = 128
ENC_LAYERS = 2
ENC_HEADS = 4
MLP_HIDDEN_DIM = 512  # For MLP policy
TEMP_KD = 2.0

# Policy training config
POLICY_WARMUP_EPOCHS = 2
POLICY_TRAIN_EPOCHS = 10
POLICY_LR = 1e-3
WEIGHT_DECAY = 0.0
MLP_WEIGHT_DECAY = 1e-4  # Higher for MLP to prevent overfitting
SEED = 42

# Budget training config
MIN_RATIO = 0.1
MAX_RATIO = 0.8
RATIO_WEIGHT = 25.0
GATE_TEMP_START = 5.0
GATE_TEMP_END = 0.3

# Fine-tuning config
FT_EPOCHS = 10
FT_LR = 1e-3
CALIB_ITERS = 100

EVAL_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
L1_M_WEIGHT = 1e-3

# ResNet-50 has 4 stages with [3, 4, 6, 3] Bottleneck blocks = 16 blocks total
NUM_BLOCKS = 16
BLOCKS_PER_STAGE = [3, 4, 6, 3]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

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

    # Check if dataset exists
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
def build_resnet50(num_classes=200):
    """Build ResNet-50 model with custom classifier"""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_teacher(train_loader, val_loader, num_classes, epochs=10):
    """Train teacher ResNet-50 if checkpoint doesn't exist"""
    print("\nTraining Teacher ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
# Pruned Block for Bottleneck
# =========================
class PrunedBottleneck(nn.Module):
    """
    Physical replacement for a pruned Bottleneck block.
    Only runs the skip connection (with optional downsample).
    """
    expansion = 4  # Bottleneck expansion factor

    def __init__(self, downsample_module):
        super().__init__()
        self.downsample = downsample_module

    def forward(self, x):
        if self.downsample is not None:
            return F.relu(self.downsample(x))
        else:
            return F.relu(x)

# =========================
# Policy Components
# =========================
class Summarizer(nn.Module):
    """Summarizes a Bottleneck Block: Inputs vs Residual Output"""
    def __init__(self, sum_dim=SUM_DIM, k=64):
        super().__init__()
        self.pool1d = nn.AdaptiveAvgPool1d(k)
        self.proj = nn.Linear(4 * k, 256)
        self.mlp = nn.Sequential(
            nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, sum_dim), nn.LayerNorm(sum_dim)
        )

    def _pool_vec(self, v):
        v = v.unsqueeze(1)
        v = self.pool1d(v)
        return v.squeeze(1)

    def forward(self, h_in, r_out):
        gap_h = h_in.mean(dim=(2,3)); gmp_h, _ = h_in.flatten(2).max(dim=2)
        gap_r = r_out.mean(dim=(2,3)); gmp_r, _ = r_out.flatten(2).max(dim=2)
        parts = [self._pool_vec(p) for p in [gap_h, gmp_h, gap_r, gmp_r]]
        feats = torch.cat(parts, dim=1)
        z = self.mlp(self.proj(feats))
        return z.mean(dim=0)

class CompressionAwareEncoder(nn.Module):
    """Transformer-based policy encoder for 16 blocks"""
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
    """
    MLP-based policy (no self-attention).
    Flattens all block tokens, concatenates budget, predicts all gates at once.
    """
    def __init__(self, num_blocks=NUM_BLOCKS, token_dim=TOKEN_DIM, hidden_dim=MLP_HIDDEN_DIM):
        super().__init__()

        # Budget embedding: Map 1 scalar -> 64 dim
        self.budget_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU()
        )

        # Input: (num_blocks * token_dim) + 64 budget dim
        input_dim = (num_blocks * token_dim) + 64

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_blocks)  # Output num_blocks logits
        )
        self.num_blocks = num_blocks

    def forward(self, tokens, target_ratio):
        # tokens shape: [Batch, num_blocks, token_dim]
        B = tokens.size(0)

        # Flatten tokens
        flat_tokens = tokens.view(B, -1)

        if not torch.is_tensor(target_ratio):
            target_ratio = torch.tensor([[float(target_ratio)]], dtype=torch.float32, device=tokens.device)
        else:
            target_ratio = target_ratio.float().view(1, 1).to(tokens.device)

        # Embed budget
        b_emb = self.budget_embed(target_ratio)
        if b_emb.size(0) != B:
            b_emb = b_emb.expand(B, -1)

        # Concatenate
        x = torch.cat([flat_tokens, b_emb], dim=1)

        # MLP
        logits = self.net(x)
        return logits

# =========================
# FLOPs Calculation for Bottleneck
# =========================
def conv_flops(H, W, Cin, Cout, k, stride):
    return (H // stride) * (W // stride) * Cin * Cout * (k * k)

def get_bottleneck_flops(block, input_shape):
    """
    Calculate FLOPs for a Bottleneck block's residual branch.
    Bottleneck has: conv1 (1x1) -> conv2 (3x3) -> conv3 (1x1)
    """
    B, C, H, W = input_shape

    # Conv1: 1x1 reducing channels
    f1 = conv_flops(H, W, block.conv1.in_channels, block.conv1.out_channels, 1, 1)

    # Conv2: 3x3 (possibly with stride)
    stride2 = block.conv2.stride[0]
    H2, W2 = H // stride2, W // stride2
    f2 = conv_flops(H, W, block.conv2.in_channels, block.conv2.out_channels, 3, stride2)

    # Conv3: 1x1 expanding channels
    f3 = conv_flops(H2, W2, block.conv3.in_channels, block.conv3.out_channels, 1, 1)

    return float(f1 + f2 + f3)

# =========================
# Forward Pass Helpers
# =========================
def forward_resnet50_collect_blocks(model, x, collect_grads=False):
    """
    Forward pass through ResNet-50, collecting info for all 16 Bottleneck blocks.
    """
    infos = []

    # Stem
    h = model.conv1(x); h = model.bn1(h); h = model.relu(h); h = model.maxpool(h)

    stages = [model.layer1, model.layer2, model.layer3, model.layer4]

    for s_idx, stage in enumerate(stages):
        for b_idx, block in enumerate(stage):
            h_in = h
            if collect_grads: h_in.retain_grad()

            # --- Residual Branch (Bottleneck: conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu -> conv3 -> bn3) ---
            out = block.conv1(h_in)
            out = block.bn1(out)
            out = block.relu(out)

            out = block.conv2(out)
            out = block.bn2(out)
            out = block.relu(out)

            out = block.conv3(out)
            r_out = block.bn3(out)  # Final residual before addition
            if collect_grads: r_out.retain_grad()

            # --- Skip ---
            if block.downsample is not None:
                skip = block.downsample(h_in)
            else:
                skip = h_in

            # Store info
            infos.append({
                "h_in": h_in,
                "r_out": r_out,
                "block_obj": block,
                "stage": s_idx,
                "block_idx": b_idx,
                "H": h_in.size(2),
                "W": h_in.size(3),
                "flops": get_bottleneck_flops(block, h_in.shape)
            })

            # Final activation
            h = F.relu(skip + r_out)

    # Final classifier
    h = model.avgpool(h)
    h = torch.flatten(h, 1)
    logits = model.fc(h)

    return logits, infos

def forward_resnet50_gated_blocks(model, x, block_gates):
    """
    Forward pass with 16 gates (one per Bottleneck block).
    """
    gate_idx = 0

    h = model.conv1(x); h = model.bn1(h); h = model.relu(h); h = model.maxpool(h)

    for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in stage:
            # Residual Branch
            r = block.conv1(h); r = block.bn1(r); r = block.relu(r)
            r = block.conv2(r); r = block.bn2(r); r = block.relu(r)
            r = block.conv3(r); r = block.bn3(r)

            # Apply Gate
            g = block_gates[gate_idx].view(1,1,1,1)
            r = r * g
            gate_idx += 1

            # Skip Connection
            if block.downsample is not None:
                skip = block.downsample(h)
            else:
                skip = h

            h = F.relu(skip + r)

    h = model.avgpool(h)
    h = torch.flatten(h, 1)
    return model.fc(h)

def build_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device, num_blocks=NUM_BLOCKS):
    """
    Builds tokens for all blocks with Taylor scores.
    """
    x = x.requires_grad_(True)

    with torch.no_grad():
        _, teacher_infos = forward_resnet50_collect_blocks(teacher, x, collect_grads=False)

    logits_S, student_infos = forward_resnet50_collect_blocks(student, x, collect_grads=True)
    loss = F.kl_div(F.log_softmax(logits_S/TEMP_KD, dim=1),
                   F.softmax(y_T/TEMP_KD, dim=1), reduction='batchmean') * (TEMP_KD**2)
    loss.backward()

    token_list = []
    flops_list = []

    # Normalize block_idx based on blocks per stage
    max_blocks_in_stage = max(BLOCKS_PER_STAGE)

    for i in range(num_blocks):
        t_info = teacher_infos[i]
        s_info = student_infos[i]

        feats = summarizer(t_info["h_in"], t_info["r_out"]).to(device)

        grad_r = s_info["r_out"].grad
        if grad_r is not None:
            taylor = (grad_r * s_info["r_out"]).abs().mean().detach().item()
        else:
            taylor = 0.0

        # Metadata (normalized)
        meta = torch.tensor([
            t_info["stage"] / 3.0,
            t_info["block_idx"] / max_blocks_in_stage,
            t_info["H"] / IMG_SIZE,
            t_info["W"] / IMG_SIZE
        ], dtype=torch.float32, device=device)

        tay_tensor = torch.tensor([math.log1p(taylor)], dtype=torch.float32, device=device)
        tok = torch.cat([feats, meta, tay_tensor], dim=0)

        token_list.append(tok)
        flops_list.append(t_info["flops"])

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

def recalibrate_bn(model, loader, steps=CALIB_ITERS):
    """Recalibrate BN statistics after pruning"""
    print(f"Recalibrating BN statistics ({steps} batches)...")
    model.train()
    for p in model.parameters(): p.requires_grad = False

    cnt = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _ = model(x)
            cnt += 1
            if cnt >= steps: break

    for p in model.parameters(): p.requires_grad = True

def physically_prune_model(original_model, mask):
    """
    Creates a new model where dropped Bottleneck blocks are replaced
    with PrunedBottleneck (skip connection only).
    """
    pruned_model = copy.deepcopy(original_model)
    mask_idx = 0

    stages = [pruned_model.layer1, pruned_model.layer2, pruned_model.layer3, pruned_model.layer4]

    for stage in stages:
        for b_idx in range(len(stage)):
            keep = mask[mask_idx]
            if keep == 0:
                original_block = stage[b_idx]
                new_block = PrunedBottleneck(original_block.downsample)
                stage[b_idx] = new_block
            mask_idx += 1

    return pruned_model.to(device)

# =========================
# Checkpoint Management
# =========================
def load_checkpoint():
    """Load Phase 2 checkpoint if exists"""
    checkpoint_path = os.path.join(OUT_DIR, CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None

def save_checkpoint(checkpoint_data):
    """Save Phase 2 checkpoint"""
    checkpoint_path = os.path.join(OUT_DIR, CHECKPOINT_FILE)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"Checkpoint saved to {checkpoint_path}")

def clear_checkpoint():
    """Remove checkpoint after completion"""
    checkpoint_path = os.path.join(OUT_DIR, CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint cleared.")

# =========================
# L2 BASELINE METHOD
# =========================
def compute_l2_scores(model):
    """
    Compute L2-norm based importance scores for each Bottleneck block.
    Score = ||W||_F / sqrt(#params) for conv1, conv2, conv3 combined
    """
    scores = []
    stages = [model.layer1, model.layer2, model.layer3, model.layer4]

    for stage in stages:
        for block in stage:
            # Sum L2 norms of conv1, conv2, conv3
            total_norm = 0.0
            total_params = 0

            for conv in [block.conv1, block.conv2, block.conv3]:
                w = conv.weight.data
                total_norm += w.norm(p=2).item() ** 2
                total_params += w.numel()

            # Normalized score
            score = math.sqrt(total_norm) / math.sqrt(total_params)
            scores.append(score)

    return scores

def get_block_flops_list(model, input_shape):
    """Get FLOPs for each block"""
    flops_list = []
    B, C, H, W = input_shape

    stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    spatial_sizes = [(56, 56), (28, 28), (14, 14), (7, 7)]  # For 224x224 input

    for s_idx, stage in enumerate(stages):
        H, W = spatial_sizes[s_idx]
        for b_idx, block in enumerate(stage):
            # Adjust H, W for first block of stage (might have stride)
            if b_idx == 0 and s_idx > 0:
                input_H, input_W = spatial_sizes[s_idx - 1]
            else:
                input_H, input_W = H, W

            flops = get_bottleneck_flops(block, (1, block.conv1.in_channels, input_H, input_W))
            flops_list.append(flops)

    return flops_list

def l2_baseline_materialize(teacher, train_loader, val_loader, ratio):
    """L2 baseline: select blocks based on L2 norm scores"""
    print(f"\n[L2 Baseline] Target Ratio: {ratio}")

    # Compute L2 scores
    l2_scores = compute_l2_scores(teacher)

    # Get FLOPs
    x_sample, _ = next(iter(train_loader))
    flops_list = get_block_flops_list(teacher, x_sample.shape)
    flops = torch.tensor(flops_list, dtype=torch.float32)

    # Efficiency = score / flops (higher is better to keep)
    scores_tensor = torch.tensor(l2_scores, dtype=torch.float32)
    eff = (scores_tensor / (flops + 1e-9)).numpy()
    indices = sorted(range(NUM_BLOCKS), key=lambda i: eff[i], reverse=True)

    # Greedy selection
    mask = [0] * NUM_BLOCKS
    current_flops = 0
    total_flops = flops.sum().item()

    for idx in indices:
        f = flops[idx].item()
        if (current_flops + f) / total_flops <= ratio:
            mask[idx] = 1
            current_flops += f

    if sum(mask) == 0:
        mask[indices[0]] = 1
        current_flops = flops[indices[0]].item()

    real_ratio = current_flops / total_flops
    print(f"Mask: {mask}")
    print(f"Actual Ratio: {real_ratio:.3f}, Kept: {sum(mask)}/{NUM_BLOCKS}")

    # Prune and fine-tune
    pruned_model = physically_prune_model(teacher, mask)
    recalibrate_bn(pruned_model, train_loader)

    acc_pre = run_evaluation(pruned_model, val_loader)
    print(f"Accuracy Pre-FT: {acc_pre*100:.2f}%")

    # Fine-tune
    opt = torch.optim.AdamW(pruned_model.parameters(), lr=FT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)

    best_acc = acc_pre
    ft_loss_history = []

    for ep in range(FT_EPOCHS):
        pruned_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): y_T = teacher(x)
            y_S = pruned_model(x)
            loss = kd_loss(y_S, y_T, TEMP_KD)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        acc = run_evaluation(pruned_model, val_loader)
        if acc > best_acc: best_acc = acc

        avg_loss = epoch_loss / n_batches
        ft_loss_history.append({"epoch": ep, "loss": avg_loss, "accuracy": acc})
        print(f" [FT Ep {ep+1}] Loss: {avg_loss:.4f} Acc: {acc*100:.2f}%")

    return best_acc, real_ratio, mask, l2_scores, acc_pre, ft_loss_history

# =========================
# ENCODER METHOD
# =========================
def train_encoder_policy(teacher, train_loader):
    """Train transformer encoder policy"""
    print("\nPhase 1: Training Block-Wise Policy Encoder for ResNet-50")

    student = build_resnet50(num_classes=200).to(device)
    student.load_state_dict(teacher.state_dict())
    student.eval()

    summarizer = Summarizer(sum_dim=SUM_DIM).to(device)
    token_proj = nn.Sequential(
        nn.LayerNorm(SUM_DIM + 5),
        nn.Linear(SUM_DIM + 5, TOKEN_DIM)
    ).to(device)

    encoder = CompressionAwareEncoder(num_blocks=NUM_BLOCKS, dim=ENC_WIDTH).to(device)

    params = list(summarizer.parameters()) + list(token_proj.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=POLICY_LR, weight_decay=WEIGHT_DECAY)

    total_epochs = POLICY_WARMUP_EPOCHS + POLICY_TRAIN_EPOCHS
    loss_history = []

    for epoch in range(total_epochs):
        encoder.train(); summarizer.train(); token_proj.train()

        t = epoch / max(1, total_epochs-1)
        temp = GATE_TEMP_START * (1-t) + GATE_TEMP_END * t

        acc_loss_kd = 0; acc_loss_ratio = 0; acc_loss_l1 = 0; acc_loss_total = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                y_T = teacher(x)

            tokens, flops = build_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device)

            target_ratio = random.uniform(MIN_RATIO, MAX_RATIO)
            logits = encoder(tokens, target_ratio).squeeze(0)

            gates = torch.sigmoid(logits / temp)

            full_flops = flops.sum()
            kept_flops = (gates * flops).sum()
            curr_ratio = kept_flops / (full_flops + 1e-6)

            y_S = forward_resnet50_gated_blocks(student, x, gates)

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
        print(f"Epoch {epoch} Summary - KD: {epoch_losses['loss_kd']:.4f}, Ratio: {epoch_losses['loss_ratio']:.4f}, Total: {epoch_losses['loss_total']:.4f}")

    return encoder, summarizer, token_proj, loss_history

def encoder_materialize_and_finetune(teacher, encoder, summarizer, token_proj, train_loader, val_loader, ratio):
    """Materialize pruned model using encoder scores"""
    print(f"\n[Encoder] Target Ratio: {ratio}")

    x_sample, _ = next(iter(train_loader))
    x_sample = x_sample[:32].to(device)
    with torch.no_grad(): y_T = teacher(x_sample)

    temp_student = build_resnet50(200).to(device)
    temp_student.load_state_dict(teacher.state_dict())
    temp_student.eval()

    tokens, flops = build_block_tokens(teacher, temp_student, summarizer, token_proj, x_sample, y_T, device)
    logits = encoder(tokens, ratio).squeeze(0)
    scores = torch.sigmoid(logits)

    # Greedy Selection
    eff = (scores / (flops + 1e-9)).detach().cpu().numpy()
    indices = sorted(range(NUM_BLOCKS), key=lambda i: eff[i], reverse=True)

    mask = [0] * NUM_BLOCKS
    current_flops = 0
    total_flops = flops.sum().item()

    for idx in indices:
        f = flops[idx].item()
        if (current_flops + f) / total_flops <= ratio:
            mask[idx] = 1
            current_flops += f

    if sum(mask) == 0:
        mask[indices[0]] = 1
        current_flops = flops[indices[0]].item()

    real_ratio = current_flops / total_flops
    print(f"Mask: {mask}")
    print(f"Actual Ratio: {real_ratio:.3f}, Kept: {sum(mask)}/{NUM_BLOCKS}")

    # Prune
    pruned_model = physically_prune_model(teacher, mask)
    recalibrate_bn(pruned_model, train_loader)

    acc_pre = run_evaluation(pruned_model, val_loader)
    print(f"Accuracy Pre-FT: {acc_pre*100:.2f}%")

    # Fine-tune
    opt = torch.optim.AdamW(pruned_model.parameters(), lr=FT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)

    best_acc = acc_pre
    ft_loss_history = []

    for ep in range(FT_EPOCHS):
        pruned_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): y_T = teacher(x)
            y_S = pruned_model(x)
            loss = kd_loss(y_S, y_T, TEMP_KD)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        acc = run_evaluation(pruned_model, val_loader)
        if acc > best_acc: best_acc = acc

        avg_loss = epoch_loss / n_batches
        ft_loss_history.append({"epoch": ep, "loss": avg_loss, "accuracy": acc})
        print(f" [FT Ep {ep+1}] Loss: {avg_loss:.4f} Acc: {acc*100:.2f}%")

    return best_acc, real_ratio, mask, scores.detach().cpu().tolist(), acc_pre, ft_loss_history

# =========================
# MLP METHOD
# =========================
def train_mlp_policy(teacher, train_loader):
    """Train MLP-based policy (no transformer)"""
    print("\nPhase 1: Training MLP Policy for ResNet-50")

    student = build_resnet50(num_classes=200).to(device)
    student.load_state_dict(teacher.state_dict())
    student.eval()

    summarizer = Summarizer(sum_dim=SUM_DIM).to(device)
    token_proj = nn.Sequential(
        nn.LayerNorm(SUM_DIM + 5),
        nn.Linear(SUM_DIM + 5, TOKEN_DIM)
    ).to(device)

    policy = GlobalMLPPolicy(num_blocks=NUM_BLOCKS, token_dim=TOKEN_DIM).to(device)

    params = list(summarizer.parameters()) + list(token_proj.parameters()) + list(policy.parameters())
    opt = torch.optim.AdamW(params, lr=POLICY_LR, weight_decay=MLP_WEIGHT_DECAY)

    total_epochs = POLICY_WARMUP_EPOCHS + POLICY_TRAIN_EPOCHS
    loss_history = []

    for epoch in range(total_epochs):
        policy.train(); summarizer.train(); token_proj.train()

        t = epoch / max(1, total_epochs-1)
        temp = GATE_TEMP_START * (1-t) + GATE_TEMP_END * t

        acc_loss_kd = 0; acc_loss_ratio = 0; acc_loss_l1 = 0; acc_loss_total = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                y_T = teacher(x)

            tokens, flops = build_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device)

            target_ratio = random.uniform(MIN_RATIO, MAX_RATIO)
            logits = policy(tokens, target_ratio).squeeze(0)

            gates = torch.sigmoid(logits / temp)

            full_flops = flops.sum()
            kept_flops = (gates * flops).sum()
            curr_ratio = kept_flops / (full_flops + 1e-6)

            y_S = forward_resnet50_gated_blocks(student, x, gates)

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
        print(f"Epoch {epoch} Summary - KD: {epoch_losses['loss_kd']:.4f}, Ratio: {epoch_losses['loss_ratio']:.4f}, Total: {epoch_losses['loss_total']:.4f}")

    return policy, summarizer, token_proj, loss_history

def mlp_materialize_and_finetune(teacher, policy, summarizer, token_proj, train_loader, val_loader, ratio):
    """Materialize pruned model using MLP policy scores"""
    print(f"\n[MLP] Target Ratio: {ratio}")

    x_sample, _ = next(iter(train_loader))
    x_sample = x_sample[:32].to(device)
    with torch.no_grad(): y_T = teacher(x_sample)

    temp_student = build_resnet50(200).to(device)
    temp_student.load_state_dict(teacher.state_dict())
    temp_student.eval()

    tokens, flops = build_block_tokens(teacher, temp_student, summarizer, token_proj, x_sample, y_T, device)
    logits = policy(tokens, ratio).squeeze(0)
    scores = torch.sigmoid(logits)

    # Greedy Selection
    eff = (scores / (flops + 1e-9)).detach().cpu().numpy()
    indices = sorted(range(NUM_BLOCKS), key=lambda i: eff[i], reverse=True)

    mask = [0] * NUM_BLOCKS
    current_flops = 0
    total_flops = flops.sum().item()

    for idx in indices:
        f = flops[idx].item()
        if (current_flops + f) / total_flops <= ratio:
            mask[idx] = 1
            current_flops += f

    if sum(mask) == 0:
        mask[indices[0]] = 1
        current_flops = flops[indices[0]].item()

    real_ratio = current_flops / total_flops
    print(f"Mask: {mask}")
    print(f"Actual Ratio: {real_ratio:.3f}, Kept: {sum(mask)}/{NUM_BLOCKS}")

    # Prune
    pruned_model = physically_prune_model(teacher, mask)
    recalibrate_bn(pruned_model, train_loader)

    acc_pre = run_evaluation(pruned_model, val_loader)
    print(f"Accuracy Pre-FT: {acc_pre*100:.2f}%")

    # Fine-tune
    opt = torch.optim.AdamW(pruned_model.parameters(), lr=FT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)

    best_acc = acc_pre
    ft_loss_history = []

    for ep in range(FT_EPOCHS):
        pruned_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): y_T = teacher(x)
            y_S = pruned_model(x)
            loss = kd_loss(y_S, y_T, TEMP_KD)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        acc = run_evaluation(pruned_model, val_loader)
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
    print("ResNet-50 Block-Wise Pruning (16 Bottleneck Blocks)")
    print("Methods: Transformer Encoder + MLP + L2 Baseline")
    print("="*60)

    train_loader, val_loader, num_classes = get_loaders()

    # Load or train teacher
    if os.path.exists(CKPT_TEACHER):
        print(f"\nLoading teacher from {CKPT_TEACHER}")
        teacher = build_resnet50(num_classes).to(device)
        ckpt = torch.load(CKPT_TEACHER, map_location='cpu')
        teacher.load_state_dict(ckpt['state_dict'])
    else:
        print("\nTeacher checkpoint not found. Training teacher...")
        teacher, _ = train_teacher(train_loader, val_loader, num_classes, epochs=10)

    teacher.eval()
    teacher_acc = run_evaluation(teacher, val_loader)
    print(f"\nTeacher Accuracy: {teacher_acc*100:.2f}%")

    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"\nResuming from checkpoint...")
        print(f"  Completed methods: {checkpoint.get('completed_methods', [])}")
        print(f"  Current method: {checkpoint.get('current_method', 'None')}")
        print(f"  Completed ratios: {checkpoint.get('completed_ratios', [])}")

    # Initialize results storage
    l2_results = checkpoint.get('l2_results', []) if checkpoint else []
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

        # Determine which ratios are already done
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
                "kept": sum(mask),
                "accuracy_before_ft": acc_before_ft,
                "accuracy_after_ft": best_acc,
                "accuracy_drop": teacher_acc - best_acc,
                "mask": mask,
                "scores": scores,
                "ft_loss_history": ft_losses
            })

            # Save checkpoint after each ratio
            save_checkpoint({
                'teacher_accuracy': teacher_acc,
                'completed_methods': completed_methods,
                'current_method': 'l2',
                'completed_ratios': [r['target_ratio'] for r in l2_results],
                'l2_results': l2_results,
                'mlp_results': mlp_results,
                'encoder_results': encoder_results
            })

        completed_methods.append('l2')

        # Save L2 results
        l2_results_path = os.path.join(OUT_DIR, RESULTS_FILE_L2)
        with open(l2_results_path, 'w') as f:
            json.dump({
                "method": "L2 Baseline",
                "teacher_accuracy": teacher_acc,
                "num_blocks": NUM_BLOCKS,
                "config": {"ft_epochs": FT_EPOCHS, "temp_kd": TEMP_KD},
                "results": l2_results
            }, f, indent=2)
        print(f"\nL2 Baseline results saved to {l2_results_path}")

    # =========================
    # Run MLP Method
    # =========================
    if 'mlp' not in completed_methods:
        print("\n" + "="*60)
        print("Running MLP Policy Method")
        print("="*60)

        mlp_policy_path = os.path.join(OUT_DIR, "resnet50_mlp_policy.pth")
        mlp_loss_path = os.path.join(OUT_DIR, "resnet50_mlp_policy_losses.json")

        if os.path.exists(mlp_policy_path):
            print("Loading MLP Policy...")
            state = torch.load(mlp_policy_path)
            mlp_policy = GlobalMLPPolicy(num_blocks=NUM_BLOCKS).to(device)
            mlp_policy.load_state_dict(state['pol'])
            mlp_summarizer = Summarizer().to(device)
            mlp_summarizer.load_state_dict(state['sum'])
            mlp_token_proj = nn.Sequential(nn.LayerNorm(SUM_DIM+5), nn.Linear(SUM_DIM+5, TOKEN_DIM)).to(device)
            mlp_token_proj.load_state_dict(state['proj'])

            if os.path.exists(mlp_loss_path):
                with open(mlp_loss_path, 'r') as f:
                    mlp_policy_loss_history = json.load(f)
            else:
                mlp_policy_loss_history = []
        else:
            mlp_policy, mlp_summarizer, mlp_token_proj, mlp_policy_loss_history = train_mlp_policy(teacher, train_loader)
            torch.save({
                'pol': mlp_policy.state_dict(),
                'sum': mlp_summarizer.state_dict(),
                'proj': mlp_token_proj.state_dict()
            }, mlp_policy_path)
            with open(mlp_loss_path, 'w') as f:
                json.dump(mlp_policy_loss_history, f, indent=2)
            print(f"MLP Policy losses saved to {mlp_loss_path}")

        # Determine which ratios are already done
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
                "kept": sum(mask),
                "accuracy_before_ft": acc_before_ft,
                "accuracy_after_ft": best_acc,
                "accuracy_drop": teacher_acc - best_acc,
                "mask": mask,
                "scores": scores,
                "ft_loss_history": ft_losses
            })

            # Save checkpoint after each ratio
            save_checkpoint({
                'teacher_accuracy': teacher_acc,
                'completed_methods': completed_methods,
                'current_method': 'mlp',
                'completed_ratios': [r['target_ratio'] for r in mlp_results],
                'l2_results': l2_results,
                'mlp_results': mlp_results,
                'encoder_results': encoder_results
            })

        completed_methods.append('mlp')

        # Save MLP results
        mlp_results_path = os.path.join(OUT_DIR, RESULTS_FILE_MLP)
        with open(mlp_results_path, 'w') as f:
            json.dump({
                "method": "MLP Policy",
                "teacher_accuracy": teacher_acc,
                "num_blocks": NUM_BLOCKS,
                "config": {
                    "policy_warmup_epochs": POLICY_WARMUP_EPOCHS,
                    "policy_train_epochs": POLICY_TRAIN_EPOCHS,
                    "ft_epochs": FT_EPOCHS,
                    "ratio_weight": RATIO_WEIGHT,
                    "l1_weight": L1_M_WEIGHT,
                    "temp_kd": TEMP_KD,
                    "mlp_hidden_dim": MLP_HIDDEN_DIM
                },
                "policy_loss_history": mlp_policy_loss_history,
                "results": mlp_results
            }, f, indent=2)
        print(f"\nMLP results saved to {mlp_results_path}")

    # =========================
    # Run Encoder Method
    # =========================
    if 'encoder' not in completed_methods:
        print("\n" + "="*60)
        print("Running Transformer Encoder Method")
        print("="*60)

        encoder_path = os.path.join(OUT_DIR, "resnet50_encoder_policy.pth")
        policy_loss_path = os.path.join(OUT_DIR, "resnet50_encoder_policy_losses.json")

        if os.path.exists(encoder_path):
            print("Loading Encoder Policy...")
            state = torch.load(encoder_path)
            encoder = CompressionAwareEncoder(num_blocks=NUM_BLOCKS).to(device)
            encoder.load_state_dict(state['enc'])
            summarizer = Summarizer().to(device)
            summarizer.load_state_dict(state['sum'])
            token_proj = nn.Sequential(nn.LayerNorm(SUM_DIM+5), nn.Linear(SUM_DIM+5, TOKEN_DIM)).to(device)
            token_proj.load_state_dict(state['proj'])

            if os.path.exists(policy_loss_path):
                with open(policy_loss_path, 'r') as f:
                    policy_loss_history = json.load(f)
            else:
                policy_loss_history = []
        else:
            encoder, summarizer, token_proj, policy_loss_history = train_encoder_policy(teacher, train_loader)
            torch.save({
                'enc': encoder.state_dict(),
                'sum': summarizer.state_dict(),
                'proj': token_proj.state_dict()
            }, encoder_path)
            with open(policy_loss_path, 'w') as f:
                json.dump(policy_loss_history, f, indent=2)
            print(f"Encoder Policy losses saved to {policy_loss_path}")

        # Determine which ratios are already done
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
                "kept": sum(mask),
                "accuracy_before_ft": acc_before_ft,
                "accuracy_after_ft": best_acc,
                "accuracy_drop": teacher_acc - best_acc,
                "mask": mask,
                "scores": scores,
                "ft_loss_history": ft_losses
            })

            # Save checkpoint after each ratio
            save_checkpoint({
                'teacher_accuracy': teacher_acc,
                'completed_methods': completed_methods,
                'current_method': 'encoder',
                'completed_ratios': [r['target_ratio'] for r in encoder_results],
                'l2_results': l2_results,
                'mlp_results': mlp_results,
                'encoder_results': encoder_results
            })

        completed_methods.append('encoder')

        # Save Encoder results
        encoder_results_path = os.path.join(OUT_DIR, RESULTS_FILE_ENCODER)
        with open(encoder_results_path, 'w') as f:
            json.dump({
                "method": "Transformer Encoder",
                "teacher_accuracy": teacher_acc,
                "num_blocks": NUM_BLOCKS,
                "config": {
                    "policy_warmup_epochs": POLICY_WARMUP_EPOCHS,
                    "policy_train_epochs": POLICY_TRAIN_EPOCHS,
                    "ft_epochs": FT_EPOCHS,
                    "ratio_weight": RATIO_WEIGHT,
                    "l1_weight": L1_M_WEIGHT,
                    "temp_kd": TEMP_KD
                },
                "policy_loss_history": policy_loss_history,
                "results": encoder_results
            }, f, indent=2)
        print(f"\nEncoder results saved to {encoder_results_path}")

    # =========================
    # Save Comparison
    # =========================
    comparison_path = os.path.join(OUT_DIR, RESULTS_FILE_COMPARISON)
    with open(comparison_path, 'w') as f:
        json.dump({
            "teacher_accuracy": teacher_acc,
            "num_blocks": NUM_BLOCKS,
            "eval_ratios": EVAL_RATIOS,
            "l2_baseline": l2_results,
            "mlp": mlp_results,
            "encoder": encoder_results
        }, f, indent=2)
    print(f"\nComparison results saved to {comparison_path}")

    # Clear checkpoint since we're done
    clear_checkpoint()

    # =========================
    # Print Summary
    # =========================
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Ratio':<8} {'L2 Acc':<12} {'MLP Acc':<12} {'Encoder Acc':<12} {'Best':<10}")
    print("-"*54)
    for i, r in enumerate(EVAL_RATIOS):
        l2_acc = l2_results[i]["accuracy_after_ft"] * 100
        mlp_acc = mlp_results[i]["accuracy_after_ft"] * 100
        enc_acc = encoder_results[i]["accuracy_after_ft"] * 100
        best = max(l2_acc, mlp_acc, enc_acc)
        best_method = "L2" if best == l2_acc else ("MLP" if best == mlp_acc else "Enc")
        print(f"{r:<8.1f} {l2_acc:<12.2f} {mlp_acc:<12.2f} {enc_acc:<12.2f} {best_method:<10}")

    print("\n" + "="*70)
    print("Detailed Results:")
    print("="*70)

    print("\nL2 Baseline:")
    for res in l2_results:
        print(f"  Target: {res['target_ratio']:.1f} | Actual: {res['actual_ratio']:.3f} | "
              f"Kept: {res['kept']}/{NUM_BLOCKS} | Acc: {res['accuracy_after_ft']*100:.2f}%")

    print("\nMLP Policy:")
    for res in mlp_results:
        print(f"  Target: {res['target_ratio']:.1f} | Actual: {res['actual_ratio']:.3f} | "
              f"Kept: {res['kept']}/{NUM_BLOCKS} | Acc: {res['accuracy_after_ft']*100:.2f}%")

    print("\nTransformer Encoder:")
    for res in encoder_results:
        print(f"  Target: {res['target_ratio']:.1f} | Actual: {res['actual_ratio']:.3f} | "
              f"Kept: {res['kept']}/{NUM_BLOCKS} | Acc: {res['accuracy_after_ft']*100:.2f}%")

if __name__ == "__main__":
    main()

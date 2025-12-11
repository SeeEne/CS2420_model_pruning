"""
AZ-NAS Encoder Overfitting Experiment (BLOCK-WISE Pruning)

This experiment tests whether using AZ-NAS loss (instead of KD loss) for encoder training
reduces overfitting when transferring pruning policies across datasets.

AZ-NAS Loss = -Expressivity - 0.1 * Progressivity
(Replaces KD loss in the original encoder training)

Uses BLOCK-WISE pruning: 8 gates for 8 BasicBlocks in ResNet-18
(Not unit-wise with 16 gates for 16 convs)

Pipeline phases:
- Phase 1: Train AZ-NAS Encoder on Tiny-ImageNet
- Phase 2: Fine-tune Student on Tiny-ImageNet (ratios: 0.1, 0.3, 0.5, 0.7, 0.9)
- Phase 3: Train/Load CIFAR-100 Teacher
- Phase 4: Old AZ-NAS Policy on CIFAR-100 (ratio=0.5) - Overfitting test
- Phase 5: New AZ-NAS Policy on CIFAR-100 (ratio=0.5) - Baseline comparison

With checkpoint/resume capability at each phase.
"""

import os, math, random, gc, json
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms, models
import numpy as np

# =========================
# Config
# =========================
OUT_DIR = "checkpoints"
RESULTS_FILE = "aznas_encoder_results_hybird.json"
CHECKPOINT_FILE = "aznas_encoder_checkpoint_hybird.json"

# Tiny-ImageNet paths
TINYIMAGENET_ROOT = "data/tiny-imagenet-200"
TINYIMAGENET_TEACHER_CKPT = "checkpoints/teacher.pth"
AZNAS_ENCODER_CKPT = "checkpoints/aznas_encoder_tinyimagenet.pth"

# CIFAR-100 paths
CIFAR100_TEACHER_CKPT = "checkpoints/cifar100_teacher_resnet18.pth"
AZNAS_ENCODER_CIFAR100_CKPT = "checkpoints/aznas_encoder_cifar100.pth"

BATCH_SIZE = 256
NUM_WORKERS = 2
IMG_SIZE = 224
SEED = 42

# Tiny-ImageNet config
TIN_NUM_CLASSES = 200
TIN_MEAN = (0.485, 0.456, 0.406)
TIN_STD = (0.229, 0.224, 0.225)

# CIFAR-100 config
CIFAR_NUM_CLASSES = 100
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)

# Encoder architecture
TOKEN_DIM = 128
SUM_DIM = 64
ENC_WIDTH = 128
ENC_LAYERS = 2
ENC_HEADS = 4
NUM_BLOCKS = 8  # 8 BasicBlocks in ResNet-18 (block-wise pruning)

# Policy training config
POLICY_WARMUP_EPOCHS = 1
POLICY_TRAIN_EPOCHS = 5
POLICY_LR = 1e-4
WEIGHT_DECAY = 0.0

# Budget training config
MIN_RATIO = 0.1
MAX_RATIO = 0.9
RATIO_WEIGHT = 25.0
GATE_TEMP_START = 5.0
GATE_TEMP_END = 0.3
L1_M_WEIGHT = 1e-3

# === NEW STRATEGY CONFIG ===
KD_WARMUP_EPOCHS = 1      # Epochs to train only with KD to establish semantic anchor
AZNAS_WEIGHT = 0.5        # Weight for AZ-NAS regularization in Co-training phase
# ===========================

# AZ-NAS loss weight (instead of KD)
AZNAS_COMPLEXITY_WEIGHT = 0.001  # Penalty for complexity

# Fine-tuning config
FT_EPOCHS = 10
FT_LR = 1e-3
TEMP_KD = 2.0
TEACHER_EPOCHS = 15

# Evaluation ratios
TINYIMAGENET_RATIOS = [0.3, 0.5, 0.7]
CIFAR100_RATIO = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_vram_usage(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM {tag}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# =========================
# Custom Dataset for Tiny ImageNet Validation
# =========================
class TinyImageNetVal(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        annotations_file = os.path.join(root, 'val_annotations.txt')
        self.images = []
        self.labels = []
        train_dir = os.path.join(os.path.dirname(root), 'train')
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(train_dir)))}
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_id = parts[1]
                self.images.append(os.path.join(root, 'images', img_name))
                self.labels.append(self.class_to_idx[class_id])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# =========================
# Data Loading
# =========================
def get_tinyimagenet_loaders():
    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(TIN_MEAN, TIN_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(TIN_MEAN, TIN_STD),
    ])
    train_ds = datasets.ImageFolder(root=os.path.join(TINYIMAGENET_ROOT, 'train'), transform=train_tf)
    val_ds = TinyImageNetVal(root=os.path.join(TINYIMAGENET_ROOT, 'val'), transform=test_tf)
    num_classes = len(train_ds.classes)
    print(f"Tiny-ImageNet: {len(train_ds)} train, {len(val_ds)} val, {num_classes} classes")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, num_classes

def get_cifar100_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_ds = datasets.CIFAR100(root="data/cifar100", train=True, download=True, transform=train_tf)
    val_ds = datasets.CIFAR100(root="data/cifar100", train=False, download=True, transform=test_tf)
    print(f"CIFAR-100: {len(train_ds)} train, {len(val_ds)} val, 100 classes")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, CIFAR_NUM_CLASSES

# =========================
# Models
# =========================
def build_resnet18(num_classes=200):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class Summarizer(nn.Module):
    """Summarizes a Block: Inputs vs Residual Output (from 3_encoder_block.py)"""
    def __init__(self, sum_dim=SUM_DIM, k=64):
        super().__init__()
        self.pool1d = nn.AdaptiveAvgPool1d(k)
        # Input (h_in) + Residual Output (r_out) -> 4 stats each * 2 = 8
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
        # Compare what went INTO the block vs what came out of the RESIDUAL branch
        gap_h = h_in.mean(dim=(2, 3))
        gmp_h, _ = h_in.flatten(2).max(dim=2)
        gap_r = r_out.mean(dim=(2, 3))
        gmp_r, _ = r_out.flatten(2).max(dim=2)
        parts = [self._pool_vec(p) for p in [gap_h, gmp_h, gap_r, gmp_r]]
        feats = torch.cat(parts, dim=1)
        z = self.mlp(self.proj(feats))
        return z.mean(dim=0)  # [sum_dim]

class TokenProj(nn.Module):
    def __init__(self, in_dim, out_dim=TOKEN_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.net(x)

class CompressionAwareEncoder(nn.Module):
    def __init__(self, dim=ENC_WIDTH, depth=ENC_LAYERS, heads=ENC_HEADS, num_blocks=NUM_BLOCKS):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * 2.0),
            batch_first=True, activation='gelu', norm_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos = nn.Parameter(torch.zeros(1, num_blocks + 1, dim))
        self.budget_embed = nn.Sequential(
            nn.Linear(1, dim * 2), nn.LayerNorm(dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim)
        )
        self.head = nn.Linear(dim, 1)

    def forward(self, tokens, target_ratio):
        if not torch.is_tensor(target_ratio):
            target_ratio = torch.tensor([[float(target_ratio)]], dtype=torch.float32, device=tokens.device)
        else:
            target_ratio = target_ratio.float().view(1, 1).to(tokens.device)
        budget_tok = self.budget_embed(target_ratio)
        x = torch.cat([budget_tok.unsqueeze(1), tokens], dim=1)
        x = x + self.pos[:, :x.size(1)]
        h = self.enc(x)
        block_h = h[:, 1:, :]  # Skip budget token
        logits = self.head(block_h).squeeze(-1)
        return logits

# =========================
# FLOPs Computation (Block-wise)
# =========================
def conv_flops(H, W, Cin, Cout, k, stride):
    return (H // stride) * (W // stride) * Cin * Cout * (k * k)

def get_block_flops(block, input_shape):
    """Calculate FLOPs for the RESIDUAL branch only (Conv1 + Conv2)"""
    B, C, H, W = input_shape
    # Conv1
    f1 = conv_flops(H, W, block.conv1.in_channels, block.conv1.out_channels, 3, block.conv1.stride[0])
    # Conv2 (Output size depends on stride of conv1)
    H2, W2 = H // block.conv1.stride[0], W // block.conv1.stride[0]
    f2 = conv_flops(H2, W2, block.conv2.in_channels, block.conv2.out_channels, 3, block.conv2.stride[0])
    return float(f1 + f2)

# =========================
# Block-wise Forward Pass Helpers
# =========================
def forward_resnet_collect_blocks(model, x, collect_grads=False):
    """
    Runs model, collects (h_in, r_out) pairs for every block.
    Also collects gradients if collect_grads=True for Taylor calculation.
    """
    infos = []

    # Stem
    h = model.conv1(x)
    h = model.bn1(h)
    h = model.relu(h)
    h = model.maxpool(h)

    stages = [model.layer1, model.layer2, model.layer3, model.layer4]

    for s_idx, stage in enumerate(stages):
        for b_idx, block in enumerate(stage):
            h_in = h
            if collect_grads:
                h_in.retain_grad()

            # --- Residual Branch ---
            out = block.conv1(h_in)
            out = block.bn1(out)
            out = block.relu(out)
            if collect_grads:
                out.retain_grad()

            out = block.conv2(out)
            r_out = block.bn2(out)  # Final residual before addition
            if collect_grads:
                r_out.retain_grad()

            # --- Skip ---
            if block.downsample is not None:
                skip = block.downsample(h_in)
            else:
                skip = h_in

            # Store info
            infos.append({
                "h_in": h_in,      # Input to block
                "r_out": r_out,    # Residual output
                "block_obj": block,
                "stage": s_idx,
                "block_idx": b_idx,
                "H": h_in.size(2),
                "W": h_in.size(3),
                "flops": get_block_flops(block, h_in.shape)
            })

            # Final activation
            h = F.relu(skip + r_out)

    # Final classifier
    h = model.avgpool(h)
    h = torch.flatten(h, 1)
    logits = model.fc(h)

    return logits, infos

def forward_resnet_gated_blocks(model, x, block_gates):
    """
    Forward pass with 8 gates (one per block).
    If gate is close to 0, the residual branch is suppressed.
    """
    gate_idx = 0

    h = model.conv1(x)
    h = model.bn1(h)
    h = model.relu(h)
    h = model.maxpool(h)

    for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in stage:
            # 1. Compute Residual Branch
            r = block.conv1(h)
            r = block.bn1(r)
            r = block.relu(r)
            r = block.conv2(r)
            r = block.bn2(r)

            # 2. Apply Gate (Suppress Block)
            g = block_gates[gate_idx].view(1, 1, 1, 1)
            r = r * g
            gate_idx += 1

            # 3. Skip Connection
            if block.downsample is not None:
                skip = block.downsample(h)
            else:
                skip = h

            h = F.relu(skip + r)

    h = model.avgpool(h)
    h = torch.flatten(h, 1)
    return model.fc(h)

# =========================
# AZ-NAS Score Computation (Block-wise)
# =========================
# def compute_aznas_scores_gated_blocks(model, x, gates):
#     """Compute AZ-NAS scores with block-gated forward pass"""
#     model.eval()

#     # Collect features during gated forward
#     features = []
#     g_idx = 0

#     h = model.conv1(x)
#     h = model.bn1(h)
#     h = model.relu(h)
#     features.append(h.detach())
#     h = model.maxpool(h)

#     for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
#         for block in layer:
#             # Residual branch
#             r = block.conv1(h)
#             r = block.bn1(r)
#             r = block.relu(r)
#             r = block.conv2(r)
#             r = block.bn2(r)

#             # Apply block gate
#             g = gates[g_idx].view(1, 1, 1, 1)
#             r = r * g
#             g_idx += 1

#             skip = block.downsample(h) if block.downsample is not None else h
#             h = F.relu(skip + r)
#             features.append(h.detach())

#     # Compute expressivity (entropy of feature covariance eigenvalues)
#     expressivity_scores = []
#     for feat in features:
#         if feat.numel() == 0:
#             continue
#         b, c, fh, fw = feat.shape
#         X = feat.permute(0, 2, 3, 1).reshape(b * fh * fw, c)
#         mu = X.mean(dim=0, keepdim=True)
#         Xc = X - mu
#         sigma = (Xc.T @ Xc) / max(1, Xc.shape[0])
#         s = torch.linalg.eigvalsh(sigma)
#         s = torch.relu(s) + 1e-12
#         p = s / s.sum()
#         exp_score = (-p * torch.log(p)).sum().item()
#         expressivity_scores.append(exp_score)

#     expressivity = np.mean(expressivity_scores) if expressivity_scores else 0.0

#     # Progressivity (min increase in expressivity across layers)
#     if len(expressivity_scores) >= 2:
#         diffs = [expressivity_scores[i + 1] - expressivity_scores[i] for i in range(len(expressivity_scores) - 1)]
#         progressivity = min(diffs)
#     else:
#         progressivity = 0.0

#     return {
#         "expressivity": expressivity,
#         "progressivity": progressivity,
#     }

def compute_aznas_scores_gated_blocks(model, x, gates):
    """
    Debug Version:
    1. Removed Downsampling (Full Resolution) -> Expect VRAM spike!
    2. Dynamic Jitter -> Better numerical stability.
    """
    scores = []
    g_idx = 0
    
    # --- Stem ---
    # 注意：为了节省显存，Stem 部分我们还是不开梯度追踪，只在 Block 内部开
    with torch.no_grad():
        h = model.conv1(x)
        h = model.bn1(h)
        h = model.relu(h)
        h = model.maxpool(h)
    
    # 必须把 h 重新设为 requires_grad，否则前面的 no_grad 会切断链条？
    # 不，不对。Gates 有梯度就够了。h 在这里只是基底。
    # 但为了保险，让 h detach 一下作为纯输入，防止梯度试图往 Stem 传（虽然传不到因为 weights frozen）
    h = h.detach()

    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            # 1. Residual Branch
            # Block 的权重是 frozen 的，但是我们需要中间变量建立计算图
            r = block.conv1(h)
            r = block.bn1(r)
            r = block.relu(r)
            r = block.conv2(r)
            r = block.bn2(r)

            # 2. Apply Gate (Gradient Source!)
            g = gates[g_idx].view(1, 1, 1, 1)
            
            # [关键检查点] r*g 这一步必须产生 grad_fn
            r_gated = r * g 
            g_idx += 1

            # 3. Skip Connection
            skip = block.downsample(h) if block.downsample is not None else h
            h = F.relu(skip + r_gated)

            # --- Score Calculation ---
            b, c, fh, fw = h.shape
            
            # [恢复全分辨率] 删除了 downsampling，显存应该会涨
            X = h.permute(0, 2, 3, 1).reshape(-1, c)
            
            # Covariance
            mu = X.mean(dim=0, keepdim=True)
            Xc = X - mu
            n = Xc.shape[0]
            sigma = (Xc.T @ Xc) / max(1, n)
            
            # [动态 Jitter] 根据矩阵的平均值来加噪声，防止噪声淹没信号
            sigma_norm = torch.norm(sigma, p='fro')
            epsilon = 1e-5 * sigma_norm.detach() + 1e-6
            jitter = epsilon * torch.eye(c, device=x.device)
            sigma = sigma + jitter
            
            # Eigendecomposition
            try:
                # 这里的梯度计算是显存杀手
                s = torch.linalg.eigvalsh(sigma)
            except RuntimeError:
                # 假如炸了，打印一下
                print(f"Eigen decomposition failed at Layer {g_idx}")
                s = torch.ones(c, device=x.device)

            s = torch.relu(s) + 1e-12
            p = s / s.sum()
            
            entropy = -torch.sum(p * torch.log(p))
            scores.append(entropy)

    if len(scores) > 0:
        scores_stack = torch.stack(scores)
        expressivity = torch.mean(scores_stack)
        if len(scores) >= 2:
            diffs = scores_stack[1:] - scores_stack[:-1]
            progressivity = torch.min(diffs)
        else:
            progressivity = torch.tensor(0.0, device=x.device)
    else:
        expressivity = torch.tensor(0.0, device=x.device)
        progressivity = torch.tensor(0.0, device=x.device)

    return {"expressivity": expressivity, "progressivity": progressivity}

# =========================
# Helper Functions (Block-wise, from 3_encoder_block.py)
# =========================
def kd_loss(student_logits, teacher_logits, T=TEMP_KD):
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction='batchmean') * (T * T)

def build_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device):
    """
    Builds 8 tokens representing the 8 blocks.
    Taylor score = Sum of activation*grad magnitude for block residual output.
    """
    # Enable gradients for input when we need to collect gradients
    x = x.requires_grad_(True)

    # 1. Get Teacher Stats (for summary features)
    with torch.no_grad():
        _, teacher_infos = forward_resnet_collect_blocks(teacher, x, collect_grads=False)

    # 2. Get Student Gradients (for Taylor score)
    logits_S, student_infos = forward_resnet_collect_blocks(student, x, collect_grads=True)
    loss = F.kl_div(F.log_softmax(logits_S / TEMP_KD, dim=1),
                    F.softmax(y_T / TEMP_KD, dim=1), reduction='batchmean') * (TEMP_KD ** 2)
    loss.backward()

    token_list = []
    flops_list = []

    for i in range(NUM_BLOCKS):
        # Teacher info for features (stable)
        t_info = teacher_infos[i]
        s_info = student_infos[i]  # Student info for grads

        # Features from Summarizer
        feats = summarizer(t_info["h_in"], t_info["r_out"]).to(device)

        # Taylor Score calculation
        grad_r = s_info["r_out"].grad
        if grad_r is not None:
            taylor = (grad_r * s_info["r_out"]).abs().mean().detach().item()
        else:
            taylor = 0.0

        # Metadata (4 values for block-wise)
        meta = torch.tensor([
            t_info["stage"] / 3.0,
            t_info["block_idx"] / 1.0,
            t_info["H"] / IMG_SIZE,
            t_info["W"] / IMG_SIZE
        ], dtype=torch.float32, device=device)

        # Combine
        tay_tensor = torch.tensor([math.log1p(taylor)], dtype=torch.float32, device=device)
        tok = torch.cat([feats, meta, tay_tensor], dim=0)

        token_list.append(tok)
        flops_list.append(t_info["flops"])

    # Clean up gradients
    for info in student_infos:
        if info["r_out"].grad is not None:
            info["r_out"].grad = None

    # Project to Token Dim
    tokens = torch.stack(token_list).unsqueeze(0)  # [1, 8, InputDim]
    # Note: InputDim = SUM_DIM + 4 (meta) + 1 (taylor)
    tokens = token_proj(tokens)
    flops = torch.tensor(flops_list, dtype=torch.float32, device=device)

    student.zero_grad()
    return tokens, flops

@torch.no_grad()
def evaluate_accuracy(model, loader, mask=None):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if mask is not None:
            logits = forward_resnet_gated_blocks(model, x, mask)
        else:
            logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

# =========================
# Phase 1: Train AZ-NAS Encoder (Block-wise)
# =========================
def train_aznas_encoder(teacher, train_loader, val_loader, num_classes, save_path):
    print("\n" + "=" * 70)
    print("Training AZ-NAS Block-Wise Encoder (replacing KD loss with AZ-NAS loss)")
    print("=" * 70)

    student = build_resnet18(num_classes).to(device)
    student.load_state_dict(teacher.state_dict())
    for p in student.parameters():
        p.requires_grad = False
    student.eval()

    summarizer = Summarizer(sum_dim=SUM_DIM).to(device)
    # Input dim = SUM_DIM + 4 (meta) + 1 (taylor) = SUM_DIM + 5
    token_proj = TokenProj(SUM_DIM + 5, TOKEN_DIM).to(device)
    encoder = CompressionAwareEncoder(dim=ENC_WIDTH, depth=ENC_LAYERS, heads=ENC_HEADS, num_blocks=NUM_BLOCKS).to(device)

    params = list(summarizer.parameters()) + list(token_proj.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=POLICY_LR, weight_decay=WEIGHT_DECAY)

    total_epochs = POLICY_WARMUP_EPOCHS + POLICY_TRAIN_EPOCHS
    loss_history = []

    for epoch in range(total_epochs):
        encoder.train()
        summarizer.train()
        token_proj.train()
        
        # Determine Phase
        is_warmup = epoch < KD_WARMUP_EPOCHS
        phase_name = "KD Warmup" if is_warmup else "Co-Training"

        lr = POLICY_LR * 0.5 * (1 + math.cos(math.pi * epoch / max(1, total_epochs - 1)))
        for g in opt.param_groups: g['lr'] = lr
        t = epoch / max(1, total_epochs - 1)
        gate_temp = GATE_TEMP_START * (1 - t) + GATE_TEMP_END * t

        run_expr = run_aznas = run_ratio = run_kd = 0.0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            # Teacher forward (No grad)
            with torch.no_grad():
                y_T = teacher(x)

            # Build Tokens (Uses gradients internally on student but cleans up)
            tokens, flops = build_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device)

            # Predict Gates
            target_ratio = random.uniform(MIN_RATIO, MAX_RATIO)
            logits = encoder(tokens, target_ratio).squeeze(0)
            gates = torch.sigmoid(logits / gate_temp)

            # --- FORWARD PASS FOR LOSS ---
            # 1. Get Student Logits using predicted Gates (Differentiable)
            y_S = forward_resnet_gated_blocks(student, x, gates)

            # 2. Semantic Loss (KD) - Always active to prevent shallow bias
            loss_kd = kd_loss(y_S, y_T, T=TEMP_KD)

            # 3. Structure Loss (FLOPs & L1)
            exp_flops = (gates * flops).sum()
            flops_ratio = exp_flops / (flops.sum() + 1e-6)
            loss_ratio = (flops_ratio - target_ratio) ** 2
            loss_l1 = gates.mean()

            # 4. AZ-NAS Loss (Only after Warmup)
            if not is_warmup:
                aznas_info = compute_aznas_scores_gated_blocks(student, x, gates)
                expr = aznas_info['expressivity']
                prog = aznas_info['progressivity']
                loss_expr = torch.exp(-expr / 5.0)
                loss_prog = torch.exp(-prog / 1.0)
                loss_aznas = loss_expr + 0.1 * loss_prog
                current_az_val = loss_aznas
            else:
                loss_aznas = torch.tensor(0.0, device=device)
                current_az_val = torch.tensor(0.0, device=device)
                expr = torch.tensor(0.0, device=device)

            # Total Loss Combination
            # Warmup: KD + Ratio
            # Co-train: KD + AZ-NAS + Ratio
            loss = loss_kd + RATIO_WEIGHT * loss_ratio + L1_M_WEIGHT * loss_l1
            
            if not is_warmup:
                loss += AZNAS_WEIGHT * loss_aznas

            opt.zero_grad(set_to_none=True)
            loss.backward()
            
            # ================= [DEBUG PROBE START] =================
            # 检查 Gates 是否真的收到了梯度
            # 我们需要 retain_grad 才能检查中间变量，但 gates 是通过 logits 算出来的
            # 最直接的是检查 encoder 的参数有没有梯度
            
            # encoder_grad_norm = 0.0
            # for p in encoder.parameters():
            #     if p.grad is not None:
            #         encoder_grad_norm += p.grad.norm().item()
            
            # if not is_warmup and i % 10 == 0:
            #     print(f"\n[DEBUG] Step {i}:")
            #     print(f"  -> AZ-NAS Loss Value: {loss_aznas.item():.6f}")
            #     print(f"  -> Encoder Grad Norm: {encoder_grad_norm:.6f}")
                
            #     # 如果 grad norm 是 0，说明梯度真的断了
            #     if encoder_grad_norm == 0.0:
            #         print("  !!! ALERT: GRADIENTS ARE ZERO. GRAPH IS BROKEN. !!!")
                    
            #     # 检查一下 expressivity 的值，如果它一直是 4.x 不动，说明特征太强了
            #     print(f"  -> Raw Expressivity: {expr.item():.4f}")
            # ================= [DEBUG PROBE END] =================
            
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            opt.step()

            # Logging (Use .item() to save memory)
            run_kd += loss_kd.item()
            run_aznas += current_az_val.item() if not is_warmup else 0.0
            run_expr += expr.item() if not is_warmup else 0.0
            run_ratio += loss_ratio.item()

            if (i + 1) % 50 == 0:
                print(f"[Ep {epoch + 1}][{i + 1}][{phase_name}] "
                      f"KD={run_kd/50:.4f} AZ={run_aznas/50:.4f} "
                      f"Expr={run_expr/50:.2f} RatioL={run_ratio/50:.4f} Temp={gate_temp:.2f}")
                run_kd = run_aznas = run_expr = run_ratio = 0.0

        loss_history.append({"epoch": epoch + 1, "gate_temp": gate_temp})

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "summarizer": summarizer.state_dict(),
        "token_proj": token_proj.state_dict(),
        "num_classes": num_classes,
        "loss_history": loss_history
    }, save_path)
    print(f"\nAZ-NAS Block-Wise Encoder saved to {save_path}")
    return encoder, summarizer, token_proj

# =========================
# Phase 2: Materialize and Fine-tune (Block-wise)
# =========================
def materialize_mask(encoder, summarizer, token_proj, teacher, train_loader, target_ratio, num_classes):
    """Generate a binary block mask using greedy selection based on learned scores."""
    encoder.eval()
    summarizer.eval()
    token_proj.eval()
    teacher.eval()

    student_once = build_resnet18(num_classes).to(device)
    student_once.load_state_dict(teacher.state_dict())
    for p in student_once.parameters():
        p.requires_grad = False
    student_once.eval()

    all_scores = []
    all_flops = None
    iters = 20
    itr = 0

    for x, _ in train_loader:
        x = x.to(device)
        with torch.no_grad():
            y_T = teacher(x)
        # Build block tokens
        tokens, flops = build_block_tokens(teacher, student_once, summarizer, token_proj, x, y_T, device)
        logits = encoder(tokens, target_ratio).squeeze(0)  # [8]
        scores = torch.sigmoid(logits)
        all_scores.append(scores)
        if all_flops is None:
            all_flops = flops
        itr += 1
        if itr >= iters:
            break

    avg_scores = torch.stack(all_scores, dim=0).mean(dim=0)
    flops = all_flops

    # Greedy selection by efficiency (score / flops)
    eff = (avg_scores.detach() / (flops + 1e-9)).cpu().numpy().tolist()
    idx_sorted = sorted(range(NUM_BLOCKS), key=lambda i: eff[i], reverse=True)

    mask = torch.zeros(NUM_BLOCKS, device=device)
    full_flops = flops.sum()
    acc_flops = 0.0

    for i in idx_sorted:
        if (acc_flops + flops[i]) / full_flops <= target_ratio or mask.sum().item() == 0:
            mask[i] = 1.0
            acc_flops += flops[i].item()
        if acc_flops / full_flops >= target_ratio * 0.98:
            break

    # Ensure at least one block is kept
    if mask.sum().item() < 1:
        mask[idx_sorted[0]] = 1.0
        acc_flops = flops[idx_sorted[0]].item()

    actual_ratio = (mask * flops).sum() / full_flops
    return mask, actual_ratio.item(), avg_scores.tolist()

def finetune_pruned_model(teacher, mask, train_loader, val_loader, num_classes):
    """Fine-tune pruned model with KD loss using block-wise gates."""
    student = build_resnet18(num_classes).to(device)
    student.load_state_dict(teacher.state_dict())
    for p in student.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(student.parameters(), lr=FT_LR)

    print(f"Fine-tuning (kept={int(mask.sum().item())}/{NUM_BLOCKS} blocks)...")
    best_acc = 0.0

    for epoch in range(FT_EPOCHS):
        student.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_T = teacher(x)
            y_S = forward_resnet_gated_blocks(student, x, mask)
            loss = kd_loss(y_S, y_T, T=TEMP_KD)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        acc = evaluate_accuracy(student, val_loader, mask)
        best_acc = max(best_acc, acc)
        print(f"  [FT {epoch + 1}/{FT_EPOCHS}] acc={acc * 100:.2f}%")

    return student, best_acc

def train_teacher_model(train_loader, val_loader, num_classes, epochs, save_path):
    print(f"\nTraining Teacher ({num_classes} classes)...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = GradScaler()

    best_acc = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                out = model(x)
                loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        scheduler.step()
        acc = evaluate_accuracy(model, val_loader)
        best_acc = max(best_acc, acc)
        print(f"  [Teacher Ep {ep + 1}/{epochs}] acc={acc * 100:.2f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'val_acc': best_acc}, save_path)
    print(f"Teacher saved to {save_path}")
    return model, best_acc

# =========================
# Checkpoint Management
# =========================
def load_checkpoint():
    path = os.path.join(OUT_DIR, CHECKPOINT_FILE)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def save_checkpoint(data):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, CHECKPOINT_FILE)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def clear_checkpoint():
    path = os.path.join(OUT_DIR, CHECKPOINT_FILE)
    if os.path.exists(path):
        os.remove(path)

# =========================
# Main
# =========================
def main():
    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("AZ-NAS Encoder Overfitting Experiment")
    print("=" * 70)

    checkpoint = load_checkpoint()
    results = checkpoint.get('results', {}) if checkpoint else {}

    # =========================
    # Phase 1: Train AZ-NAS Encoder on Tiny-ImageNet
    # =========================
    if checkpoint is None or checkpoint.get('phase', 0) < 1:
        print("\n" + "=" * 70)
        print("Phase 1: Train AZ-NAS Encoder on Tiny-ImageNet")
        print("=" * 70)

        train_loader, val_loader, num_classes = get_tinyimagenet_loaders()

        # Load teacher
        teacher = build_resnet18(TIN_NUM_CLASSES).to(device)
        ckpt = torch.load(TINYIMAGENET_TEACHER_CKPT, map_location="cpu")
        teacher.load_state_dict(ckpt["state_dict"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        teacher_acc = evaluate_accuracy(teacher, val_loader)
        print(f"Tiny-ImageNet Teacher: {teacher_acc * 100:.2f}%")
        results['tinyimagenet_teacher_acc'] = teacher_acc

        encoder, summarizer, token_proj = train_aznas_encoder(
            teacher, train_loader, val_loader, TIN_NUM_CLASSES, AZNAS_ENCODER_CKPT
        )

        save_checkpoint({'phase': 1, 'results': results})
        print_vram_usage("after Phase 1")
        clear_vram()

    # =========================
    # Phase 2: Fine-tune on Tiny-ImageNet
    # =========================
    if checkpoint is None or checkpoint.get('phase', 0) < 2:
        print("\n" + "=" * 70)
        print("Phase 2: Fine-tune Student on Tiny-ImageNet")
        print("=" * 70)

        train_loader, val_loader, _ = get_tinyimagenet_loaders()

        teacher = build_resnet18(TIN_NUM_CLASSES).to(device)
        ckpt = torch.load(TINYIMAGENET_TEACHER_CKPT, map_location="cpu")
        teacher.load_state_dict(ckpt["state_dict"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher_acc = evaluate_accuracy(teacher, val_loader)

        # Load encoder
        enc_ckpt = torch.load(AZNAS_ENCODER_CKPT, map_location="cpu")
        encoder = CompressionAwareEncoder(dim=ENC_WIDTH, depth=ENC_LAYERS, heads=ENC_HEADS, num_blocks=NUM_BLOCKS).to(device)
        encoder.load_state_dict(enc_ckpt["encoder"])
        summarizer = Summarizer(sum_dim=SUM_DIM).to(device)
        summarizer.load_state_dict(enc_ckpt["summarizer"])
        token_proj = TokenProj(SUM_DIM + 5, TOKEN_DIM).to(device)
        token_proj.load_state_dict(enc_ckpt["token_proj"])

        results['tinyimagenet_students'] = {}

        for ratio in TINYIMAGENET_RATIOS:
            print(f"\n=== Ratio {ratio} ===")
            mask, actual_ratio, scores = materialize_mask(
                encoder, summarizer, token_proj, teacher, train_loader, ratio, TIN_NUM_CLASSES
            )
            print(f"Mask: {mask.int().tolist()}, Actual ratio: {actual_ratio:.3f}")

            acc_before = evaluate_accuracy(
                build_resnet18(TIN_NUM_CLASSES).to(device).eval(), val_loader, mask
            )

            _, best_acc = finetune_pruned_model(teacher, mask, train_loader, val_loader, TIN_NUM_CLASSES)

            results['tinyimagenet_students'][str(ratio)] = {
                'target_ratio': ratio,
                'actual_ratio': actual_ratio,
                'mask': mask.int().tolist(),
                'accuracy_before_ft': acc_before,
                'accuracy_after_ft': best_acc,
                'accuracy_drop': teacher_acc - best_acc
            }

        save_checkpoint({'phase': 2, 'results': results})
        print_vram_usage("after Phase 2")
        clear_vram()

    # =========================
    # Phase 3: CIFAR-100 Teacher
    # =========================
    if checkpoint is None or checkpoint.get('phase', 0) < 3:
        print("\n" + "=" * 70)
        print("Phase 3: Train/Load CIFAR-100 Teacher")
        print("=" * 70)

        train_loader, val_loader, _ = get_cifar100_loaders()

        if os.path.exists(CIFAR100_TEACHER_CKPT):
            teacher = build_resnet18(CIFAR_NUM_CLASSES).to(device)
            ckpt = torch.load(CIFAR100_TEACHER_CKPT, map_location="cpu")
            teacher.load_state_dict(ckpt["state_dict"])
            teacher.eval()
            teacher_acc = evaluate_accuracy(teacher, val_loader)
            print(f"Loaded CIFAR-100 Teacher: {teacher_acc * 100:.2f}%")
        else:
            teacher, teacher_acc = train_teacher_model(
                train_loader, val_loader, CIFAR_NUM_CLASSES, TEACHER_EPOCHS, CIFAR100_TEACHER_CKPT
            )

        results['cifar100_teacher_acc'] = teacher_acc

        save_checkpoint({'phase': 3, 'results': results})
        print_vram_usage("after Phase 3")
        clear_vram()

    # =========================
    # Phase 4: Old AZ-NAS on CIFAR-100
    # =========================
    if checkpoint is None or checkpoint.get('phase', 0) < 4:
        print("\n" + "=" * 70)
        print("Phase 4: Old AZ-NAS Policy on CIFAR-100 (Overfitting Test)")
        print("=" * 70)

        train_loader, val_loader, _ = get_cifar100_loaders()

        teacher = build_resnet18(CIFAR_NUM_CLASSES).to(device)
        ckpt = torch.load(CIFAR100_TEACHER_CKPT, map_location="cpu")
        teacher.load_state_dict(ckpt["state_dict"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher_acc = evaluate_accuracy(teacher, val_loader)

        # Load OLD encoder (trained on Tiny-ImageNet)
        enc_ckpt = torch.load(AZNAS_ENCODER_CKPT, map_location="cpu")
        encoder = CompressionAwareEncoder(dim=ENC_WIDTH, depth=ENC_LAYERS, heads=ENC_HEADS, num_blocks=NUM_BLOCKS).to(device)
        encoder.load_state_dict(enc_ckpt["encoder"])
        summarizer = Summarizer(sum_dim=SUM_DIM).to(device)
        summarizer.load_state_dict(enc_ckpt["summarizer"])
        token_proj = TokenProj(SUM_DIM + 5, TOKEN_DIM).to(device)
        token_proj.load_state_dict(enc_ckpt["token_proj"])

        mask, actual_ratio, _ = materialize_mask(
            encoder, summarizer, token_proj, teacher, train_loader, CIFAR100_RATIO, CIFAR_NUM_CLASSES
        )
        print(f"Old Policy Mask: {mask.int().tolist()}")

        _, best_acc = finetune_pruned_model(teacher, mask, train_loader, val_loader, CIFAR_NUM_CLASSES)

        results['old_aznas_cifar100'] = {
            'target_ratio': CIFAR100_RATIO,
            'actual_ratio': actual_ratio,
            'mask': mask.int().tolist(),
            'accuracy_after_ft': best_acc,
            'accuracy_drop': teacher_acc - best_acc
        }

        save_checkpoint({'phase': 4, 'results': results})
        print_vram_usage("after Phase 4")
        clear_vram()

    # =========================
    # Phase 5: New AZ-NAS on CIFAR-100
    # =========================
    if checkpoint is None or checkpoint.get('phase', 0) < 5:
        print("\n" + "=" * 70)
        print("Phase 5: New AZ-NAS Policy on CIFAR-100")
        print("=" * 70)

        train_loader, val_loader, _ = get_cifar100_loaders()

        teacher = build_resnet18(CIFAR_NUM_CLASSES).to(device)
        ckpt = torch.load(CIFAR100_TEACHER_CKPT, map_location="cpu")
        teacher.load_state_dict(ckpt["state_dict"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher_acc = evaluate_accuracy(teacher, val_loader)

        # Train NEW encoder on CIFAR-100
        encoder, summarizer, token_proj = train_aznas_encoder(
            teacher, train_loader, val_loader, CIFAR_NUM_CLASSES, AZNAS_ENCODER_CIFAR100_CKPT
        )

        mask, actual_ratio, _ = materialize_mask(
            encoder, summarizer, token_proj, teacher, train_loader, CIFAR100_RATIO, CIFAR_NUM_CLASSES
        )
        print(f"New Policy Mask: {mask.int().tolist()}")

        _, best_acc = finetune_pruned_model(teacher, mask, train_loader, val_loader, CIFAR_NUM_CLASSES)

        results['new_aznas_cifar100'] = {
            'target_ratio': CIFAR100_RATIO,
            'actual_ratio': actual_ratio,
            'mask': mask.int().tolist(),
            'accuracy_after_ft': best_acc,
            'accuracy_drop': teacher_acc - best_acc
        }

        save_checkpoint({'phase': 5, 'results': results})

    # =========================
    # Final Summary
    # =========================
    results_path = os.path.join(OUT_DIR, RESULTS_FILE)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    clear_checkpoint()

    print("\n" + "=" * 70)
    print("SUMMARY (Block-Wise Pruning: 8 blocks)")
    print("=" * 70)
    print(f"Tiny-ImageNet Teacher: {results.get('tinyimagenet_teacher_acc', 0) * 100:.2f}%")
    print(f"CIFAR-100 Teacher: {results.get('cifar100_teacher_acc', 0) * 100:.2f}%")

    print("\nTiny-ImageNet Results (Block-wise):")
    for ratio in TINYIMAGENET_RATIOS:
        data = results.get('tinyimagenet_students', {}).get(str(ratio), {})
        if data:
            kept = sum(data.get('mask', []))
            print(f"  r={ratio}: {data['accuracy_after_ft'] * 100:.2f}% (kept {kept}/{NUM_BLOCKS} blocks)")

    print("\nCIFAR-100 Overfitting Test (ratio=0.5, block-wise):")
    old = results.get('old_aznas_cifar100', {})
    new = results.get('new_aznas_cifar100', {})
    old_kept = sum(old.get('mask', []))
    new_kept = sum(new.get('mask', []))
    print(f"  Old AZ-NAS (Tiny-ImageNet): {old.get('accuracy_after_ft', 0) * 100:.2f}% ({old_kept}/{NUM_BLOCKS} blocks)")
    print(f"  New AZ-NAS (CIFAR-100):     {new.get('accuracy_after_ft', 0) * 100:.2f}% ({new_kept}/{NUM_BLOCKS} blocks)")
    diff = new.get('accuracy_after_ft', 0) - old.get('accuracy_after_ft', 0)
    print(f"  Difference: {diff * 100:+.2f}%")

    if diff > 0:
        print("\n  -> Evidence of OVERFITTING in old policy")
    else:
        print("\n  -> Good GENERALIZATION from old policy")

    print("=" * 70)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

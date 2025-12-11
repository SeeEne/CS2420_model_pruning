"""
Overfitting/Generalization Experiment for Pruned Models (CIFAR-100)

This experiment compares Encoder and MLP policies (old vs new):

ENCODER:
1. "Old Encoder Student": New teacher (CIFAR-100) -> Old encoder policy (from Tiny-ImageNet) -> FT
2. "New Encoder Student": New teacher (CIFAR-100) -> Train new encoder on CIFAR-100 -> FT

MLP:
3. "Old MLP Student": New teacher (CIFAR-100) -> Old MLP policy (from Tiny-ImageNet) -> FT
4. "New MLP Student": New teacher (CIFAR-100) -> Train new MLP on CIFAR-100 -> FT

Hypothesis: The old policies may overfit to Tiny-ImageNet's structure, leading to worse
pruning decisions on CIFAR-100 compared to policies trained directly on CIFAR-100.

Pipeline phases:
- Phase 1: Load/Train CIFAR-100 teacher
- Phase 2: Old Encoder Student (using old encoder masks)
- Phase 3: Train New Encoder on CIFAR-100
- Phase 4: New Encoder Student (using new encoder masks)
- Phase 5: Old MLP Student (using old MLP masks)
- Phase 6: Train New MLP on CIFAR-100 + New MLP Student

With checkpoint/resume capability at each phase.
"""

import os, math, random, gc, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import json
import copy

# =========================
# Config
# =========================
OUT_DIR = "checkpoints"
RESULTS_FILE = "cifar100_overfit_results.json"
CHECKPOINT_FILE = "cifar100_overfit_checkpoint.json"

# Model checkpoints from Phase 1 (trained on Tiny-ImageNet)
CKPT_TEACHER_ORIGINAL = "checkpoints/teacher_resnet50.pth"  # Tiny-ImageNet teacher
ENCODER_RESULTS_FILE = "checkpoints/resnet50_encoder_block_results.json"  # Old encoder masks
MLP_RESULTS_FILE = "checkpoints/resnet50_mlp_block_results.json"  # Old MLP masks

BATCH_SIZE = 128
NUM_WORKERS = 0
IMG_SIZE = 224
SEED = 42

# Policy training config (for new encoder)
POLICY_WARMUP_EPOCHS = 2
POLICY_TRAIN_EPOCHS = 10
POLICY_LR = 1e-5
MIN_RATIO = 0.1
MAX_RATIO = 0.8
RATIO_WEIGHT = 25.0
L1_M_WEIGHT = 1e-3
GATE_TEMP_START = 5.0
GATE_TEMP_END = 0.3
TEMP_KD = 2.0

# Fine-tuning config
FT_EPOCHS = 10
FT_LR = 1e-4
FT_WEIGHT_DECAY = 1e-4
NEW_TEACHER_EPOCHS = 15  # Epochs to train new teacher on CIFAR-100

# Compression ratios to test
EVAL_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]

# ResNet-50 has 16 Bottleneck blocks
NUM_BLOCKS = 16
BLOCKS_PER_STAGE = [3, 4, 6, 3]
TOKEN_DIM = 128
SUM_DIM = 64
ENC_WIDTH = 128
ENC_LAYERS = 2
ENC_HEADS = 4
MLP_HIDDEN_DIM = 512
MLP_WEIGHT_DECAY = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def clear_vram():
    """Clear VRAM by collecting garbage and emptying CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_vram_usage(tag=""):
    """Print current VRAM usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM {tag}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# =========================
# Pruned Block for Bottleneck
# =========================
class PrunedBottleneck(nn.Module):
    """Physical replacement for a pruned Bottleneck block."""
    expansion = 4

    def __init__(self, downsample_module):
        super().__init__()
        self.downsample = downsample_module

    def forward(self, x):
        if self.downsample is not None:
            return F.relu(self.downsample(x))
        else:
            return F.relu(x)

# =========================
# Policy Components (from 5_resnet50_block.py)
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
    """MLP-based policy (no self-attention)."""
    def __init__(self, num_blocks=NUM_BLOCKS, token_dim=TOKEN_DIM, hidden_dim=MLP_HIDDEN_DIM):
        super().__init__()
        self.budget_embed = nn.Sequential(nn.Linear(1, 64), nn.GELU())
        input_dim = (num_blocks * token_dim) + 64
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
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
        return self.net(x)

# =========================
# FLOPs Calculation
# =========================
def conv_flops(H, W, Cin, Cout, k, stride):
    return (H // stride) * (W // stride) * Cin * Cout * (k * k)

def get_bottleneck_flops(block, input_shape):
    B, C, H, W = input_shape
    f1 = conv_flops(H, W, block.conv1.in_channels, block.conv1.out_channels, 1, 1)
    stride2 = block.conv2.stride[0]
    H2, W2 = H // stride2, W // stride2
    f2 = conv_flops(H, W, block.conv2.in_channels, block.conv2.out_channels, 3, stride2)
    f3 = conv_flops(H2, W2, block.conv3.in_channels, block.conv3.out_channels, 1, 1)
    return float(f1 + f2 + f3)

# =========================
# Forward Pass Helpers
# =========================
def forward_resnet50_collect_blocks(model, x, collect_grads=False):
    """Forward pass through ResNet-50, collecting info for all 16 Bottleneck blocks."""
    infos = []
    h = model.conv1(x); h = model.bn1(h); h = model.relu(h); h = model.maxpool(h)
    stages = [model.layer1, model.layer2, model.layer3, model.layer4]

    for s_idx, stage in enumerate(stages):
        for b_idx, block in enumerate(stage):
            h_in = h
            if collect_grads: h_in.retain_grad()
            out = block.conv1(h_in); out = block.bn1(out); out = block.relu(out)
            out = block.conv2(out); out = block.bn2(out); out = block.relu(out)
            out = block.conv3(out); r_out = block.bn3(out)
            if collect_grads: r_out.retain_grad()
            if block.downsample is not None:
                skip = block.downsample(h_in)
            else:
                skip = h_in
            infos.append({
                "h_in": h_in, "r_out": r_out, "block_obj": block,
                "stage": s_idx, "block_idx": b_idx,
                "H": h_in.size(2), "W": h_in.size(3),
                "flops": get_bottleneck_flops(block, h_in.shape)
            })
            h = F.relu(skip + r_out)

    h = model.avgpool(h); h = torch.flatten(h, 1); logits = model.fc(h)
    return logits, infos

def forward_resnet50_gated_blocks(model, x, block_gates):
    """Forward pass with 16 gates (one per Bottleneck block)."""
    gate_idx = 0
    h = model.conv1(x); h = model.bn1(h); h = model.relu(h); h = model.maxpool(h)

    for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in stage:
            r = block.conv1(h); r = block.bn1(r); r = block.relu(r)
            r = block.conv2(r); r = block.bn2(r); r = block.relu(r)
            r = block.conv3(r); r = block.bn3(r)
            g = block_gates[gate_idx].view(1,1,1,1)
            r = r * g
            gate_idx += 1
            if block.downsample is not None:
                skip = block.downsample(h)
            else:
                skip = h
            h = F.relu(skip + r)

    h = model.avgpool(h); h = torch.flatten(h, 1)
    return model.fc(h)

def build_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device, num_blocks=NUM_BLOCKS):
    """Builds tokens for all blocks with Taylor scores."""
    x = x.requires_grad_(True)
    with torch.no_grad():
        _, teacher_infos = forward_resnet50_collect_blocks(teacher, x, collect_grads=False)
    logits_S, student_infos = forward_resnet50_collect_blocks(student, x, collect_grads=True)
    loss = F.kl_div(F.log_softmax(logits_S/TEMP_KD, dim=1),
                   F.softmax(y_T/TEMP_KD, dim=1), reduction='batchmean') * (TEMP_KD**2)
    loss.backward()

    token_list = []
    flops_list = []
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

    for info in student_infos:
        if info["r_out"].grad is not None:
            info["r_out"].grad = None

    tokens = torch.stack(token_list).unsqueeze(0)
    tokens = token_proj(tokens)
    flops = torch.tensor(flops_list, dtype=torch.float32, device=device)
    student.zero_grad()
    return tokens, flops

def kd_loss(student_logits, teacher_logits, T):
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction='batchmean') * (T*T)

# =========================
# CIFAR-100 Dataset
# =========================
def get_cifar100_loaders(data_root="data"):
    """Get CIFAR-100 data loaders"""
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR100(root=os.path.join(data_root, "cifar100"),
                                  train=True, download=True, transform=train_tf)
    val_ds = datasets.CIFAR100(root=os.path.join(data_root, "cifar100"),
                                train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, 100

# =========================
# Model Building
# =========================
def build_resnet50(num_classes=100):
    """Build ResNet-50 model with custom classifier"""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def physically_prune_model(original_model, mask):
    """Creates a new model with pruned blocks replaced by skip-only blocks."""
    pruned_model = copy.deepcopy(original_model)
    mask_idx = 0
    stages = [pruned_model.layer1, pruned_model.layer2, pruned_model.layer3, pruned_model.layer4]
    for stage in stages:
        for b_idx in range(len(stage)):
            if mask[mask_idx] == 0:
                original_block = stage[b_idx]
                stage[b_idx] = PrunedBottleneck(original_block.downsample)
            mask_idx += 1
    return pruned_model.to(device)

# =========================
# Evaluation & Utilities
# =========================
def run_evaluation(model, loader):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

def recalibrate_bn(model, loader, steps=50):
    """Recalibrate BN statistics"""
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

# =========================
# Train New Teacher on CIFAR-100 (with resume support)
# =========================
TEACHER_TRAINING_CKPT = "cifar100_teacher_training.pth"  # Intermediate checkpoint

def train_cifar100_teacher(train_loader, val_loader, epochs=NEW_TEACHER_EPOCHS):
    """Train a new ResNet-50 teacher on CIFAR-100 with resume support"""
    print("\n" + "="*60)
    print("Training New Teacher on CIFAR-100")
    print("="*60)

    training_ckpt_path = os.path.join(OUT_DIR, TEACHER_TRAINING_CKPT)
    start_epoch = 0
    best_acc = 0.0
    loss_history = []

    # Check for existing training checkpoint
    if os.path.exists(training_ckpt_path):
        print(f"Resuming teacher training from {training_ckpt_path}")
        ckpt = torch.load(training_ckpt_path, map_location='cpu')
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 100)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        loss_history = ckpt['loss_history']
        print(f"Resuming from epoch {start_epoch}, best_acc: {best_acc*100:.2f}%")
    else:
        # Start from ImageNet pretrained weights
        print("Starting fresh from ImageNet pretrained weights")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 100)
        model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = GradScaler()

    # Fast-forward scheduler if resuming
    for _ in range(start_epoch):
        scheduler.step()

    for ep in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                out = model(x)
                loss = F.cross_entropy(out, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        acc = run_evaluation(model, val_loader)
        avg_loss = epoch_loss / n_batches

        if acc > best_acc:
            best_acc = acc

        loss_history.append({"epoch": ep, "loss": avg_loss, "accuracy": acc})
        print(f"[Teacher Ep {ep+1}/{epochs}] Loss: {avg_loss:.4f} Acc: {acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

        # Save training checkpoint after each epoch
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'best_acc': best_acc,
            'loss_history': loss_history
        }, training_ckpt_path)

    print(f"\nNew Teacher Best Accuracy: {best_acc*100:.2f}%")

    # Remove training checkpoint after completion
    if os.path.exists(training_ckpt_path):
        os.remove(training_ckpt_path)
        print("Teacher training checkpoint cleared.")

    del optimizer, scheduler, scaler
    clear_vram()

    return model, best_acc, loss_history

# =========================
# Train New Encoder Policy on CIFAR-100
# =========================
def train_new_encoder_policy(teacher, train_loader):
    """Train a NEW encoder policy on CIFAR-100 data"""
    print("\n" + "="*60)
    print("Training NEW Encoder Policy on CIFAR-100")
    print("="*60)

    student = build_resnet50(num_classes=100).to(device)
    student.load_state_dict(teacher.state_dict())
    student.eval()

    summarizer = Summarizer(sum_dim=SUM_DIM).to(device)
    token_proj = nn.Sequential(
        nn.LayerNorm(SUM_DIM + 5),
        nn.Linear(SUM_DIM + 5, TOKEN_DIM)
    ).to(device)
    encoder = CompressionAwareEncoder(num_blocks=NUM_BLOCKS).to(device)

    params = list(summarizer.parameters()) + list(token_proj.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=POLICY_LR)

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
                print(f"[Ep {epoch}][{i}] KD: {acc_loss_kd/(i+1):.4f} Ratio: {acc_loss_ratio/(i+1):.4f} Temp: {temp:.2f}")

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
        print(f"Epoch {epoch} Summary - KD: {epoch_losses['loss_kd']:.4f}, Ratio: {epoch_losses['loss_ratio']:.4f}")

    del student
    clear_vram()

    return encoder, summarizer, token_proj, loss_history

# =========================
# Materialize and Fine-tune
# =========================
def materialize_and_finetune(teacher, mask, train_loader, val_loader, model_name="model"):
    """Materialize pruned model using mask and fine-tune"""
    print(f"\n--- {model_name} ---")
    print(f"Mask: {mask} (kept: {sum(mask)}/{NUM_BLOCKS})")
    print_vram_usage(f"before {model_name}")

    pruned_model = physically_prune_model(teacher, mask)
    recalibrate_bn(pruned_model, train_loader)

    acc_pre = run_evaluation(pruned_model, val_loader)
    print(f"Accuracy Pre-FT: {acc_pre*100:.2f}%")

    # Fine-tune with KD
    optimizer = torch.optim.AdamW(pruned_model.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, FT_EPOCHS)
    scaler = GradScaler()

    best_acc = acc_pre
    ft_loss_history = []

    for ep in range(FT_EPOCHS):
        pruned_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_T = teacher(x)

            with autocast():
                y_S = pruned_model(x)
                loss = kd_loss(y_S, y_T, TEMP_KD)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        acc = run_evaluation(pruned_model, val_loader)
        avg_loss = epoch_loss / n_batches

        if acc > best_acc:
            best_acc = acc

        ft_loss_history.append({"epoch": ep, "loss": avg_loss, "accuracy": acc})
        print(f"  [{model_name} FT Ep {ep+1}/{FT_EPOCHS}] Loss: {avg_loss:.4f} Acc: {acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

    del pruned_model, optimizer, scheduler, scaler
    clear_vram()
    print_vram_usage(f"after {model_name} cleanup")

    return best_acc, acc_pre, ft_loss_history

def get_new_encoder_mask(teacher, encoder, summarizer, token_proj, train_loader, ratio):
    """Generate mask using the new encoder policy"""
    x_sample, _ = next(iter(train_loader))
    x_sample = x_sample[:32].to(device)
    with torch.no_grad():
        y_T = teacher(x_sample)

    temp_student = build_resnet50(100).to(device)
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

    actual_ratio = current_flops / total_flops

    del temp_student
    clear_vram()

    return mask, actual_ratio, scores.detach().cpu().tolist()

# =========================
# Train New MLP Policy on CIFAR-100
# =========================
def train_new_mlp_policy(teacher, train_loader):
    """Train a NEW MLP policy on CIFAR-100 data"""
    print("\n" + "="*60)
    print("Training NEW MLP Policy on CIFAR-100")
    print("="*60)

    student = build_resnet50(num_classes=100).to(device)
    student.load_state_dict(teacher.state_dict())
    student.eval()

    summarizer = Summarizer(sum_dim=SUM_DIM).to(device)
    token_proj = nn.Sequential(
        nn.LayerNorm(SUM_DIM + 5),
        nn.Linear(SUM_DIM + 5, TOKEN_DIM)
    ).to(device)
    mlp = GlobalMLPPolicy(num_blocks=NUM_BLOCKS, token_dim=TOKEN_DIM, hidden_dim=MLP_HIDDEN_DIM).to(device)

    params = list(summarizer.parameters()) + list(token_proj.parameters()) + list(mlp.parameters())
    opt = torch.optim.AdamW(params, lr=POLICY_LR, weight_decay=MLP_WEIGHT_DECAY)

    total_epochs = POLICY_WARMUP_EPOCHS + POLICY_TRAIN_EPOCHS
    loss_history = []

    for epoch in range(total_epochs):
        mlp.train(); summarizer.train(); token_proj.train()
        t = epoch / max(1, total_epochs-1)
        temp = GATE_TEMP_START * (1-t) + GATE_TEMP_END * t

        acc_loss_kd = 0; acc_loss_ratio = 0; acc_loss_l1 = 0; acc_loss_total = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_T = teacher(x)

            tokens, flops = build_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device)
            target_ratio = random.uniform(MIN_RATIO, MAX_RATIO)
            logits = mlp(tokens, target_ratio).squeeze(0)
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
                print(f"[MLP Ep {epoch}][{i}] KD: {acc_loss_kd/(i+1):.4f} Ratio: {acc_loss_ratio/(i+1):.4f} Temp: {temp:.2f}")

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
        print(f"MLP Epoch {epoch} Summary - KD: {epoch_losses['loss_kd']:.4f}, Ratio: {epoch_losses['loss_ratio']:.4f}")

    del student
    clear_vram()

    return mlp, summarizer, token_proj, loss_history

def get_new_mlp_mask(teacher, mlp, summarizer, token_proj, train_loader, ratio):
    """Generate mask using the new MLP policy"""
    x_sample, _ = next(iter(train_loader))
    x_sample = x_sample[:32].to(device)
    with torch.no_grad():
        y_T = teacher(x_sample)

    temp_student = build_resnet50(100).to(device)
    temp_student.load_state_dict(teacher.state_dict())
    temp_student.eval()

    tokens, flops = build_block_tokens(teacher, temp_student, summarizer, token_proj, x_sample, y_T, device)
    logits = mlp(tokens, ratio).squeeze(0)
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

    actual_ratio = current_flops / total_flops

    del temp_student
    clear_vram()

    return mask, actual_ratio, scores.detach().cpu().tolist()

# =========================
# Load Old MLP Results
# =========================
def load_old_mlp_results():
    """Load MLP pruning results (masks) from Tiny-ImageNet experiment"""
    if os.path.exists(MLP_RESULTS_FILE):
        with open(MLP_RESULTS_FILE, 'r') as f:
            data = json.load(f)
            return {r['target_ratio']: r for r in data['results']}
    else:
        print(f"Warning: Old MLP results not found at {MLP_RESULTS_FILE}")
        return None

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
# Load Old Encoder Results
# =========================
def load_old_encoder_results():
    """Load encoder pruning results (masks) from Tiny-ImageNet experiment"""
    if os.path.exists(ENCODER_RESULTS_FILE):
        with open(ENCODER_RESULTS_FILE, 'r') as f:
            data = json.load(f)
            return {r['target_ratio']: r for r in data['results']}
    else:
        print(f"Warning: Old encoder results not found at {ENCODER_RESULTS_FILE}")
        return None

# =========================
# Main Experiment
# =========================
def main():
    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("="*70)
    print("CIFAR-100 Overfitting Experiment")
    print("Comparing: Old Student (old policy) vs New Student (new policy)")
    print("="*70)

    # Load checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"\nResuming from checkpoint...")
        print(f"  Phase: {checkpoint.get('phase', 'unknown')}")
        print(f"  Completed old student ratios: {checkpoint.get('completed_old_ratios', [])}")
        print(f"  Completed new student ratios: {checkpoint.get('completed_new_ratios', [])}")

    # Get CIFAR-100 data
    train_loader, val_loader, num_classes = get_cifar100_loaders()
    print(f"\nCIFAR-100: {num_classes} classes")

    # Load old encoder results
    old_encoder_results = load_old_encoder_results()
    if old_encoder_results is None:
        print("Error: Need old encoder results from 5_resnet50_block.py")
        return

    # Load old Tiny-ImageNet teacher for reference
    old_teacher_acc = 0.0
    if os.path.exists(CKPT_TEACHER_ORIGINAL):
        print(f"\nLoading old Tiny-ImageNet teacher from {CKPT_TEACHER_ORIGINAL}")
        old_ckpt = torch.load(CKPT_TEACHER_ORIGINAL, map_location='cpu')
        old_teacher_acc = old_ckpt.get('val_acc', 0)
        print(f"Old Teacher Accuracy (Tiny-ImageNet): {old_teacher_acc*100:.2f}%")
    else:
        print(f"Warning: Old teacher checkpoint not found at {CKPT_TEACHER_ORIGINAL}")

    # Load old MLP results
    old_mlp_results = load_old_mlp_results()
    if old_mlp_results is None:
        print("Warning: Old MLP results not found, MLP phases will be skipped")

    # Initialize results
    default_results = {
        "old_teacher": {"accuracy": old_teacher_acc},
        "teacher": {},
        "old_encoder_student": {},
        "new_encoder_student": {},
        "old_mlp_student": {},
        "new_mlp_student": {},
        "new_encoder_loss_history": [],
        "new_mlp_loss_history": []
    }

    if checkpoint:
        results = checkpoint.get('results', default_results)
        # Backward compatibility: migrate old key names to new ones
        if 'old_student' in results and 'old_encoder_student' not in results:
            results['old_encoder_student'] = results.pop('old_student')
        if 'new_student' in results and 'new_encoder_student' not in results:
            results['new_encoder_student'] = results.pop('new_student')
        # Ensure new keys exist
        if 'old_mlp_student' not in results:
            results['old_mlp_student'] = {}
        if 'new_mlp_student' not in results:
            results['new_mlp_student'] = {}
        if 'new_mlp_loss_history' not in results:
            results['new_mlp_loss_history'] = []
    else:
        results = default_results

    # Update old teacher accuracy in results
    results["old_teacher"] = {"accuracy": old_teacher_acc}

    # =========================
    # Phase 1: Load or Train CIFAR-100 Teacher
    # =========================
    cifar100_teacher_path = os.path.join(OUT_DIR, "cifar100_teacher.pth")

    if os.path.exists(cifar100_teacher_path):
        # Load existing CIFAR-100 teacher
        print(f"\nLoading existing CIFAR-100 teacher from {cifar100_teacher_path}")
        ckpt = torch.load(cifar100_teacher_path, map_location='cpu')
        teacher = build_resnet50(100).to(device)
        teacher.load_state_dict(ckpt['state_dict'])
        teacher_acc = ckpt.get('val_acc', 0)
        results["teacher"] = {"accuracy": teacher_acc}
    else:
        # Train new CIFAR-100 teacher
        print("\nCIFAR-100 teacher not found, training new teacher...")
        teacher, teacher_acc, teacher_loss = train_cifar100_teacher(train_loader, val_loader)
        results["teacher"] = {
            "accuracy": teacher_acc,
            "loss_history": teacher_loss
        }
        # Save teacher checkpoint
        torch.save({
            'state_dict': teacher.state_dict(),
            'val_acc': teacher_acc
        }, cifar100_teacher_path)
        print(f"Teacher saved to {cifar100_teacher_path}")

    teacher.eval()
    print(f"CIFAR-100 Teacher Accuracy: {teacher_acc*100:.2f}%")

    # =========================
    # Phase 2: Old Encoder Student (using old encoder masks)
    # =========================
    print("\n" + "="*70)
    print("Phase 2: OLD ENCODER STUDENT (using masks from Tiny-ImageNet encoder)")
    print("="*70)

    # Backward compatibility: check both old and new key names
    completed_old_enc_ratios = checkpoint.get('completed_old_enc_ratios', checkpoint.get('completed_old_ratios', [])) if checkpoint else []

    for ratio in EVAL_RATIOS:
        if ratio in completed_old_enc_ratios:
            print(f"\n[Old Encoder Student] Ratio {ratio} already completed, skipping...")
            continue

        if ratio not in old_encoder_results:
            print(f"\n[Old Encoder Student] Ratio {ratio} not found in old encoder results, skipping...")
            continue

        old_mask = old_encoder_results[ratio]['mask']
        print(f"\n[Old Encoder Student] Ratio {ratio}")

        best_acc, acc_pre, ft_loss = materialize_and_finetune(
            teacher, old_mask, train_loader, val_loader, model_name=f"OldEncStudent-r{ratio}"
        )

        results["old_encoder_student"][str(ratio)] = {
            "target_ratio": ratio,
            "mask": old_mask,
            "kept_blocks": sum(old_mask),
            "accuracy_before_ft": acc_pre,
            "accuracy_after_ft": best_acc,
            "accuracy_drop": teacher_acc - best_acc,
            "ft_loss_history": ft_loss
        }

        completed_old_enc_ratios.append(ratio)
        save_checkpoint({
            'phase': 'old_encoder_student',
            'completed_old_enc_ratios': completed_old_enc_ratios,
            'completed_new_enc_ratios': checkpoint.get('completed_new_enc_ratios', []) if checkpoint else [],
            'completed_old_mlp_ratios': checkpoint.get('completed_old_mlp_ratios', []) if checkpoint else [],
            'completed_new_mlp_ratios': checkpoint.get('completed_new_mlp_ratios', []) if checkpoint else [],
            'results': results
        })

    # VRAM cleanup after Phase 2
    print_vram_usage("after Phase 2")
    clear_vram()

    # =========================
    # Phase 3: Train New Encoder on CIFAR-100
    # =========================
    new_encoder_path = os.path.join(OUT_DIR, "cifar100_new_encoder.pth")

    if not os.path.exists(new_encoder_path):
        print("\n" + "="*70)
        print("Phase 3: Training NEW Encoder Policy on CIFAR-100")
        print("="*70)

        new_encoder, new_summarizer, new_token_proj, new_loss_history = train_new_encoder_policy(
            teacher, train_loader
        )

        torch.save({
            'encoder': new_encoder.state_dict(),
            'summarizer': new_summarizer.state_dict(),
            'token_proj': new_token_proj.state_dict()
        }, new_encoder_path)

        results["new_encoder_loss_history"] = new_loss_history

        save_checkpoint({
            'phase': 'new_encoder_done',
            'completed_old_enc_ratios': completed_old_enc_ratios,
            'completed_new_enc_ratios': [],
            'completed_old_mlp_ratios': [],
            'completed_new_mlp_ratios': [],
            'results': results
        })
    else:
        print("\nLoading existing new encoder...")
        state = torch.load(new_encoder_path, map_location='cpu')
        new_encoder = CompressionAwareEncoder(num_blocks=NUM_BLOCKS).to(device)
        new_encoder.load_state_dict(state['encoder'])
        new_summarizer = Summarizer(sum_dim=SUM_DIM).to(device)
        new_summarizer.load_state_dict(state['summarizer'])
        new_token_proj = nn.Sequential(
            nn.LayerNorm(SUM_DIM + 5),
            nn.Linear(SUM_DIM + 5, TOKEN_DIM)
        ).to(device)
        new_token_proj.load_state_dict(state['token_proj'])

    new_encoder.eval()
    new_summarizer.eval()
    new_token_proj.eval()

    # VRAM cleanup after Phase 3
    print_vram_usage("after Phase 3")
    clear_vram()

    # =========================
    # Phase 4: New Encoder Student (using new encoder masks)
    # =========================
    print("\n" + "="*70)
    print("Phase 4: NEW ENCODER STUDENT (using masks from CIFAR-100 encoder)")
    print("="*70)

    # Backward compatibility: check both old and new key names
    completed_new_enc_ratios = checkpoint.get('completed_new_enc_ratios', checkpoint.get('completed_new_ratios', [])) if checkpoint else []

    for ratio in EVAL_RATIOS:
        if ratio in completed_new_enc_ratios:
            print(f"\n[New Encoder Student] Ratio {ratio} already completed, skipping...")
            continue

        print(f"\n[New Encoder Student] Ratio {ratio}")

        # Get mask from new encoder
        new_mask, actual_ratio, scores = get_new_encoder_mask(
            teacher, new_encoder, new_summarizer, new_token_proj, train_loader, ratio
        )
        print(f"New Mask: {new_mask} (actual ratio: {actual_ratio:.3f})")

        best_acc, acc_pre, ft_loss = materialize_and_finetune(
            teacher, new_mask, train_loader, val_loader, model_name=f"NewEncStudent-r{ratio}"
        )

        results["new_encoder_student"][str(ratio)] = {
            "target_ratio": ratio,
            "actual_ratio": actual_ratio,
            "mask": new_mask,
            "kept_blocks": sum(new_mask),
            "scores": scores,
            "accuracy_before_ft": acc_pre,
            "accuracy_after_ft": best_acc,
            "accuracy_drop": teacher_acc - best_acc,
            "ft_loss_history": ft_loss
        }

        completed_new_enc_ratios.append(ratio)
        save_checkpoint({
            'phase': 'new_encoder_student',
            'completed_old_enc_ratios': completed_old_enc_ratios,
            'completed_new_enc_ratios': completed_new_enc_ratios,
            'completed_old_mlp_ratios': checkpoint.get('completed_old_mlp_ratios', []) if checkpoint else [],
            'completed_new_mlp_ratios': checkpoint.get('completed_new_mlp_ratios', []) if checkpoint else [],
            'results': results
        })

    # VRAM cleanup after Phase 4 - also free encoder components no longer needed
    del new_encoder, new_summarizer, new_token_proj
    print_vram_usage("after Phase 4")
    clear_vram()

    # =========================
    # Phase 5: Old MLP Student (using old MLP masks)
    # =========================
    if old_mlp_results is not None:
        print("\n" + "="*70)
        print("Phase 5: OLD MLP STUDENT (using masks from Tiny-ImageNet MLP)")
        print("="*70)

        completed_old_mlp_ratios = checkpoint.get('completed_old_mlp_ratios', []) if checkpoint else []

        for ratio in EVAL_RATIOS:
            if ratio in completed_old_mlp_ratios:
                print(f"\n[Old MLP Student] Ratio {ratio} already completed, skipping...")
                continue

            if ratio not in old_mlp_results:
                print(f"\n[Old MLP Student] Ratio {ratio} not found in old MLP results, skipping...")
                continue

            old_mask = old_mlp_results[ratio]['mask']
            print(f"\n[Old MLP Student] Ratio {ratio}")

            best_acc, acc_pre, ft_loss = materialize_and_finetune(
                teacher, old_mask, train_loader, val_loader, model_name=f"OldMLPStudent-r{ratio}"
            )

            results["old_mlp_student"][str(ratio)] = {
                "target_ratio": ratio,
                "mask": old_mask,
                "kept_blocks": sum(old_mask),
                "accuracy_before_ft": acc_pre,
                "accuracy_after_ft": best_acc,
                "accuracy_drop": teacher_acc - best_acc,
                "ft_loss_history": ft_loss
            }

            completed_old_mlp_ratios.append(ratio)
            save_checkpoint({
                'phase': 'old_mlp_student',
                'completed_old_enc_ratios': completed_old_enc_ratios,
                'completed_new_enc_ratios': completed_new_enc_ratios,
                'completed_old_mlp_ratios': completed_old_mlp_ratios,
                'completed_new_mlp_ratios': checkpoint.get('completed_new_mlp_ratios', []) if checkpoint else [],
                'results': results
            })
    else:
        completed_old_mlp_ratios = []

    # VRAM cleanup after Phase 5
    print_vram_usage("after Phase 5")
    clear_vram()

    # =========================
    # Phase 6: Train New MLP on CIFAR-100 and New MLP Student
    # =========================
    new_mlp_path = os.path.join(OUT_DIR, "cifar100_new_mlp.pth")

    if not os.path.exists(new_mlp_path):
        print("\n" + "="*70)
        print("Phase 6a: Training NEW MLP Policy on CIFAR-100")
        print("="*70)

        new_mlp, new_mlp_summarizer, new_mlp_token_proj, new_mlp_loss_history = train_new_mlp_policy(
            teacher, train_loader
        )

        torch.save({
            'mlp': new_mlp.state_dict(),
            'summarizer': new_mlp_summarizer.state_dict(),
            'token_proj': new_mlp_token_proj.state_dict()
        }, new_mlp_path)

        results["new_mlp_loss_history"] = new_mlp_loss_history

        save_checkpoint({
            'phase': 'new_mlp_done',
            'completed_old_enc_ratios': completed_old_enc_ratios,
            'completed_new_enc_ratios': completed_new_enc_ratios,
            'completed_old_mlp_ratios': completed_old_mlp_ratios,
            'completed_new_mlp_ratios': [],
            'results': results
        })
    else:
        print("\nLoading existing new MLP...")
        state = torch.load(new_mlp_path, map_location='cpu')
        new_mlp = GlobalMLPPolicy(num_blocks=NUM_BLOCKS, token_dim=TOKEN_DIM, hidden_dim=MLP_HIDDEN_DIM).to(device)
        new_mlp.load_state_dict(state['mlp'])
        new_mlp_summarizer = Summarizer(sum_dim=SUM_DIM).to(device)
        new_mlp_summarizer.load_state_dict(state['summarizer'])
        new_mlp_token_proj = nn.Sequential(
            nn.LayerNorm(SUM_DIM + 5),
            nn.Linear(SUM_DIM + 5, TOKEN_DIM)
        ).to(device)
        new_mlp_token_proj.load_state_dict(state['token_proj'])

    new_mlp.eval()
    new_mlp_summarizer.eval()
    new_mlp_token_proj.eval()

    # VRAM cleanup after Phase 6a
    print_vram_usage("after Phase 6a (MLP training)")
    clear_vram()

    # Phase 6b: New MLP Student (using new MLP masks)
    print("\n" + "="*70)
    print("Phase 6b: NEW MLP STUDENT (using masks from CIFAR-100 MLP)")
    print("="*70)

    completed_new_mlp_ratios = checkpoint.get('completed_new_mlp_ratios', []) if checkpoint else []

    for ratio in EVAL_RATIOS:
        if ratio in completed_new_mlp_ratios:
            print(f"\n[New MLP Student] Ratio {ratio} already completed, skipping...")
            continue

        print(f"\n[New MLP Student] Ratio {ratio}")

        # Get mask from new MLP
        new_mask, actual_ratio, scores = get_new_mlp_mask(
            teacher, new_mlp, new_mlp_summarizer, new_mlp_token_proj, train_loader, ratio
        )
        print(f"New MLP Mask: {new_mask} (actual ratio: {actual_ratio:.3f})")

        best_acc, acc_pre, ft_loss = materialize_and_finetune(
            teacher, new_mask, train_loader, val_loader, model_name=f"NewMLPStudent-r{ratio}"
        )

        results["new_mlp_student"][str(ratio)] = {
            "target_ratio": ratio,
            "actual_ratio": actual_ratio,
            "mask": new_mask,
            "kept_blocks": sum(new_mask),
            "scores": scores,
            "accuracy_before_ft": acc_pre,
            "accuracy_after_ft": best_acc,
            "accuracy_drop": teacher_acc - best_acc,
            "ft_loss_history": ft_loss
        }

        completed_new_mlp_ratios.append(ratio)
        save_checkpoint({
            'phase': 'new_mlp_student',
            'completed_old_enc_ratios': completed_old_enc_ratios,
            'completed_new_enc_ratios': completed_new_enc_ratios,
            'completed_old_mlp_ratios': completed_old_mlp_ratios,
            'completed_new_mlp_ratios': completed_new_mlp_ratios,
            'results': results
        })

    # VRAM cleanup after Phase 6b - free MLP components and teacher
    del new_mlp, new_mlp_summarizer, new_mlp_token_proj, teacher
    print_vram_usage("after Phase 6b (all phases complete)")
    clear_vram()

    # =========================
    # Save Final Results
    # =========================
    results_path = os.path.join(OUT_DIR, RESULTS_FILE)
    final_results = {
        "experiment": "CIFAR-100 Old vs New Student Comparison",
        "old_teacher_accuracy_tinyimagenet": old_teacher_acc,
        "new_teacher_accuracy_cifar100": teacher_acc,
        "compression_ratios": EVAL_RATIOS,
        "results": results
    }
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    clear_checkpoint()

    # =========================
    # Print Summary
    # =========================
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY: Old vs New Student (Encoder & MLP)")
    print("="*80)
    print(f"Old Teacher Accuracy (Tiny-ImageNet): {old_teacher_acc*100:.2f}%")
    print(f"New Teacher Accuracy (CIFAR-100):     {teacher_acc*100:.2f}%")
    print()

    # Encoder Comparison
    print("\n--- ENCODER COMPARISON ---")
    print(f"{'Ratio':<8} {'Old Enc Acc':<14} {'New Enc Acc':<14} {'Winner':<12}")
    print("-"*50)

    for ratio in EVAL_RATIOS:
        old_data = results["old_encoder_student"].get(str(ratio), {})
        new_data = results["new_encoder_student"].get(str(ratio), {})

        old_acc = old_data.get("accuracy_after_ft", 0) * 100
        new_acc = new_data.get("accuracy_after_ft", 0) * 100

        if old_acc > 0 and new_acc > 0:
            winner = "OLD_ENC" if old_acc > new_acc else "NEW_ENC"
            diff = abs(old_acc - new_acc)
            winner = f"{winner} (+{diff:.1f}%)"
        else:
            winner = "N/A"

        print(f"{ratio:<8.1f} {old_acc:<14.2f} {new_acc:<14.2f} {winner:<12}")

    # MLP Comparison
    print("\n--- MLP COMPARISON ---")
    print(f"{'Ratio':<8} {'Old MLP Acc':<14} {'New MLP Acc':<14} {'Winner':<12}")
    print("-"*50)

    for ratio in EVAL_RATIOS:
        old_data = results["old_mlp_student"].get(str(ratio), {})
        new_data = results["new_mlp_student"].get(str(ratio), {})

        old_acc = old_data.get("accuracy_after_ft", 0) * 100
        new_acc = new_data.get("accuracy_after_ft", 0) * 100

        if old_acc > 0 and new_acc > 0:
            winner = "OLD_MLP" if old_acc > new_acc else "NEW_MLP"
            diff = abs(old_acc - new_acc)
            winner = f"{winner} (+{diff:.1f}%)"
        else:
            winner = "N/A"

        print(f"{ratio:<8.1f} {old_acc:<14.2f} {new_acc:<14.2f} {winner:<12}")

    # Cross-Method Comparison (New Encoder vs New MLP)
    print("\n--- NEW ENCODER vs NEW MLP (both trained on CIFAR-100) ---")
    print(f"{'Ratio':<8} {'New Enc Acc':<14} {'New MLP Acc':<14} {'Better Method':<15}")
    print("-"*55)

    for ratio in EVAL_RATIOS:
        enc_data = results["new_encoder_student"].get(str(ratio), {})
        mlp_data = results["new_mlp_student"].get(str(ratio), {})

        enc_acc = enc_data.get("accuracy_after_ft", 0) * 100
        mlp_acc = mlp_data.get("accuracy_after_ft", 0) * 100

        if enc_acc > 0 and mlp_acc > 0:
            winner = "ENCODER" if enc_acc > mlp_acc else "MLP"
            diff = abs(enc_acc - mlp_acc)
            winner = f"{winner} (+{diff:.1f}%)"
        else:
            winner = "N/A"

        print(f"{ratio:<8.1f} {enc_acc:<14.2f} {mlp_acc:<14.2f} {winner:<15}")

    print("\n" + "="*80)
    print("Detailed Results:")
    print("="*80)

    print("\nOld Encoder Student (Tiny-ImageNet policy):")
    for ratio in EVAL_RATIOS:
        data = results["old_encoder_student"].get(str(ratio), {})
        if data:
            print(f"  r={ratio}: Acc={data['accuracy_after_ft']*100:.2f}%, Drop={data['accuracy_drop']*100:.2f}%, Kept={data['kept_blocks']}")

    print("\nNew Encoder Student (CIFAR-100 policy):")
    for ratio in EVAL_RATIOS:
        data = results["new_encoder_student"].get(str(ratio), {})
        if data:
            print(f"  r={ratio}: Acc={data['accuracy_after_ft']*100:.2f}%, Drop={data['accuracy_drop']*100:.2f}%, Kept={data['kept_blocks']}")

    print("\nOld MLP Student (Tiny-ImageNet policy):")
    for ratio in EVAL_RATIOS:
        data = results["old_mlp_student"].get(str(ratio), {})
        if data:
            print(f"  r={ratio}: Acc={data['accuracy_after_ft']*100:.2f}%, Drop={data['accuracy_drop']*100:.2f}%, Kept={data['kept_blocks']}")

    print("\nNew MLP Student (CIFAR-100 policy):")
    for ratio in EVAL_RATIOS:
        data = results["new_mlp_student"].get(str(ratio), {})
        if data:
            print(f"  r={ratio}: Acc={data['accuracy_after_ft']*100:.2f}%, Drop={data['accuracy_drop']*100:.2f}%, Kept={data['kept_blocks']}")

    print("\n" + "="*80)
    print("Experiment complete!")
    print("="*80)

if __name__ == "__main__":
    main()

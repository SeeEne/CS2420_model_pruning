"""


Compression-Aware Encoder Training with 16-Gate Pruning for ResNet-18 on CIFAR-10


"""

import os, math, random, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json

# =========================
# Config
# =========================
DATA_DIR = "data"
CKPT = "checkpoints/teacher.pth"
OUT_DIR = "checkpoints"
RESULTS_FILE = "checkpoints/encoder_16gate_results.json"

BATCH_SIZE = 128
NUM_WORKERS = 0
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
POLICY_LR = 1e-3
WEIGHT_DECAY = 0.0
SEED = 42

# Budget training config
MIN_RATIO = 0.1
MAX_RATIO = 0.8
RATIO_WEIGHT = 25.0
GATE_TEMP_START = 5.0
GATE_TEMP_END = 0.3

# Fine-tuning config per ratio
FT_EPOCHS = 10
FT_LR = 1e-3

# Target ratios to evaluate
EVAL_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# L1 sparsity on gates
L1_M_WEIGHT = 1e-3

NUM_BLOCKS = 8
NUM_UNITS = 16   # 2 per block

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# =========================
# Data
# =========================
def get_loaders():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.CIFAR10(root=DATA_DIR, train=True, transform=train_tf, download=True)
    test_ds  = datasets.CIFAR10(root=DATA_DIR, train=False, transform=test_tf, download=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, test_loader

# =========================
# Models
# =========================
def build_resnet18_teacher(num_classes=10):
    return models.resnet18(weights=None, num_classes=num_classes)

class Summarizer(nn.Module):
    def __init__(self, sum_dim=SUM_DIM, k=64):
        super().__init__()
        self.pool1d = nn.AdaptiveAvgPool1d(k)
        self.proj = nn.Linear(4 * k, 256)
        self.mlp  = nn.Sequential(
            nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, sum_dim), nn.LayerNorm(sum_dim)
        )

    def _pool_vec(self, v):
        v = v.unsqueeze(1)
        v = self.pool1d(v)
        return v.squeeze(1)

    def forward(self, h, r):
        gap_h = h.mean(dim=(2,3)); gmp_h, _ = h.flatten(2).max(dim=2)
        gap_r = r.mean(dim=(2,3)); gmp_r, _ = r.flatten(2).max(dim=2)
        parts = [self._pool_vec(p) for p in [gap_h, gmp_h, gap_r, gmp_r]]
        feats = torch.cat(parts, dim=1)
        z = self.mlp(self.proj(feats))
        return z.mean(dim=0)

class TokenProj(nn.Module):
    def __init__(self, in_dim, out_dim=TOKEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )
    def forward(self, x): return self.net(x)

class CompressionAwareEncoder(nn.Module):
    def __init__(self, dim=ENC_WIDTH, depth=ENC_LAYERS, heads=ENC_HEADS,
                 mlp_ratio=2.0, num_blocks=NUM_UNITS):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim*mlp_ratio),
            batch_first=True, activation='gelu', norm_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos = nn.Parameter(torch.zeros(1, num_blocks + 1, dim))

        self.budget_embed = nn.Sequential(
            nn.Linear(1, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        for m in self.budget_embed.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=2.0)

        self.head = nn.Linear(dim, 1)

    def forward(self, tokens, target_ratio):
        # FIX1: device/shape safe
        if not torch.is_tensor(target_ratio):
            target_ratio = torch.tensor([[float(target_ratio)]], dtype=torch.float32, device=tokens.device)
        else:
            target_ratio = target_ratio.float().view(1, 1).to(tokens.device)

        budget_tok = self.budget_embed(target_ratio)
        x = torch.cat([budget_tok.unsqueeze(1), tokens], dim=1)
        x = x + self.pos[:, :x.size(1)]
        h = self.enc(x)
        block_h = h[:, 1:, :]
        logits = self.head(block_h).squeeze(-1)   # [B(=1), NUM_UNITS] -> [NUM_UNITS]
        return logits

# =========================
# Helper Functions
# =========================
def kd_loss(student_logits, teacher_logits, T=TEMP_KD):
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction='batchmean') * (T*T)

def conv_flops(H, W, Cin, Cout, k, downsample_stride=1):
    return H * W * Cin * Cout * (k * k) // (downsample_stride * downsample_stride)

@torch.no_grad()
def collect_block_infos_teacher(teacher: models.ResNet, x: torch.Tensor):
    """
    Teacher forward to collect per-block tensors (h_in, r1, r2) + statics.
    We'll produce two "unit" entries per BasicBlock (sub_id=0 for conv1, 1 for conv2).
    """
    infos = []
    h = teacher.conv1(x); h = teacher.bn1(h); h = teacher.relu(h); h = teacher.maxpool(h)
    stages = [teacher.layer1, teacher.layer2, teacher.layer3, teacher.layer4]
    for s, layer in enumerate(stages):
        for i, block in enumerate(layer):
            h_in = h
            # conv1 path
            r1 = block.conv1(h_in); r1 = block.bn1(r1); r1_relu = block.relu(r1)
            # conv2 path
            r2 = block.conv2(r1_relu); r2 = block.bn2(r2)
            # skip
            skip = block.downsample(h_in) if block.downsample is not None else h_in
            out_pre = skip + r2
            h = block.relu(out_pre)

            # BN magnitude (two bn layers)
            gamma1 = block.bn1.weight.abs().sum().item()
            gamma2 = block.bn2.weight.abs().sum().item()

            # stride=2 if downsample exists (in torchvision BasicBlock, it's the stage transition)
            down = (block.downsample is not None)

            # unit 0 = conv1
            infos.append({
                "stage": s, "idx": i, "sub": 0,
                "h_in": h_in, "r": r1,       # summarizer(h_in, r1)
                "C_in": h_in.size(1), "C_mid": r1.size(1),
                "H": h_in.size(2), "W": h_in.size(3),
                "down": down,
                "bn_gamma": gamma1,
            })
            # unit 1 = conv2
            infos.append({
                "stage": s, "idx": i, "sub": 1,
                "h_in": r1_relu, "r": r2,    # summarizer(r1_relu, r2)
                "C_in": r1.size(1), "C_mid": r2.size(1),  # both are Cout
                "H": r1.size(2), "W": r1.size(3),
                "down": False,               # conv2 stride=1
                "bn_gamma": gamma2,
            })
    return infos

def estimate_unit_flops(info):
    """
    Per-unit FLOPs:
      - unit 0 (conv1): conv3x3 on (H/stride,W/stride) + downsample(1x1,stride) if exists
      - unit 1 (conv2): conv3x3 on (H/stride=1) -> use the conv1's output spatial size
    We approximate conv2's H,W as conv1 output H1,W1 (if conv1 had stride=2).
    To do that, we compute H1,W1 locally using 'down' flag from unit 0.
    For unit 1, we must know whether its block had stride=2 in conv1; we infer by stage boundary:
    Here we pass unit dicts in order, so the pair is contiguous: unit0 then unit1.
    To avoid cross-couple, we store H,W for each unit as "its input spatial size".
    For conv1 unit, output H1,W1 = ceil(H/stride). For conv2 unit, input is r1_relu, already H1,W1.
    """
    Cin = info["C_in"]
    Cout = info["C_mid"]
    H, W = info["H"], info["W"]
    if info["sub"] == 0:
        stride = 2 if info["down"] else 1
        H1, W1 = math.ceil(H/stride), math.ceil(W/stride)
        f  = conv_flops(H1, W1, Cin, Cout, 3)     # conv1
        if info["down"]:
            f += H1 * W1 * Cin * Cout             # downsample 1x1
        return float(f)
    else:
        # conv2 (stride=1), spatial is whatever came out of conv1 (already reflected in H,W)
        f = conv_flops(H, W, Cin, Cout, 3)
        return float(f)

def resnet18_forward_collect_r1r2(student: models.ResNet, x: torch.Tensor, gates: torch.Tensor=None,
                                  need_lists: bool=False):
    """
    Student forward with 16 gates: per BasicBlock, gate on conv1 output and conv2 output.
    If need_lists=True, return r1_list and r2_list (requires_grad) for Taylor.
    """
    if gates is None:
        gates = torch.ones(NUM_UNITS, device=x.device)
    assert len(gates) == NUM_UNITS

    r1_list, r2_list = [], []
    g_idx = 0

    h = student.conv1(x); h = student.bn1(h); h = student.relu(h); h = student.maxpool(h)
    for layer in [student.layer1, student.layer2, student.layer3, student.layer4]:
        for block in layer:
            
            h = h.detach()
            h.requires_grad_(True)
            
            # conv1
            r1 = block.conv1(h); r1 = block.bn1(r1); r1 = block.relu(r1)
            if need_lists:
                r1.retain_grad()
                r1_list.append(r1)
            g1 = gates[g_idx].view(1,1,1,1); g_idx += 1
            r1 = r1 * g1

            # conv2
            r2 = block.conv2(r1); r2 = block.bn2(r2)
            if need_lists:
                r2.retain_grad()
                r2_list.append(r2)
            g2 = gates[g_idx].view(1,1,1,1); g_idx += 1
            r2 = r2 * g2

            # skip
            skip = block.downsample(h) if block.downsample is not None else h
            out_pre = skip + r2
            h = block.relu(out_pre)

    h = student.avgpool(h)
    h = torch.flatten(h, 1)
    logits = student.fc(h)
    return logits, r1_list, r2_list

def build_tokens_with_taylor16(teacher, student, summarizer, token_proj, x, y_T, device):
    """
    Build 16 tokens (2 per block) + FLOPs vector length=16 + Taylor sensitivities for r1 & r2.
    """
    teacher.eval(); student.eval()
    with torch.no_grad():
        _ = teacher(x)
        infos = collect_block_infos_teacher(teacher, x)  # len=16

    # KD forward/backward to get grads on r1 and r2
    logits_S, r1_list, r2_list = resnet18_forward_collect_r1r2(student, x, gates=None, need_lists=True)
    kd = kd_loss(logits_S, y_T, T=TEMP_KD)
    kd.backward()

    taylor_vals = []
    for r in r1_list + r2_list:
        if r.grad is None:
            taylor_vals.append(0.0)
        else:
            taylor_vals.append((r.grad * r).abs().mean().detach().item())
    # clear grads on r tensors
    for r in r1_list + r2_list:
        if r.grad is not None:
            r.grad.detach_(); r.grad.zero_()
    del r1_list, r2_list

    # assemble tokens
    taylor_tensor = torch.tensor([math.log1p(v) for v in taylor_vals], dtype=torch.float32, device=device)
    token_list, flops_list = [], []
    for idx, info in enumerate(infos):
        s = summarizer(info["h_in"], info["r"]).to(device)
        C = float(info["C_in"] + info["C_mid"])  # scale for gamma
        static = torch.tensor([
            info["stage"]/3.0,
            info["idx"]/1.0,
            float(info["sub"]),                   # NEW: sub_id (0 for conv1, 1 for conv2)
            info["bn_gamma"] / (C + 1e-6),
            info["H"]/IMG_SIZE, info["W"]/IMG_SIZE
        ], dtype=torch.float32, device=device)
        tay = taylor_tensor[idx].view(1)
        tok = torch.cat([s, static, tay], dim=0)  # SUM_DIM + 6 + 1
        token_list.append(tok)
        flops_list.append(estimate_unit_flops(info))

    tokens = torch.stack(token_list, dim=0).unsqueeze(0)              # [1, 16, SUM_DIM+7]
    tokens = token_proj(tokens)                                       # [1, 16, TOKEN_DIM]
    flops = torch.tensor(flops_list, dtype=torch.float32, device=device)  # [16]
    return tokens, flops

def resnet18_masked_forward_with_gates16(student: models.ResNet, x: torch.Tensor, gates: torch.Tensor):
    assert len(gates) == NUM_UNITS
    g_idx = 0
    h = student.conv1(x); h = student.bn1(h); h = student.relu(h); h = student.maxpool(h)
    for layer in [student.layer1, student.layer2, student.layer3, student.layer4]:
        for block in layer:
            # conv1
            r1 = block.conv1(h); r1 = block.bn1(r1); r1 = block.relu(r1)
            g1 = gates[g_idx].view(1,1,1,1); g_idx += 1
            r1 = r1 * g1
            # conv2
            r2 = block.conv2(r1); r2 = block.bn2(r2)
            g2 = gates[g_idx].view(1,1,1,1); g_idx += 1
            r2 = r2 * g2
            # skip
            skip = block.downsample(h) if block.downsample is not None else h
            out_pre = skip + r2
            h = block.relu(out_pre)
    h = student.avgpool(h)
    h = torch.flatten(h, 1)
    logits = student.fc(h)
    return logits

@torch.no_grad()
def evaluate_accuracy(model, loader, mask=None):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if mask is not None:
            logits = resnet18_masked_forward_with_gates16(model, x, mask)
        else:
            logits = model(x)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    return correct / total

# =========================
# PHASE 1: Train Policy Encoder
# =========================
def train_policy_encoder(teacher, train_loader, test_loader):
    print("\n" + "="*80)
    print("PHASE 1: Training Compression-Aware Policy Encoder (16-gate)")
    print("="*80)

    # frozen student (for Taylor & masked KD forward)
    student = build_resnet18_teacher(num_classes=10).to(device)
    student.load_state_dict(teacher.state_dict())
    for p in student.parameters(): p.requires_grad = False
    student.eval()

    summarizer = Summarizer(sum_dim=SUM_DIM, k=64).to(device)
    token_proj = TokenProj(SUM_DIM + 7, TOKEN_DIM).to(device)  # SUM + [stage,idx,sub,gamma,H,W]=6 + taylor=1  => +7
    encoder = CompressionAwareEncoder(dim=ENC_WIDTH, depth=ENC_LAYERS, heads=ENC_HEADS, num_blocks=NUM_UNITS).to(device)

    params = (list(summarizer.parameters()) +
              list(token_proj.parameters()) +
              list(encoder.parameters()))
    opt = torch.optim.AdamW(params, lr=POLICY_LR, weight_decay=WEIGHT_DECAY)

    total_epochs = POLICY_WARMUP_EPOCHS + POLICY_TRAIN_EPOCHS
    for epoch in range(total_epochs):
        encoder.train(); summarizer.train(); token_proj.train()
        student.eval()

        lr = POLICY_LR * 0.5 * (1 + math.cos(math.pi * epoch / max(1, total_epochs-1)))
        for g in opt.param_groups: g['lr'] = lr

        t = epoch / max(1, total_epochs - 1)
        gate_temp = GATE_TEMP_START * (1 - t) + GATE_TEMP_END * t

        run_kd = run_ratio = run_l1 = 0.0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                y_T = teacher(x)

            tokens, flops = build_tokens_with_taylor16(teacher, student, summarizer, token_proj, x, y_T, device)

            target_ratio = random.uniform(MIN_RATIO, MAX_RATIO)
            logits = encoder(tokens, target_ratio).squeeze(0)     # [16]
            m = torch.sigmoid(logits / gate_temp)                 # [16]

            exp_flops = (m * flops).sum()
            full_flops = flops.sum() + 1e-6
            flops_ratio = exp_flops / full_flops

            y_S = resnet18_masked_forward_with_gates16(student, x, m)
            loss_kd = kd_loss(y_S, y_T, T=TEMP_KD)
            loss_ratio = (flops_ratio - target_ratio) ** 2
            loss_l1 = m.abs().mean()
            loss = loss_kd + RATIO_WEIGHT * loss_ratio + L1_M_WEIGHT * loss_l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            run_kd  += float(loss_kd.detach())
            run_ratio += float(loss_ratio.detach())
            run_l1 += float(loss_l1.detach())

            if (i+1) % 100 == 0:
                print(f"[Ep {epoch+1}/{total_epochs}] step {i+1}/{len(train_loader)} "
                      f"KD={run_kd/100:.3f}  ratio²={run_ratio/100:.5f} (×{RATIO_WEIGHT}={(run_ratio/100)*RATIO_WEIGHT:.3f}) "
                      f"L1={run_l1/100:.4f}  temp={gate_temp:.2f}")
                run_kd = run_ratio = run_l1 = 0.0

    print("\nPolicy encoder training complete!")
    return encoder, summarizer, token_proj

# =========================
# PHASE 2: Materialize Masks and Fine-tune
# =========================
def materialize_mask_for_ratio(encoder, summarizer, token_proj, teacher, train_loader, target_ratio):
    """
    Generate a 16-d binary mask with cost-aware greedy.
    Reuse a single frozen student_once for Taylor.
    """
    encoder.eval(); summarizer.eval(); token_proj.eval(); teacher.eval()

    student_once = build_resnet18_teacher(num_classes=10).to(device)
    student_once.load_state_dict(teacher.state_dict())
    for p in student_once.parameters(): p.requires_grad = False
    student_once.eval()

    all_scores = []
    all_flops = None

    iters = 20
    itr = 0
    for x, _ in train_loader:
        x = x.to(device)
        with torch.no_grad():
            y_T = teacher(x)
        tokens, flops = build_tokens_with_taylor16(teacher, student_once, summarizer, token_proj, x, y_T, device)
        logits = encoder(tokens, target_ratio).squeeze(0)   # [16]
        scores = torch.sigmoid(logits)
        all_scores.append(scores)
        if all_flops is None:
            all_flops = flops
        itr += 1
        if itr >= iters: break

    avg_scores = torch.stack(all_scores, dim=0).mean(dim=0) # [16]
    flops = all_flops

    eff = (avg_scores.detach() / (flops + 1e-9)).cpu().numpy().tolist()
    idx_sorted = sorted(range(NUM_UNITS), key=lambda i: eff[i], reverse=True)

    mask = torch.zeros(NUM_UNITS, device=device)
    full_flops = flops.sum()
    acc_flops = 0.0

    for i in idx_sorted:
        if (acc_flops + flops[i]) / full_flops <= target_ratio or mask.sum().item() == 0:
            mask[i] = 1.0
            acc_flops += flops[i].item()
        current_ratio = acc_flops / full_flops
        if current_ratio >= target_ratio * 0.98:
            break

    if mask.sum().item() < 1:
        mask[idx_sorted[0]] = 1.0
        acc_flops = flops[idx_sorted[0]].item()

    actual_ratio = (mask * flops).sum() / full_flops
    return mask, actual_ratio.item(), avg_scores.tolist()

def resnet18_masked_forward_with_binary_mask16(student, x, mask):
    # convenience wrapper that expects 0/1 mask
    return resnet18_masked_forward_with_gates16(student, x, mask)

def finetune_pruned_model(teacher, mask, actual_ratio, train_loader, test_loader):
    student = build_resnet18_teacher(num_classes=10).to(device)
    student.load_state_dict(teacher.state_dict())
    for p in student.parameters(): p.requires_grad = True

    opt = torch.optim.AdamW(student.parameters(), lr=FT_LR)

    print(f"\nFine-tuning pruned model (ratio={actual_ratio:.3f}, kept={int(mask.sum().item())}/16)...")
    best_acc = 0.0
    for epoch in range(FT_EPOCHS):
        student.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_T = teacher(x)
            y_S = resnet18_masked_forward_with_binary_mask16(student, x, mask)
            loss = kd_loss(y_S, y_T, T=TEMP_KD)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        acc = evaluate_accuracy(student, test_loader, mask)
        best_acc = max(best_acc, acc)
        print(f"  [FT {epoch+1}/{FT_EPOCHS}] acc={acc*100:.2f}%")
    return student, best_acc

# =========================
# MAIN
# =========================
def main():
    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)

    train_loader, test_loader = get_loaders()

    teacher = build_resnet18_teacher(num_classes=10)
    ckpt = torch.load(CKPT, map_location="cpu")
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.to(device).eval()

    teacher_acc = evaluate_accuracy(teacher, test_loader)
    print(f"\nTeacher accuracy (unpruned): {teacher_acc*100:.2f}%")

    if not os.path.exists(os.path.join(OUT_DIR, "compression_aware_encoder_16gate.pth")):
        encoder, summarizer, token_proj = train_policy_encoder(teacher, train_loader, test_loader)

        torch.save({
            "encoder": encoder.state_dict(),
            "summarizer": summarizer.state_dict(),
            "token_proj": token_proj.state_dict(),
        }, os.path.join(OUT_DIR, "compression_aware_encoder_16gate.pth"))
        print(f"\nSaved policy to {os.path.join(OUT_DIR, 'compression_aware_encoder_16gate.pth')}")

        print("\n" + "="*80)
    else:
        print("\nLoading pre-trained policy encoder...")
        ckpt = torch.load(os.path.join(OUT_DIR, "compression_aware_encoder_16gate.pth"), map_location="cpu")
        encoder = CompressionAwareEncoder(dim=ENC_WIDTH, depth=ENC_LAYERS, heads=ENC_HEADS, num_blocks=NUM_UNITS).to(device)
        encoder.load_state_dict(ckpt["encoder"])
        summarizer = Summarizer(sum_dim=SUM_DIM, k=64).to(device)
        summarizer.load_state_dict(ckpt["summarizer"])
        token_proj = TokenProj(SUM_DIM + 7, TOKEN_DIM).to(device)
        token_proj.load_state_dict(ckpt["token_proj"])
        print("Policy encoder loaded.\n")
    
    
    print("PHASE 2: Materializing Masks and Fine-tuning Pruned Models (16-gate)")
    print("="*80)

    results = {"teacher_accuracy": teacher_acc * 100, "eval_ratios": []}

    for target_ratio in EVAL_RATIOS:
        print(f"\n{'='*80}\nTarget Ratio: {target_ratio:.1f}\n{'='*80}")

        mask, actual_ratio, scores = materialize_mask_for_ratio(
            encoder, summarizer, token_proj, teacher, train_loader, target_ratio
        )
        print(f"Scores(sigmoid, first 8): {[f'{v:.2f}' for v in scores[:8]]} ...")
        print(f"Mask keep count: {int(mask.sum().item())}/16")
        print(f"Actual ratio: {actual_ratio:.3f} (target: {target_ratio:.1f})")

        student_before = build_resnet18_teacher(num_classes=10).to(device)
        student_before.load_state_dict(teacher.state_dict())
        acc_before = evaluate_accuracy(student_before, test_loader, mask)
        print(f"\nAccuracy before fine-tuning: {acc_before*100:.2f}%")

        student_after, best_acc = finetune_pruned_model(
            teacher, mask, actual_ratio, train_loader, test_loader
        )
        final_acc = evaluate_accuracy(student_after, test_loader, mask)
        print(f"\nFinal accuracy after fine-tuning: {final_acc*100:.2f}%")

        results["eval_ratios"].append({
            "target_ratio": target_ratio,
            "actual_ratio": actual_ratio,
            "mask": mask.int().tolist(),
            "kept": int(mask.sum().item()),
            "scores": [float(s) for s in scores],
            "accuracy_before_ft": acc_before * 100,
            "accuracy_after_ft": final_acc * 100,
            "teacher_accuracy": teacher_acc * 100,
            "accuracy_drop": (teacher_acc - final_acc) * 100
        })

        print(f"\n{'─'*80}")
        print(f"SUMMARY ratio {target_ratio:.1f}: "
              f"Teacher={teacher_acc*100:.2f}%  BeforeFT={acc_before*100:.2f}%  AfterFT={final_acc*100:.2f}%  "
              f"Actual={actual_ratio:.3f}  Kept={int(mask.sum().item())}/16")
        print(f"{'─'*80}")

    results_path = os.path.join(OUT_DIR, "..", "compression_analysis", RESULTS_FILE)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*80}\nAll results saved to: {results_path}\n{'='*80}")

    print("\n" + "="*80)
    print("FINAL COMPARISON TABLE (16-gate)")
    print("="*80)
    print(f"{'Ratio':<8} {'Actual':<8} {'Kept':<6} {'Before FT':<12} {'After FT':<12} {'Drop':<10}")
    print("-"*80)
    for r in results["eval_ratios"]:
        print(f"{r['target_ratio']:<8.1f} {r['actual_ratio']:<8.3f} {r['kept']:<6d} "
              f"{r['accuracy_before_ft']:<12.2f} {r['accuracy_after_ft']:<12.2f} {r['accuracy_drop']:<10.2f}")
    print("="*80)
    print(f"Teacher (unpruned): {results['teacher_accuracy']:.2f}%")
    print("="*80)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

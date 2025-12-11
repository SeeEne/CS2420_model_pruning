import os, math, random, gc, json, time, argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler
import numpy as np

OUT_DIR = "checkpoints"
RESULTS_FILE = "aznas_component_ablation_results.json"

CIFAR100_ROOT = "data/cifar100"
CIFAR100_NUM_CLASSES = 100
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

BATCH_SIZE = 256
NUM_WORKERS = 4
IMG_SIZE = 128
SEED = 42

TOKEN_DIM = 128
SUM_DIM = 64
ENC_WIDTH = 128
ENC_LAYERS = 2
ENC_HEADS = 4
NUM_BLOCKS = 8

POLICY_WARMUP_EPOCHS = 2
POLICY_TRAIN_EPOCHS = 3
POLICY_LR = 1e-4
WEIGHT_DECAY = 0.0

MIN_RATIO = 0.1
MAX_RATIO = 0.9
RATIO_WEIGHT = 25.0
GATE_TEMP_START = 5.0
GATE_TEMP_END = 0.3
L1_M_WEIGHT = 1e-3

KD_WARMUP_EPOCHS = 1
AZNAS_WEIGHT = 0.5

FT_EPOCHS = 5
FT_LR = 1e-3
TEMP_KD = 2.0

TEST_RATIOS = [0.5]

MATERIALIZE_ITERS = 3

QUICK_TEST = False
QUICK_TEST_ITERS = 10
FAST_FULL = False
MAX_BATCHES_PER_EPOCH = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ABLATION_CONFIGS = {
    'expr_only': {
        'name': 'Expressivity Only',
        'expr_weight': 1.0,
        'prog_weight': 0.0,
        'complex_weight': 0.0,
    },
    'prog_only': {
        'name': 'Progressivity Only',
        'expr_weight': 0.0,
        'prog_weight': 1.0,
        'complex_weight': 0.0,
    },
    'complex_only': {
        'name': 'Complexity Only',
        'expr_weight': 0.0,
        'prog_weight': 0.0,
        'complex_weight': 1.0,
    },
    'expr_prog': {
        'name': 'Expr + Prog',
        'expr_weight': 1.0,
        'prog_weight': 0.1,
        'complex_weight': 0.0,
    },
    'full': {
        'name': 'Full (E+P+C)',
        'expr_weight': 1.0,
        'prog_weight': 0.1,
        'complex_weight': 0.01,
    },
}

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

def get_cifar100_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_ds = datasets.CIFAR100(root=CIFAR100_ROOT, train=True, download=True, transform=train_tf)
    val_ds = datasets.CIFAR100(root=CIFAR100_ROOT, train=False, download=True, transform=test_tf)
    print(f"CIFAR-100: {len(train_ds)} train, {len(val_ds)} val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader

def build_resnet18(num_classes=CIFAR100_NUM_CLASSES):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class Summarizer(nn.Module):
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
        gap_h = h_in.mean(dim=(2, 3))
        gmp_h, _ = h_in.flatten(2).max(dim=2)
        gap_r = r_out.mean(dim=(2, 3))
        gmp_r, _ = r_out.flatten(2).max(dim=2)
        parts = [self._pool_vec(p) for p in [gap_h, gmp_h, gap_r, gmp_r]]
        feats = torch.cat(parts, dim=1)
        z = self.mlp(self.proj(feats))
        return z.mean(dim=0)

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
        block_h = h[:, 1:, :]
        logits = self.head(block_h).squeeze(-1)
        return logits

def conv_flops(H, W, Cin, Cout, k, stride):
    return (H // stride) * (W // stride) * Cin * Cout * (k * k)

def get_block_flops(block, input_shape):
    B, C, H, W = input_shape
    f1 = conv_flops(H, W, block.conv1.in_channels, block.conv1.out_channels, 3, block.conv1.stride[0])
    H2, W2 = H // block.conv1.stride[0], W // block.conv1.stride[0]
    f2 = conv_flops(H2, W2, block.conv2.in_channels, block.conv2.out_channels, 3, block.conv2.stride[0])
    return float(f1 + f2)

def forward_resnet_collect_blocks(model, x, collect_grads=False):
    infos = []

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

            out = block.conv1(h_in)
            out = block.bn1(out)
            out = block.relu(out)
            if collect_grads:
                out.retain_grad()

            out = block.conv2(out)
            r_out = block.bn2(out)
            if collect_grads:
                r_out.retain_grad()

            if block.downsample is not None:
                skip = block.downsample(h_in)
            else:
                skip = h_in

            infos.append({
                "h_in": h_in,
                "r_out": r_out,
                "block_obj": block,
                "stage": s_idx,
                "block_idx": b_idx,
                "H": h_in.size(2),
                "W": h_in.size(3),
                "flops": get_block_flops(block, h_in.shape)
            })

            h = F.relu(skip + r_out)

    h = model.avgpool(h)
    h = torch.flatten(h, 1)
    logits = model.fc(h)

    return logits, infos

def forward_resnet_gated_blocks(model, x, block_gates):
    gate_idx = 0

    h = model.conv1(x)
    h = model.bn1(h)
    h = model.relu(h)
    h = model.maxpool(h)

    for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in stage:
            r = block.conv1(h)
            r = block.bn1(r)
            r = block.relu(r)
            r = block.conv2(r)
            r = block.bn2(r)

            g = block_gates[gate_idx].view(1, 1, 1, 1)
            r = r * g
            gate_idx += 1

            if block.downsample is not None:
                skip = block.downsample(h)
            else:
                skip = h

            h = F.relu(skip + r)

    h = model.avgpool(h)
    h = torch.flatten(h, 1)
    return model.fc(h)

def compute_aznas_scores_gated_blocks(model, x, gates, ablation_config):
    expressivity_scores = []
    block_outputs = []
    flops_list = []
    g_idx = 0

    with torch.no_grad():
        h = model.conv1(x)
        h = model.bn1(h)
        h = model.relu(h)
        h = model.maxpool(h)

    h = h.detach().requires_grad_(True)
    block_outputs.append(h)

    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            h_in = h

            r = block.conv1(h)
            r = block.bn1(r)
            r = block.relu(r)
            r = block.conv2(r)
            r = block.bn2(r)

            g = gates[g_idx].view(1, 1, 1, 1)
            r_gated = r * g
            g_idx += 1

            skip = block.downsample(h) if block.downsample is not None else h
            h = F.relu(skip + r_gated)

            block_outputs.append(h)

            if ablation_config['expr_weight'] > 0:
                b, c, fh, fw = h.shape
                X = h.permute(0, 2, 3, 1).reshape(-1, c)

                mu = X.mean(dim=0, keepdim=True)
                Xc = X - mu
                n = Xc.shape[0]
                sigma = (Xc.T @ Xc) / max(1, n)

                sigma_norm = torch.norm(sigma, p='fro')
                epsilon = 1e-5 * sigma_norm.detach() + 1e-6
                jitter = epsilon * torch.eye(c, device=x.device)
                sigma = sigma + jitter

                try:
                    s = torch.linalg.eigvalsh(sigma)
                except RuntimeError:
                    s = torch.ones(c, device=x.device)

                s = torch.relu(s) + 1e-12
                p = s / s.sum()
                entropy = -torch.sum(p * torch.log(p))
                expressivity_scores.append(entropy)

            block_flops = get_block_flops(block, h_in.shape)
            flops_list.append(block_flops)

    expressivity = torch.tensor(0.0, device=x.device)
    progressivity = torch.tensor(0.0, device=x.device)

    if ablation_config['expr_weight'] > 0 and len(expressivity_scores) > 0:
        scores_stack = torch.stack(expressivity_scores)

        normalized_scores = []
        for stage_idx in range(4):
            start_idx = stage_idx * 2
            end_idx = start_idx + 2

            if end_idx <= len(scores_stack):
                stage_scores = scores_stack[start_idx:end_idx]
                stage_mean = stage_scores.mean()
                stage_std = stage_scores.std()

                if stage_std > 1e-6:
                    stage_normalized = (stage_scores - stage_mean) / stage_std
                else:
                    stage_normalized = stage_scores - stage_mean

                normalized_scores.append(stage_normalized)

        if len(normalized_scores) > 0:
            normalized_stack = torch.cat(normalized_scores)
            expressivity = torch.mean(normalized_stack)
        else:
            expressivity = torch.mean(scores_stack)

        if ablation_config['prog_weight'] > 0 and len(scores_stack) >= 2:
            if len(normalized_scores) > 0:
                normalized_stack = torch.cat(normalized_scores)
                diffs = normalized_stack[1:] - normalized_stack[:-1]
            else:
                diffs = scores_stack[1:] - scores_stack[:-1]
            progressivity = torch.min(diffs)

    complexity = torch.tensor(0.0, device=x.device)
    if ablation_config['complex_weight'] > 0:
        total_flops = sum(flops_list)
        flops_tensor = torch.tensor(flops_list, device=gates.device, dtype=gates.dtype)
        effective_flops = torch.sum(gates * flops_tensor)
        total_flops_tensor = torch.tensor(total_flops, device=gates.device, dtype=gates.dtype)
        complexity = effective_flops / (total_flops_tensor + 1e-9)

    trainability_scores = []

    for i in reversed(range(1, len(block_outputs))):
        f_out = block_outputs[i]
        f_in = block_outputs[i-1]

        dummy_out = f_out.sum()

        try:
            g_out = torch.autograd.grad(
                outputs=dummy_out,
                inputs=f_in,
                retain_graph=True,
                create_graph=False
            )[0]
        except RuntimeError:
            trainability_scores.append(-1e6)
            continue

        g_in = f_in
        if g_out.size() == g_in.size() and torch.all(g_in == g_out):
            trainability_scores.append(-1e6)
            continue

        if g_out.size(2) != g_in.size(2) or g_out.size(3) != g_in.size(3):
            bo, co, ho, wo = g_out.size()
            bi, ci, hi, wi = g_in.size()

            if ho < hi:
                g_in = F.adaptive_avg_pool2d(g_in, (ho, wo))
            elif ho > hi:
                g_out = F.adaptive_avg_pool2d(g_out, (hi, wi))

        bo, co, ho, wo = g_out.size()
        bi, ci, hi, wi = g_in.size()

        g_out_flat = g_out.permute(0, 2, 3, 1).reshape(-1, co)
        g_in_flat = g_in.permute(0, 2, 3, 1).reshape(-1, ci)

        jac_mat = (g_in_flat.T @ g_out_flat) / max(1, g_in_flat.size(0))

        if jac_mat.size(0) < jac_mat.size(1):
            jac_mat = jac_mat.transpose(0, 1)

        try:
            s = torch.linalg.svdvals(jac_mat)
            score = (-s.max() - 1.0/(s.max() + 1e-6) + 2.0).item()
            trainability_scores.append(score)
        except RuntimeError:
            trainability_scores.append(-1e6)

    if len(trainability_scores) > 0:
        trainability = torch.tensor(np.mean(trainability_scores), device=x.device)
    else:
        trainability = torch.tensor(0.0, device=x.device)

    return {
        "expressivity": expressivity,
        "progressivity": progressivity,
        "trainability": trainability,
        "complexity": complexity,
    }

def kd_loss(student_logits, teacher_logits, T=TEMP_KD):
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction='batchmean') * (T * T)

def build_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device):
    x = x.requires_grad_(True)

    with torch.no_grad():
        _, teacher_infos = forward_resnet_collect_blocks(teacher, x, collect_grads=False)

    logits_S, student_infos = forward_resnet_collect_blocks(student, x, collect_grads=True)
    loss = F.kl_div(F.log_softmax(logits_S / TEMP_KD, dim=1),
                    F.softmax(y_T / TEMP_KD, dim=1), reduction='batchmean') * (TEMP_KD ** 2)
    loss.backward()

    token_list = []
    flops_list = []

    for i in range(NUM_BLOCKS):
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
            t_info["block_idx"] / 1.0,
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

def train_aznas_encoder_ablation(teacher, train_loader, val_loader, ablation_config, config_name):
    print(f"\n{'='*70}")
    print(f"Training: {ablation_config['name']}")
    print(f"Config: expr={ablation_config['expr_weight']}, prog={ablation_config['prog_weight']}, complex={ablation_config['complex_weight']}")
    print(f"{'='*70}")

    student = build_resnet18(CIFAR100_NUM_CLASSES).to(device)
    student.load_state_dict(teacher.state_dict())
    for p in student.parameters():
        p.requires_grad = False
    student.eval()

    summarizer = Summarizer(sum_dim=SUM_DIM).to(device)
    token_proj = TokenProj(SUM_DIM + 5, TOKEN_DIM).to(device)
    encoder = CompressionAwareEncoder(dim=ENC_WIDTH, depth=ENC_LAYERS, heads=ENC_HEADS, num_blocks=NUM_BLOCKS).to(device)

    params = list(summarizer.parameters()) + list(token_proj.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=POLICY_LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()

    total_epochs = POLICY_WARMUP_EPOCHS + POLICY_TRAIN_EPOCHS

    for epoch in range(total_epochs):
        encoder.train()
        summarizer.train()
        token_proj.train()

        is_warmup = epoch < KD_WARMUP_EPOCHS
        phase_name = "KD Warmup" if is_warmup else "Co-Training"

        lr = POLICY_LR * 0.5 * (1 + math.cos(math.pi * epoch / max(1, total_epochs - 1)))
        for g in opt.param_groups:
            g['lr'] = lr
        t = epoch / max(1, total_epochs - 1)
        gate_temp = GATE_TEMP_START * (1 - t) + GATE_TEMP_END * t

        run_kd = run_aznas = run_ratio = 0.0
        run_expr = run_prog = run_complex = 0.0

        for i, (x, y) in enumerate(train_loader):
            if MAX_BATCHES_PER_EPOCH is not None and i >= MAX_BATCHES_PER_EPOCH:
                break

            x = x.to(device)
            with torch.no_grad():
                y_T = teacher(x)

            tokens, flops = build_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device)
            target_ratio = random.uniform(MIN_RATIO, MAX_RATIO)

            with autocast():
                logits = encoder(tokens, target_ratio).squeeze(0)
                gates = torch.sigmoid(logits / gate_temp)

                y_S = forward_resnet_gated_blocks(student, x, gates)
                loss_kd = kd_loss(y_S, y_T, T=TEMP_KD)

                exp_flops = (gates * flops).sum()
                flops_ratio = exp_flops / (flops.sum() + 1e-6)
                loss_ratio = (flops_ratio - target_ratio) ** 2
                loss_l1 = gates.mean()

                if not is_warmup:
                    aznas_info = compute_aznas_scores_gated_blocks(student, x, gates, ablation_config)
                    expr = aznas_info['expressivity']
                    prog = aznas_info['progressivity']
                    train_score = aznas_info['trainability']
                    complexity_score = aznas_info['complexity']

                    loss_expr = torch.exp(-expr / 5.0) if ablation_config['expr_weight'] > 0 else torch.tensor(0.0, device=device)
                    loss_prog = torch.exp(-prog / 1.0) if ablation_config['prog_weight'] > 0 else torch.tensor(0.0, device=device)
                    loss_complexity = complexity_score if ablation_config['complex_weight'] > 0 else torch.tensor(0.0, device=device)

                    loss_aznas = (ablation_config['expr_weight'] * loss_expr +
                                  ablation_config['prog_weight'] * loss_prog +
                                  ablation_config['complex_weight'] * loss_complexity)
                else:
                    loss_aznas = torch.tensor(0.0, device=device)
                    expr = prog = complexity_score = torch.tensor(0.0, device=device)

                loss = loss_kd + RATIO_WEIGHT * loss_ratio + L1_M_WEIGHT * loss_l1
                if not is_warmup:
                    loss += AZNAS_WEIGHT * loss_aznas

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            run_kd += loss_kd.item()
            if not is_warmup:
                run_aznas += loss_aznas.item()
                run_expr += expr.item()
                run_prog += prog.item()
                run_complex += complexity_score.item()
            run_ratio += loss_ratio.item()

            if (i + 1) % 50 == 0:
                if not is_warmup:
                    print(
                        f"[Ep {epoch + 1}][{i + 1}][{phase_name}] "
                        f"KD={run_kd/50:.4f} AZ={run_aznas/50:.4f} "
                        f"Expr={run_expr/50:.2f} Prog={run_prog/50:.2f} "
                        f"Compl={run_complex/50:.3f} RatioL={run_ratio/50:.4f}"
                    )
                else:
                    print(
                        f"[Ep {epoch + 1}][{i + 1}][{phase_name}] "
                        f"KD={run_kd/50:.4f} RatioL={run_ratio/50:.4f}"
                    )
                run_kd = run_aznas = run_ratio = 0.0
                run_expr = run_prog = run_complex = 0.0

    print(f"Encoder training complete for {config_name}")
    return encoder, summarizer, token_proj

def materialize_mask_ablation(encoder, summarizer, token_proj, teacher, train_loader, target_ratio, ablation_config):
    encoder.eval()
    summarizer.eval()
    token_proj.eval()
    teacher.eval()

    student_once = build_resnet18(CIFAR100_NUM_CLASSES).to(device)
    student_once.load_state_dict(teacher.state_dict())
    for p in student_once.parameters():
        p.requires_grad = False
    student_once.eval()

    all_scores = []
    all_flops = None
    itr = 0
    sample_batch = None

    for x, _ in train_loader:
        if MAX_BATCHES_PER_EPOCH is not None and itr >= min(MATERIALIZE_ITERS, MAX_BATCHES_PER_EPOCH):
            break

        x = x.to(device)
        with torch.no_grad():
            y_T = teacher(x)
        tokens, flops = build_block_tokens(teacher, student_once, summarizer, token_proj, x, y_T, device)
        logits = encoder(tokens, target_ratio).squeeze(0)
        scores = torch.sigmoid(logits)
        all_scores.append(scores)
        if all_flops is None:
            all_flops = flops
        if sample_batch is None:
            sample_batch = x[:2]
        itr += 1
        if itr >= MATERIALIZE_ITERS:
            break

    avg_scores = torch.stack(all_scores, dim=0).mean(dim=0)
    flops = all_flops

    print(f"  Computing AZ-NAS ablation importance...")

    with torch.no_grad():
        baseline_gates = torch.ones(NUM_BLOCKS, device=device)
        baseline_aznas = compute_aznas_scores_gated_blocks(student_once, sample_batch, baseline_gates, ablation_config)

    block_importance = []
    for i in range(NUM_BLOCKS):
        with torch.no_grad():
            probe_mask = torch.ones(NUM_BLOCKS, device=device)
            probe_mask[i] = 0.0

            ablated_aznas = compute_aznas_scores_gated_blocks(student_once, sample_batch, probe_mask, ablation_config)

            expr_drop = (baseline_aznas['expressivity'] - ablated_aznas['expressivity']).item()
            prog_drop = (baseline_aznas['progressivity'] - ablated_aznas['progressivity']).item()
            complexity_penalty = (flops[i] / flops.sum()).item()

            importance = (
                ablation_config['expr_weight'] * expr_drop +
                ablation_config['prog_weight'] * prog_drop -
                ablation_config['complex_weight'] * complexity_penalty
            )

            block_importance.append(importance)

    print(f"  Block importances: {[f'{imp:.3f}' for imp in block_importance]}")

    idx_sorted = sorted(range(NUM_BLOCKS), key=lambda i: block_importance[i], reverse=True)

    mask = torch.zeros(NUM_BLOCKS, device=device)
    full_flops = flops.sum()
    acc_flops = 0.0

    for i in idx_sorted:
        if (acc_flops + flops[i]) / full_flops <= target_ratio or mask.sum().item() == 0:
            mask[i] = 1.0
            acc_flops += flops[i].item()
        if acc_flops / full_flops >= target_ratio * 0.98:
            break

    if mask.sum().item() < 1:
        mask[idx_sorted[0]] = 1.0
        acc_flops = flops[idx_sorted[0]].item()

    actual_ratio = (mask * flops).sum() / full_flops

    print(f"  Selected blocks: {[i for i in range(NUM_BLOCKS) if mask[i] > 0.5]}")
    print(f"  Target ratio: {target_ratio:.2f}, Actual ratio: {actual_ratio:.2f}")

    return mask, actual_ratio.item(), avg_scores.tolist()

def finetune_pruned_model(teacher, mask, train_loader, val_loader):
    student = build_resnet18(CIFAR100_NUM_CLASSES).to(device)
    student.load_state_dict(teacher.state_dict())
    for p in student.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(student.parameters(), lr=FT_LR)
    scaler = GradScaler()

    print(f"  Fine-tuning (kept={int(mask.sum().item())}/{NUM_BLOCKS} blocks)...")
    best_acc = 0.0

    for epoch in range(FT_EPOCHS):
        student.train()
        for i, (x, y) in enumerate(train_loader):
            if MAX_BATCHES_PER_EPOCH is not None and i >= MAX_BATCHES_PER_EPOCH:
                break

            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_T = teacher(x)

            with autocast():
                y_S = forward_resnet_gated_blocks(student, x, mask)
                loss = kd_loss(y_S, y_T, T=TEMP_KD)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        acc = evaluate_accuracy(student, val_loader, mask)
        best_acc = max(best_acc, acc)
        print(f"    [FT {epoch + 1}/{FT_EPOCHS}] acc={acc * 100:.2f}%")

    return student, best_acc

def main():
    global QUICK_TEST, POLICY_WARMUP_EPOCHS, POLICY_TRAIN_EPOCHS, FT_EPOCHS, FAST_FULL, MAX_BATCHES_PER_EPOCH

    parser = argparse.ArgumentParser(description='AZ-NAS Component Ablation Study')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test mode: 2 configs, 1 epoch each (~5-10 mins)')
    parser.add_argument('--fast_full', action='store_true',
                       help='Fast full mode: all configs, reduced epochs/batches (~20 mins)')
    args = parser.parse_args()

    QUICK_TEST = args.quick_test
    FAST_FULL = args.fast_full

    if QUICK_TEST:
        print("\n🚀 QUICK TEST MODE ENABLED 🚀")
        print("  - Testing only 2 configs (expr_only, full)")
        print("  - 1 epoch per stage")
        print("  - 10 batches per epoch")
        print("  - Estimated time: ~5-10 minutes\n")
        POLICY_WARMUP_EPOCHS = 1
        POLICY_TRAIN_EPOCHS = 1
        FT_EPOCHS = 1
        MAX_BATCHES_PER_EPOCH = QUICK_TEST_ITERS
    elif FAST_FULL:
        print("\n⚡ FAST FULL MODE ENABLED ⚡")
        print("  - Testing all ablation configs")
        print("  - 1 warmup + 1 train epoch (2 total)")
        print("  - 1 FT epoch")
        print("  - ~40 batches per epoch\n")
        POLICY_WARMUP_EPOCHS = 1
        POLICY_TRAIN_EPOCHS = 1
        FT_EPOCHS = 1
        MAX_BATCHES_PER_EPOCH = 40

    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("="*70)
    print("AZ-NAS Component Ablation Study (Part 1) - CIFAR-100")
    print("="*70)

    start_time = time.time()

    train_loader, val_loader = get_cifar100_loaders()

    teacher_path = os.path.join(OUT_DIR, "cifar100_teacher.pth")
    if not os.path.exists(teacher_path):
        print(f"\nERROR: CIFAR-100 teacher not found at {teacher_path}")
        print("Please train and save the CIFAR-100 teacher checkpoint before running this script.")
        return

    print(f"\nLoading teacher from {teacher_path}")
    teacher = build_resnet18(CIFAR100_NUM_CLASSES).to(device)
    ckpt = torch.load(teacher_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        teacher.load_state_dict(ckpt["state_dict"])
        teacher_acc = ckpt.get("val_acc", None)
    else:
        teacher.load_state_dict(ckpt)
        teacher_acc = ckpt.get("val_acc", None) if isinstance(ckpt, dict) else None

    teacher.to(device)
    teacher.eval()

    if teacher_acc is None:
        teacher_acc = evaluate_accuracy(teacher, val_loader)
        print(f"Teacher Accuracy (computed): {teacher_acc * 100:.2f}%")
    else:
        print(f"Teacher Accuracy (stored): {teacher_acc * 100:.2f}%")

    for p in teacher.parameters():
        p.requires_grad = False

    all_results = {
        'teacher_acc': teacher_acc,
        'test_ratios': TEST_RATIOS,
        'configs': {}
    }

    configs_to_test = ABLATION_CONFIGS
    if QUICK_TEST:
        configs_to_test = {
            'expr_only': ABLATION_CONFIGS['expr_only'],
            'full': ABLATION_CONFIGS['full']
        }

    for config_key, config_spec in configs_to_test.items():
        print(f"\n{'='*70}")
        print(f"Running Configuration: {config_spec['name']}")
        print(f"{'='*70}")

        config_results = {
            'name': config_spec['name'],
            'weights': {
                'expr': config_spec['expr_weight'],
                'prog': config_spec['prog_weight'],
                'complex': config_spec['complex_weight'],
            },
            'ratios': {}
        }

        encoder, summarizer, token_proj = train_aznas_encoder_ablation(
            teacher, train_loader, val_loader, config_spec, config_key
        )

        for ratio in TEST_RATIOS:
            print(f"\n--- Testing ratio={ratio} ---")

            mask, actual_ratio, scores = materialize_mask_ablation(
                encoder, summarizer, token_proj, teacher, train_loader, ratio, config_spec
            )

            temp_student = build_resnet18(CIFAR100_NUM_CLASSES).to(device)
            temp_student.load_state_dict(teacher.state_dict())
            temp_student.eval()
            acc_before = evaluate_accuracy(temp_student, val_loader, mask)
            print(f"  Before FT: {acc_before * 100:.2f}%")

            _, best_acc = finetune_pruned_model(teacher, mask, train_loader, val_loader)

            config_results['ratios'][str(ratio)] = {
                'target_ratio': ratio,
                'actual_ratio': actual_ratio,
                'mask': mask.int().tolist(),
                'kept_blocks': int(mask.sum().item()),
                'accuracy_before_ft': acc_before,
                'accuracy_after_ft': best_acc,
                'accuracy_drop': teacher_acc - best_acc
            }

            print(f"  After FT: {best_acc * 100:.2f}% (drop: {(teacher_acc - best_acc) * 100:.2f}%)")

        all_results['configs'][config_key] = config_results

        del encoder, summarizer, token_proj
        clear_vram()

    results_path = os.path.join(OUT_DIR, RESULTS_FILE)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY COMPLETE!")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"{'='*70}")

    print(f"\n{'='*100}")
    print(f"{'Config':<20} {'Ratio':<8} {'Kept':<8} {'Acc Before':<12} {'Acc After':<12} {'Drop':<10}")
    print(f"{'='*100}")

    for config_key, config_data in all_results['configs'].items():
        for ratio_key, ratio_data in config_data['ratios'].items():
            print(f"{config_data['name']:<20} {ratio_key:<8} "
                  f"{ratio_data['kept_blocks']}/{NUM_BLOCKS:<5} "
                  f"{ratio_data['accuracy_before_ft']*100:<12.2f} "
                  f"{ratio_data['accuracy_after_ft']*100:<12.2f} "
                  f"{ratio_data['accuracy_drop']*100:<10.2f}")

    print(f"{'='*100}")
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

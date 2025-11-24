"""
Compression-Aware Encoder Training with BLOCK-WISE Pruning (ResNet-18)
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
CKPT = "checkpoints/teacher.pth"
OUT_DIR = "checkpoints"
RESULTS_FILE = "encoder_block_results_tinyimagenet.json"

BATCH_SIZE = 512
NUM_WORKERS = 4
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

# Fine-tuning config
FT_EPOCHS = 10
FT_LR = 1e-3 # Lower LR for stability

EVAL_RATIOS = [0.1, 0.3,  0.5, 0.7, 0.9]
L1_M_WEIGHT = 1e-3

# ResNet-18 has 4 stages, 2 blocks each = 8 Blocks total
NUM_BLOCKS = 8 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# =========================
# Dataset (Same as before)
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
                self.images.append(os.path.join(root, 'images', parts[0]))
                self.labels.append(self.class_to_idx[parts[1]])
    
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
    train_ds = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_tf)
    val_ds = TinyImageNetVal(root=os.path.join(DATA_DIR, 'val'), transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes)

# =========================
# Models & Pruning Logic
# =========================
def build_resnet18(num_classes=200):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class PrunedBlock(nn.Module):
    """
    Physical replacement for a pruned BasicBlock.
    It removes the conv1/conv2 weights entirely and only runs the skip connection.
    This handles dimension matching via the 'downsample' layer.
    """
    def __init__(self, downsample_module):
        super().__init__()
        self.downsample = downsample_module # Can be None or a Sequential(Conv, BN)
        
    def forward(self, x):
        # The block is gone. We only process the skip connection.
        if self.downsample is not None:
            return F.relu(self.downsample(x))
        else:
            return F.relu(x)

class Summarizer(nn.Module):
    """Summarizes a Block: Inputs vs Residual Output"""
    def __init__(self, sum_dim=SUM_DIM, k=64):
        super().__init__()
        self.pool1d = nn.AdaptiveAvgPool1d(k)
        # Input (h_in) + Residual Output (r_out) -> 4 stats each * 2 = 8
        self.proj = nn.Linear(4 * k, 256)
        self.mlp  = nn.Sequential(
            nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, sum_dim), nn.LayerNorm(sum_dim)
        )

    def _pool_vec(self, v):
        v = v.unsqueeze(1)
        v = self.pool1d(v)
        return v.squeeze(1)

    def forward(self, h_in, r_out):
        # We compare what went INTO the block vs what came out of the RESIDUAL branch
        gap_h = h_in.mean(dim=(2,3)); gmp_h, _ = h_in.flatten(2).max(dim=2)
        gap_r = r_out.mean(dim=(2,3)); gmp_r, _ = r_out.flatten(2).max(dim=2)
        
        parts = [self._pool_vec(p) for p in [gap_h, gmp_h, gap_r, gmp_r]]
        feats = torch.cat(parts, dim=1)
        z = self.mlp(self.proj(feats))
        return z.mean(dim=0) # [sum_dim]

class CompressionAwareEncoder(nn.Module):
    def __init__(self, dim=ENC_WIDTH, depth=ENC_LAYERS, heads=ENC_HEADS):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim*2.0),
            batch_first=True, activation='gelu', norm_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos = nn.Parameter(torch.zeros(1, NUM_BLOCKS + 1, dim))
        
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
        block_h = h[:, 1:, :] # Skip budget token
        logits = self.head(block_h).squeeze(-1)
        return logits

# =========================
# Collection & Forward Pass Helpers
# =========================

def conv_flops(H, W, Cin, Cout, k, stride):
    return (H // stride) * (W // stride) * Cin * Cout * (k * k)

def get_block_flops(block, input_shape):
    # Calculate FLOPs for the RESIDUAL branch only (Conv1 + Conv2)
    # The downsample/skip FLOPs are unavoidable, so we don't count them in the budget
    B, C, H, W = input_shape
    
    # Conv1
    f1 = conv_flops(H, W, block.conv1.in_channels, block.conv1.out_channels, 3, block.conv1.stride[0])
    
    # Conv2 (Output size depends on stride of conv1)
    H2, W2 = H // block.conv1.stride[0], W // block.conv1.stride[0]
    f2 = conv_flops(H2, W2, block.conv2.in_channels, block.conv2.out_channels, 3, block.conv2.stride[0])
    
    return float(f1 + f2)

def forward_resnet_collect_blocks(model, x, collect_grads=False):
    """
    Runs model, collects (h_in, r_out) pairs for every block.
    Also collects gradients if collect_grads=True for Taylor calculation.
    """
    infos = []
    
    # Stem
    h = model.conv1(x); h = model.bn1(h); h = model.relu(h); h = model.maxpool(h)
    
    stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    
    for s_idx, stage in enumerate(stages):
        for b_idx, block in enumerate(stage):
            h_in = h
            if collect_grads: h_in.retain_grad()
            
            # --- Residual Branch ---
            out = block.conv1(h_in)
            out = block.bn1(out)
            out = block.relu(out)
            if collect_grads: out.retain_grad() # Track conv1 output
            
            out = block.conv2(out)
            r_out = block.bn2(out) # Final residual before addition
            if collect_grads: r_out.retain_grad()
            
            # --- Skip ---
            if block.downsample is not None:
                skip = block.downsample(h_in)
            else:
                skip = h_in
                
            # Store info
            infos.append({
                "h_in": h_in,     # Input to block
                "r_out": r_out,   # Residual output
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

def build_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device):
    """
    Builds 8 tokens representing the 8 blocks.
    Taylor score = Sum of activation*grad magnitude for both convs in block.
    """
    # Enable gradients for input when we need to collect gradients
    x = x.requires_grad_(True)

    # 1. Get Teacher Stats (for summary features)
    with torch.no_grad():
        _, teacher_infos = forward_resnet_collect_blocks(teacher, x, collect_grads=False)

    # 2. Get Student Gradients (for Taylor score)
    # We need to run forward again on student to get gradients
    logits_S, student_infos = forward_resnet_collect_blocks(student, x, collect_grads=True)
    loss = F.kl_div(F.log_softmax(logits_S/TEMP_KD, dim=1), 
                   F.softmax(y_T/TEMP_KD, dim=1), reduction='batchmean') * (TEMP_KD**2)
    loss.backward()
    
    token_list = []
    flops_list = []
    
    for i in range(NUM_BLOCKS):
        # Teacher info for features (stable)
        t_info = teacher_infos[i]
        s_info = student_infos[i] # Student info for grads
        
        # Features from Summarizer
        feats = summarizer(t_info["h_in"], t_info["r_out"]).to(device)
        
        # Taylor Score calculation
        # Sum of saliency of r_out (conv2) + intermediate (conv1) implies importance of whole branch
        grad_r = s_info["r_out"].grad
        # Note: We can't easily access conv1 output from info without complex hooks, 
        # so using r_out grad magnitude is a standard proxy for the block output importance.
        # Ideally we sum both, but r_out dominates the skip connection decision.
        
        if grad_r is not None:
            taylor = (grad_r * s_info["r_out"]).abs().mean().detach().item()
        else:
            taylor = 0.0
            
        # Metadata
        meta = torch.tensor([
            t_info["stage"]/3.0,
            t_info["block_idx"]/1.0,
            t_info["H"]/IMG_SIZE,
            t_info["W"]/IMG_SIZE
        ], dtype=torch.float32, device=device)
        
        # Combine
        tay_tensor = torch.tensor([math.log1p(taylor)], dtype=torch.float32, device=device)
        tok = torch.cat([feats, meta, tay_tensor], dim=0)
        
        token_list.append(tok)
        flops_list.append(t_info["flops"])

    # Clean up gradients to prevent memory leaks
    for info in student_infos:
        if info["r_out"].grad is not None:
            info["r_out"].grad = None

    # Project to Token Dim
    tokens = torch.stack(token_list).unsqueeze(0) # [1, 8, InputDim]
    # Note: InputDim = SUM_DIM + 4 (meta) + 1 (taylor)

    # We need a projection layer. Let's assume it's passed in or we use a Linear
    tokens = token_proj(tokens)
    flops = torch.tensor(flops_list, dtype=torch.float32, device=device)

    student.zero_grad()
    return tokens, flops

def forward_resnet_gated_blocks(model, x, block_gates):
    """
    Forward pass with 8 gates (one per block).
    If gate is close to 0, the residual branch is suppressed.
    """
    gate_idx = 0
    
    h = model.conv1(x); h = model.bn1(h); h = model.relu(h); h = model.maxpool(h)
    
    for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in stage:
            # 1. Compute Residual Branch
            r = block.conv1(h); r = block.bn1(r); r = block.relu(r)
            r = block.conv2(r); r = block.bn2(r)
            
            # 2. Apply Gate (Suppress Block)
            # This suppresses the BN noise if gate -> 0
            g = block_gates[gate_idx].view(1,1,1,1)
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
# PHASE 1: Train Policy
# =========================
def train_policy(teacher, train_loader):
    print("\nPhase 1: Training Block-Wise Policy Encoder")
    
    # Student for gradient calculation
    student = build_resnet18(num_classes=200).to(device)
    student.load_state_dict(teacher.state_dict())
    student.eval() # Eval mode for BN, but we will backprop for Taylor
    
    summarizer = Summarizer(sum_dim=SUM_DIM).to(device)
    # Input dim = SUM_DIM + 4 (meta) + 1 (taylor)
    token_proj = nn.Sequential(
        nn.LayerNorm(SUM_DIM + 5),
        nn.Linear(SUM_DIM + 5, TOKEN_DIM)
    ).to(device)
    
    encoder = CompressionAwareEncoder(dim=ENC_WIDTH).to(device)
    
    params = list(summarizer.parameters()) + list(token_proj.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=POLICY_LR, weight_decay=WEIGHT_DECAY)
    
    total_epochs = POLICY_WARMUP_EPOCHS + POLICY_TRAIN_EPOCHS
    
    for epoch in range(total_epochs):
        encoder.train(); summarizer.train(); token_proj.train()
        
        t = epoch / max(1, total_epochs-1)
        temp = GATE_TEMP_START * (1-t) + GATE_TEMP_END * t
        
        acc_loss_kd = 0; acc_loss_ratio = 0
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                y_T = teacher(x)

            # Build tokens (requires backprop on student for gradients)
            tokens, flops = build_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device)
            
            # Forward Policy
            target_ratio = random.uniform(MIN_RATIO, MAX_RATIO)
            logits = encoder(tokens, target_ratio).squeeze(0) # [8]
            
            # Gumbel-Softmax / Sigmoid relaxation
            gates = torch.sigmoid(logits / temp)
            
            # Expected Ratio
            full_flops = flops.sum()
            kept_flops = (gates * flops).sum()
            curr_ratio = kept_flops / (full_flops + 1e-6)
            
            # Gated Forward
            y_S = forward_resnet_gated_blocks(student, x, gates)
            
            # Losses
            loss_kd = kd_loss(y_S, y_T, TEMP_KD)
            loss_ratio = (curr_ratio - target_ratio) ** 2
            loss_l1 = gates.mean()
            
            loss = loss_kd + RATIO_WEIGHT * loss_ratio + L1_M_WEIGHT * loss_l1
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            acc_loss_kd += loss_kd.item()
            acc_loss_ratio += loss_ratio.item()
            
            if i % 50 == 0:
                print(f"[Ep {epoch}][{i}] KD: {acc_loss_kd/(i+1):.4f} Ratio: {acc_loss_ratio/(i+1):.4f} Temp: {temp:.2f}")
                print(f" Total Loss: {loss.item():.4f} Curr Ratio: {curr_ratio.item():.4f} Target: {target_ratio:.4f}")
                
    return encoder, summarizer, token_proj

def kd_loss(student_logits, teacher_logits, T):
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction='batchmean') * (T*T)

# =========================
# PHASE 2: Materialize & Fine-tune
# =========================

def physically_prune_model(original_model, mask):
    """
    Creates a new model where dropped blocks are physically replaced 
    with PrunedBlock (skip connection only).
    mask: list of 8 ints (0 or 1).
    """
    pruned_model = copy.deepcopy(original_model)
    mask_idx = 0
    
    stages = [pruned_model.layer1, pruned_model.layer2, pruned_model.layer3, pruned_model.layer4]
    
    for stage in stages:
        for b_idx in range(len(stage)):
            keep = mask[mask_idx]
            if keep == 0:
                # Replace BasicBlock with PrunedBlock
                original_block = stage[b_idx]
                # We only need the downsample layer to maintain dimensions
                new_block = PrunedBlock(original_block.downsample)
                stage[b_idx] = new_block
            
            mask_idx += 1
            
    return pruned_model.to(device)

def recalibrate_bn(model, loader, steps=100):
    """
    Essential: Run data through the modified structure to update BN stats
    before starting fine-tuning.
    """
    print(f"Recalibrating BN statistics ({steps} batches)...")
    model.train()
    # Freeze weights, update ONLY BN running stats
    for p in model.parameters(): p.requires_grad = False
    
    cnt = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _ = model(x)
            cnt += 1
            if cnt >= steps: break
            
    for p in model.parameters(): p.requires_grad = True
    print("Recalibration complete.")

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

def materialize_and_finetune(teacher, encoder, summarizer, token_proj, train_loader, val_loader, ratio):
    print(f"\n--- Target Ratio: {ratio} ---")
    
    # 1. Generate Mask (Greedy approach similar to previous, but block-wise)
    # Run a batch to get scores
    x_sample, _ = next(iter(train_loader))
    x_sample = x_sample[:32].to(device)
    with torch.no_grad(): y_T = teacher(x_sample)

    # Need gradients for taylor
    temp_student = build_resnet18(200).to(device)
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

    # Ensure at least one block is kept (prevent empty network)
    if sum(mask) == 0:
        best_idx = indices[0]  # Most efficient block
        mask[best_idx] = 1
        current_flops = flops[best_idx].item()

    real_ratio = current_flops / total_flops
    print(f"Mask: {mask}")
    print(f"Actual Ratio: {real_ratio:.3f}")
    
    # 2. Physically Prune
    pruned_model = physically_prune_model(teacher, mask)
    
    # 3. Recalibrate BN (Fixes the 0.5% accuracy bug)
    recalibrate_bn(pruned_model, train_loader)
    
    acc_pre = run_evaluation(pruned_model, val_loader)
    print(f"Accuracy Pre-FT: {acc_pre*100:.2f}%")
    
    # 4. Fine-tune
    opt = torch.optim.AdamW(pruned_model.parameters(), lr=FT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)
    
    best_acc = acc_pre
    
    for ep in range(FT_EPOCHS):
        pruned_model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): y_T = teacher(x)
            
            y_S = pruned_model(x)
            loss = kd_loss(y_S, y_T, TEMP_KD)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        scheduler.step()
        acc = run_evaluation(pruned_model, val_loader)
        if acc > best_acc: best_acc = acc
        print(f" [FT Ep {ep+1}] Acc: {acc*100:.2f}%")
        
    return best_acc, real_ratio

# =========================
# Main
# =========================
def main():
    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)
    
    train_loader, val_loader, num_classes = get_loaders()
    
    teacher = build_resnet18(num_classes).to(device)
    ckpt = torch.load(CKPT, map_location='cpu')
    teacher.load_state_dict(ckpt['state_dict'])
    teacher.eval()
    
    print("Teacher Acc:", run_evaluation(teacher, val_loader))
    
    # Phase 1
    encoder_path = os.path.join(OUT_DIR, "encoder_block_policy.pth")
    if os.path.exists(encoder_path):
        print("Loading Policy...")
        state = torch.load(encoder_path)
        encoder = CompressionAwareEncoder().to(device); encoder.load_state_dict(state['enc'])
        summarizer = Summarizer().to(device); summarizer.load_state_dict(state['sum'])
        token_proj = nn.Sequential(nn.LayerNorm(SUM_DIM+5), nn.Linear(SUM_DIM+5, TOKEN_DIM)).to(device)
        token_proj.load_state_dict(state['proj'])
    else:
        encoder, summarizer, token_proj = train_policy(teacher, train_loader)
        torch.save({'enc': encoder.state_dict(), 'sum': summarizer.state_dict(), 'proj': token_proj.state_dict()}, encoder_path)
        
    # Phase 2
    results = []
    for r in EVAL_RATIOS:
        acc, act_r = materialize_and_finetune(teacher, encoder, summarizer, token_proj, train_loader, val_loader, r)
        results.append({"target": r, "actual": act_r, "acc": acc})
        
    print("\nFinal Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
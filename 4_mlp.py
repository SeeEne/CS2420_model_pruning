"""
Baseline: MLP Policy (Non-Transformer) for Block-Wise Pruning (ResNet-18)
This tests if simple Dense Layers can perform as well as Self-Attention.
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
RESULTS_FILE = "mlp_policy_results.json"

BATCH_SIZE = 512
NUM_WORKERS = 4
IMG_SIZE = 224
TOKEN_DIM = 128
SUM_DIM = 64
# MLP Config
MLP_HIDDEN_DIM = 512 

TEMP_KD = 2.0
POLICY_WARMUP_EPOCHS = 2
POLICY_TRAIN_EPOCHS = 10
POLICY_LR = 1e-3
WEIGHT_DECAY = 1e-4 # Slightly higher WD for MLP to prevent overfitting
SEED = 42

MIN_RATIO = 0.1
MAX_RATIO = 0.8
RATIO_WEIGHT = 25.0
GATE_TEMP_START = 5.0
GATE_TEMP_END = 0.3

FT_EPOCHS = 10
FT_LR = 1e-3
EVAL_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
L1_M_WEIGHT = 1e-3
NUM_BLOCKS = 8 

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
        if not os.path.exists(train_dir):
            self.class_to_idx = {}
        else:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(train_dir)))}
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
        dummy_ds = datasets.FakeData(size=1000, image_size=(3, IMG_SIZE, IMG_SIZE), num_classes=200, transform=transforms.ToTensor())
        return DataLoader(dummy_ds, 32), DataLoader(dummy_ds, 32), 200

    train_ds = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_tf)
    val_ds = TinyImageNetVal(root=os.path.join(DATA_DIR, 'val'), transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes)

# =========================
# MLP Policy Architecture
# =========================
class GlobalMLPPolicy(nn.Module):
    """
    Replaces the Transformer. 
    Flattens all block tokens into one vector, concatenates budget, 
    and predicts all gates at once.
    """
    def __init__(self, num_blocks=NUM_BLOCKS, token_dim=TOKEN_DIM, hidden_dim=MLP_HIDDEN_DIM):
        super().__init__()
        
        # Budget embedding: Map 1 scalar -> 64 dim
        self.budget_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU()
        )
        
        # Input: (8 blocks * 128 dim) + 64 budget dim
        input_dim = (num_blocks * token_dim) + 64
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_blocks) # Output 8 logits directly
        )

    def forward(self, tokens, target_ratio):
        # tokens shape: [Batch, 8, 128]
        B = tokens.size(0)
        
        # Flatten tokens: [Batch, 1024]
        flat_tokens = tokens.view(B, -1)
        
        if not torch.is_tensor(target_ratio):
            target_ratio = torch.tensor([[float(target_ratio)]], dtype=torch.float32, device=tokens.device)
        else:
            target_ratio = target_ratio.float().view(1, 1).to(tokens.device)
            
        # Embed budget: [Batch, 64]
        b_emb = self.budget_embed(target_ratio)
        if b_emb.size(0) != B:
            b_emb = b_emb.expand(B, -1)
            
        # Concatenate: [Batch, 1088]
        x = torch.cat([flat_tokens, b_emb], dim=1)
        
        # MLP
        logits = self.net(x) # [Batch, 8]
        return logits

# =========================
# Shared Components (Summarizer, ResNet, etc.)
# =========================
def build_resnet18(num_classes=200):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class PrunedBlock(nn.Module):
    def __init__(self, downsample_module):
        super().__init__()
        self.downsample = downsample_module 
    def forward(self, x):
        return F.relu(self.downsample(x)) if self.downsample else F.relu(x)

class Summarizer(nn.Module):
    def __init__(self, sum_dim=SUM_DIM, k=64):
        super().__init__()
        self.pool1d = nn.AdaptiveAvgPool1d(k)
        self.proj = nn.Linear(4 * k, 256)
        self.mlp  = nn.Sequential(nn.LayerNorm(256), nn.GELU(), nn.Linear(256, sum_dim), nn.LayerNorm(sum_dim))
    def _pool_vec(self, v): return self.pool1d(v.unsqueeze(1)).squeeze(1)
    def forward(self, h_in, r_out):
        gap_h = h_in.mean(dim=(2,3)); gmp_h, _ = h_in.flatten(2).max(dim=2)
        gap_r = r_out.mean(dim=(2,3)); gmp_r, _ = r_out.flatten(2).max(dim=2)
        feats = torch.cat([self._pool_vec(p) for p in [gap_h, gmp_h, gap_r, gmp_r]], dim=1)
        return self.mlp(self.proj(feats)).mean(dim=0)

def conv_flops(H, W, Cin, Cout, k, stride): return (H // stride) * (W // stride) * Cin * Cout * (k * k)
def get_block_flops(block, input_shape):
    B, C, H, W = input_shape
    f1 = conv_flops(H, W, block.conv1.in_channels, block.conv1.out_channels, 3, block.conv1.stride[0])
    H2, W2 = H // block.conv1.stride[0], W // block.conv1.stride[0]
    f2 = conv_flops(H2, W2, block.conv2.in_channels, block.conv2.out_channels, 3, block.conv2.stride[0])
    return float(f1 + f2)

def forward_resnet_collect_blocks(model, x, collect_grads=False):
    infos = []
    h = model.conv1(x); h = model.bn1(h); h = model.relu(h); h = model.maxpool(h)
    stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    for s_idx, stage in enumerate(stages):
        for b_idx, block in enumerate(stage):
            h_in = h
            if collect_grads: h_in.retain_grad()
            out = block.conv1(h_in); out = block.bn1(out); out = block.relu(out)
            out = block.conv2(out); r_out = block.bn2(out)
            if collect_grads: r_out.retain_grad()
            skip = block.downsample(h_in) if block.downsample else h_in
            infos.append({"h_in": h_in, "r_out": r_out, "stage": s_idx, "block_idx": b_idx, "H": h_in.size(2), "W": h_in.size(3), "flops": get_block_flops(block, h_in.shape)})
            h = F.relu(skip + r_out)
    h = model.avgpool(h); h = torch.flatten(h, 1)
    return model.fc(h), infos

def build_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device):
    with torch.no_grad(): _, teacher_infos = forward_resnet_collect_blocks(teacher, x, collect_grads=False)
    for p in student.parameters(): p.requires_grad = False
    x_student = x.detach().clone().requires_grad_(True)
    logits_S, student_infos = forward_resnet_collect_blocks(student, x_student, collect_grads=True)
    loss = F.kl_div(F.log_softmax(logits_S/TEMP_KD, dim=1), F.softmax(y_T/TEMP_KD, dim=1), reduction='batchmean') * (TEMP_KD**2)
    loss.backward()
    
    token_list, flops_list = [], []
    for i in range(NUM_BLOCKS):
        t_info = teacher_infos[i]; s_info = student_infos[i]
        feats = summarizer(t_info["h_in"], t_info["r_out"]).to(device)
        grad_r = s_info["r_out"].grad
        taylor = (grad_r * s_info["r_out"]).abs().mean().detach().item() if grad_r is not None else 0.0
        s_info["r_out"].grad = None
        meta = torch.tensor([t_info["stage"]/3.0, t_info["block_idx"]/1.0, t_info["H"]/IMG_SIZE, t_info["W"]/IMG_SIZE], dtype=torch.float32, device=device)
        tay_tensor = torch.tensor([math.log1p(taylor)], dtype=torch.float32, device=device)
        token_list.append(torch.cat([feats, meta, tay_tensor], dim=0))
        flops_list.append(t_info["flops"])
    
    if x_student.grad is not None: x_student.grad = None
    tokens = torch.stack(token_list).unsqueeze(0)
    tokens = token_proj(tokens)
    flops = torch.tensor(flops_list, dtype=torch.float32, device=device)
    return tokens, flops

def forward_resnet_gated_blocks(model, x, block_gates):
    gate_idx = 0
    h = model.conv1(x); h = model.bn1(h); h = model.relu(h); h = model.maxpool(h)
    for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in stage:
            r = block.conv1(h); r = block.bn1(r); r = block.relu(r)
            r = block.conv2(r); r = block.bn2(r)
            g = block_gates[gate_idx].view(1,1,1,1); r = r * g; gate_idx += 1
            skip = block.downsample(h) if block.downsample else h
            h = F.relu(skip + r)
    h = model.avgpool(h); h = torch.flatten(h, 1)
    return model.fc(h)

def kd_loss(student_logits, teacher_logits, T):
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction='batchmean') * (T*T)

# =========================
# Phase 1: Train MLP Policy
# =========================
def train_policy(teacher, train_loader):
    print("\nPhase 1: Training MLP Policy (No Transformer)")
    student = build_resnet18(num_classes=200).to(device)
    student.load_state_dict(teacher.state_dict()); student.eval()

    summarizer = Summarizer(sum_dim=SUM_DIM).to(device)
    token_proj = nn.Sequential(nn.LayerNorm(SUM_DIM + 5), nn.Linear(SUM_DIM + 5, TOKEN_DIM)).to(device)

    # --- SWAP: Use MLP instead of Transformer ---
    policy = GlobalMLPPolicy(num_blocks=NUM_BLOCKS, token_dim=TOKEN_DIM).to(device)

    params = list(summarizer.parameters()) + list(token_proj.parameters()) + list(policy.parameters())
    opt = torch.optim.AdamW(params, lr=POLICY_LR, weight_decay=WEIGHT_DECAY)

    # Track losses for logging
    loss_history = []

    total_epochs = POLICY_WARMUP_EPOCHS + POLICY_TRAIN_EPOCHS
    for epoch in range(total_epochs):
        policy.train(); summarizer.train(); token_proj.train()
        t = epoch / max(1, total_epochs-1)
        temp = GATE_TEMP_START * (1-t) + GATE_TEMP_END * t

        acc_loss_kd = 0; acc_loss_ratio = 0; acc_loss_l1 = 0; acc_loss_total = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): y_T = teacher(x)

            tokens, flops = build_block_tokens(teacher, student, summarizer, token_proj, x, y_T, device)

            target_ratio = random.uniform(MIN_RATIO, MAX_RATIO)
            # MLP Forward
            logits = policy(tokens, target_ratio).squeeze(0)

            gates = torch.sigmoid(logits / temp)
            full_flops = flops.sum()
            kept_flops = (gates * flops).sum()
            curr_ratio = kept_flops / (full_flops + 1e-6)

            y_S = forward_resnet_gated_blocks(student, x, gates)

            # Separate loss components
            loss_kd = kd_loss(y_S, y_T, TEMP_KD)
            loss_ratio = (curr_ratio - target_ratio) ** 2
            loss_l1 = gates.mean()
            loss = loss_kd + RATIO_WEIGHT * loss_ratio + L1_M_WEIGHT * loss_l1

            opt.zero_grad(); loss.backward(); opt.step()

            acc_loss_kd += loss_kd.item()
            acc_loss_ratio += loss_ratio.item()
            acc_loss_l1 += loss_l1.item()
            acc_loss_total += loss.item()

            if i % 100 == 0:
                print(f"[Ep {epoch}][{i}] KD: {acc_loss_kd/(i+1):.4f} Ratio: {acc_loss_ratio/(i+1):.4f} L1: {acc_loss_l1/(i+1):.4f}")
                print(f" Total Loss: {loss.item():.4f} Curr Ratio: {curr_ratio.item():.2f} / {target_ratio:.2f}")

        # Log epoch statistics
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
        print(f"Epoch {epoch} Summary - KD: {epoch_losses['loss_kd']:.4f}, Ratio: {epoch_losses['loss_ratio']:.4f}, L1: {epoch_losses['loss_l1']:.4f}, Total: {epoch_losses['loss_total']:.4f}")

    return policy, summarizer, token_proj, loss_history

# =========================
# Phase 2: Materialize
# =========================
def physically_prune_model(original_model, mask):
    pruned_model = copy.deepcopy(original_model)
    mask_idx = 0
    stages = [pruned_model.layer1, pruned_model.layer2, pruned_model.layer3, pruned_model.layer4]
    for stage in stages:
        for b_idx in range(len(stage)):
            if mask[mask_idx] == 0:
                stage[b_idx] = PrunedBlock(stage[b_idx].downsample)
            mask_idx += 1
    return pruned_model.to(device)

def recalibrate_bn(model, loader, steps=100):
    print(f"Recalibrating BN...")
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

def run_evaluation(model, loader):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total

def materialize_and_finetune(teacher, policy, summarizer, token_proj, train_loader, val_loader, ratio):
    print(f"\n--- Target Ratio: {ratio} ---")
    x_sample, _ = next(iter(train_loader))
    x_sample = x_sample[:32].to(device)
    with torch.no_grad(): y_T = teacher(x_sample)

    temp_student = build_resnet18(200).to(device)
    temp_student.load_state_dict(teacher.state_dict()); temp_student.eval()

    tokens, flops = build_block_tokens(teacher, temp_student, summarizer, token_proj, x_sample, y_T, device)
    logits = policy(tokens, ratio).squeeze(0)
    scores = torch.sigmoid(logits)

    eff = (scores / (flops + 1e-9)).detach().cpu().numpy()
    indices = sorted(range(NUM_BLOCKS), key=lambda i: eff[i], reverse=True)

    mask = [0] * NUM_BLOCKS; current_flops = 0; total_flops = flops.sum().item()
    for idx in indices:
        f = flops[idx].item()
        if (current_flops + f) / total_flops <= ratio:
            mask[idx] = 1; current_flops += f
    if sum(mask) == 0: mask[indices[0]] = 1; current_flops += flops[indices[0]].item()

    real_ratio = current_flops / total_flops
    print(f"Mask: {mask}, Actual: {real_ratio:.3f}")
    print(f"Kept Blocks: {sum(mask)}/{NUM_BLOCKS}")

    pruned_model = physically_prune_model(teacher, mask)
    recalibrate_bn(pruned_model, train_loader)
    acc_pre = run_evaluation(pruned_model, val_loader)
    print(f"Accuracy Pre-FT: {acc_pre*100:.2f}%")

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
    train_loader, val_loader, num_classes = get_loaders()

    teacher = build_resnet18(num_classes).to(device)
    if os.path.exists(CKPT): teacher.load_state_dict(torch.load(CKPT, map_location='cpu')['state_dict'])
    teacher.eval()

    teacher_acc = run_evaluation(teacher, val_loader)
    print(f"Teacher Acc: {teacher_acc*100:.2f}%")

    # Phase 1: Train or Load Policy
    encoder_path = os.path.join(OUT_DIR, "mlp_policy.pth")
    policy_loss_path = os.path.join(OUT_DIR, "mlp_policy_losses.json")

    if os.path.exists(encoder_path):
        print("Loading MLP Policy...")
        state = torch.load(encoder_path)
        policy = GlobalMLPPolicy().to(device); policy.load_state_dict(state['pol'])
        summarizer = Summarizer().to(device); summarizer.load_state_dict(state['sum'])
        token_proj = nn.Sequential(nn.LayerNorm(SUM_DIM+5), nn.Linear(SUM_DIM+5, TOKEN_DIM)).to(device)
        token_proj.load_state_dict(state['proj'])
        # Load loss history if available
        if os.path.exists(policy_loss_path):
            with open(policy_loss_path, 'r') as f:
                policy_loss_history = json.load(f)
        else:
            policy_loss_history = []
    else:
        policy, summarizer, token_proj, policy_loss_history = train_policy(teacher, train_loader)
        torch.save({'pol': policy.state_dict(), 'sum': summarizer.state_dict(), 'proj': token_proj.state_dict()}, encoder_path)
        # Save policy training losses
        with open(policy_loss_path, 'w') as f:
            json.dump(policy_loss_history, f, indent=2)
        print(f"Policy losses saved to {policy_loss_path}")

    # Phase 2: Materialize & Fine-tune
    results = []
    for r in EVAL_RATIOS:
        best_acc, actual_ratio, mask, scores, acc_before_ft, ft_losses = materialize_and_finetune(
            teacher, policy, summarizer, token_proj, train_loader, val_loader, r
        )

        result_entry = {
            "target_ratio": r,
            "actual_ratio": actual_ratio,
            "kept": sum(mask),
            "accuracy_before_ft": acc_before_ft,
            "accuracy_after_ft": best_acc,
            "accuracy_drop": teacher_acc - best_acc,
            "mask": mask,
            "scores": scores,
            "ft_loss_history": ft_losses
        }
        results.append(result_entry)

    # Save full results
    results_path = os.path.join(OUT_DIR, RESULTS_FILE)
    with open(results_path, 'w') as f:
        json.dump({
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
            "policy_loss_history": policy_loss_history,
            "results": results
        }, f, indent=2)

    print(f"\nFull results saved to {results_path}")
    print("\nFinal Results Summary:")
    for res in results:
        print(f"Target: {res['target_ratio']:.1f} | Actual: {res['actual_ratio']:.3f} | "
              f"Kept: {res['kept']}/{NUM_BLOCKS} | "
              f"Acc: {res['accuracy_after_ft']*100:.2f}% | "
              f"Drop: {res['accuracy_drop']*100:.2f}%")

if __name__ == "__main__":
    main()
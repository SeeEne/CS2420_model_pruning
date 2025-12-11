"""
Baseline: Block-Wise Pruning based on L2 Magnitude for ResNet-18 (Tiny ImageNet)
Strategy:
1. Calculate L2 Norm of weights for each of the 8 blocks.
2. Sort blocks by Magnitude (Smallest -> Prune, Largest -> Keep).
3. Physically replace pruned blocks with Skip-Connection-Only blocks.
4. Recalibrate BN -> Fine-tune.
"""

import os, math, json, copy
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image

# =========================
# Config
# =========================
DATA_DIR = "data/tiny-imagenet-200"
TEACHER_CKPT = "checkpoints/teacher.pth"
OUT_JSON = "checkpoints/baseline_l2_block_results.json"

IMG_SIZE = 224
BATCH_SIZE = 512
NUM_WORKERS = 4
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_BLOCKS = 8

# Training / eval knobs
FT_EPOCHS = 10          
FT_LR = 1e-3
TEMP_KD = 2.0
CALIB_ITERS = 100       # BN recalibration iters
RATIO_LIST = [0.1, 0.3, 0.5, 0.7, 0.9]

# =========================
# Dataset (Tiny ImageNet)
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
                if self.class_to_idx:
                    self.labels.append(self.class_to_idx[parts[1]])
                else:
                    self.labels.append(0)

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
    
    # Dummy fallback
    if not os.path.exists(os.path.join(DATA_DIR, 'train')):
        dummy_ds = datasets.FakeData(size=1000, image_size=(3, IMG_SIZE, IMG_SIZE), num_classes=200, transform=transforms.ToTensor())
        return DataLoader(dummy_ds, 32), DataLoader(dummy_ds, 32), 200

    train_ds = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_tf)
    val_ds = TinyImageNetVal(root=os.path.join(DATA_DIR, 'val'), transform=test_tf)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes)

# =========================
# Model & Helpers
# =========================
def build_resnet18(num_classes=200):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class PrunedBlock(nn.Module):
    """
    Physical replacement for a pruned BasicBlock.
    Runs ONLY the skip connection (and downsample if it exists).
    """
    def __init__(self, downsample_module):
        super().__init__()
        self.downsample = downsample_module 
        
    def forward(self, x):
        if self.downsample is not None:
            return F.relu(self.downsample(x))
        else:
            return F.relu(x)

def kd_loss(student_logits, teacher_logits, T=TEMP_KD):
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction='batchmean') * (T*T)

def conv_flops(H, W, Cin, Cout, k, stride=1):
    return (H // stride) * (W // stride) * Cin * Cout * (k * k)

def get_block_info(model, img_size=IMG_SIZE):
    """
    Returns 2 lists of length 8 (NUM_BLOCKS):
    1. flops_list: Total FLOPs of the RESIDUAL branch (conv1+conv2)
    2. l2_scores: Combined normalized L2 norm of conv1+conv2 weights
    """
    model.eval()
    # Dummy forward to calculate Feature Map sizes
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    h = model.conv1(x); h = model.bn1(h); h = model.relu(h); h = model.maxpool(h)
    
    flops_list = []
    l2_scores = []
    
    stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    
    for stage in stages:
        for block in stage:
            # --- FLOPs Calc ---
            Cin, H, W = h.size(1), h.size(2), h.size(3)
            
            # Conv1
            stride1 = block.conv1.stride[0]
            Cmid = block.conv1.out_channels
            f1 = conv_flops(H, W, Cin, Cmid, 3, stride1)
            
            # Conv2
            H2, W2 = H // stride1, W // stride1
            Cout = block.conv2.out_channels
            stride2 = block.conv2.stride[0] # usually 1
            f2 = conv_flops(H2, W2, Cmid, Cout, 3, stride2)
            
            flops_list.append(float(f1 + f2))
            
            # --- L2 Score Calc ---
            # Score = Normalized L2(Conv1) + Normalized L2(Conv2)
            w1 = block.conv1.weight.data
            w2 = block.conv2.weight.data
            
            s1 = w1.pow(2).sum().sqrt().item() / math.sqrt(w1.numel() + 1e-8)
            s2 = w2.pow(2).sum().sqrt().item() / math.sqrt(w2.numel() + 1e-8)
            
            l2_scores.append(s1 + s2)

            # Update h for next block simulation
            if block.downsample is not None:
                skip = block.downsample(h)
            else:
                skip = h
            
            # Assume output size matches skip connection (H2, W2)
            h = torch.zeros(1, Cout, H2, W2, device=device)

    return torch.tensor(l2_scores, device=device), torch.tensor(flops_list, device=device)

# =========================
# Selection & Pruning
# =========================
def select_blocks_l2(scores, flops, target_ratio):
    """
    Sort blocks by L2 score (High to Low).
    Keep adding blocks until the FLOPs budget is met.
    """
    total_flops = flops.sum().item()
    
    # Sort: High Score (Important) -> Low Score (Prunable)
    vals, idx_desc = torch.sort(scores, descending=True)
    
    mask = [0] * NUM_BLOCKS
    current_flops = 0.0
    
    for i in idx_desc:
        idx = i.item()
        block_cost = flops[idx].item()
        
        # Check if adding this block exceeds budget
        if (current_flops + block_cost) / total_flops <= target_ratio:
            mask[idx] = 1
            current_flops += block_cost
            
    # Safety: Ensure at least one block if ratio is very small but non-zero
    if sum(mask) == 0 and target_ratio > 0.01:
        best_idx = idx_desc[0].item()
        mask[best_idx] = 1
        current_flops += flops[best_idx].item()
            
    actual_ratio = current_flops / total_flops
    return mask, actual_ratio, sum(mask)

def physically_prune_model(original_model, mask):
    """
    Physically remove blocks where mask[i] == 0
    """
    pruned_model = copy.deepcopy(original_model)
    mask_idx = 0
    stages = [pruned_model.layer1, pruned_model.layer2, pruned_model.layer3, pruned_model.layer4]
    
    for stage in stages:
        for b_idx in range(len(stage)):
            keep = mask[mask_idx]
            if keep == 0:
                original_block = stage[b_idx]
                new_block = PrunedBlock(original_block.downsample)
                stage[b_idx] = new_block
            mask_idx += 1
            
    return pruned_model.to(device)

def recalibrate_bn(model, loader, steps=CALIB_ITERS):
    """Update BN stats on the pruned structure"""
    print(f"Recalibrating BN ({steps} iters)...")
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

def evaluate(model, loader):
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
# Main Routine
# =========================
def run_one_ratio(r, teacher, train_loader, val_loader):
    # 1. Calculate Scores on the UNPRUNED teacher/base model
    # We use a copy so we don't mess up the teacher instance
    base_model = copy.deepcopy(teacher)
    scores, flops = get_block_info(base_model)
    
    # 2. Select Mask
    mask, actual_r, kept = select_blocks_l2(scores, flops, r)
    
    # 3. Physically Prune
    student = physically_prune_model(teacher, mask)
    
    # 4. BN Recalibration (Critical for baseline fairness)
    recalibrate_bn(student, train_loader)
    acc_before = evaluate(student, val_loader) * 100.0
    
    # 5. Fine-tune
    opt = torch.optim.AdamW(student.parameters(), lr=FT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)
    
    best_acc = acc_before
    print(f"FT Start: Ratio={actual_r:.3f} Kept={kept} Acc={acc_before:.2f}%")
    
    for ep in range(FT_EPOCHS):
        student.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): y_T = teacher(x)
            
            y_S = student(x)
            loss = kd_loss(y_S, y_T)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        scheduler.step()
        acc = evaluate(student, val_loader) * 100.0
        if acc > best_acc: 
            best_acc = acc
            print(f"  New Best Acc: {best_acc:.2f}%")
            
        print(f"  Ep {ep+1} Acc: {acc:.2f}%")
        print("-" * 30)
        
    return {
        "target_ratio": r,
        "actual_ratio": actual_r,
        "kept": kept,
        "before_ft": acc_before,
        "after_ft": best_acc,
        "drop": (evaluate(teacher, val_loader)*100.0) - best_acc,
        "mask": mask
    }

def main():
    torch.manual_seed(SEED)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    
    train_loader, val_loader, num_classes = get_loaders()
    
    # Load Teacher
    teacher = build_resnet18(num_classes).to(device)
    if os.path.exists(TEACHER_CKPT):
        ckpt = torch.load(TEACHER_CKPT, map_location="cpu")
        teacher.load_state_dict(ckpt["state_dict"])
    else:
        print("Warning: Teacher checkpoint not found, using random weights.")
    teacher.eval()
    
    teacher_acc = evaluate(teacher, val_loader) * 100.0
    print(f"Teacher Accuracy: {teacher_acc:.2f}%")
    
    results = []
    print(f"{'Ratio':<6} {'Actual':<7} {'Kept':<6} {'Before':<8} {'After':<8}")
    
    for r in RATIO_LIST:
        res = run_one_ratio(r, teacher, train_loader, val_loader)
        res['teacher_acc'] = teacher_acc
        results.append(res)
        print(f"{res['target_ratio']:<6.1f} {res['actual_ratio']:<7.3f} {res['kept']:<6d} "
              f"{res['before_ft']:<8.2f} {res['after_ft']:<8.2f}")
        
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUT_JSON}")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
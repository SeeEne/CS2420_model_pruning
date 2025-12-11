# run_batch_baseline_l2_16gate.py
import os, math, json, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ------------------------------
# Paths / hyperparams
# ------------------------------
DATA_DIR = "data"
TEACHER_CKPT = "checkpoints/teacher.pth"
OUT_JSON = "checkpoints/baseline_l2_16gate_results.json"

IMG_SIZE = 224
BATCH_SIZE = 256
NUM_WORKERS = 0
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_BLOCKS = 8
NUM_UNITS  = 16

# Training / eval knobs
FT_EPOCHS = 10          
LR = 1e-3
TEMP_KD = 2.0
CALIB_ITERS = 30        # BN recalibration iters
RATIO_LIST = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

# ------------------------------
# Data
# ------------------------------
def loaders():
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

# ------------------------------
# Model + utils
# ------------------------------
def build_resnet18(num_classes=10):
    return models.resnet18(weights=None, num_classes=num_classes)

def kd_loss(student_logits, teacher_logits, T=TEMP_KD):
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction='batchmean') * (T*T)

def conv_flops(H, W, Cin, Cout, k, stride=1):
    return H * W * Cin * Cout * (k * k) // (stride * stride)

@torch.no_grad()
def estimate_unit_flops_map(model, img_size=IMG_SIZE):
    """
    返回 16 个单元（每个block的conv1、conv2）的 FLOPs：
      - conv1 单元：3x3 在 (H/stride,W/stride) + 若有 downsample 则加 1x1(stride) 的成本
      - conv2 单元：3x3 在 conv1 输出空间 (H1,W1)
      
    Returns FLOPs for 16 units (conv1 and conv2 of each block):
      - conv1 unit: 3x3 at (H/stride, W/stride) + if downsample exists, add 1x1 (stride) cost
        - conv2 unit: 3x3 at conv1 output space (H1, W1)
    """
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    h = model.conv1(x); h = model.bn1(h); h = model.relu(h); h = model.maxpool(h)
    flops_list = []
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            Cin, H, W = h.size(1), h.size(2), h.size(3)
            stride = 2 if block.downsample is not None else 1
            H1, W1 = math.ceil(H/stride), math.ceil(W/stride)

            r1 = block.conv1(h); r1 = block.bn1(r1); r1_relu = block.relu(r1)
            Cout = r1_relu.size(1)
            f_conv1 = conv_flops(H1, W1, Cin, Cout, 3, stride=1)
            f_down  = H1 * W1 * Cin * Cout if block.downsample is not None else 0
            flops_list.append(float(f_conv1 + f_down))

            r2 = block.conv2(r1_relu); r2 = block.bn2(r2)
            f_conv2 = conv_flops(H1, W1, Cout, Cout, 3, stride=1)
            flops_list.append(float(f_conv2))

            skip = block.downsample(h) if block.downsample is not None else h
            out_pre = skip + r2
            h = block.relu(out_pre)
    assert len(flops_list) == NUM_UNITS
    return torch.tensor(flops_list, dtype=torch.float32, device=device)

@torch.no_grad()
def unit_l2_scores_16(model):
    """
    纯 L2：对每个 conv（conv1/conv2）计算 ||W||_F / sqrt(#params)，共 16 个分数。
    顺序：[b0.conv1, b0.conv2, b1.conv1, b1.conv2, ...]
    
    Pure L2: for each conv (conv1/conv2), compute ||W||_F / sqrt(#params), total 16 scores.
    Order: [b0.conv1, b0.conv2, b1.conv1, b1.conv2, ...]
    """
    scores = []
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            for conv in [block.conv1, block.conv2]:
                w = conv.weight.data
                s = w.pow(2).sum().sqrt().item() / math.sqrt(w.numel() + 1e-8)
                scores.append(s)
    return torch.tensor(scores, dtype=torch.float32, device=device)

def select_mask_topk_match_ratio(scores, flops, target_ratio):
    """
    仅用 L2 排序（scores 降序），尝试 k=1..16，让 FLOPs 比例最接近 target_ratio。
    不使用 score/FLOPs 效率，不加层级约束。
    
    Use L2 scores only (descending), try k=1..16 to match FLOPs ratio closest to target_ratio.
    No efficiency or structural constraints.
    """
    assert scores.numel() == flops.numel() == NUM_UNITS
    vals, idx_desc = torch.sort(scores, descending=True)

    total = flops.sum()
    best_k, best_diff, best_ratio = 1, float('inf'), None
    for k in range(1, NUM_UNITS+1):
        mask = torch.zeros(NUM_UNITS, dtype=torch.float32, device=device)
        mask[idx_desc[:k]] = 1.0
        ratio = (mask * flops).sum() / total
        diff = abs(ratio.item() - target_ratio)
        if diff < best_diff:
            best_diff = diff
            best_k = k
            best_ratio = ratio.item()

    mask = torch.zeros(NUM_UNITS, dtype=torch.float32, device=device)
    mask[idx_desc[:best_k]] = 1.0
    actual = (mask * flops).sum() / total
    return mask, float(actual.item()), int(best_k)

def resnet18_masked_forward_with_gates16(model: models.ResNet, x: torch.Tensor, gates: torch.Tensor):
    assert len(gates) == NUM_UNITS
    g_idx = 0
    h = model.conv1(x); h = model.bn1(h); h = model.relu(h); h = model.maxpool(h)
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            r1 = block.conv1(h); r1 = block.bn1(r1); r1 = block.relu(r1)
            g1 = gates[g_idx].view(1,1,1,1); g_idx += 1
            r1 = r1 * g1

            r2 = block.conv2(r1); r2 = block.bn2(r2)
            g2 = gates[g_idx].view(1,1,1,1); g_idx += 1
            r2 = r2 * g2

            skip = block.downsample(h) if block.downsample is not None else h
            out_pre = skip + r2
            h = block.relu(out_pre)
    h = model.avgpool(h)
    h = torch.flatten(h, 1)
    logits = model.fc(h)
    return logits

@torch.no_grad()
def evaluate(model, loader, mask16):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = resnet18_masked_forward_with_gates16(model, x, mask16)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    return correct / total

@torch.no_grad()
def bn_recalibrate(model, loader, mask16, max_iters=CALIB_ITERS):
    model.train()  # 让BN更新running stats
    it = 0
    for x, _ in loader:
        x = x.to(device)
        _ = resnet18_masked_forward_with_gates16(model, x, mask16)
        it += 1
        if it >= max_iters: break
    model.eval()

# ------------------------------
# One-ratio pipeline
# ------------------------------
def run_one_ratio(r, teacher_ckpt, train_loader, test_loader):
    # 构建 teacher / student
    # Build teacher / student
    teacher = build_resnet18(num_classes=10).to(device)
    ckpt = torch.load(teacher_ckpt, map_location="cpu")
    teacher.load_state_dict(ckpt["state_dict"]); teacher.eval()

    student = build_resnet18(num_classes=10).to(device)
    student.load_state_dict(ckpt["state_dict"])

    # 计算 L2 分数（16）与 FLOPs（16）
    # Compute L2 scores (16) and FLOPs (16)
    scores16 = unit_l2_scores_16(student)
    flops16  = estimate_unit_flops_map(student)

    # 仅用 L2 排序，选取最接近目标 FLOPs 的 top-k
    # Use L2 scores only, select top-k to match target FLOPs
    mask16, actual_ratio, kept = select_mask_topk_match_ratio(scores16, flops16, r)

    # BN 重校准，再评估
    # BN recalibration, then evaluate
    bn_recalibrate(student, train_loader, mask16, max_iters=CALIB_ITERS)
    acc_before = evaluate(student, test_loader, mask16) * 100.0

    # KD-only 微调
    # KD-only finetune
    for p in student.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(student.parameters(), lr=LR)

    for epoch in range(FT_EPOCHS):
        student.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_T = teacher(x)
            logits = resnet18_masked_forward_with_gates16(student, x, mask16)
            loss = kd_loss(logits, y_T, T=TEMP_KD)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    acc_after = evaluate(student, test_loader, mask16) * 100.0
    return {
        "target_ratio": r,
        "actual_ratio": float(actual_ratio),
        "kept": kept,
        "before_ft": float(acc_before),
        "after_ft": float(acc_after),
        "mask": mask16.int().tolist()
    }

# ------------------------------
# Main: batch run
# ------------------------------
def main():
    # seeds
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = loaders()

    # Teacher (unpruned) acc
    teacher = build_resnet18(num_classes=10).to(device)
    ckpt = torch.load(TEACHER_CKPT, map_location="cpu")
    teacher.load_state_dict(ckpt["state_dict"]); teacher.eval()
    teacher_acc = evaluate(teacher, test_loader, torch.ones(NUM_UNITS, device=device)) * 100.0
    print(f"Teacher (unpruned) top1: {teacher_acc:.2f}%")

    rows = []
    for r in RATIO_LIST:
        print("\n" + "="*80)
        print(f"Target Ratio: {r:.1f}")
        res = run_one_ratio(r, TEACHER_CKPT, train_loader, test_loader)
        drop = teacher_acc - res["after_ft"]
        rows.append({**res, "drop": float(drop)})
        print(f"actual={res['actual_ratio']:.3f} kept={res['kept']:02d}/16  "
              f"before={res['before_ft']:.2f}%  after={res['after_ft']:.2f}%  drop={drop:.2f}%")

    # 打印表格
    print("\n" + "="*80)
    print("BASELINE L2-ONLY (16-gate) — FINAL TABLE")
    print("="*80)
    header = f"{'Ratio':<6} {'Actual':<7} {'Kept':<6} {'Before FT':<10} {'After FT':<9} {'Drop':<8}"
    print(header); print("-"*len(header))
    for r in rows:
        print(f"{r['target_ratio']:<6.1f} {r['actual_ratio']:<7.3f} {r['kept']:<6d} "
              f"{r['before_ft']:<10.2f} {r['after_ft']:<9.2f} {r['drop']:<8.2f}")
    print("="*80)
    print(f"Teacher (unpruned): {teacher_acc:.2f}%")

    # 保存 JSON
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"teacher_accuracy": float(teacher_acc), "rows": rows}, f, indent=2)
    print(f"\nSaved baseline results to: {OUT_JSON}")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

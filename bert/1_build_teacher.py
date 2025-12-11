"""_summary_


This is the first step in a knowledge distillation pipeline.
Train a ResNet-18 teacher model on CIFAR-10.



"""


# %%
import argparse, os, time, math, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# %% [markdown]
# ### Build the teacher model ###

# %%
def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_dataloaders(data_dir="data", batch_size=256, num_workers=4):
    # CIFAR-10 stats
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.Resize(224),  # Resize to 224x224 for ResNet-18
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(224),  # Resize to 224x224 for ResNet-18
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.CIFAR10(root=data_dir, train=True,  transform=train_tf, download=True)
    test_ds  = datasets.CIFAR10(root=data_dir, train=False, transform=test_tf,  download=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def build_teacher(num_classes=10, pretrained=True):
    # Use ResNet18 and replace the final FC
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, epochs):
    model.train()
    ce = nn.CrossEntropyLoss()
    running = 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(x)
            loss = ce(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
        if (i+1) % 100 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] step {i+1}/{len(loader)}  loss={running/100:.4f}")
            running = 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--out", type=str, default="checkpoints")
    
    # This handles Jupyter's extra arguments
    import sys
    if any('ipykernel' in arg for arg in sys.argv):
        args = parser.parse_args([])  # Use defaults in Jupyter
    else:
        args = parser.parse_args()    # Normal CLI parsing

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    train_loader, test_loader = get_dataloaders(args.data, args.batch_size)

    model = build_teacher(pretrained=not args.no_pretrained).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_acc = 0.0
    for epoch in range(args.epochs):
        # cosine lr
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * epoch / max(1, args.epochs-1)))
        for g in optimizer.param_groups: g["lr"] = lr

        train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args.epochs)
        val_loss, val_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  lr={lr:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(),
                        "val_acc": best_acc}, os.path.join(args.out, "teacher.pth"))
            print(f"✓ Saved checkpoint with acc {best_acc*100:.2f}%")

    print(f"Best teacher acc: {best_acc*100:.2f}%  (weights: {os.path.join(args.out,'teacher.pth')})")


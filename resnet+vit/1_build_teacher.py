"""
Train a ResNet-18 teacher model on Tiny ImageNet-200.
Handles the special validation folder structure.
"""

# %%
import argparse, os, time, math, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image

# %% [markdown]
# ### Custom Dataset for Tiny ImageNet Validation ###

# %%
class TinyImageNetVal(Dataset):
    """Custom dataset for Tiny ImageNet validation set."""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        # Read val_annotations.txt
        annotations_file = os.path.join(root, 'val_annotations.txt')
        self.images = []
        self.labels = []
        
        # Build class to index mapping from train folder
        train_dir = os.path.join(os.path.dirname(root), 'train')
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(train_dir)))}
        
        # Parse annotations
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

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_dataloaders(data_dir="data/tiny-imagenet-200", batch_size=4096, num_workers=4):
    # ImageNet stats (use these for any ImageNet variant)
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    
    train_tf = transforms.Compose([
        transforms.Resize(224),  # Upsample from 64x64 to 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    test_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Training: use ImageFolder (class-based structure)
    train_ds = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_tf)
    
    # Validation: use custom dataset (flat structure with annotations)
    val_ds = TinyImageNetVal(root=os.path.join(data_dir, 'val'), transform=test_tf)
    
    # Get number of classes
    num_classes = len(train_ds.classes)
    print(f"✓ Dataset loaded: {len(train_ds)} train images, {len(val_ds)} val images, {num_classes} classes")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  
                             num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, 
                             num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    return train_loader, test_loader, num_classes

def build_teacher(num_classes=200, pretrained=True):
    """Build ResNet-18 teacher model."""
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)
    
    # Replace final FC layer for num_classes
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
    parser.add_argument("--data", type=str, default="data/tiny-imagenet-200")
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
    print(f"✓ Using device: {device}")
    os.makedirs(args.out, exist_ok=True)

    # Load data - now returns num_classes too
    train_loader, test_loader, num_classes = get_dataloaders(args.data, args.batch_size)

    # Build model with correct number of classes
    model = build_teacher(num_classes=num_classes, pretrained=not args.no_pretrained).to(device)
    print(f"✓ Model built with {num_classes} output classes")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_acc = 0.0
    for epoch in range(args.epochs):
        # cosine lr schedule
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * epoch / max(1, args.epochs-1)))
        for g in optimizer.param_groups: g["lr"] = lr

        train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args.epochs)
        val_loss, val_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  lr={lr:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(),
                        "val_acc": best_acc,
                        "num_classes": num_classes}, 
                      os.path.join(args.out, "teacher.pth"))
            print(f"✓ Saved checkpoint with acc {best_acc*100:.2f}%")

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best teacher acc: {best_acc*100:.2f}%")
    print(f"Weights saved to: {os.path.join(args.out,'teacher.pth')}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
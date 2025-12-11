"""
Fine-tune ViT-Small with pre-computed masks from encoder validation.
Uses masks from 10_vit_small_block_az_regularization.py validation output.

Fine-tuning config matches 9_vit_small_block.py:
- FT_EPOCHS = 10
- FT_LR = 1e-4
- KD-only training (no task loss)
"""
import warnings
warnings.filterwarnings("ignore")

import os
import json
import copy
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import timm

# =========================
# Config
# =========================
DATA_DIR = "data/tiny-imagenet-200"
CKPT_TEACHER = "checkpoints/teacher_vit_small.pth"
OUT_DIR = "checkpoints"
RESULTS_FILE = "vit_small_az_finetune_results.json"

BATCH_SIZE = 256
NUM_WORKERS = 2
IMG_SIZE = 224
TEMP_KD = 2.0

# Fine-tuning config (same as 9_vit_small_block.py)
FT_EPOCHS = 8
FT_LR = 1e-4

# Standard KD loss weights
# Total loss = ALPHA * task_loss + (1 - ALPHA) * kd_loss
ALPHA = 0.5  # Weight for task loss (cross-entropy with ground truth)

# Masks from encoder validation output
# Format: {ratio: mask_string}
MASKS = {
    0.3: [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # 110100000001
    0.4: [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],  # 110101000001
    0.5: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # 111111000000
    0.6: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],  # 111111000010
    0.7: [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1],  # 111101000111
    0.8: [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],  # 111111001111
}

NUM_BLOCKS = 12
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        if os.path.exists(train_dir):
            self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(train_dir)))}
        else:
            self.class_to_idx = {}
        if os.path.exists(annotations_file):
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
        print(f"Warning: Dataset not found at {DATA_DIR}. Using FakeData for testing.")
        dummy_ds = datasets.FakeData(size=1000, image_size=(3, IMG_SIZE, IMG_SIZE), num_classes=200, transform=transforms.ToTensor())
        return DataLoader(dummy_ds, 32), DataLoader(dummy_ds, 32), 200

    train_ds = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_tf)
    val_ds = TinyImageNetVal(root=os.path.join(DATA_DIR, 'val'), transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes)

# =========================
# Model
# =========================
def build_vit_small(num_classes=200, pretrained=False):
    if pretrained:
        model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
    else:
        model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
    return model

# =========================
# Pruned ViT Model (physically removes blocks)
# =========================
class PrunedViT(nn.Module):
    """
    ViT model with blocks physically removed based on mask.
    Only keeps blocks where mask[i] == 1.
    This gives real FLOPs and memory savings.
    """
    def __init__(self, vit_model, mask):
        super().__init__()
        self.patch_embed = vit_model.patch_embed
        self.cls_token = vit_model.cls_token
        self.pos_embed = vit_model.pos_embed
        self.pos_drop = vit_model.pos_drop
        self.norm = vit_model.norm
        self.head = vit_model.head

        # Only keep blocks where mask == 1
        kept_blocks = []
        self.kept_indices = []
        for i, block in enumerate(vit_model.blocks):
            if i < len(mask) and mask[i] == 1:
                kept_blocks.append(block)
                self.kept_indices.append(i)
        self.blocks = nn.ModuleList(kept_blocks)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Add cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply only kept blocks
        for block in self.blocks:
            x = block(x)

        # Final norm and head
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

# =========================
# Loss and Evaluation
# =========================
def kd_loss(student_logits, teacher_logits, T):
    """Soft target KD loss (KL divergence with temperature scaling)"""
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction='batchmean') * (T*T)


def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    Standard Knowledge Distillation loss combining:
    - Soft target loss (KD): KL divergence between student and teacher soft outputs
    - Hard target loss (Task): Cross-entropy with ground truth labels

    Total loss = alpha * task_loss + (1 - alpha) * kd_loss

    Args:
        student_logits: Student model outputs
        teacher_logits: Teacher model outputs
        labels: Ground truth labels
        T: Temperature for softening distributions
        alpha: Weight for task loss (1-alpha for KD loss)
    """
    # Soft target loss (KD)
    loss_kd = kd_loss(student_logits, teacher_logits, T)

    # Hard target loss (Task)
    loss_task = F.cross_entropy(student_logits, labels)

    # Combined loss
    total_loss = alpha * loss_task + (1 - alpha) * loss_kd

    return total_loss, loss_kd, loss_task

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

def run_evaluation_pruned(model, loader):
    """Evaluate pruned model (already has blocks removed)"""
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def cleanup_vram():
    """Clean up VRAM between runs"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# =========================
# Fine-tuning
# =========================
def finetune_with_mask(teacher, mask, train_loader, val_loader, ratio):
    """Fine-tune student with standard KD loss (KD + Task loss)"""
    print(f"\n{'='*60}")
    print(f"Fine-tuning for Ratio {ratio}")
    print(f"{'='*60}")
    print(f"Mask: {''.join(str(int(m)) for m in mask)}")
    print(f"Kept Blocks: {sum(mask)}/{NUM_BLOCKS}")
    print(f"Loss: alpha={ALPHA} (task) + {1-ALPHA:.1f} (KD), T={TEMP_KD}")

    # Create pruned student (physically remove blocks)
    student_full = copy.deepcopy(teacher)
    student = PrunedViT(student_full, mask).to(device)
    del student_full  # Free memory from full model copy
    cleanup_vram()

    print(f"Pruned model has {len(student.blocks)} blocks (indices: {student.kept_indices})")

    # Evaluate before fine-tuning
    acc_before = run_evaluation_pruned(student, val_loader)
    print(f"Accuracy Before FT: {acc_before*100:.2f}%")

    # Optimizer and scheduler
    opt = torch.optim.AdamW(student.parameters(), lr=FT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)

    best_acc = acc_before
    ft_loss_history = []

    for ep in range(FT_EPOCHS):
        student.train()
        epoch_loss_total = 0.0
        epoch_loss_kd = 0.0
        epoch_loss_task = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_T = teacher(x)
            y_S = student(x)

            # Standard KD loss: alpha * task_loss + (1-alpha) * kd_loss
            loss, loss_kd, loss_task = distillation_loss(y_S, y_T, y, TEMP_KD, ALPHA)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss_total += loss.item()
            epoch_loss_kd += loss_kd.item()
            epoch_loss_task += loss_task.item()
            n_batches += 1

        scheduler.step()
        acc = run_evaluation_pruned(student, val_loader)
        if acc > best_acc:
            best_acc = acc

        avg_loss_total = epoch_loss_total / n_batches
        avg_loss_kd = epoch_loss_kd / n_batches
        avg_loss_task = epoch_loss_task / n_batches
        ft_loss_history.append({
            "epoch": ep,
            "loss_total": avg_loss_total,
            "loss_kd": avg_loss_kd,
            "loss_task": avg_loss_task,
            "accuracy": acc
        })
        print(f"  [FT Ep {ep+1}/{FT_EPOCHS}] Total: {avg_loss_total:.4f} (KD: {avg_loss_kd:.4f}, Task: {avg_loss_task:.4f}) Acc: {acc*100:.2f}%")

    print(f"Best Accuracy: {best_acc*100:.2f}%")

    # Cleanup student after fine-tuning
    del student, opt, scheduler
    cleanup_vram()

    return {
        "ratio": ratio,
        "mask": mask,
        "kept_blocks": int(sum(mask)),
        "accuracy_before_ft": acc_before,
        "accuracy_after_ft": best_acc,
        "improvement": best_acc - acc_before,
        "ft_loss_history": ft_loss_history
    }

# =========================
# Resume Logic
# =========================
def load_existing_results():
    """Load existing results if available for resume"""
    results_path = os.path.join(OUT_DIR, RESULTS_FILE)
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def get_completed_ratios(existing_results):
    """Get set of ratios that have already been completed"""
    if existing_results is None:
        return set()
    completed = set()
    for r in existing_results.get("results", []):
        completed.add(r["ratio"])
    return completed


def save_results(all_results):
    """Save results to file"""
    results_path = os.path.join(OUT_DIR, RESULTS_FILE)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")


# =========================
# Main
# =========================
def main():
    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("="*60)
    print("ViT-Small Fine-tuning with Pre-computed Masks")
    print("="*60)
    print(f"Device: {device}")
    print(f"FT Epochs: {FT_EPOCHS}")
    print(f"FT LR: {FT_LR}")
    print(f"Ratios: {list(MASKS.keys())}")

    # Check for existing results (resume logic)
    existing_results = load_existing_results()
    completed_ratios = get_completed_ratios(existing_results)

    if completed_ratios:
        print(f"\n[RESUME] Found existing results for ratios: {sorted(completed_ratios)}")
        print(f"[RESUME] Will skip these and continue with remaining ratios")

    # Load data
    train_loader, val_loader, num_classes = get_loaders()

    # Load teacher
    if not os.path.exists(CKPT_TEACHER):
        print(f"Error: Teacher checkpoint not found at {CKPT_TEACHER}")
        return

    print(f"\nLoading teacher from {CKPT_TEACHER}")
    teacher = build_vit_small(num_classes).to(device)
    ckpt = torch.load(CKPT_TEACHER, map_location='cpu')
    teacher.load_state_dict(ckpt['state_dict'])
    teacher.eval()

    teacher_acc = run_evaluation(teacher, val_loader)
    print(f"Teacher Accuracy: {teacher_acc*100:.2f}%")

    # Initialize results (use existing or create new)
    if existing_results is not None:
        all_results = existing_results
        # Update teacher accuracy in case it changed
        all_results["teacher_accuracy"] = teacher_acc
    else:
        all_results = {
            "teacher_accuracy": teacher_acc,
            "ft_epochs": FT_EPOCHS,
            "ft_lr": FT_LR,
            "results": []
        }

    # Fine-tune for each ratio
    all_ratios = sorted(MASKS.keys())
    remaining_ratios = [r for r in all_ratios if r not in completed_ratios]

    if not remaining_ratios:
        print("\n[RESUME] All ratios already completed!")
    else:
        print(f"\n[RESUME] Remaining ratios to process: {remaining_ratios}")

    for ratio in remaining_ratios:
        mask = MASKS[ratio]
        total_idx = all_ratios.index(ratio) + 1
        print(f"\n[{total_idx}/{len(MASKS)}] Processing ratio {ratio}...")

        result = finetune_with_mask(teacher, mask, train_loader, val_loader, ratio)
        result["accuracy_drop"] = teacher_acc - result["accuracy_after_ft"]
        all_results["results"].append(result)

        # Save after each ratio (for resume capability)
        save_results(all_results)

        # VRAM cleanup between ratios
        print(f"Cleaning up VRAM after ratio {ratio}...")
        cleanup_vram()

    # Sort results by ratio for consistent output
    all_results["results"] = sorted(all_results["results"], key=lambda x: x["ratio"])

    # Final save
    save_results(all_results)

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"{'Ratio':<8} {'Kept':<6} {'Before FT':<12} {'After FT':<12} {'Drop':<10}")
    print("-"*60)
    for r in all_results["results"]:
        print(f"{r['ratio']:<8.1f} {r['kept_blocks']:<6d} "
              f"{r['accuracy_before_ft']*100:<12.2f} "
              f"{r['accuracy_after_ft']*100:<12.2f} "
              f"{r['accuracy_drop']*100:<10.2f}")
    print("="*60)

if __name__ == "__main__":
    main()

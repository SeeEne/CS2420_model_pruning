# ============================================================================
# SINGLE FILE COLAB VERSION - Phase 1 Encoder Training
# All dependencies included in this one file
# ============================================================================

import os, shutil, random, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Sampler
from torchvision import transforms, datasets
import torchvision
import numpy as np


# ============================================================================
# SECTION 1: DEVICE & CONSTANTS
# ============================================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

NUM_CLASSES = 200
IMG_SIZE = 224
BATCH_TRAIN = 256
BATCH_VAL = 256
BATCH_SCORE = 128
NUM_WORKERS = 0
TIN_MEAN = (0.4802, 0.4481, 0.3975)
TIN_STD = (0.2770, 0.2691, 0.2821)


# ============================================================================
# SECTION 2: DATASET (Google Drive version)
# ============================================================================

def mount_drive():
    """Mount Google Drive if not already mounted."""
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
            print("Google Drive mounted successfully!")
        else:
            print("Google Drive already mounted.")
    except ImportError:
        print("Not running in Colab, skipping drive mount.")

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def organize_tiny_imagenet_val(val_root: str):
    """Organize val folder into class subfolders."""
    val_root = Path(val_root)
    images_dir = val_root / "images"
    annot_path = val_root / "val_annotations.txt"
    if not annot_path.exists():
        return
    if images_dir.exists():
        with open(annot_path, "r") as f:
            for line in f:
                img, wnid = line.strip().split("\t")[:2]
                (val_root / wnid).mkdir(exist_ok=True)
        moved = 0
        with open(annot_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img, wnid = parts[0], parts[1]
                src = images_dir / img
                dst = val_root / wnid / img
                if src.exists() and not dst.exists():
                    shutil.move(str(src), str(dst))
                    moved += 1
        if images_dir.exists() and not any(images_dir.iterdir()):
            images_dir.rmdir()
        print(f"[val organize] moved {moved} files.")

class _FixedOrderSampler(Sampler[int]):
    def __init__(self, indices: List[int]):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

def _build_scoring_subset_indices(dataset, size: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    idxs = list(range(len(dataset)))
    rng.shuffle(idxs)
    return sorted(idxs[:min(size, len(dataset))])

def make_dataloaders(data_root: str, batch_train: int = BATCH_TRAIN, batch_val: int = BATCH_VAL,
                     batch_score: int = BATCH_SCORE, scoring_size: int = 4096,
                     scoring_seed: int = 123, num_workers: int = NUM_WORKERS):
    set_seed(42)
    mount_drive()
    organize_tiny_imagenet_val(os.path.join(data_root, "val"))

    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomCrop(IMG_SIZE, padding=int(IMG_SIZE * 0.0625)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(TIN_MEAN, TIN_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(TIN_MEAN, TIN_STD),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=eval_tf)
    class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_val, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

    scoring_indices = _build_scoring_subset_indices(train_ds, size=scoring_size, seed=scoring_seed)
    scoring_subset = Subset(datasets.ImageFolder(os.path.join(data_root, "train"), transform=eval_tf),
                            scoring_indices)
    scoring_loader = DataLoader(scoring_subset, batch_size=batch_score, shuffle=False,
                               sampler=_FixedOrderSampler(list(range(len(scoring_subset)))),
                               num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, scoring_loader, class_names


# ============================================================================
# SECTION 3: MODELS
# ============================================================================

def build_resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    m = torchvision.models.resnet18(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)
    return m

def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[7:]] = v
        elif k.startswith("model."):
            out[k[6:]] = v
        else:
            out[k] = v
    return out

def load_teacher(teacher_ckpt: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    m = build_resnet18(num_classes).to(DEVICE)
    sd = torch.load(teacher_ckpt, map_location=DEVICE)
    sd = sd.get("state_dict", sd)
    sd = _strip_module_prefix(sd)
    m.load_state_dict(sd, strict=False)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m

def freeze(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False

def unfreeze(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


# ============================================================================
# SECTION 4: RANKER (Transformer)
# ============================================================================

from dataclasses import dataclass

@dataclass
class RankerConfig:
    d_model: int = 192
    nhead: int = 6
    num_layers: int = 3
    dim_ff: int = 384
    dropout: float = 0.1
    use_uncertainty: bool = False

class FeatureProjector(nn.Module):
    def __init__(self, in_dims: Dict[str, int], d_model: int):
        super().__init__()
        self.keys = list(in_dims.keys())
        self.proj = nn.ModuleDict({k: nn.Linear(in_dims[k], d_model) for k in self.keys})
        self.gates = nn.ParameterDict({k: nn.Parameter(torch.ones(1)) for k in self.keys})
        self.norm = nn.LayerNorm(d_model)
    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        xs = []
        for k in self.keys:
            if k in feats and feats[k] is not None:
                x = self.proj[k](feats[k]) * self.gates[k]
                xs.append(x)
        x = torch.stack(xs, dim=0).sum(dim=0)
        return self.norm(x)

class PositionalStructuralEmbedding(nn.Module):
    def __init__(self, d_model: int, max_depth: int = 64, max_blocks: int = 64, num_branches: int = 3, num_stages: int = 5):
        super().__init__()
        self.depth = nn.Embedding(max_depth, d_model)
        self.block = nn.Embedding(max_blocks, d_model)
        self.branch = nn.Embedding(num_branches, d_model)
        self.stage = nn.Embedding(num_stages, d_model)
        self.gamma = nn.Parameter(torch.ones(4))
        self.norm = nn.LayerNorm(d_model)
    def forward(self, depth_idx, block_idx, branch_idx, stage_idx):
        e = self.gamma[0]*self.depth(depth_idx) + self.gamma[1]*self.block(block_idx) + \
            self.gamma[2]*self.branch(branch_idx) + self.gamma[3]*self.stage(stage_idx)
        return self.norm(e)

class TransformerRanker(nn.Module):
    def __init__(self, ranker_cfg: RankerConfig, in_dims: Dict[str, int], max_depth: int = 64,
                 max_blocks: int = 64, num_branches: int = 3, num_stages: int = 5):
        super().__init__()
        self.cfg = ranker_cfg
        self.feat_proj = FeatureProjector(in_dims, ranker_cfg.d_model)
        self.struct_emb = PositionalStructuralEmbedding(ranker_cfg.d_model, max_depth, max_blocks, num_branches, num_stages)
        enc_layer = nn.TransformerEncoderLayer(d_model=ranker_cfg.d_model, nhead=ranker_cfg.nhead,
                                               dim_feedforward=ranker_cfg.dim_ff, dropout=ranker_cfg.dropout,
                                               batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=ranker_cfg.num_layers)
        self.head = nn.Sequential(nn.LayerNorm(ranker_cfg.d_model), nn.Linear(ranker_cfg.d_model, ranker_cfg.d_model),
                                 nn.GELU(), nn.Linear(ranker_cfg.d_model, 1))
        self.unc_head = nn.Sequential(nn.LayerNorm(ranker_cfg.d_model), nn.Linear(ranker_cfg.d_model, 1)) \
                        if ranker_cfg.use_uncertainty else None
        self.global_token = nn.Parameter(torch.zeros(1, 1, ranker_cfg.d_model))
        nn.init.trunc_normal_(self.global_token, std=0.02)

    def forward(self, feats: Dict[str, torch.Tensor], depth_idx, block_idx, branch_idx, stage_idx, global_feat=None):
        x = self.feat_proj(feats)
        s = self.struct_emb(depth_idx, block_idx, branch_idx, stage_idx)
        x = x + s
        x = x.unsqueeze(0)
        if global_feat is not None:
            g = global_feat
        else:
            B = x.shape[0]
            g = self.global_token.expand(B, -1, -1).squeeze(1)
        seq = torch.cat([g.unsqueeze(1), x], dim=1)
        out = self.encoder(seq)
        ch_tokens = out[:, 1:, :]
        scores = self.head(ch_tokens).squeeze(-1)
        unc = self.unc_head(ch_tokens).squeeze(-1) if self.unc_head is not None else None
        return scores, unc

class RankerAPI:
    def __init__(self, in_dims: Dict[str, int], cfg: RankerConfig = RankerConfig(),
                 max_depth: int = 64, max_blocks: int = 64, num_branches: int = 3, num_stages: int = 5):
        self.model = TransformerRanker(cfg, in_dims, max_depth, max_blocks, num_branches, num_stages).to(DEVICE)
        self.cfg = cfg


# ============================================================================
# SECTION 5: AZ-NAS SCORE
# ============================================================================

def kaiming_normal_fanin_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    return model

class _ResNet18FeatureGrabber:
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self.features = []

    def _hook_stem(self, module, inp, out):
        self.features.append(out)

    def _hook_block(self, module, inp, out):
        self.features.append(out)

    def register(self):
        self.clear()
        self.handles.append(self.model.relu.register_forward_hook(self._hook_stem))
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(self.model, layer_name)
            for block in layer:
                self.handles.append(block.register_forward_hook(self._hook_block))
        return self

    def get_features(self):
        return self.features

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.features = []

def _conv2d_flops(m: nn.Conv2d, x_shape, y_shape):
    Cin, Hin, Win = x_shape
    Cout, Hout, Wout = y_shape
    kH, kW = m.kernel_size
    groups = m.groups
    macs = Cout * Hout * Wout * (Cin // groups) * kH * kW
    flops = 2 * macs
    if m.bias is not None:
        flops += Cout * Hout * Wout
    return int(flops)

def _linear_flops(m: nn.Linear):
    macs = m.in_features * m.out_features
    flops = 2 * macs + (m.out_features if m.bias is not None else 0)
    return int(flops)

@torch.no_grad()
def estimate_flops_resnet18(model: nn.Module, resolution: int = 224, device=None) -> int:
    model = model.to(device or next(model.parameters()).device)
    model.eval()
    flops = 0
    handles = []

    def hook(m, x, y):
        nonlocal flops
        x0 = x[0]
        y0 = y if not isinstance(y, (list, tuple)) else y[0]
        if isinstance(m, nn.Conv2d):
            xs = tuple(x0.shape[-3:])
            ys = tuple(y0.shape[-3:])
            flops += _conv2d_flops(m, xs, ys)
        elif isinstance(m, nn.Linear):
            flops += _linear_flops(m)

    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            handles.append(mod.register_forward_hook(hook))

    dummy = torch.zeros(1, 3, resolution, resolution, device=next(model.parameters()).device)
    _ = model(dummy)

    for h in handles:
        h.remove()

    return int(flops)

def compute_aznas_score_resnet18(model: nn.Module, gpu: int = None, trainloader = None,
                                 resolution: int = 224, batch_size: int = 96, fp16: bool = False, init: bool = True):
    model.train()
    if gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    dtype = torch.half if fp16 else torch.float32
    if init:
        init_model(model, 'kaiming_norm_fanin')

    if trainloader is None:
        inputs = torch.randn(batch_size, 3, resolution, resolution, device=device, dtype=dtype, requires_grad=True)
        targets = None
    else:
        xb, yb = next(iter(trainloader))
        inputs = xb.to(device=device, dtype=dtype)
        targets = yb.to(device=device)

    grabber = _ResNet18FeatureGrabber(model).register()

    outputs = model(inputs)
    if targets is not None and outputs.shape[-1] >= int(targets.max().item()) + 1:
        loss = nn.functional.cross_entropy(outputs.float(), targets.long())
    else:
        loss = (outputs.float() ** 2).mean()

    layer_features = grabber.get_features()
    if len(layer_features) < 2:
        layer_features = [outputs]

    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)

    # Expressivity & Progressivity
    expressivity_scores = []
    with torch.no_grad():
        for feat in layer_features:
            b, c, h, w = feat.shape
            X = feat.detach().permute(0, 2, 3, 1).reshape(b * h * w, c)
            mu = X.mean(dim=0, keepdim=True)
            Xc = X - mu
            sigma = (Xc.T @ Xc) / max(1, Xc.shape[0])
            s = torch.linalg.eigvalsh(sigma)
            s = torch.relu(s) + 1e-12
            p = s / s.sum()
            exp_score = (-p * torch.log(p)).sum().item()
            expressivity_scores.append(exp_score)

    expressivity_scores = np.array(expressivity_scores, dtype=np.float64)
    expressivity = float(np.nansum(expressivity_scores))
    if len(expressivity_scores) >= 2:
        progressivity = float(np.nanmin(expressivity_scores[1:] - expressivity_scores[:-1]))
    else:
        progressivity = float('-inf')

    # Trainability
    train_scores = []
    for i in reversed(range(1, len(layer_features))):
        f_out = layer_features[i]
        f_in = layer_features[i - 1]

        g_out = torch.ones_like(f_out, dtype=f_out.dtype, device=f_out.device) * 0.5
        g_out = (torch.bernoulli(g_out) - 0.5) * 2

        g_in = torch.autograd.grad(outputs=f_out, inputs=f_in, grad_outputs=g_out,
                                   retain_graph=True, allow_unused=True)[0]

        if g_in is None:
            train_scores.append(-np.inf)
            continue

        if g_out.shape[2:] != g_in.shape[2:]:
            ho, wo = g_out.shape[2:]
            hi, wi = g_in.shape[2:]
            stride_h = max(1, hi // ho)
            stride_w = max(1, wi // wo)
            if stride_h == stride_w:
                pixel_unshuffle = nn.PixelUnshuffle(stride_h)
                g_in = pixel_unshuffle(g_in)
            else:
                g_in = nn.functional.adaptive_avg_pool2d(g_in, (ho, wo))

        bo, co, ho, wo = g_out.shape
        bi, ci, hi, wi = g_in.shape

        Go = g_out.permute(0, 2, 3, 1).contiguous().view(bo * ho * wo, co)
        Gi = g_in.permute(0, 2, 3, 1).contiguous().view(bi * hi * wi, ci)
        M = (Gi.T @ Go) / max(1, (bo * ho * wo))

        if M.shape[0] < M.shape[1]:
            M = M.T
        svals = torch.linalg.svdvals(M)
        smax = svals.max().item() if svals.numel() > 0 else 0.0
        train_scores.append(-smax - 1.0 / (smax + 1e-6) + 2.0)

    trainability = float(np.mean(train_scores)) if len(train_scores) else float('-inf')

    # Complexity
    complexity = float(estimate_flops_resnet18(model, resolution=resolution, device=device))

    grabber.clear()
    model.zero_grad(set_to_none=True)

    info = {
        "expressivity": expressivity if not np.isnan(expressivity) else float('-inf'),
        "progressivity": progressivity if not np.isnan(progressivity) else float('-inf'),
        "trainability": trainability if not np.isnan(trainability) else float('-inf'),
        "complexity": complexity,
    }
    return info


# ============================================================================
# SECTION 6: PIPELINE UTILITIES
# ============================================================================

def _expand_out_mask_for_weight(weight: torch.Tensor, out_mask: torch.Tensor) -> torch.Tensor:
    m = out_mask.to(weight.device, dtype=weight.dtype)
    m = m.view(-1, 1, 1, 1)
    return m.expand_as(weight)

@contextmanager
def apply_temp_outmask_conv(conv: nn.Conv2d, out_mask: torch.Tensor):
    W = conv.weight
    orig = W.data
    try:
        temp = orig.clone()
        M = _expand_out_mask_for_weight(temp, out_mask)
        temp.mul_(M)
        conv.weight = nn.Parameter(temp, requires_grad=W.requires_grad)
        yield
    finally:
        conv.weight = nn.Parameter(orig, requires_grad=W.requires_grad)

def build_channel_features_for_layer(conv: nn.Conv2d, bn: nn.BatchNorm2d = None) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        W = conv.weight.detach()
        C_out = W.shape[0]

        w_l2 = W.view(C_out, -1).pow(2).sum(dim=1).sqrt().unsqueeze(1)
        feats = {"w_l2": w_l2.to(DEVICE)}

        if bn is not None and bn.affine:
            gamma = bn.weight.detach().view(-1, 1)
            beta = bn.bias.detach().view(-1, 1)
            feats["bn_gamma"] = gamma.to(DEVICE)
            feats["bn_beta"] = beta.to(DEVICE)

        if conv.bias is not None:
            b = conv.bias.detach().view(-1, 1)
            feats["bias"] = b.to(DEVICE)
    return feats

def get_conv_bn_pairs(model: nn.Module) -> List[Tuple[nn.Conv2d, nn.BatchNorm2d]]:
    pairs = []
    pairs.append((model.conv1, model.bn1))
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for block in layer:
            pairs.append((block.conv1, block.bn1))
            pairs.append((block.conv2, block.bn2))
    return pairs

def assemble_aznas_rank_score(candidate_infos: List[Dict], complexity_weight: float = 2.0) -> List[float]:
    if not candidate_infos:
        return []

    exp = torch.tensor([ci.get("expressivity", float("-inf")) for ci in candidate_infos], dtype=torch.float64)
    prog = torch.tensor([ci.get("progressivity", float("-inf")) for ci in candidate_infos], dtype=torch.float64)
    trn = torch.tensor([ci.get("trainability", float("-inf")) for ci in candidate_infos], dtype=torch.float64)
    comp = torch.tensor([ci.get("complexity", float("inf")) for ci in candidate_infos], dtype=torch.float64)

    def _ranks(arr: torch.Tensor, descending: bool = True) -> torch.Tensor:
        vals = arr.clone()
        if descending:
            order = torch.argsort(vals, dim=0, descending=True)
        else:
            order = torch.argsort(vals, dim=0, descending=False)
        ranks = torch.empty_like(order, dtype=torch.float64)
        ranks[order] = torch.arange(1, len(arr) + 1, dtype=torch.float64)
        return ranks

    r_exp = _ranks(exp, descending=True)
    r_prog = _ranks(prog, descending=True)
    r_trn = _ranks(trn, descending=True)
    r_comp = _ranks(comp, descending=False)

    final = r_exp + r_prog + r_trn + complexity_weight * r_comp
    return final.tolist()


# ============================================================================
# SECTION 7: TRAINING LOGIC
# ============================================================================

def extract_layer_metadata(layer_idx: int, total_layers: int) -> Dict[str, int]:
    if layer_idx == 0:
        stage_id = 0
    elif layer_idx <= 4:
        stage_id = 1
    elif layer_idx <= 8:
        stage_id = 2
    elif layer_idx <= 12:
        stage_id = 3
    else:
        stage_id = 4

    return {
        "depth": layer_idx,
        "block_id": layer_idx // 2,
        "branch_type": layer_idx % 2,
        "stage_id": stage_id
    }

def train_encoder_phase1(student: nn.Module, train_loader, scoring_loader, ranker: RankerAPI,
                         epochs: int = 8, sample_layers_per_iter: int = 2, lr: float = 1e-3,
                         weight_decay: float = 1e-4, candidate_ratios: List[float] = [0.6, 0.8],
                         complexity_weight: float = 2.0, save_path: str = "/content/encoder_aznas_phase1.pth",
                         max_batches_per_epoch: int = 15):

    print("\n" + "="*80)
    print("Phase 1: Training Encoder with AZ-NAS Loss")
    print("="*80)

    freeze(student)
    student.eval()
    print(f"Student frozen and in eval mode")

    conv_bn_pairs = get_conv_bn_pairs(student)
    print(f"Found {len(conv_bn_pairs)} conv-bn pairs")

    optimizer = torch.optim.AdamW(ranker.model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history = []

    for epoch in range(epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*80}")

        ranker.model.train()
        epoch_losses = []

        for batch_idx, (x, y) in enumerate(train_loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break

            num_layers_to_sample = random.randint(3, sample_layers_per_iter) if sample_layers_per_iter > 3 else sample_layers_per_iter
            sampled_indices = random.sample(range(len(conv_bn_pairs)), num_layers_to_sample)

            batch_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

            for layer_idx in sampled_indices:
                conv, bn = conv_bn_pairs[layer_idx]
                n_channels = conv.out_channels

                features = build_channel_features_for_layer(conv, bn)
                metadata = extract_layer_metadata(layer_idx, len(conv_bn_pairs))

                ranker.model.train()
                feats_dict = features
                depth_idx = torch.full((n_channels,), metadata["depth"], dtype=torch.long, device=DEVICE)
                block_idx = torch.full((n_channels,), metadata["block_id"], dtype=torch.long, device=DEVICE)
                branch_idx = torch.full((n_channels,), metadata["branch_type"], dtype=torch.long, device=DEVICE)
                stage_idx = torch.full((n_channels,), metadata["stage_id"], dtype=torch.long, device=DEVICE)

                scores, _ = ranker.model(feats_dict, depth_idx, block_idx, branch_idx, stage_idx)
                scores = scores.squeeze(0)

                candidate_masks = []
                for ratio in candidate_ratios:
                    keep = max(8, int(ratio * n_channels))
                    order = torch.argsort(scores, descending=True)
                    mask = torch.zeros(n_channels, dtype=torch.bool, device=DEVICE)
                    mask[order[:keep]] = True
                    candidate_masks.append(mask)

                candidate_infos = []
                for mask in candidate_masks:
                    with apply_temp_outmask_conv(conv, mask):
                        unfreeze(student)
                        aznas_info = compute_aznas_score_resnet18(model=student, trainloader=scoring_loader,
                                                                  resolution=IMG_SIZE, init=False)
                        freeze(student)
                    candidate_infos.append(aznas_info)

                rank_scores = assemble_aznas_rank_score(candidate_infos, complexity_weight=complexity_weight)

                if rank_scores:
                    # Find best mask index (lowest rank score = best)
                    best_idx = rank_scores.index(min(rank_scores))

                    # Create differentiable loss:
                    # Compute weighted score for each mask based on encoder scores
                    mask_scores_list = []
                    for mask in candidate_masks:
                        # Sum of encoder scores for kept channels
                        mask_score = (scores * mask.float()).sum() / mask.float().sum()
                        mask_scores_list.append(mask_score)

                    # Stack into tensor and create cross-entropy loss
                    # We want the encoder to give highest total score to the best mask
                    mask_scores_tensor = torch.stack(mask_scores_list)
                    target = torch.tensor([best_idx], dtype=torch.long, device=DEVICE)
                    layer_loss = F.cross_entropy(mask_scores_tensor.unsqueeze(0), target)
                else:
                    layer_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

                batch_loss = batch_loss + layer_loss if isinstance(batch_loss, torch.Tensor) else layer_loss

            if len(sampled_indices) > 0:
                batch_loss = batch_loss / len(sampled_indices)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_losses.append(batch_loss.item())

            if (batch_idx + 1) % 5 == 0:
                avg_loss = sum(epoch_losses[-5:]) / len(epoch_losses[-5:])
                print(f"  [Epoch {epoch+1}][Batch {batch_idx+1}] Loss: {avg_loss:.4f} | Layers: {len(sampled_indices)}")

        epoch_avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        loss_history.append({"epoch": epoch + 1, "avg_loss": epoch_avg_loss, "num_batches": len(epoch_losses)})
        print(f"\nEpoch {epoch+1} Summary: Avg Loss = {epoch_avg_loss:.4f}")

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '/content', exist_ok=True)
    torch.save({"encoder": ranker.model.state_dict(), "config": ranker.cfg.__dict__,
                "loss_history": loss_history}, save_path)
    print(f"\n{'='*80}")
    print(f"Encoder saved to: {save_path}")
    print(f"{'='*80}\n")

    loss_json_path = save_path.replace(".pth", "_losses.json")
    with open(loss_json_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"Loss history saved to: {loss_json_path}\n")

    config_json_path = save_path.replace(".pth", "_config.json")
    config_dict = {"encoder_architecture": ranker.cfg.__dict__,
                   "feature_dimensions": {"w_l2": 1, "bn_gamma": 1, "bn_beta": 1},
                   "training_config": {"epochs": epochs, "lr": lr, "weight_decay": weight_decay,
                                      "sample_layers_per_iter": sample_layers_per_iter,
                                      "candidate_ratios": candidate_ratios,
                                      "complexity_weight": complexity_weight}}
    with open(config_json_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Architecture config saved to: {config_json_path}\n")

    return ranker, loss_history


# ============================================================================
# SECTION 8: MAIN
# ============================================================================

def main():
    print(f"Using device: {DEVICE}")
    print(f"Image size: {IMG_SIZE}")

    # Hardcoded paths for Colab - EVERYTHING in Google Drive
    DATA_ROOT = "/content/drive/MyDrive/CS242/Final_Presentation_Code/tiny-imagenet-200"
    TEACHER_CKPT = "/content/drive/MyDrive/CS242/Final_Presentation_Code/teacher.pth"
    SAVE_PATH = "/content/drive/MyDrive/CS242/Final_Presentation_Code/encoder_aznas_phase1.pth"

    print("\n⚠️ IMPORTANT: Make sure teacher.pth is in your Google Drive at:")
    print("  /content/drive/MyDrive/CS242/Final_Presentation_Code/teacher.pth")

    print("\nLoading data...")
    train_loader, val_loader, scoring_loader, class_names = make_dataloaders(
        data_root=DATA_ROOT, batch_train=256, batch_val=256, batch_score=256, num_workers=2
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Scoring batches: {len(scoring_loader)}")

    print(f"\nLoading teacher from: {TEACHER_CKPT}")
    student = load_teacher(TEACHER_CKPT, num_classes=len(class_names))
    print(f"Teacher loaded successfully")

    print("\nInitializing ranker...")
    in_dims = {"w_l2": 1, "bn_gamma": 1, "bn_beta": 1}
    ranker_cfg = RankerConfig(d_model=192, nhead=6, num_layers=3, dim_ff=384, dropout=0.1, use_uncertainty=False)
    ranker = RankerAPI(in_dims=in_dims, cfg=ranker_cfg, max_depth=64, max_blocks=64, num_branches=3, num_stages=5)
    print(f"Ranker initialized with {sum(p.numel() for p in ranker.model.parameters())/1e6:.2f}M parameters")

    print("\nStarting training...")
    print("⚡ Training: 3 epochs, 10 batches/epoch, 2 layers, ratios=[0.5, 0.6] (~1 hour)")
    trained_ranker, loss_history = train_encoder_phase1(
        student=student, train_loader=train_loader, scoring_loader=scoring_loader, ranker=ranker,
        epochs=3, sample_layers_per_iter=2, lr=1e-3, weight_decay=1e-4,
        candidate_ratios=[0.5, 0.6], complexity_weight=2.0, save_path=SAVE_PATH, max_batches_per_epoch=10
    )

    print("\n" + "="*80)
    print("Phase 1 Training Complete!")
    print("="*80)
    print(f"Final loss: {loss_history[-1]['avg_loss']:.4f}")
    print(f"Encoder saved to: {SAVE_PATH}")
    print("\nAll files saved to Google Drive at:")
    print("  - /content/drive/MyDrive/CS242/Final_Presentation_Code/encoder_aznas_phase1.pth")
    print("  - /content/drive/MyDrive/CS242/Final_Presentation_Code/encoder_aznas_phase1_losses.json")
    print("  - /content/drive/MyDrive/CS242/Final_Presentation_Code/encoder_aznas_phase1_config.json")
    print("\nThese will persist after session ends!")


if __name__ == "__main__":
    main()

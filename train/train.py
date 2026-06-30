"""
Train DeepLabV3+ (ResNet50) for Indian road segmentation.

Optimised for: NVIDIA RTX 4060 Laptop GPU (8 GB VRAM, Ada Lovelace Tensor Cores)

Usage:
    python train/train.py                          # RTX 4060 defaults (batch=4, epochs=40)
    python train/train.py --batch 2 --workers 2   # RTX 3050 / 4 GB VRAM fallback
    python train/train.py --resume models/road_seg.pth --epochs 20   # resume

The trained weights are saved to models/road_seg.pth and can be loaded
directly by pipeline/segmentation.py.

Loss      : BCE + Dice (50/50 blend) — handles class imbalance
            (roads are ~5–15% of pixels in typical satellite patches).
Precision : FP16 mixed precision via torch.autocast + GradScaler
            → ~1.8–2× speed-up on Ada Tensor Cores, ~50% VRAM saving.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

from train.dataset import IndianRoadDataset


# ─────────────────────────────────────────────────────────────────────────────
# Loss: BCE + Dice
# ─────────────────────────────────────────────────────────────────────────────

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss  = self.bce(logits, targets)
        dice_loss = self._dice(logits, targets)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

    @staticmethod
    def _dice(logits, targets, smooth: float = 1.0):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        tgts  = targets.view(-1)
        intersection = (probs * tgts).sum()
        return 1 - (2 * intersection + smooth) / (probs.sum() + tgts.sum() + smooth)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def iou_score(logits, targets, threshold: float = 0.45):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum() - inter
    return (inter + 1) / (union + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="  train", leave=False):
        images, masks = images.to(device), masks.to(device)

        # set_to_none=True releases gradient memory immediately (faster + less VRAM
        # than filling tensors with zero, especially with larger batch sizes).
        optimizer.zero_grad(set_to_none=True)

        # torch.autocast is the modern, non-deprecated AMP API (replaces
        # torch.cuda.amp.autocast). FP16 on CUDA leverages Ada Lovelace Tensor
        # Cores; falls back to FP32 transparently on CPU.
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0
    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images, masks = images.to(device), masks.to(device)
        # AMP during validation reduces memory for large batch sizes;
        # no scaler needed here because we never call .backward().
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, masks)
        total_loss += loss.item()
        total_iou  += iou_score(logits, masks).item()
    n = len(loader)
    return total_loss / n, total_iou / n


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train road segmentation model for Indian roads (optimised for RTX 4060 8 GB)"
    )
    parser.add_argument("--data",    default="data/indian_roads",   help="Dataset directory")
    # ── RTX 4060 defaults ────────────────────────────────────────────────────────
    # batch=4  : fits comfortably in 8 GB VRAM at 512×512 with FP16 AMP
    #            (use --batch 2 for RTX 3050 / 4 GB)
    # epochs=40: well-past convergence for ~500–1000 image datasets
    # workers=4: 4 CPU threads keep the GPU saturated without starving the OS
    # ─────────────────────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",  type=int,   default=40,   help="Number of epochs")
    parser.add_argument("--batch",   type=int,   default=4,    help="Batch size (use 2 for 4 GB VRAM)")
    parser.add_argument("--lr",      type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--workers", type=int,   default=4,    help="DataLoader worker threads")
    parser.add_argument("--resume",  default=None,              help="Resume from checkpoint (.pth)")
    parser.add_argument("--output",  default="models/road_seg.pth", help="Output weights path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}")
        print(f"VRAM   : {props.total_memory / 1024**3:.1f} GB")
        print(f"CUDA   : {torch.version.cuda}")
        print(f"AMP    : FP16 enabled (Tensor Cores active)")
    else:
        print("AMP    : disabled (CPU run — FP32 only)")

    # Datasets
    train_ds = IndianRoadDataset(args.data, split="train")
    val_ds   = IndianRoadDataset(args.data, split="val")
    print(f"\nTrain  : {len(train_ds)} samples")
    print(f"Val    : {len(val_ds)} samples")
    print(f"Batch  : {args.batch}  |  Workers: {args.workers}  |  Epochs: {args.epochs}\n")

    # persistent_workers=True keeps worker processes alive between epochs,
    # eliminating the fork/join overhead (~0.5–1 s per epoch on Windows).
    # prefetch_factor=2 lets each worker pre-load 2 batches ahead so the GPU
    # never stalls waiting for the next batch.
    _pin  = device.type == "cuda"
    _pw   = args.workers > 0          # persistent_workers requires workers > 0
    _pf   = 2 if args.workers > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=_pin,
        persistent_workers=_pw,
        prefetch_factor=_pf,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=_pin,
        persistent_workers=_pw,
        prefetch_factor=_pf,
    )

    # Model
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",   # start from ImageNet pretrained weights
        in_channels=3,
        classes=1,
    ).to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine annealing — gradually reduces LR so model fine-tunes smoothly
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # torch.amp.GradScaler is the modern, non-deprecated API (torch ≥ 2.0).
    # It automatically scales FP16 gradients to prevent underflow, then
    # unscales before the optimiser step — transparent to the rest of the loop.
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))

    best_iou = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs…\n")

    for epoch in range(1, args.epochs + 1):
        train_loss           = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_iou    = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_IoU={val_iou:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), output_path)
            print(f"  ✅ Best model saved  (IoU={best_iou:.4f})")

    print(f"\nTraining complete.  Best val IoU: {best_iou:.4f}")
    print(f"Weights saved to: {output_path}")
    print("Run the app: streamlit run app/main.py")


if __name__ == "__main__":
    main()

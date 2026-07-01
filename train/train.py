"""
Train DeepLabV3+ (ResNet50) for Indian road segmentation.

Optimised for: NVIDIA RTX 4060 Laptop GPU (8 GB VRAM, Ada Lovelace Tensor Cores)

Usage:
    python train/train.py                          # RTX 4060 defaults (batch=4, epochs=40)
    python train/train.py --batch 2 --workers 2   # RTX 3050 / 4 GB VRAM fallback
    python train/train.py --resume models/road_seg.pth --epochs 20   # resume

The trained weights are saved to models/road_seg.pth and can be loaded
directly by pipeline/segmentation.py.

Loss      : Dice + Focal (combined) — designed for class-imbalanced road segmentation.
            Dice loss handles the road/background imbalance (~5-15% road pixels).
            Focal loss up-weights hard examples (roads hidden under trees/shadows)
            so the model cannot ignore them by predicting all-background.
Metrics   : Val Dice score used for best-model checkpointing (more robust than IoU
            for tiny, occluded road segments).
Scheduler : Dual schedule — CosineAnnealingLR drives the global LR curve;
            ReduceLROnPlateau halves LR after `patience` epochs of no val-loss
            improvement (safety net for noisy satellite data plateaus).
Norm      : ImageNet mean/std [0.485,0.456,0.406] / [0.229,0.224,0.225]
            applied via albumentations.Normalize in dataset.py.
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
# Loss: Dice + Focal (smp native implementations)
# ─────────────────────────────────────────────────────────────────────────────
#
# Why Dice + Focal for obscured roads?
# - DiceLoss: directly optimises the overlap between predicted and true road
#   pixels, handling the severe class imbalance (roads are ~5-15% of image)
#   without needing manual class weights.
# - FocalLoss: down-weights easy examples (clear background) and up-weights
#   hard examples (roads hidden under tree canopy, shadows, or cloud patches).
#   gamma=2 means a pixel with 90% confidence contributes ~1/100th the loss
#   of a pixel at 50% confidence, forcing the model to focus on hard cases.
# Combined: Dice stabilises training; Focal drives precision on hard examples.

def build_criterion():
    dice  = smp.losses.DiceLoss(mode="binary")
    focal = smp.losses.FocalLoss(mode="binary")
    def combined(logits, targets):
        return dice(logits, targets) + focal(logits, targets)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def iou_score(logits, targets, threshold: float = 0.45):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum() - inter
    return (inter + 1) / (union + 1)


def dice_score(logits, targets, threshold: float = 0.45):
    """Dice / F1 score — used for best-model checkpointing."""
    preds = (torch.sigmoid(logits) > threshold).long()
    tgts  = targets.long()
    # smp.metrics.f1_score expects (N, C, H, W) with reduce_labels=False
    tp, fp, fn, tn = smp.metrics.get_stats(
        preds, tgts, mode="binary", threshold=None
    )
    return smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()


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
    total_dice = 0.0
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
        total_dice += dice_score(logits, masks)
    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


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
    parser.add_argument("--epochs",   type=int,   default=40,   help="Number of epochs")
    parser.add_argument("--batch",    type=int,   default=4,    help="Batch size (use 2 for 4 GB VRAM)")
    parser.add_argument("--lr",       type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--workers",  type=int,   default=4,    help="DataLoader worker threads")
    parser.add_argument("--patience", type=int,   default=3,    help="ReduceLROnPlateau patience (epochs)")
    parser.add_argument("--resume",   default=None,              help="Resume from checkpoint (.pth)")
    parser.add_argument("--output",   default="models/road_seg.pth", help="Output weights path")
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

    criterion = build_criterion()
    print("Loss      : DiceLoss + FocalLoss (smp native, binary mode)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Dual LR schedule:
    # 1. CosineAnnealingLR — smoothly decays LR from initial to eta_min over all epochs.
    #    Prevents the optimizer overshooting fine-grained satellite road features.
    # 2. ReduceLROnPlateau — halves LR if val_loss doesn’t improve for `patience` epochs.
    #    Acts as a safety net when the model plateaus on noisy tree-occluded roads.
    #    Both schedulers step every epoch; Plateau fires only when loss stalls.
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",          # monitor val_loss (lower = better)
        factor=0.5,          # halve LR on plateau
        patience=args.patience,
        min_lr=1e-6,
        # verbose removed — was dropped in PyTorch 2.x
    )
    print(f"Scheduler : CosineAnnealingLR + ReduceLROnPlateau(patience={args.patience})")

    # torch.amp.GradScaler is the modern, non-deprecated API (torch ≥ 2.0).
    # It automatically scales FP16 gradients to prevent underflow, then
    # unscales before the optimiser step — transparent to the rest of the loop.
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))

    best_dice = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs…\n")

    for epoch in range(1, args.epochs + 1):
        train_loss                    = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_iou, val_dice   = validate(model, val_loader, criterion, device)

        # Step both schedulers every epoch
        cosine_sched.step()
        plateau_sched.step(val_loss)   # Plateau monitors val_loss; fires only on stall

        current_lr = cosine_sched.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_IoU={val_iou:.4f}  "
            f"val_Dice={val_dice:.4f}  "
            f"lr={current_lr:.2e}"
        )

        # Save best model based on val Dice score (more robust than IoU for
        # tiny, occluded road segments under tree canopy or cloud shadow)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), output_path)
            print(f"  ✅ Best model saved  (Dice={best_dice:.4f}, IoU={val_iou:.4f})")

    print(f"\nTraining complete.  Best val Dice: {best_dice:.4f}")
    print(f"Weights saved to: {output_path}")
    print("Run the app: streamlit run app/main.py")


if __name__ == "__main__":
    main()

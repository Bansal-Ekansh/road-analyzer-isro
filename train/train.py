"""
Train DeepLabV3+ (ResNet50) for Indian road segmentation.

Usage:
    python train/train.py --data data/indian_roads --epochs 25 --batch 8

The trained weights are saved to models/road_seg.pth and can be loaded
directly by pipeline/segmentation.py.

Loss: BCE + Dice (standard for road segmentation — handles class imbalance
      since roads are ~5–15% of pixels in typical satellite patches).
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
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
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
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
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
    parser = argparse.ArgumentParser(description="Train road segmentation model for Indian roads")
    parser.add_argument("--data",       default="data/indian_roads", help="Dataset directory")
    parser.add_argument("--epochs",     type=int,   default=25,    help="Number of epochs")
    parser.add_argument("--batch",      type=int,   default=8,     help="Batch size")
    parser.add_argument("--lr",         type=float, default=3e-4,  help="Initial learning rate")
    parser.add_argument("--workers",    type=int,   default=2,     help="DataLoader workers")
    parser.add_argument("--resume",     default=None,               help="Resume from checkpoint")
    parser.add_argument("--output",     default="models/road_seg.pth", help="Output weights path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Datasets
    train_ds = IndianRoadDataset(args.data, split="train")
    val_ds   = IndianRoadDataset(args.data, split="val")
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
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

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

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

"""
PyTorch Dataset for road segmentation with Indian-specific augmentations.
"""

import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    """
    Augmentation pipeline tuned for Indian satellite imagery challenges:
    - Colour shift handles laterite (red) roads, concrete (white), asphalt (black)
    - RandomShadow simulates tree canopy cover
    - Occlusion patches simulate buildings / cloud shadow
    - Fog simulates haze/pollution common over Indian cities
    """
    return A.Compose([
        A.RandomResizedCrop(512, 512, scale=(0.7, 1.0), ratio=(0.8, 1.2), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Colour augmentations — cover laterite (reddish), concrete (bright), asphalt (dark)
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=10, b_shift_limit=10, p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=1.0),
        ], p=0.8),

        # Simulate tree canopy / shadow over roads
        A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, p=0.4),

        # Simulate haze/pollution over dense Indian cities
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.25, alpha_coef=0.08, p=0.3),

        # Simulate cloud patches / construction occlusion
        A.CoarseDropout(
            num_holes_range=(1, 5),
            hole_height_range=(20, 60),
            hole_width_range=(20, 60),
            fill=0, p=0.3,
        ),

        # Simulate JPEG compression artefacts in low-res Bhuvan tiles
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),

        # Blur — simulates motion blur in satellite or resampling artefacts
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.2),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


class IndianRoadDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", val_fraction: float = 0.15):
        data_dir  = Path(data_dir)
        img_paths = sorted((data_dir / "images").glob("*.png"))
        msk_paths = sorted((data_dir / "masks").glob("*.png"))

        assert len(img_paths) == len(msk_paths), (
            f"Image/mask count mismatch: {len(img_paths)} vs {len(msk_paths)}"
        )
        assert len(img_paths) > 0, f"No images found in {data_dir / 'images'}"

        n_val = max(1, int(len(img_paths) * val_fraction))
        if split == "train":
            self.img_paths = img_paths[n_val:]
            self.msk_paths = msk_paths[n_val:]
            self.transforms = get_train_transforms()
        else:
            self.img_paths = img_paths[:n_val]
            self.msk_paths = msk_paths[:n_val]
            self.transforms = get_val_transforms()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.img_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(str(self.msk_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask  = (mask > 127).astype(np.float32)

        aug   = self.transforms(image=image, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)

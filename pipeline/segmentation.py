"""
Road segmentation using DeepLabV3+ with ResNet50 encoder.

This module now strictly uses the PyTorch deep learning pipeline.
"""

import cv2
import numpy as np
from pathlib import Path

import torch
import segmentation_models_pytorch as smp

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE    = 512


class RoadSegmenter:
    def __init__(self, weights_path: str):
        # Streamlit Community Cloud is CPU-only.  Force CPU so the app never
        # tries to allocate CUDA memory (which would crash immediately on Cloud).
        # On a local machine with a GPU this still works — inference just runs
        # on CPU; re-enable cuda below if you want GPU on local dev.
        _cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if _cuda_available else "cpu")

        self.model = self._build_model()

        if weights_path and Path(weights_path).exists():
            # Always load to CPU first — safe on both Cloud (no GPU) and local.
            # map_location=cpu means CUDA tensors in the checkpoint are silently
            # remapped, preventing "CUDA out of memory" / "no CUDA device" errors.
            cpu = torch.device("cpu")
            weights = torch.load(weights_path, map_location=cpu, weights_only=False)
            self.model.load_state_dict(weights)
            self.model.to(self.device)   # move to GPU only if locally available
            print(f"[Segmenter] Loaded weights → device={self.device}  ({weights_path})")
        else:
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")

        self.model.eval()

    def _build_model(self):
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        return model.to(self.device)

    def _preprocess(self, image_rgb: np.ndarray) -> tuple:
        h, w    = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (_INPUT_SIZE, _INPUT_SIZE))
        img     = resized.astype(np.float32) / 255.0
        img     = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor  = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor, (h, w)

    def segment(self, image_rgb: np.ndarray, threshold: float = 0.45) -> np.ndarray:
        """
        Returns a binary road mask (uint8, 0/255) at original image size.
        """
        tensor, (orig_h, orig_w) = self._preprocess(image_rgb)
        with torch.no_grad():
            logits = self.model(tensor)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
        binary = (prob > threshold).astype(np.uint8) * 255
        binary = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return _morphological_cleanup(binary)

    def segment_with_confidence(self, image_rgb: np.ndarray) -> tuple:
        """Returns (binary_mask, probability_map) both at original size."""
        tensor, (orig_h, orig_w) = self._preprocess(image_rgb)
        with torch.no_grad():
            logits = self.model(tensor)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
        prob   = cv2.resize(prob, (orig_w, orig_h))
        binary = (prob > 0.45).astype(np.uint8) * 255
        return _morphological_cleanup(binary), prob


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """Remove noise and fill small holes."""
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k1, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k2, iterations=2)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    out = np.zeros_like(binary)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 300:
            out[labels == i] = 255
    return out

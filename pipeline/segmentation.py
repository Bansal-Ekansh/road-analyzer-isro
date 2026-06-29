"""
Road segmentation using DeepLabV3+ with ResNet50 encoder.

If torch / segmentation_models_pytorch are not installed, the class
automatically falls back to a colour-heuristic extractor so the app
keeps working without any ML dependencies.
"""

import cv2
import numpy as np
from pathlib import Path


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE    = 512


class RoadSegmenter:
    def __init__(self, weights_path: str | None = None):
        self._has_dl = False
        self.trained = False
        self.model   = None
        self.device  = "cpu"

        try:
            import torch
            import segmentation_models_pytorch as smp   # noqa: F401

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model  = self._build_model()
            self._has_dl = True

            if weights_path and Path(weights_path).exists():
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.trained = True
                print(f"[Segmenter] Loaded weights from {weights_path}")
            else:
                print("[Segmenter] Using ImageNet encoder + heuristic blend.")

            self.model.eval()

        except (ImportError, RuntimeError, Exception) as e:
            print(
                f"[Segmenter] DL init failed ({type(e).__name__}: {e})\n"
                "            Using heuristic road extractor (no GPU needed).\n"
                "            To fix: pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            )

    # ------------------------------------------------------------------
    def _build_model(self):
        import segmentation_models_pytorch as smp
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        return model.to(self.device)

    # ------------------------------------------------------------------
    def _preprocess(self, image_rgb: np.ndarray) -> tuple:
        import torch
        h, w    = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (_INPUT_SIZE, _INPUT_SIZE))
        img     = resized.astype(np.float32) / 255.0
        img     = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor  = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor, (h, w)

    # ------------------------------------------------------------------
    def segment(self, image_rgb: np.ndarray, threshold: float = 0.45) -> np.ndarray:
        """
        Returns a binary road mask (uint8, 0/255) at original image size.
        Uses DeepLabV3+ only when fine-tuned weights are loaded; heuristic otherwise.
        Random (untrained) DL weights produce garbage masks, so we skip them.
        """
        if not self._has_dl or not self.trained:
            mask = _heuristic_road_mask(image_rgb)
            return _morphological_cleanup(mask)

        import torch
        tensor, (orig_h, orig_w) = self._preprocess(image_rgb)
        with torch.no_grad():
            logits = self.model(tensor)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
        binary = (prob > threshold).astype(np.uint8) * 255
        binary = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return _morphological_cleanup(binary)

    # ------------------------------------------------------------------
    def segment_with_confidence(self, image_rgb: np.ndarray) -> tuple:
        """Returns (binary_mask, probability_map) both at original size."""
        if not self._has_dl:
            mask = self.segment(image_rgb)
            prob = mask.astype(np.float32) / 255.0
            return mask, prob

        import torch
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

def _heuristic_road_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Colour/texture heuristic: roads are low-saturation, mid-brightness
    regions. Works on any satellite image without ML dependencies.
    """
    hsv  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    low_sat = hsv[:, :, 1] < 55
    mid_val = (hsv[:, :, 2] > 70) & (hsv[:, :, 2] < 210)
    lap     = cv2.Laplacian(gray, cv2.CV_64F)
    smooth  = np.abs(lap) < 20

    mask    = (low_sat & mid_val & smooth).astype(np.uint8) * 255

    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=2)
    return mask


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

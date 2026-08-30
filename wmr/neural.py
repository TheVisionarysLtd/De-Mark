"""Lightweight U-Net that segments the Gemini sparkle on any background.

Trained on synthetic composites of the REAL extracted sparkle (see dev/ml/).
Inference is optional: if torch or the weights are missing, callers fall back to
the classical detector. Runs on CPU.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = Path(__file__).resolve().parent / "weights" / "sparkle_unet.pt"
INPUT = 192                 # model input side
ROI_RIGHT = 0.45           # inference region: right 45% x bottom 35% of the frame
ROI_BOTTOM = 0.35

_model = None
_unavailable = False


def torch_available() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False


def weights_available() -> bool:
    return WEIGHTS.exists()


def neural_available() -> bool:
    return torch_available() and weights_available()


# --- architecture ------------------------------------------------------------
def build_model():
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class TinyUNet(nn.Module):
        def __init__(self, base=16):
            super().__init__()
            self.e1 = block(3, base)
            self.e2 = block(base, base * 2)
            self.e3 = block(base * 2, base * 4)
            self.bott = block(base * 4, base * 8)
            self.pool = nn.MaxPool2d(2)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.d3 = block(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.d2 = block(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.d1 = block(base * 2, base)
            self.out = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            import torch
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            b = self.bott(self.pool(e3))
            d3 = self.d3(torch.cat([self.up3(b), e3], 1))
            d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
            return self.out(d1)

    return TinyUNet()


def _load():
    global _model, _unavailable
    if _model is not None:
        return _model
    if _unavailable or not neural_available():
        _unavailable = True
        return None
    try:
        import torch
        model = build_model()
        model.load_state_dict(torch.load(str(WEIGHTS), map_location="cpu"))
        model.eval()
        _model = model
        return model
    except Exception:
        _unavailable = True
        return None


def predict_full_mask(frame_bgr: np.ndarray, threshold: float = 0.5) -> np.ndarray | None:
    """Predict a full-frame watermark mask (0/255), or None if unavailable.

    Runs the net on the bottom-right region and places the result back.
    """
    model = _load()
    if model is None:
        return None
    import torch

    h, w = frame_bgr.shape[:2]
    rx, ry = int(w * (1 - ROI_RIGHT)), int(h * (1 - ROI_BOTTOM))
    roi = frame_bgr[ry:, rx:]
    rgb = cv2.cvtColor(cv2.resize(roi, (INPUT, INPUT)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1))[None]
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    prob = cv2.resize(prob, (roi.shape[1], roi.shape[0]))
    roi_mask = (prob >= threshold).astype(np.uint8) * 255
    roi_mask = _drop_small(roi_mask, min_frac=0.0006)  # kill stray false-positive specks
    if roi_mask.any():
        # grow toward full coverage of the mark — the net often lands on part of a
        # faint sparkle, and the inpainter fills any overshoot cleanly, so it's
        # safe to be generous.
        grow = _odd(max(15, int(min(roi.shape[:2]) * 0.08)))
        roi_mask = cv2.dilate(roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    full = np.zeros((h, w), np.uint8)
    full[ry:, rx:] = roi_mask
    return full


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _drop_small(mask: np.ndarray, min_frac: float) -> np.ndarray:
    """Remove connected components smaller than ``min_frac`` of the region area."""
    if not mask.any():
        return mask
    min_area = max(40, int(mask.shape[0] * mask.shape[1] * min_frac))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            out[labels == label] = 255
    return out

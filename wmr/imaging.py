"""Image encode/decode helpers and the single-image processing entry point.

Alpha is preserved end-to-end: a transparent PNG keeps its alpha channel while
only the colour planes are inpainted.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import RemovalSettings
from .pipeline import mask_overlay, process_frame


def decode_image(data: bytes):
    """Decode image bytes to (bgr, alpha_or_None). Raises ValueError if invalid."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image — file may be corrupt or unsupported.")

    if img.ndim == 2:  # grayscale
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None
    if img.shape[2] == 4:  # BGRA
        return img[:, :, :3].copy(), img[:, :, 3].copy()
    return img[:, :, :3].copy(), None


def encode_png(bgr: np.ndarray, alpha: np.ndarray | None = None) -> bytes:
    """Encode a BGR(+alpha) array to lossless PNG bytes."""
    if alpha is not None:
        bgra = np.dstack([bgr, alpha])
        ok, buf = cv2.imencode(".png", bgra)
    else:
        ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed.")
    return buf.tobytes()


def process_image_bytes(data: bytes, settings: RemovalSettings) -> dict:
    """Full still-image job: decode -> detect -> inpaint -> encode.

    Returns before/after/overlay BGR arrays plus ready-to-download PNG bytes.
    """
    bgr, alpha = decode_image(data)
    cleaned, mask = process_frame(bgr, settings)
    overlay = mask_overlay(bgr, mask)
    return {
        "before": bgr,
        "after": cleaned,
        "overlay": overlay,
        "mask": mask,
        "alpha": alpha,
        "png": encode_png(cleaned, alpha),
        "size": (bgr.shape[1], bgr.shape[0]),
    }

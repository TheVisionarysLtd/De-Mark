"""Inpainting backends.

Two tiers:

* **OpenCV** (always available) — Navier-Stokes or Telea fills. Fast and clean
  for the small regions our watermarks occupy; the default for video.
* **LaMa** (optional) — ``simple-lama-inpainting`` (deep learning). Higher
  fidelity on textured / structured backgrounds; used for stills when present.

LaMa is imported lazily and every call is guarded so the app degrades to OpenCV
if the package or its model weights are unavailable.
"""

from __future__ import annotations

import importlib.util

import cv2
import numpy as np

_lama_model = None       # cached SimpleLama instance
_lama_broken = False     # set once if construction/inference fails


def lama_available() -> bool:
    """True if the optional LaMa package is importable (weights load lazily)."""
    try:
        return importlib.util.find_spec("simple_lama_inpainting") is not None
    except Exception:
        return False


def resolve_backend(backend: str) -> str:
    """Map ``auto`` to the best available concrete backend."""
    if backend == "auto":
        return "lama" if lama_available() else "opencv_ns"
    return backend


def _get_lama():
    """Lazily build and cache the LaMa model; returns None if unavailable."""
    global _lama_model, _lama_broken
    if _lama_model is not None:
        return _lama_model
    if _lama_broken:
        return None
    try:
        from simple_lama_inpainting import SimpleLama

        _lama_model = SimpleLama()
        return _lama_model
    except Exception:
        _lama_broken = True
        return None


def _inpaint_lama(image_bgr: np.ndarray, mask: np.ndarray):
    """LaMa fill. Returns BGR array on success, or None to signal fallback."""
    model = _get_lama()
    if model is None:
        return None
    try:
        from PIL import Image

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        binary = (mask > 0).astype(np.uint8) * 255
        result = model(Image.fromarray(rgb), Image.fromarray(binary).convert("L"))

        out = np.asarray(result)[:, :, :3]
        h, w = image_bgr.shape[:2]
        if out.shape[:2] != (h, w):  # LaMa pads to a multiple of 8
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def inpaint(image_bgr: np.ndarray, mask: np.ndarray, backend: str = "auto",
            radius: int = 4) -> np.ndarray:
    """Fill the masked region of ``image_bgr`` and return a new BGR array.

    Never mutates the input. Falls back to OpenCV Navier-Stokes if a requested
    LaMa backend cannot run, so a result is always produced.
    """
    if mask is None or not mask.any():
        return image_bgr.copy()

    resolved = resolve_backend(backend)
    if resolved == "lama":
        filled = _inpaint_lama(image_bgr, mask)
        if filled is not None:
            return filled
        resolved = "opencv_ns"  # graceful degradation

    flag = cv2.INPAINT_TELEA if resolved == "opencv_telea" else cv2.INPAINT_NS
    binary = (mask > 0).astype(np.uint8) * 255
    return cv2.inpaint(image_bgr, binary, radius, flag)

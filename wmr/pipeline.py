"""Frame-level pipeline shared by the image and video paths.

Everything here works on BGR ``uint8`` numpy arrays (OpenCV convention). The
video reader converts RGB->BGR at its boundary so there is exactly one code
path for masking and inpainting.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import neural, sparkle
from .config import FEATHER_SIGMA, INPAINT_EXPAND_PX, RemovalSettings
from .inpaint import inpaint as inpaint_region
from .mask import detect_badge_mask, detect_core_mask, pad_mask
from .roi import Rect, compute_manual_roi, compute_roi


def _use_sparkle(settings: RemovalSettings) -> bool:
    """Deterministic sparkle matcher — the default for auto/corner mode."""
    return (settings.region_mode == "corner"
            and settings.detector in ("auto", "neural")
            and sparkle.sparkle_available())


def _use_neural(settings: RemovalSettings) -> bool:
    """Learned detector — opt-in only ('neural'); adds recall on faint marks."""
    return (settings.region_mode == "corner"
            and settings.detector == "neural"
            and neural.neural_available())


def resolve_roi(frame_w: int, frame_h: int, settings: RemovalSettings) -> Rect:
    """The active ROI: user-placed box in manual mode, else the bottom-right corner."""
    if settings.region_mode == "manual":
        return compute_manual_roi(frame_w, frame_h, settings.center_x, settings.center_y,
                                  settings.box_w, settings.box_h)
    return compute_roi(frame_w, frame_h, settings.bottom_fraction, settings.right_fraction)


def detect_roi_core(roi_bgr: np.ndarray, settings: RemovalSettings) -> np.ndarray:
    """Unpadded watermark mask for one ROI crop, honouring manual overrides.

    Manual ``force_fill`` inpaints the whole box (the reliable last resort for a
    mark auto-detection can't segment); otherwise the normal detector runs inside
    whatever region (corner or manual box) is active. In manual mode the user has
    already localised the mark by drawing the box, so if the detector finds
    nothing inside it we fill the whole box rather than leaving it untouched —
    the box IS the instruction to remove something here.
    """
    if settings.region_mode == "manual" and settings.force_fill:
        return np.full(roi_bgr.shape[:2], 255, np.uint8)
    core = detect_core_mask(roi_bgr, settings.sensitivity)
    if settings.region_mode == "manual" and not core.any():
        return np.full(roi_bgr.shape[:2], 255, np.uint8)
    return core


def build_frame_mask(frame_bgr: np.ndarray, settings: RemovalSettings) -> np.ndarray:
    """Return a full-frame padded uint8 mask (0/255); non-zero only in the ROI."""
    h, w = frame_bgr.shape[:2]

    # Corner mode with a smart detector (the default): union the deterministic
    # sparkle matcher (works on any palette) with the badge template matcher
    # (fixed 'Gemini Notebook' logo). Together they cover both watermark types.
    # The learned net is opt-in ('neural') and only adds recall. There is NO
    # classical fallback here: it fired on watermark-free frames (e.g. plain slides
    # in a deck), causing false removals. If the smart pass finds nothing, nothing
    # is removed — the user picks "Select area" for anything it misses.
    if settings.region_mode == "corner" and settings.detector in ("auto", "neural"):
        combined = np.zeros((h, w), np.uint8)
        if _use_sparkle(settings):
            found = sparkle.locate_sparkle(frame_bgr, settings.sensitivity)
            if found is not None:
                combined = cv2.bitwise_or(combined, found)
        if _use_neural(settings):
            predicted = neural.predict_full_mask(frame_bgr)
            if predicted is not None:
                combined = cv2.bitwise_or(combined, predicted)
        badge = detect_badge_mask(frame_bgr)
        if badge is not None:
            combined = cv2.bitwise_or(combined, badge)
        return pad_mask(combined, settings.padding_px) if combined.any() else combined

    # Classical corner detector (detector='classic') or manual box mode.
    roi = resolve_roi(w, h, settings)
    roi_bgr = frame_bgr[roi.y:roi.y1, roi.x:roi.x1]
    core = pad_mask(detect_roi_core(roi_bgr, settings), settings.padding_px)

    full = np.zeros((h, w), np.uint8)
    full[roi.y:roi.y1, roi.x:roi.x1] = core
    return full


def apply_inpaint(frame_bgr: np.ndarray, full_mask: np.ndarray,
                  settings: RemovalSettings) -> np.ndarray:
    """Inpaint the masked region and return a new frame (input untouched).

    Only a padded bounding box around the mask is handed to the inpainter — this
    bounds cost (critical per-frame for video) while giving the fill enough
    surrounding context to reconstruct the background seamlessly.
    """
    if not full_mask.any():
        return frame_bgr.copy()

    ys, xs = np.nonzero(full_mask)
    h, w = frame_bgr.shape[:2]
    pad = INPAINT_EXPAND_PX
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + 1 + pad)

    crop = frame_bgr[y0:y1, x0:x1]
    crop_mask = full_mask[y0:y1, x0:x1]
    filled = inpaint_region(crop, crop_mask, settings.backend, settings.inpaint_radius)

    # Feathered composite: blend the fill over the original through a soft-edged
    # mask so there is no hard rectangular seam — only the watermark area changes,
    # and the transition into the surrounding texture is smooth.
    soft = cv2.GaussianBlur((crop_mask > 0).astype(np.float32), (0, 0), sigmaX=FEATHER_SIGMA)
    soft = np.clip(soft, 0.0, 1.0)[..., None]
    blended = soft * filled.astype(np.float32) + (1.0 - soft) * crop.astype(np.float32)

    out = frame_bgr.copy()
    out[y0:y1, x0:x1] = blended.astype(np.uint8)
    return out


def process_frame(frame_bgr: np.ndarray, settings: RemovalSettings):
    """Detect + inpaint in one call. Returns (cleaned_frame, full_mask)."""
    mask = build_frame_mask(frame_bgr, settings)
    cleaned = apply_inpaint(frame_bgr, mask, settings)
    return cleaned, mask


def mask_overlay(frame_bgr: np.ndarray, full_mask: np.ndarray,
                 color=(0, 0, 255), alpha: float = 0.45) -> np.ndarray:
    """Return a preview with the detected mask tinted over the frame."""
    overlay = frame_bgr.copy()
    overlay[full_mask > 0] = color
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0)


def overlay_with_roi(frame_bgr: np.ndarray, settings: RemovalSettings):
    """Preview showing the detected mask (red) and the active ROI box outline.

    Returns (overlay_bgr, full_mask).
    """
    mask = build_frame_mask(frame_bgr, settings)
    out = mask_overlay(frame_bgr, mask)
    h, w = frame_bgr.shape[:2]
    roi = resolve_roi(w, h, settings)
    thickness = max(2, w // 400)
    cv2.rectangle(out, (roi.x, roi.y), (roi.x1 - 1, roi.y1 - 1), (0, 200, 255), thickness)
    return out, mask

"""Deterministic Gemini-sparkle locator — works on any background palette.

The Gemini "sparkle" is a *fixed* glyph: a concave four-point star rendered as a
near-opaque (~88%) grey/white overlay in the bottom-right corner. Both the shape
and its defining optical trait are constant across every image Gemini produces —
so instead of *learning* to recognise it (a small U-Net generalised poorly
across palettes), we exploit what never changes:

1.  **Shape** — multi-scale template matching of the canonical glyph against two
    palette-invariant response maps: a local-brightness *top-hat* (finds the mark
    on dark/mid backgrounds) and a *whiteness* map ``V·(1-S)`` (finds it on light
    or colourful backgrounds). ``TM_CCOEFF_NORMED`` is contrast/-offset
    invariant, so the same template scores highly whether the corner is black,
    grey, or vivid.
2.  **Signature** — every candidate is then verified against the sparkle's tells:
    the covered pixels are near-grey (low saturation, because a ~88%-opaque white
    overlay desaturates whatever is beneath it), locally brighter than their
    immediate surround, and located in the bottom-right corner.

This is deterministic, explainable, needs no training data, and — crucially —
returns ``None`` when nothing passes verification, so a clean image (or a busy
one with no watermark) is never damaged by a false removal. Verification
thresholds relax with the ``sensitivity`` knob for faint marks over colour.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

import cv2
import numpy as np

CANON_PATH = Path(__file__).resolve().parent / "weights" / "sparkle_canon.npy"

# Each corner's search window as a fraction of the frame (generous; a corner
# prior narrows it). The mark almost always sits bottom-right, but some exports
# place it in another corner — so all four are searched.
SEARCH_FRAC = 0.45

# The four corners as (vertical edge, horizontal edge). Bottom-right first: it is
# by far the most common, so it wins ties.
_CORNERS = (("bottom", "right"), ("bottom", "left"), ("top", "right"), ("top", "left"))

# Glyph side as a fraction of the frame's shorter dimension. Real sparkles sit
# around 0.07; the spread covers small and large exports.
SCALE_FRACS = (0.035, 0.045, 0.055, 0.07, 0.09, 0.11, 0.13)

# Corner prior — a genuine mark's centre sits within this fraction of its corner.
# Deliberately generous (real samples land within ~0.08 of the corner) yet it
# rejects mid-frame texture matches.
CORNER_BAND = 0.22

_MAX_CANDIDATES_PER_MAP = 3   # top matches kept per (scale, map) before NMS
_ON_THR = 0.5                 # template alpha above this = the glyph's solid body
_RING_LO, _RING_HI = 0.02, 0.45  # template alpha band = the immediate surround

_canon: Optional[np.ndarray] = None
_loaded = False


class _Candidate(NamedTuple):
    corr: float
    x: int          # top-left within the search ROI
    y: int
    size: int
    template: np.ndarray


def _load_canon() -> Optional[np.ndarray]:
    global _canon, _loaded
    if not _loaded:
        _loaded = True
        try:
            _canon = np.load(str(CANON_PATH)).astype(np.float32) if CANON_PATH.exists() else None
        except Exception:
            _canon = None
    return _canon


def sparkle_available() -> bool:
    """True if the canonical template shipped with the package can be loaded."""
    return _load_canon() is not None


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _norm(a: np.ndarray) -> np.ndarray:
    """Min-max stretch to [0,1] float32 (flat input -> zeros)."""
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _response_maps(bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Two palette-invariant maps the sparkle stands out in regardless of colour."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    whiteness = (val / 255.0) * (1.0 - sat / 255.0)   # bright AND desaturated
    return {"tophat": _norm(tophat), "whiteness": _norm(whiteness)}


def _collect_candidates(maps: dict[str, np.ndarray], base: int,
                        roi_h: int, roi_w: int) -> list[_Candidate]:
    """Multi-scale, multi-map template matches with per-map non-max suppression."""
    canon = _load_canon()
    out: list[_Candidate] = []
    for frac in SCALE_FRACS:
        size = int(round(base * frac))
        if size < 12 or size >= min(roi_h, roi_w):
            continue
        template = cv2.resize(canon, (size, size), interpolation=cv2.INTER_AREA)
        for fmap in maps.values():
            result = cv2.matchTemplate(fmap, template, cv2.TM_CCOEFF_NORMED)
            for _ in range(_MAX_CANDIDATES_PER_MAP):
                _, max_val, _, loc = cv2.minMaxLoc(result)
                if max_val < 0.4:
                    break
                out.append(_Candidate(max_val, loc[0], loc[1], size, template))
                # suppress a neighbourhood so the next peak is a distinct location
                x0, y0 = max(0, loc[0] - size // 2), max(0, loc[1] - size // 2)
                result[y0:loc[1] + size // 2, x0:loc[0] + size // 2] = 0.0
    out.sort(key=lambda c: -c.corr)
    return out


def _dedupe(cands: list[_Candidate]) -> list[_Candidate]:
    """Drop candidates that overlap an already-kept (higher-corr) location."""
    kept: list[_Candidate] = []
    for c in cands:
        cx, cy = c.x + c.size / 2, c.y + c.size / 2
        if any(abs(cx - (k.x + k.size / 2)) < 0.5 * c.size
               and abs(cy - (k.y + k.size / 2)) < 0.5 * c.size for k in kept):
            continue
        kept.append(c)
    return kept


def _verify(roi_bgr: np.ndarray, cand: _Candidate, corr_min: float,
            sat_max: float) -> bool:
    """True if the candidate carries the sparkle's optical signature."""
    if cand.corr < corr_min:
        return False
    patch = roi_bgr[cand.y:cand.y + cand.size, cand.x:cand.x + cand.size]
    if patch.shape[:2] != (cand.size, cand.size):
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    on = cand.template > _ON_THR
    ring = (cand.template > _RING_LO) & (cand.template < _RING_HI)
    if not on.any() or not ring.any():
        return False

    s_on = float(sat[on].mean())
    if s_on > sat_max:                      # covered pixels must be near-grey
        return False
    # must stand out from its immediate surround (brighter, or greyer)
    d_val = float(val[on].mean() - val[ring].mean())
    d_sat = float(sat[ring].mean() - sat[on].mean())
    return d_val >= 6.0 or d_sat >= 15.0


def _corner_window(h: int, w: int, vedge: str, hedge: str):
    """Pixel bounds of a corner's search window and its centre-prior test."""
    x0 = 0 if hedge == "left" else int(w * (1 - SEARCH_FRAC))
    x1 = int(w * SEARCH_FRAC) if hedge == "left" else w
    y0 = 0 if vedge == "top" else int(h * (1 - SEARCH_FRAC))
    y1 = int(h * SEARCH_FRAC) if vedge == "top" else h

    def in_corner(cx: float, cy: float) -> bool:
        ok_x = cx <= CORNER_BAND if hedge == "left" else cx >= 1 - CORNER_BAND
        ok_y = cy <= CORNER_BAND if vedge == "top" else cy >= 1 - CORNER_BAND
        return ok_x and ok_y

    return x0, y0, x1, y1, in_corner


def _best_in_corner(frame_bgr: np.ndarray, bounds, corr_min: float,
                    sat_max: float) -> Optional[tuple]:
    """Highest-correlation verified candidate in one corner, or None.

    Returns (corr, abs_x, abs_y, size, template) in full-frame coordinates.
    """
    x0, y0, x1, y1, in_corner = bounds
    h, w = frame_bgr.shape[:2]
    roi = frame_bgr[y0:y1, x0:x1]
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 16 or roi_w < 16:
        return None

    maps = _response_maps(roi)
    candidates = _dedupe(_collect_candidates(maps, min(h, w), roi_h, roi_w))
    for cand in candidates:               # highest correlation first
        cx = (x0 + cand.x + cand.size / 2) / w
        cy = (y0 + cand.y + cand.size / 2) / h
        if not in_corner(cx, cy):
            continue
        if not _verify(roi, cand, corr_min, sat_max):
            continue
        return cand.corr, x0 + cand.x, y0 + cand.y, cand.size, cand.template
    return None


def locate_sparkle(frame_bgr: np.ndarray, sensitivity: float = 0.55,
                   grow_frac: float = 0.10) -> Optional[np.ndarray]:
    """Locate the Gemini sparkle in any corner; return a full-frame 0/255 mask.

    Searches all four corners (bottom-right is the common case and wins ties)
    and returns the mask for the highest-confidence verified hit, or None when
    nothing passes — so clean images are never touched.

    Args:
        frame_bgr: BGR image.
        sensitivity: 0..1. Higher relaxes the correlation and saturation gates
            to catch faint marks over colour (at a small false-positive risk).
        grow_frac: dilate the stamped glyph by this fraction of its size so the
            inpainter comfortably overshoots the soft edge.
    """
    canon = _load_canon()
    if canon is None or frame_bgr.size == 0:
        return None

    sens = float(np.clip(sensitivity, 0.0, 1.0))
    corr_min = 0.70 - 0.16 * sens          # 0.55 -> 0.61
    sat_max = 25.0 + 40.0 * sens           # 0.55 -> 47

    h, w = frame_bgr.shape[:2]
    best = None
    for vedge, hedge in _CORNERS:
        hit = _best_in_corner(frame_bgr, _corner_window(h, w, vedge, hedge),
                              corr_min, sat_max)
        if hit is not None and (best is None or hit[0] > best[0]):
            best = hit
    if best is None:
        return None

    _, ax, ay, size, template = best
    mask = np.zeros((h, w), np.uint8)
    stamp = (template > 0.30).astype(np.uint8) * 255
    mask[ay:ay + size, ax:ax + size][stamp > 0] = 255
    grow = _odd(max(3, int(size * grow_frac)))
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))

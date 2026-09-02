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

# Every confirmed Gemini sparkle sits in the BOTTOM-RIGHT corner, so that is the
# only region searched. (Searching all four corners was tried but only ever
# produced false positives on star-shaped gaps in title text elsewhere.)
_CORNERS = (("bottom", "right"),)

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
# The real sparkle is a bright LOCAL structure, so it lights up a top-hat filter.
# A flat bright region (e.g. a star-shaped gap between dark title letters on a
# light slide) has almost no top-hat response — this floor rejects those.
_TOPHAT_MIN = 9.0             # min mean top-hat (0-255) over the glyph pixels

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


def _verify(roi_bgr: np.ndarray, tophat: np.ndarray, cand: _Candidate,
            corr_min: float, sat_max: float) -> bool:
    """True if the candidate carries the sparkle's optical signature."""
    if cand.corr < corr_min:
        return False
    y, x, sz = cand.y, cand.x, cand.size
    patch = roi_bgr[y:y + sz, x:x + sz]
    th_patch = tophat[y:y + sz, x:x + sz]
    if patch.shape[:2] != (sz, sz) or th_patch.shape[:2] != (sz, sz):
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
    # Must be a bright LOCAL structure, not a flat bright patch: reject star-shaped
    # gaps in title text / plain light backgrounds, which have ~no top-hat response.
    if float(th_patch[on].mean()) < _TOPHAT_MIN:
        return False
    # ...and stand out from its immediate surround (brighter, or greyer).
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
    # Raw top-hat (0-255) for verifying a candidate is a real bright structure,
    # not a flat bright region. Kernel ~ a typical sparkle so the mark stands out.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    tk = _odd(max(9, int(min(h, w) * 0.06)))
    tophat = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tk, tk))
    ).astype(np.float32)

    candidates = _dedupe(_collect_candidates(maps, min(h, w), roi_h, roi_w))
    for cand in candidates:               # highest correlation first
        cx = (x0 + cand.x + cand.size / 2) / w
        cy = (y0 + cand.y + cand.size / 2) / h
        if not in_corner(cx, cy):
            continue
        if not _verify(roi, tophat, cand, corr_min, sat_max):
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
    # Real Gemini sparkles correlate ~0.76-0.94; incidental matches (star-shaped
    # gaps in title text, etc.) top out around 0.70. The floor sits between them
    # at the default and only drops for a user who deliberately raises sensitivity.
    corr_min = 0.82 - 0.16 * sens          # 0.55 -> 0.73
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


# --- Un-blend removal --------------------------------------------------------
# The Gemini sparkle is a SEMI-TRANSPARENT white glyph composited over the image
# (obs = a*255 + (1-a)*orig). So instead of inpainting (which paints a guess and
# smears busy backgrounds), we can RECOVER the original pixels by inverting the
# blend: orig = (obs - a*255) / (1 - a). That keeps the real detail that sat under
# the mark — the clean result you get from a purpose-built Gemini remover. We use
# the canonical glyph as the alpha shape and estimate the per-image opacity so the
# un-blended core blends into its surroundings (opacity shifts by output format).
_UNBLEND_MAX_RESIDUAL = 0.32   # if the un-blend still matches the star shape above
#                                this, it ghosted/over-removed -> inpaint instead.


def unblend_sparkle(frame_bgr: np.ndarray, sensitivity: float = 0.78,
                    grow: float = 1.15):
    """Recover the pixels under the sparkle. Returns (new_frame, box or None).

    ``box`` is ``(x, y, size)`` of the region touched (for the caller's mask /
    "changed" signal); ``None`` means no sparkle was found and the frame is
    returned unchanged.
    """
    canon = _load_canon()
    if canon is None or frame_bgr.size == 0:
        return frame_bgr, None
    h, w = frame_bgr.shape[:2]
    sens = float(np.clip(sensitivity, 0.0, 1.0))
    hit = _best_in_corner(frame_bgr, _corner_window(h, w, "bottom", "right"),
                          0.82 - 0.16 * sens, 25.0 + 40.0 * sens)
    if hit is None:
        return frame_bgr, None

    _, ax, ay, size, _ = hit
    sz = int(round(size * grow))
    ox = max(0, ax + (size - sz) // 2)
    oy = max(0, ay + (size - sz) // 2)
    sz = min(sz, w - ox, h - oy)
    if sz < 8:
        return frame_bgr, None

    a = cv2.resize(canon, (sz, sz), interpolation=cv2.INTER_CUBIC)
    a = cv2.GaussianBlur(a, (0, 0), max(0.5, sz * 0.02))
    amax = float(a.max()) or 1.0
    core = a > 0.5 * amax

    # surrounding-ring median (box excluded) — the target the core should blend into
    pad = max(4, sz // 4)
    ry0, rx0 = max(0, oy - pad), max(0, ox - pad)
    ry1, rx1 = min(h, oy + sz + pad), min(w, ox + sz + pad)
    ring_region = frame_bgr[ry0:ry1, rx0:rx1]
    keep = np.ones(ring_region.shape[:2], bool)
    keep[oy - ry0:oy - ry0 + sz, ox - rx0:ox - rx0 + sz] = False
    ring_med = np.median(ring_region[keep].reshape(-1, 3).astype(np.float32), axis=0)

    reg = frame_bgr[oy:oy + sz, ox:ox + sz].astype(np.float32)
    best_k, best_err = 0.5, 1e18
    for k in np.arange(0.35, 0.851, 0.05):
        al = np.clip(a * k, 0.0, 0.92)[..., None]
        ub = np.clip((reg - al * 255.0) / (1.0 - al), 0.0, 255.0)
        cm = np.median(ub[core], axis=0)
        err = float(np.abs(cm - ring_med).sum())
        if err < best_err:
            best_err, best_k = err, float(k)

    al = np.clip(a * best_k, 0.0, 0.92)[..., None]
    ub = np.clip((reg - al * 255.0) / (1.0 - al), 0.0, 255.0).astype(np.uint8)

    # SAFETY: un-blending only wins when it actually erases the star. Without
    # Gemini's exact alpha, a wrong opacity can leave a ghost or over-subtract into
    # a dark star. Measure how much the result still matches the sparkle SHAPE; if
    # a sparkle-shaped pattern remains, fall back to inpainting the box so the
    # output is never worse than the old behaviour.
    ub_gray = cv2.cvtColor(ub, cv2.COLOR_BGR2GRAY).astype(np.float32)
    af = a.astype(np.float32) - float(a.mean())
    gz = ub_gray - float(ub_gray.mean())
    denom = (float(np.linalg.norm(af)) * float(np.linalg.norm(gz))) or 1.0
    residual = abs(float((af * gz).sum() / denom))

    if residual < _UNBLEND_MAX_RESIDUAL:
        out = frame_bgr.copy()
        out[oy:oy + sz, ox:ox + sz] = ub          # clean recovery — keep it
    else:
        m = np.zeros((h, w), np.uint8)
        m[oy:oy + sz, ox:ox + sz] = ((a > 0.12).astype(np.uint8) * 255)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        out = cv2.inpaint(frame_bgr, m, 4, cv2.INPAINT_TELEA)
    return out, (ox, oy, sz)

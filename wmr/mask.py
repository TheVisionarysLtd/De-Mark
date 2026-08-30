"""Watermark mask detection inside the ROI.

Strategy: a morphological top-hat / black-hat pair isolates small bright *and*
small dark structures regardless of the underlying background brightness. That
matches both watermark styles we target:

* the bright white Gemini sparkle over mid-tone artwork, and
* the light/greyscale NotebookLM ("Gemini Notebook") badge.

The response image is thresholded (Otsu baseline, scaled by a sensitivity knob),
cleaned up, and tiny speckles are dropped. For video, per-frame masks are voted
across sampled frames so only the *persistent* logo survives (see
``votes_to_mask``), which removes flicker from transient bright background.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import (
    HALO_MIN_RADIUS,
    HALO_NEIGHBORHOOD_RATIO,
    MIN_TOPHAT_KERNEL,
    SPECK_MIN_AREA_RATIO,
    TOPHAT_KERNEL_RATIO,
)


def _odd(n: int) -> int:
    """Nearest odd integer >= n (morphological kernels want odd sizes)."""
    n = int(n)
    return n if n % 2 == 1 else n + 1


# Picture-frame border: near-white pixels forming a strip along the image edge.
# Many AI exports sit inside a white frame; it is desaturated+bright like a
# watermark, so it must be excluded or it floods the corner ROI.
_FRAME_V_MIN = 228   # near-white value
_FRAME_S_MAX = 32    # near-white saturation

# Raw top-hat response below this = no watermark present (guards flat ROIs).
_ABS_FLOOR = 10

# If the generic detector covers more than this fraction of the ROI it is
# over-firing on a busy background; fall back to shape (sparkle) matching only.
_MAX_WM_FRAC = 0.11


def _contrast_response(gray: np.ndarray, ignore: np.ndarray | None = None) -> np.ndarray:
    """Raw high-pass response (0-255) highlighting small bright/dark structures.

    Kept in raw top-hat units (not min-max normalised) so thresholding can use
    an absolute floor — a flat ROI with no watermark yields a near-zero response
    and detects nothing, instead of noise being stretched to full scale.
    ``ignore`` (e.g. the picture-frame border) is zeroed out.
    """
    h, w = gray.shape
    k = _odd(max(MIN_TOPHAT_KERNEL, int(round(min(h, w) * TOPHAT_KERNEL_RATIO))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    white_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    response = cv2.max(white_hat, black_hat)

    if ignore is not None:
        response[ignore > 0] = 0
    return cv2.GaussianBlur(response, (3, 3), 0)


def _frame_border_mask(roi_bgr: np.ndarray) -> np.ndarray:
    """Near-white regions touching the ROI's bottom/right edge (the image frame).

    The bottom and right edges of a bottom-right ROI *are* the image's outer
    edges, so a near-white component reaching them is the picture frame — not a
    watermark, which sits inset. An interior white mark never qualifies.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    near_white = ((hsv[:, :, 2] >= _FRAME_V_MIN) &
                  (hsv[:, :, 1] <= _FRAME_S_MAX)).astype(np.uint8) * 255
    if not near_white.any():
        return np.zeros(near_white.shape, np.uint8)

    h, w = near_white.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(near_white, connectivity=8)
    border = np.zeros_like(near_white)
    for label in range(1, count):
        x, y, bw, bh = stats[label, :4]
        if (y + bh) >= h or (x + bw) >= w:  # reaches the image's bottom/right edge
            border[labels == label] = 255
    return cv2.dilate(border, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def _sparkle_template(size: int) -> np.ndarray:
    """A filled four-pointed sparkle (concave star) as a float32 patch in [0,1]."""
    tmpl = np.zeros((size, size), np.float32)
    c = (size - 1) / 2.0
    r_out = size * 0.47
    r_in = r_out * 0.22
    pts = []
    for i in range(8):
        ang = np.deg2rad(i * 45)
        rad = r_out if i % 2 == 0 else r_in
        pts.append([c + rad * np.sin(ang), c - rad * np.cos(ang)])
    cv2.fillPoly(tmpl, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
    return tmpl


def _match_sparkle(gray: np.ndarray, ignore: np.ndarray, sensitivity: float) -> np.ndarray:
    """Locate the Gemini sparkle by its 4-point-star shape (multi-scale template).

    Shape matching finds the mark on busy, low-saturation backgrounds (e.g. a
    watercolour) where brightness/contrast alone cannot separate it from foliage.
    Returns a mask stamped at the single best match, or empty if none is strong
    enough.
    """
    h, w = gray.shape
    out = np.zeros((h, w), np.uint8)
    base = min(h, w)

    # Match against the white top-hat map (bright-local structure), not raw grey,
    # so a faint bright star stands out and foliage/texture correlates poorly.
    k = _odd(max(MIN_TOPHAT_KERNEL, int(round(base * TOPHAT_KERNEL_RATIO))))
    bright = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bright[ignore > 0] = 0
    if bright.max() < _ABS_FLOOR:
        return out
    src = bright.astype(np.float32)

    best_corr, best_loc, best_size = 0.0, None, 0
    for frac in (0.16, 0.22, 0.30, 0.40, 0.50):
        size = int(round(base * frac))
        if size < 9 or size > base:
            continue
        result = cv2.matchTemplate(src, _sparkle_template(size), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_corr:
            best_corr, best_loc, best_size = max_val, max_loc, size

    # Real sparkles correlate ~0.70+; incidental texture matches stay well below.
    # Threshold in between so a busy background yields nothing rather than a
    # wrong removal (use Manual mode for marks auto-detection can't isolate).
    corr_thr = 0.66 - 0.13 * float(np.clip(sensitivity, 0.0, 1.0))  # 0.55 -> 0.59
    if best_loc is not None and best_corr >= corr_thr:
        x, y = best_loc
        stamp = _sparkle_template(best_size)
        out[y:y + best_size, x:x + best_size][stamp > 0.3] = 255
    out[ignore > 0] = 0
    return out


def _keep_marks(mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
    """Keep compact, interior watermark blobs; drop noise, frame, and ROI-scale fills."""
    h, w = mask.shape
    min_area = max(6, int(h * w * min_area_ratio))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        x, y, bw, bh, area = stats[label, :5]
        if area < min_area:
            continue                      # speckle noise
        if (y + bh) >= h or (x + bw) >= w:
            continue                      # touches image bottom/right edge (frame)
        if bw >= 0.9 * w or bh >= 0.9 * h:
            continue                      # background-scale blob, not a mark
        cleaned[labels == label] = 255
    return cleaned


def _bright_halo(roi_bgr: np.ndarray, structure: np.ndarray,
                 sensitivity: float) -> np.ndarray:
    """Desaturated-bright pixels *adjacent* to detected structure.

    Captures a watermark's soft glow (e.g. the Gemini sparkle's halo) without
    firing on plain light backgrounds — a pixel only counts if it is near an
    already high-contrast detection, so uniform pale areas are ignored.
    """
    if not structure.any():
        return np.zeros_like(structure)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    value, sat = hsv[:, :, 2], hsv[:, :, 1]
    v_thr = int(235 - 55 * float(np.clip(sensitivity, 0.0, 1.0)))
    s_thr = int(28 + 44 * float(np.clip(sensitivity, 0.0, 1.0)))
    white = ((value >= v_thr) & (sat <= s_thr)).astype(np.uint8) * 255

    h, w = structure.shape
    radius = _odd(max(HALO_MIN_RADIUS, int(min(h, w) * HALO_NEIGHBORHOOD_RATIO)))
    near = cv2.dilate(structure, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius)))
    return cv2.bitwise_and(white, near)


def detect_core_mask(roi_bgr: np.ndarray, sensitivity: float) -> np.ndarray:
    """Detect watermark pixels in an ROI crop.

    The Gemini sparkle and NotebookLM badge are **desaturated** (white / grey)
    marks; the artwork around them (green, gold, pink, red) is vividly coloured.
    So detection = "locally-brighter-or-darker structure" AND "low saturation".
    The saturation gate is what stops the colourful background from ever being
    selected — the failure mode where a green corner got masked and inpainted
    green-over-green (an invisible non-fix).

    Returns an unpadded uint8 mask (0/255) the same H×W as ``roi_bgr``.
    """
    if roi_bgr.size == 0:
        return np.zeros(roi_bgr.shape[:2], np.uint8)

    sens = float(np.clip(sensitivity, 0.0, 1.0))
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    ignore = _frame_border_mask(roi_bgr)             # picture-frame border, if any
    response = _contrast_response(gray, ignore)      # raw top-hat, border zeroed

    # Saturation gate applied to the RESPONSE (before thresholding): zero out
    # colourful artwork (green/gold/pink) so a bright saturated element can't
    # inflate the peak and raise the bar above a genuinely faint grey watermark.
    s_thr = int(70 + 70 * sens)       # 0 -> 70, 0.55 -> 108, 1 -> 140
    response[sat > s_thr] = 0

    # Threshold relative to the (desaturated) peak: the watermark is the
    # strongest desaturated local structure. An absolute floor means a flat,
    # watermark-free ROI detects nothing instead of amplifying noise.
    raw_max = float(response.max())
    if raw_max < _ABS_FLOOR:
        return np.zeros(gray.shape, np.uint8)
    peak_k = 0.72 - 0.40 * sens       # fraction of peak to keep (0.55 -> 0.50)
    thr = float(np.clip(max(_ABS_FLOOR, peak_k * raw_max), _ABS_FLOOR, 252))
    _, structure = cv2.threshold(response, thr, 255, cv2.THRESH_BINARY)

    structure[ignore > 0] = 0
    structure = cv2.morphologyEx(
        structure, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    structure = _keep_marks(structure, SPECK_MIN_AREA_RATIO)   # drop frame/noise first

    # Shape-based sparkle match: robust where brightness/contrast can't separate
    # the mark from a busy, low-saturation background (e.g. a watercolour scene).
    sparkle = _match_sparkle(gray, ignore, sensitivity)

    roi_area = gray.shape[0] * gray.shape[1]
    if int((structure > 0).sum()) > _MAX_WM_FRAC * roi_area:
        base = sparkle                    # generic detector over-fired → shape only
    else:
        base = cv2.bitwise_or(structure, sparkle)

    halo = _bright_halo(roi_bgr, base, sensitivity)
    mask = cv2.bitwise_or(base, halo)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    mask[ignore > 0] = 0
    return _keep_marks(mask, SPECK_MIN_AREA_RATIO)


def pad_mask(mask: np.ndarray, padding_px: int) -> np.ndarray:
    """Dilate a mask so inpainting comfortably overshoots the watermark edge."""
    if padding_px <= 0 or not mask.any():
        return mask
    size = _odd(padding_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel)


# Real 'Gemini Notebook' badges are a grey/white pill (mean saturation <= ~25);
# reject a shape match landing on colourful content above this.
_BADGE_SAT_MAX = 45.0

_BADGE_TEMPLATE = None
_BADGE_LOADED = False


def _badge_template():
    """Lazily load the 'Gemini Notebook' badge alpha template (or None)."""
    global _BADGE_TEMPLATE, _BADGE_LOADED
    if not _BADGE_LOADED:
        _BADGE_LOADED = True
        path = Path(__file__).resolve().parent / "weights" / "badge_alpha.npy"
        try:
            _BADGE_TEMPLATE = np.load(str(path)).astype(np.float32) if path.exists() else None
        except Exception:
            _BADGE_TEMPLATE = None
    return _BADGE_TEMPLATE


def detect_badge_mask(frame_bgr: np.ndarray, right: float = 0.45, bottom: float = 0.35,
                      corr_thr: float = 0.68):
    """Locate the fixed 'Gemini Notebook' badge by multi-scale template matching.

    The badge is a constant graphic, so shape correlation finds it regardless of
    slide colour. Returns a full-frame 0/255 mask (badge box filled) or None.

    The badge is highly consistent — real badges correlate ~0.74-0.75 — so the
    threshold sits well above incidental matches on generic card/slide text
    (~0.63) or colourful content (~0.68), which would otherwise inpaint real
    artwork. A saturation gate (below) adds a second, independent guard.
    """
    badge = _badge_template()
    if badge is None or badge.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    rx, ry = int(w * (1 - right)), int(h * (1 - bottom))
    roi = frame_bgr[ry:, rx:]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    contrast = cv2.max(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k),
                       cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)).astype(np.float32)

    base = (w * 0.095) / badge.shape[1]     # badge is ~9.5% of the frame width
    best_corr, best_loc, best_size = 0.0, None, None
    for mult in (0.7, 0.85, 1.0, 1.2, 1.45):
        bw = int(badge.shape[1] * base * mult)
        bh = max(4, int(badge.shape[0] * base * mult))
        if bw < 20 or bw >= roi.shape[1] or bh >= roi.shape[0]:
            continue
        templ = cv2.resize(badge, (bw, bh), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(contrast, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_corr:
            best_corr, best_loc, best_size = max_val, max_loc, (bw, bh)

    if best_loc is None or best_corr < corr_thr:
        return None

    # Saturation gate: the 'Gemini Notebook' badge is a grey/white pill, so the
    # matched box is near-desaturated (mean S <= ~25 on real badges). A shape
    # that correlates on *colourful* content — e.g. an army uniform (mean S ~64)
    # — is a false positive and must be rejected, or it inpaints real artwork.
    bx, by = best_loc
    bw, bh = best_size
    box_hsv = cv2.cvtColor(roi[by:by + bh, bx:bx + bw], cv2.COLOR_BGR2HSV)
    if float(box_hsv[:, :, 1].mean()) > _BADGE_SAT_MAX:
        return None

    full = np.zeros((h, w), np.uint8)
    # Expand beyond the text to cover a translucent pill/panel some badges sit on
    # (wider margins vertically, where the pill padding is largest).
    ex, ey = int(bw * 0.12), int(bh * 1.1)
    x0 = max(0, rx + best_loc[0] - ex)
    y0 = max(0, ry + best_loc[1] - ey)
    x1 = min(w, rx + best_loc[0] + bw + ex)
    y1 = min(h, ry + best_loc[1] + bh + ey)
    full[y0:y1, x0:x1] = 255
    return full


def votes_to_mask(votes: np.ndarray, num_samples: int, vote_ratio: float,
                  padding_px: int) -> np.ndarray:
    """Collapse per-frame detections into one stable static-watermark mask.

    A pixel is kept only if it was flagged in at least ``vote_ratio`` of the
    sampled frames — persistent logos survive, moving background does not.
    """
    if num_samples <= 0:
        return np.zeros(votes.shape, np.uint8)
    threshold = max(1, int(round(num_samples * vote_ratio)))
    mask = np.where(votes >= threshold, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    return pad_mask(mask, padding_px)

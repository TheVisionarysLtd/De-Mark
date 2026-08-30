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


# The 'Gemini Notebook' badge is a constant logo (sparkle icon + text) that
# NotebookLM renders in ADAPTIVE contrast: light strokes on dark slides, dark
# strokes on light slides. Verify the matched region's stroke pixels are
# desaturated (grey/white/black, never coloured) and stand out strongly from
# their immediate surround in EITHER direction. This is background- and
# polarity-invariant, so it catches the badge on dark, busy and light slides
# alike, while still rejecting coloured artwork and low-contrast look-alikes.
# (An earlier version required strokes to be *bright*; that silently dropped the
# common dark-on-light badge on every pale slide — the whole point of this pass.)
_BADGE_TEXT_S_MAX = 95.0      # strokes are desaturated (grey scale)
_BADGE_TEXT_CONTRAST = 25.0   # ...and clearly separated from their surround (|Δ|)

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


def _badge_strokes_match(templ: np.ndarray, box_val: np.ndarray,
                         box_sat: np.ndarray) -> bool:
    """True if the matched region's stroke pixels read as the Gemini badge logo.

    Uses the template's own high-alpha pixels (the icon + text strokes) to sample
    the image: a real badge is *desaturated* there (grey/white/black) and stands
    out strongly from its immediate surround — either brighter (light badge on a
    dark slide) or darker (dark badge on a pale slide). Checking |contrast| rather
    than a fixed direction is what lets one detector handle both polarities, while
    the saturation gate still rejects coloured artwork and uniforms that merely
    share the badge's outline.
    """
    if box_val.shape != templ.shape or templ.max() <= 0:
        return False
    on = templ > 0.5 * float(templ.max())
    if int(on.sum()) < 10:
        return False
    off = ~on
    v_on = float(box_val[on].mean())
    s_on = float(box_sat[on].mean())
    v_off = float(box_val[off].mean()) if off.any() else v_on
    return (s_on <= _BADGE_TEXT_S_MAX
            and abs(v_on - v_off) >= _BADGE_TEXT_CONTRAST)


_BADGE_SCALES = (0.6, 0.75, 0.9, 1.05, 1.25, 1.5)   # badge width as multiples of ~9.5% frame


def _badge_contrast(gray: np.ndarray) -> np.ndarray:
    """Sign-agnostic local contrast map (bright- or dark-on-background strokes)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.max(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k),
                   cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)).astype(np.float32)


def _search_badge(frame_bgr: np.ndarray, rx: int, ry: int, roi: np.ndarray,
                  corr_thr: float):
    """Best verified badge inside ``roi`` (whose top-left is (rx, ry) in frame).

    Returns ``(corr, x0, y0, x1, y1)`` — the expanded box in full-frame pixels —
    or ``None``. Shared by the corner search and the position-locked recheck.
    """
    badge = _badge_template()
    if badge is None or badge.size == 0 or roi.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    contrast = _badge_contrast(gray)

    base = (w * 0.095) / badge.shape[1]     # badge is ~9.5% of the frame width
    cands = []                              # (corr, x, y, bw, bh, templ)
    for mult in _BADGE_SCALES:
        bw = int(badge.shape[1] * base * mult)
        bh = max(4, int(badge.shape[0] * base * mult))
        if bw < 20 or bw >= roi.shape[1] or bh >= roi.shape[0]:
            continue
        templ = cv2.resize(badge, (bw, bh), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(contrast, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        cands.append((float(max_val), max_loc[0], max_loc[1], bw, bh, templ))

    cands.sort(key=lambda c: -c[0])         # highest correlation first
    for corr, bx, by, bw, bh, templ in cands:
        if corr < corr_thr:
            break
        if not _badge_strokes_match(templ, val[by:by + bh, bx:bx + bw],
                                    sat[by:by + bh, bx:bx + bw]):
            continue
        # Expand beyond the text to cover a translucent pill/panel some badges sit
        # on (wider margins vertically, where the pill padding is largest).
        ex, ey = int(bw * 0.12), int(bh * 1.1)
        x0 = max(0, rx + bx - ex)
        y0 = max(0, ry + by - ey)
        x1 = min(w, rx + bx + bw + ex)
        y1 = min(h, ry + by + bh + ey)
        return (corr, x0, y0, x1, y1)
    return None


def detect_badge_box(frame_bgr: np.ndarray, right: float = 0.45, bottom: float = 0.35,
                     corr_thr: float = 0.42):
    """Best verified 'Gemini Notebook' badge box in the bottom-right corner.

    Returns ``(corr, x0, y0, x1, y1)`` in pixels or ``None``. This is the raw
    locator behind :func:`detect_badge_mask`; the deck pipeline also uses the
    ``corr`` and position to build a cross-page consensus (see
    :func:`badge_consensus_box`).
    """
    h, w = frame_bgr.shape[:2]
    rx, ry = int(w * (1 - right)), int(h * (1 - bottom))
    return _search_badge(frame_bgr, rx, ry, frame_bgr[ry:, rx:], corr_thr)


def detect_badge_mask(frame_bgr: np.ndarray, right: float = 0.45, bottom: float = 0.35,
                      corr_thr: float = 0.42):
    """Locate the fixed 'Gemini Notebook' badge by multi-scale template matching.

    The badge is a constant graphic, so shape correlation finds its outline
    regardless of slide colour, and a stroke check (``_badge_strokes_match``)
    confirms each candidate. Matching alone was unreliable: a high threshold
    missed the badge on dark/busy slides, while a low one fired on tables and
    uniforms. Verifying the strokes are desaturated and high-contrast (in either
    polarity) fixes both — it catches the badge whether it is light-on-dark or
    dark-on-light and rejects look-alikes. Returns a full-frame 0/255 mask (badge
    box filled) or None.
    """
    box = detect_badge_box(frame_bgr, right, bottom, corr_thr)
    if box is None:
        return None
    _, x0, y0, x1, y1 = box
    full = np.zeros(frame_bgr.shape[:2], np.uint8)
    full[y0:y1, x0:x1] = 255
    return full


# --- Deck-wide consensus -----------------------------------------------------
# A slide deck stamps the badge at ONE position on every page that carries it.
# So across pages, strong matches that agree on a position are the real badge,
# while lone matches scattered over artwork are per-slide false positives. We
# learn the consensus position from confident detections, then re-check that
# exact spot on every page — catching faint/low-contrast badges at the known
# location while never inpainting a clean corner where nothing matches.
_CONSENSUS_MIN_CORR = 0.72    # only strong, unambiguous matches vote on position
_CONSENSUS_POS_TOL = 0.04     # frac of page: how close boxes must sit to co-cluster
_CONSENSUS_MIN_FRAC = 0.12    # cluster must cover >= this share of pages (min 2)
BADGE_CONSENSUS_APPLY_CORR = 0.50  # per-page corr to accept the badge AT consensus


def badge_consensus_box(boxes, frame_w: int, frame_h: int, total_pages: int):
    """Dominant badge box (px) across a deck, or None if no consistent stamp.

    ``boxes`` is the per-page :func:`detect_badge_box` output (each ``None`` or
    ``(corr, x0, y0, x1, y1)``). Returns the median box of the largest cluster of
    strong, co-located detections when it covers enough pages, else ``None``.
    """
    strong = [b for b in boxes if b is not None and b[0] >= _CONSENSUS_MIN_CORR]
    need = max(2, int(round(total_pages * _CONSENSUS_MIN_FRAC)))
    if len(strong) < need or frame_w <= 0 or frame_h <= 0:
        return None

    best_cluster = []
    for anchor in strong:                    # cluster by top-left corner position
        ax, ay = anchor[1] / frame_w, anchor[2] / frame_h
        members = [b for b in strong
                   if abs(b[1] / frame_w - ax) <= _CONSENSUS_POS_TOL
                   and abs(b[2] / frame_h - ay) <= _CONSENSUS_POS_TOL]
        if len(members) > len(best_cluster):
            best_cluster = members
    if len(best_cluster) < need:
        return None

    arr = np.array([[b[1], b[2], b[3], b[4]] for b in best_cluster], dtype=float)
    med = np.median(arr, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]), int(med[3]))


def badge_box_at(frame_bgr: np.ndarray, consensus_box, corr_min: float = BADGE_CONSENSUS_APPLY_CORR):
    """Re-verify the badge at a known deck ``consensus_box`` on this page.

    Searches only a small window around the consensus position (not the whole
    corner), so it applies the deck's badge stamp wherever it actually recurs and
    stays silent on pages whose corner holds only artwork. Returns the expanded
    box (px) if present, else ``None``.
    """
    if consensus_box is None:
        return None
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = consensus_box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    px, py = int(bw * 0.5), int(bh * 0.5)    # pad the window generously around the box
    wx0, wy0 = max(0, x0 - px), max(0, y0 - py)
    wx1, wy1 = min(w, x1 + px), min(h, y1 + py)
    return _search_badge(frame_bgr, wx0, wy0, frame_bgr[wy0:wy1, wx0:wx1], corr_min)


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

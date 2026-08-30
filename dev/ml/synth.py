"""Synthetic training data: the REAL sparkle composited onto varied backgrounds.

Backgrounds span solids, gradients, smooth/rough noise, pastel "watercolour",
shapes, and random crops of LaMa-cleaned real plates (lotus + temple) so the
distribution includes the hard low-contrast case. Sparkles vary in scale,
opacity, rotation and colour; ~25% of samples are negatives (no watermark).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ASSETS = Path(__file__).resolve().parent / "assets"
PLATES = ASSETS / "plates"

SPARKLE = np.load(ASSETS / "sparkle_alpha.npy")  # HxW float [0,1], real shape
BADGE = np.load(ASSETS / "badge_alpha.npy")      # 'Gemini Notebook' strip
SIZE = 192


# --- clean real plates (cached) ---------------------------------------------
def ensure_plates() -> list[np.ndarray]:
    PLATES.mkdir(parents=True, exist_ok=True)
    cached = sorted(PLATES.glob("*.png"))
    if cached:
        return [cv2.imread(str(p)) for p in cached]

    from wmr.config import RemovalSettings
    from wmr.pipeline import build_frame_mask, apply_inpaint
    jobs = [
        (r"C:\Users\Chintan Kamani\Desktop\Gemini_Generated_Image_.png", None),
        (r"C:\Users\Chintan Kamani\Desktop\Gemini_Generated_Image_ (1).png",
         dict(region_mode="manual", center_x=0.845, center_y=0.878, box_w=0.11,
              box_h=0.11, force_fill=True)),
    ]
    plates = []
    for i, (path, manual) in enumerate(jobs):
        img = cv2.imread(path)
        if img is None:
            continue
        s = RemovalSettings(backend="lama", **(manual or {}))
        mask = build_frame_mask(img, s)
        plate = apply_inpaint(img, mask, s)
        cv2.imwrite(str(PLATES / f"plate_{i}.png"), plate)
        plates.append(plate)
    return plates


# --- procedural backgrounds --------------------------------------------------
def _rand_color(rng):
    return rng.integers(0, 256, 3).astype(np.float32)


def _smooth_noise(size, rng, freq):
    low = rng.integers(0, 256, (max(2, size // freq), max(2, size // freq), 3)).astype(np.uint8)
    bg = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return cv2.GaussianBlur(bg, (0, 0), sigmaX=size / (freq * 1.5) + 1)


def proc_background(size, rng):
    kind = rng.integers(0, 7)
    if kind == 0:                                   # solid
        bg = np.ones((size, size, 3), np.float32) * _rand_color(rng)
    elif kind == 1:                                 # linear gradient
        c0, c1 = _rand_color(rng), _rand_color(rng)
        t = np.linspace(0, 1, size, dtype=np.float32)
        t = t[:, None] if rng.random() < 0.5 else t[None, :]
        bg = (c0 * (1 - t[..., None]) + c1 * t[..., None]).astype(np.float32)
        bg = np.broadcast_to(bg, (size, size, 3)).copy() if bg.shape[:2] != (size, size) else bg
    elif kind in (2, 3):                            # smooth noise (low/high freq)
        bg = _smooth_noise(size, rng, rng.choice([40, 20, 10]))
    elif kind == 4:                                 # pastel / watercolour
        bg = _smooth_noise(size, rng, rng.choice([30, 18]))
        light = np.ones_like(bg) * rng.integers(150, 235, 3).astype(np.float32)
        bg = 0.55 * bg + 0.45 * light
        bg += rng.normal(0, 5, bg.shape)            # paper grain
    elif kind == 5:                                 # shapes on a base
        bg = np.ones((size, size, 3), np.float32) * _rand_color(rng)
        for _ in range(rng.integers(2, 8)):
            c = tuple(int(v) for v in _rand_color(rng))
            p = tuple(int(v) for v in rng.integers(0, size, 2))
            if rng.random() < 0.5:
                cv2.circle(bg, p, int(rng.integers(10, size // 3)), c, -1)
            else:
                q = tuple(int(v) for v in rng.integers(0, size, 2))
                cv2.rectangle(bg, p, q, c, -1)
    else:                                           # busy hi-freq
        bg = _smooth_noise(size, rng, 6)
    return np.clip(bg, 0, 255)


def random_background(size, rng, plates):
    if plates and rng.random() < 0.4:
        plate = plates[rng.integers(len(plates))]
        H, W = plate.shape[:2]
        s = rng.integers(size, min(H, W) + 1)
        y, x = rng.integers(0, H - s + 1), rng.integers(0, W - s + 1)
        return cv2.resize(plate[y:y + s, x:x + s], (size, size)).astype(np.float32)
    return proc_background(size, rng)


def _wrong_star(bg, center, r, color, rng):
    """A NON-sparkle bright shape (5-point star / diamond / flower) — hard negative."""
    pts = []
    n = int(rng.choice([4, 5, 6]))
    inner = rng.uniform(0.5, 0.95)  # convex-ish, unlike the sparkle's thin concave arms
    rot = rng.uniform(0, np.pi)
    for i in range(2 * n):
        rad = r if i % 2 == 0 else r * inner
        ang = rot + i * np.pi / n
        pts.append([center[0] + rad * np.cos(ang), center[1] + rad * np.sin(ang)])
    cv2.fillPoly(bg, [np.array(pts, np.int32)], color, cv2.LINE_AA)


def _add_distractors(bg, rng):
    """Bright/coloured blobs that are NOT the watermark (never labelled).

    Teaches the net that 'bright compact blob' is not enough — it must match the
    sparkle's specific concave 4-point shape. Fixes confusion with flowers etc.
    """
    for _ in range(int(rng.integers(0, 6))):
        p = (int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE)))
        r = int(rng.integers(6, 42))
        kind = rng.integers(0, 5)
        if kind == 0:                                   # coloured circle
            cv2.circle(bg, p, r, tuple(int(v) for v in _rand_color(rng)), -1)
        elif kind == 1:                                 # bright WHITE blob (hard)
            cv2.circle(bg, p, r, tuple(int(v) for v in rng.integers(210, 256, 3)), -1)
        elif kind == 2:                                 # wrong-shape bright star
            col = tuple(int(v) for v in rng.integers(200, 256, 3))
            _wrong_star(bg, p, r, col, rng)
        elif kind == 3:                                 # ellipse
            cv2.ellipse(bg, p, (r, int(r * rng.uniform(0.3, 1))),
                        float(rng.uniform(0, 180)), 0, 360,
                        tuple(int(v) for v in _rand_color(rng)), -1)
        else:                                           # stroke
            q = (int(p[0] + rng.integers(-45, 45)), int(p[1] + rng.integers(-45, 45)))
            cv2.line(bg, p, q, tuple(int(v) for v in _rand_color(rng)), int(rng.integers(2, 9)))
    return bg


# --- compositing -------------------------------------------------------------
def _scaled_rotated_sparkle(rng):
    # independent x/y size => robust to the ROI's aspect-ratio distortion at inference
    sx = int(rng.integers(16, 64))
    sy = int(np.clip(sx * rng.uniform(0.7, 1.4), 16, 64))
    a = cv2.resize(SPARKLE, (sx, sy), interpolation=cv2.INTER_CUBIC)
    ang = float(rng.uniform(-15, 15))
    M = cv2.getRotationMatrix2D((sx / 2, sy / 2), ang, 1.0)
    a = cv2.warpAffine(a, M, (sx, sy))
    a = a * float(rng.uniform(0.55, 2.6))           # opacity jitter (incl. very faint)
    return np.clip(a, 0, 1)


def _scaled_badge(rng):
    """The 'Gemini Notebook' strip, scaled and opacity-jittered."""
    bw = int(np.clip(size_frac(rng) * SIZE, 34, SIZE - 8))
    bh = max(5, int(round(bw * BADGE.shape[0] / BADGE.shape[1])))
    a = cv2.resize(BADGE, (bw, bh), interpolation=cv2.INTER_CUBIC)
    return np.clip(a * float(rng.uniform(0.6, 1.35)), 0, 1)


def size_frac(rng):
    return float(rng.uniform(0.28, 0.82))


def _place(bg, a, x, y, color):
    sh, sw = a.shape
    af = a[..., None]
    bg[y:y + sh, x:x + sw] = af * color + (1 - af) * bg[y:y + sh, x:x + sw]
    return sh, sw


def composite(size, rng, plates):
    bg = random_background(size, rng, plates)
    if rng.random() < 0.6:                           # clutter with non-sparkle blobs
        bg = _add_distractors(bg, rng)
    mask = np.zeros((size, size), np.float32)

    roll = rng.random()
    if roll < 0.42:                                  # sparkle (~42%)
        a = _scaled_rotated_sparkle(rng)
        sh, sw = a.shape
        x = int(rng.integers(4, size - sw - 4))
        y = int(rng.integers(4, size - sh - 4))
        white = rng.integers(238, 256, 3).astype(np.float32)
        _place(bg, a, x, y, white)
        mask[y:y + sh, x:x + sw] = (a > 0.08).astype(np.float32)
    elif roll < 0.74:                                # badge (~32%)
        a = _scaled_badge(rng)
        sh, sw = a.shape
        x_max, y_max = size - sw, size - sh           # keep it fully in-frame
        x = int(rng.integers(max(0, x_max - int(size * 0.45)), max(1, x_max + 1)))
        y = int(rng.integers(max(0, y_max - int(size * 0.30)), max(1, y_max + 1)))
        # badge polarity adapts to background: pale on dark, dark on pale
        color = (rng.integers(225, 256, 3) if rng.random() < 0.5
                 else rng.integers(70, 150, 3)).astype(np.float32)
        _place(bg, a, x, y, color)
        mask[y:y + sh, x:x + sw] = (a > 0.12).astype(np.float32)

    img = np.clip(bg, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:                            # mild global aug
        img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2)
                      + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img, mask


def make_batch(n, rng, plates):
    import torch
    imgs, masks = [], []
    for _ in range(n):
        img, m = composite(SIZE, rng, plates)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append(rgb.transpose(2, 0, 1))
        masks.append(m[None])
    return (torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(masks)))


if __name__ == "__main__":  # quick visual sanity check
    plates = ensure_plates()
    rng = np.random.default_rng(0)
    tiles = []
    for _ in range(8):
        img, m = composite(SIZE, rng, plates)
        ov = img.copy(); ov[m > 0] = (0, 0, 255)
        tiles.append(np.hstack([img, cv2.addWeighted(ov, 0.5, img, 0.5, 0)]))
    cv2.imwrite(str(ASSETS / "synth_preview.png"), np.vstack(tiles[:4]))
    print("plates:", len(plates), "-> assets/synth_preview.png")

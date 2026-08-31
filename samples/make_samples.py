"""Generate synthetic test media that reproduces the target watermarks.

Creates, next to this file:
  * sample_image.png  — a textured scene with a bright Gemini-style sparkle and a
    NotebookLM-style badge in the bottom-right corner.
  * sample_video.mp4  — an animated background with a *static* sparkle overlay,
    to exercise the video path (static-mask voting, fidelity-preserving encode).

Run:  python samples/make_samples.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

HERE = Path(__file__).resolve().parent
_WEIGHTS = HERE.parent / "wmr" / "weights"


def _overlay_alpha(img: np.ndarray, alpha: np.ndarray, x: int, y: int,
                   color=(255, 255, 255), strength: float = 1.0) -> None:
    """Alpha-composite a solid ``color`` onto ``img`` at (x,y) using an alpha map.

    ``alpha`` is a float map in [0,1]; ``strength`` scales its opacity. This is how
    the REAL watermarks are burned in — so the samples match the exact glyphs the
    detector matches against (``sparkle_canon``/``badge_alpha``) and are cleaned.
    """
    h, w = img.shape[:2]
    ah, aw = alpha.shape[:2]
    if x < 0 or y < 0 or x + aw > w or y + ah > h:
        return
    a = np.clip(alpha * strength, 0.0, 1.0)[..., None]
    roi = img[y:y + ah, x:x + aw].astype(np.float32)
    img[y:y + ah, x:x + aw] = (a * np.array(color, np.float32) + (1 - a) * roi).astype(np.uint8)


def textured_background(w: int, h: int, seed: int = 7) -> np.ndarray:
    """A smooth, colourful, photo-like background so inpainting has real texture
    to rebuild (large soft colour regions, similar to a real scene)."""
    rng = np.random.default_rng(seed)
    low = rng.integers(0, 255, size=(max(2, h // 120), max(2, w // 120), 3), dtype=np.uint8)
    bg = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=w / 45.0)

    # Diagonal gradient tint for structure the eye can check after inpainting.
    yy, xx = np.mgrid[0:h, 0:w]
    grad = ((xx + yy) / (w + h) * 120).astype(np.uint8)
    bg = cv2.addWeighted(bg, 0.7, cv2.merge([grad, grad // 2, 255 - grad]), 0.3, 0)
    return bg


def draw_sparkle(img: np.ndarray, center, radius: int) -> None:
    """Burn in the REAL Gemini sparkle glyph (canonical template) at ~88% opacity.

    Using the actual template (not a hand-drawn star) means the sample reproduces
    what Gemini stamps, so Auto Detect matches and removes it — the demo proves the
    real behaviour. Falls back to a drawn star if the template is missing.
    """
    cx, cy = center
    size = max(12, radius * 2)
    canon = _WEIGHTS / "sparkle_canon.npy"
    if canon.exists():
        sp = np.load(str(canon)).astype(np.float32)
        sp = cv2.resize(sp, (size, size), interpolation=cv2.INTER_AREA)
        _overlay_alpha(img, sp, cx - size // 2, cy - size // 2, (255, 255, 255), strength=0.88)
        return
    pts = []                                     # fallback: drawn concave star
    for i in range(8):
        ang = np.deg2rad(i * 45)
        r = radius if i % 2 == 0 else max(2, radius // 5)
        pts.append([cx + r * np.sin(ang), cy - r * np.cos(ang)])
    cv2.fillPoly(img, [np.array(pts, np.int32)], (255, 255, 255), lineType=cv2.LINE_AA)


def draw_badge(img: np.ndarray) -> None:
    """Burn in the REAL 'Gemini Notebook' badge (template) as white strokes.

    Mirrors the NotebookLM badge as it appears over imagery: white strokes on a
    faint translucent panel, bottom-right. Uses the same ``badge_alpha`` template
    the detector matches, so Auto Detect removes it.
    """
    h, w = img.shape[:2]
    badge = _WEIGHTS / "badge_alpha.npy"
    if not badge.exists():
        return
    bd = np.load(str(badge)).astype(np.float32)
    bd = bd / float(bd.max())                    # normalise alpha to [0,1]
    bw = int(w * 0.095)
    bh = max(1, int(round(bw * bd.shape[0] / bd.shape[1])))
    bd = cv2.resize(bd, (bw, bh), interpolation=cv2.INTER_AREA)
    x = w - bw - int(w * 0.03)
    y = h - bh - int(h * 0.035)

    padx, pady = int(bh * 1.4), int(bh * 0.9)    # subtle DARK pill (as NotebookLM
    panel = img.copy()                            # renders it over imagery) so the
    cv2.rectangle(panel, (x - padx, y - pady), (x + bw + int(bh * 0.4), y + bh + pady),
                  (35, 35, 35), -1)               # white strokes stay high-contrast
    cv2.addWeighted(panel, 0.30, img, 0.70, 0, dst=img)
    _overlay_alpha(img, bd, x, y, (255, 255, 255), strength=0.98)


def make_image() -> Path:
    w, h = 1024, 1024
    img = textured_background(w, h, seed=11)
    draw_sparkle(img, (int(w * 0.9), int(h * 0.9)), radius=int(h * 0.04))
    draw_badge(img)
    out = HERE / "sample_image.png"
    cv2.imwrite(str(out), img)
    return out


def make_video(seconds: int = 3, fps: int = 24) -> Path:
    w, h = 854, 480
    out = HERE / "sample_video.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(out), size=(w, h), fps=fps, pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        codec="libx264", quality=None, bitrate=None, macro_block_size=1,
        output_params=["-crf", "18", "-preset", "medium"])
    writer.send(None)

    base = textured_background(w, h, seed=3)
    total = seconds * fps
    sparkle_c = (int(w * 0.90), int(h * 0.93))  # fully inside the bottom-right ROI
    for i in range(total):
        shift = int((i / total) * w)
        frame = np.roll(base, shift, axis=1)  # animate the background horizontally
        draw_sparkle(frame, sparkle_c, radius=int(h * 0.045))  # static overlay, inside ROI
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.send(np.ascontiguousarray(rgb).tobytes())
    writer.close()
    return out


if __name__ == "__main__":
    print("image:", make_image())
    print("video:", make_video())

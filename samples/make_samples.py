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
    """Draw a bright four-pointed sparkle (concave star) with a soft glow."""
    cx, cy = center
    r_out, r_in = radius, max(2, radius // 5)
    pts = []
    for i in range(8):
        ang = np.deg2rad(i * 45)
        r = r_out if i % 2 == 0 else r_in
        pts.append([cx + r * np.sin(ang), cy - r * np.cos(ang)])
    poly = np.array(pts, np.int32)

    glow = img.copy()
    cv2.circle(glow, (cx, cy), int(radius * 1.12), (255, 255, 255), -1)
    cv2.addWeighted(glow, 0.16, img, 0.84, 0, dst=img)  # subtle, realistic halo
    cv2.fillPoly(img, [poly], (255, 255, 255), lineType=cv2.LINE_AA)


def draw_badge(img: np.ndarray) -> None:
    """Draw a translucent NotebookLM / 'Gemini Notebook' style badge, bottom-right."""
    h, w = img.shape[:2]
    bw, bh = int(w * 0.205), int(h * 0.05)
    x2, y2 = w - int(w * 0.02), h - int(h * 0.02)
    x1, y1 = x2 - bw, y2 - bh

    panel = img.copy()
    cv2.rectangle(panel, (x1, y1), (x2, y2), (245, 245, 245), -1)
    cv2.addWeighted(panel, 0.55, img, 0.45, 0, dst=img)

    cv2.circle(img, (x1 + bh // 2, (y1 + y2) // 2), bh // 4, (150, 120, 60), -1)
    cv2.putText(img, "Gemini Notebook", (x1 + bh, y2 - bh // 3),
                cv2.FONT_HERSHEY_SIMPLEX, bh / 64.0, (110, 110, 110), 1, cv2.LINE_AA)


def make_image() -> Path:
    w, h = 1024, 1024
    img = textured_background(w, h, seed=11)
    draw_sparkle(img, (int(w * 0.93), int(h * 0.92)), radius=int(h * 0.035))
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

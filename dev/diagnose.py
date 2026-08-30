"""Diagnose detection on a real image. Usage: python dev/diagnose.py <img> [sens] [bottom] [right]"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wmr.config import (                         # noqa: E402
    DEFAULT_SENSITIVITY, ROI_BOTTOM_FRACTION, ROI_RIGHT_FRACTION, RemovalSettings)
from wmr.pipeline import build_frame_mask, mask_overlay, apply_inpaint  # noqa: E402
from wmr.roi import compute_roi                 # noqa: E402

path = sys.argv[1]
sens = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SENSITIVITY
bottom = float(sys.argv[3]) if len(sys.argv) > 3 else ROI_BOTTOM_FRACTION
right = float(sys.argv[4]) if len(sys.argv) > 4 else ROI_RIGHT_FRACTION

img = cv2.imread(path)
h, w = img.shape[:2]
s = RemovalSettings(sensitivity=sens, bottom_fraction=bottom, right_fraction=right)
roi = compute_roi(w, h, s.bottom_fraction, s.right_fraction)
mask = build_frame_mask(img, s)
px = int((mask > 0).sum())
print(f"dims {w}x{h}  ROI=({roi.x},{roi.y})-({roi.x1},{roi.y1})  mask_px={px}")
if px:
    ys, xs = np.nonzero(mask)
    print(f"mask bbox x[{xs.min()}-{xs.max()}] y[{ys.min()}-{ys.max()}]")

OUT = ROOT / "samples" / "output"
OUT.mkdir(exist_ok=True)
cv2.imwrite(str(OUT / "diag_overlay.png"), mask_overlay(img, mask))
cv2.imwrite(str(OUT / "diag_after.png"), apply_inpaint(img, mask, s))
# zoom the bottom-right corner so the real sparkle is clearly visible
z = img[int(h * 0.82):, int(w * 0.62):]
cv2.imwrite(str(OUT / "diag_zoom.png"), cv2.resize(z, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_NEAREST))
print("wrote diag_overlay / diag_after / diag_zoom")

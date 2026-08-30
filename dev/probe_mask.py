"""Report the real detector's output on an image ROI. Usage: probe_mask.py <img> [sens]"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wmr.config import ROI_BOTTOM_FRACTION, ROI_RIGHT_FRACTION  # noqa: E402
from wmr import mask as M                                       # noqa: E402
from wmr.roi import compute_roi                                 # noqa: E402

img = cv2.imread(sys.argv[1])
sens = float(sys.argv[2]) if len(sys.argv) > 2 else 0.55
h, w = img.shape[:2]
roi = compute_roi(w, h, ROI_BOTTOM_FRACTION, ROI_RIGHT_FRACTION)
roi_bgr = img[roi.y:roi.y1, roi.x:roi.x1]

gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
ignore = M._frame_border_mask(roi_bgr)
resp = M._contrast_response(gray, ignore)

# best sparkle-template correlation (on the top-hat map), for threshold tuning
base = min(gray.shape)
k = M._odd(max(7, int(round(base * 0.45))))
bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
bright[ignore > 0] = 0
best = 0.0
for frac in (0.16, 0.22, 0.30, 0.40, 0.50):
    sz = int(round(base * frac))
    if 9 <= sz <= base:
        r = cv2.matchTemplate(bright.astype(np.float32), M._sparkle_template(sz), cv2.TM_CCOEFF_NORMED)
        best = max(best, cv2.minMaxLoc(r)[1])
print(f"   best_sparkle_corr={best:.3f}")

core = M.detect_core_mask(roi_bgr, sens)

px = int((core > 0).sum())
print(f"{Path(sys.argv[1]).name}: ROI {roi.w}x{roi.h}  raw_max={resp.max()}  "
      f"ignore_px={int((ignore>0).sum())}  mask_px={px}")
if px:
    ys, xs = np.nonzero(core)
    print(f"   bbox x[{xs.min()}-{xs.max()}] y[{ys.min()}-{ys.max()}]  "
          f"({xs.max()-xs.min()+1}x{ys.max()-ys.min()+1})")

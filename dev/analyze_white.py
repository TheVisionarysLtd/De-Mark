"""Compare feature maps for detecting a semi-transparent WHITE sparkle.

Hypothesis: the sparkle blends the background toward white, which RAISES the
per-pixel min(B,G,R) ("whiteness"). Coloured foliage stays low-min even when
bright. So a white top-hat on the min-channel should isolate the sparkle where a
grayscale top-hat cannot.  Usage: python dev/analyze_white.py <img> <tag>
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wmr.mask import _odd, _sparkle_template  # noqa: E402

OUT = ROOT / "samples" / "output"
OUT.mkdir(exist_ok=True)


def match(feat: np.ndarray):
    base = min(feat.shape)
    best, loc, size = 0.0, None, 0
    for frac in (0.16, 0.22, 0.30, 0.40):
        sz = int(round(base * frac))
        if 9 <= sz <= base:
            r = cv2.matchTemplate(feat.astype(np.float32), _sparkle_template(sz),
                                  cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(r)
            if mv > best:
                best, loc, size = mv, ml, sz
    return best, loc, size


def run(path: str, tag: str, roi_bottom=0.22, roi_right=0.38):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    roi = img[int(h * (1 - roi_bottom)):, int(w * (1 - roi_right)):]
    base = min(roi.shape[:2])
    minch = roi.min(axis=2)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    for name, src in [("gray", gray), ("minch", minch)]:
        k = _odd(int(base * 0.33))
        th = cv2.morphologyEx(src, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cv2.imwrite(str(OUT / f"{tag}_{name}_tophat.png"),
                    cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX))
        best, loc, size = match(th)
        print(f"{tag:8s} {name:6s} tophat: best_corr={best:.3f} loc={loc} size={size} "
              f"th_max={int(th.max())}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])

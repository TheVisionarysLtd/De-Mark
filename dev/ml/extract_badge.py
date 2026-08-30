"""Extract the 'Gemini Notebook' badge as an alpha template from the NB video.

The badge is a low-contrast grey text+icon strip on a light slide. We recover it
as a darkness map (how much darker than the local background each pixel is),
which captures the badge's shape regardless of exact colour.
Output: dev/ml/assets/badge_alpha.npy (float32 [0,1]) + preview.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ASSETS = Path(__file__).resolve().parent / "assets"
VIDEO = r"C:\Users\Chintan Kamani\Desktop\The__One_Table__Approach__England’s_Fair_Pay_Agreement_clean.mp4"


def main() -> None:
    cap = cv2.VideoCapture(VIDEO)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.5))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read frame")
    h, w = frame.shape[:2]

    # bottom-right region containing the badge
    x0, y0 = int(w * 0.855), int(h * 0.93)
    region = frame[y0:, x0:]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # local background estimate (badge is darker than the light slide)
    bg = cv2.medianBlur(gray.astype(np.uint8), 31).astype(np.float32)
    darkness = np.clip(bg - gray, 0, 255)
    if darkness.max() > 0:
        alpha = darkness / darkness.max()
    else:
        alpha = darkness
    alpha[alpha < 0.18] = 0.0
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    ys, xs = np.where(alpha > 0.2)
    if len(xs):
        pad = 3
        a0, a1 = max(0, xs.min() - pad), min(alpha.shape[1], xs.max() + 1 + pad)
        b0, b1 = max(0, ys.min() - pad), min(alpha.shape[0], ys.max() + 1 + pad)
        alpha = alpha[b0:b1, a0:a1]

    np.save(ASSETS / "badge_alpha.npy", alpha.astype(np.float32))
    grey = np.full((*alpha.shape, 3), 220, np.float32)
    dark = np.full((*alpha.shape, 3), 120, np.float32)
    comp = ((1 - alpha[..., None]) * grey + alpha[..., None] * dark).astype(np.uint8)
    cv2.imwrite(str(ASSETS / "badge_preview.png"),
                cv2.resize(comp, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    print(f"badge alpha shape {alpha.shape}  max {alpha.max():.2f} -> assets/badge_alpha.npy")


if __name__ == "__main__":
    main()

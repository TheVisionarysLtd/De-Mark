"""Extract the REAL Gemini sparkle as an alpha template from the lotus image.

The sparkle is a semi-transparent white overlay:  obs = a*255 + (1-a)*bg.
On the lotus it sits on near-uniform green, so we recover the clean background
with LaMa, then solve per-pixel for the alpha a = (obs - bg) / (255 - bg).
That alpha map IS the watermark's true shape and opacity — the thing we composite
onto many backgrounds to build training data.

Output: dev/ml/assets/sparkle_alpha.npy (float32 HxW in [0,1]) + a preview PNG.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wmr.inpaint import inpaint  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

LOTUS = r"C:\Users\Chintan Kamani\Desktop\Gemini_Generated_Image_.png"
# sparkle bbox in the lotus (from earlier analysis: centre ~ (787, 1034))
BOX = (742, 989, 92, 92)  # x, y, w, h — generous, captures faint arms


def main() -> None:
    img = cv2.imread(LOTUS)
    x, y, w, h = BOX
    crop = img[y:y + h, x:x + w].astype(np.float32)

    # mask the sparkle (bright, desaturated) inside the crop, generously
    hsv = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    spark = ((hsv[:, :, 2] > 135) & (hsv[:, :, 1] < 130)).astype(np.uint8) * 255
    spark = cv2.dilate(spark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # clean background behind the sparkle via LaMa
    full_mask = np.zeros(img.shape[:2], np.uint8)
    full_mask[y:y + h, x:x + w] = spark
    from wmr.config import RemovalSettings
    from wmr.pipeline import apply_inpaint
    bg_full = apply_inpaint(img, full_mask, RemovalSettings(backend="lama"))
    bg = bg_full[y:y + h, x:x + w].astype(np.float32)

    # solve alpha per channel, average, clamp
    denom = np.clip(255.0 - bg, 20, None)
    alpha = np.clip((crop - bg) / denom, 0, 1).mean(axis=2)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha[alpha < 0.06] = 0.0  # drop noise floor

    # tight-crop to the sparkle extent
    ys, xs = np.where(alpha > 0.08)
    if len(xs):
        pad = 4
        x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
        y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
        alpha = alpha[y0:y1, x0:x1]

    np.save(ASSETS / "sparkle_alpha.npy", alpha.astype(np.float32))
    cv2.imwrite(str(ASSETS / "sparkle_alpha.png"), (alpha * 255).astype(np.uint8))
    # preview: template composited (white) on grey
    grey = np.full((*alpha.shape, 3), 120, np.float32)
    comp = (alpha[..., None] * 255 + (1 - alpha[..., None]) * grey).astype(np.uint8)
    cv2.imwrite(str(ASSETS / "sparkle_preview.png"),
                cv2.resize(comp, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    print(f"alpha shape {alpha.shape}  max {alpha.max():.2f}  "
          f"mean(>0) {alpha[alpha>0].mean():.2f}  -> assets/sparkle_alpha.npy")


if __name__ == "__main__":
    main()

"""Full pipeline (AI detect + LaMa fill) on the real images. Run after training."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wmr.config import RemovalSettings   # noqa: E402
from wmr import imaging                  # noqa: E402
from wmr.neural import neural_available  # noqa: E402

OUT = ROOT / "samples" / "output"
IMAGES = [
    ("lotus", r"C:\Users\Chintan Kamani\Desktop\Gemini_Generated_Image_.png"),
    ("temple", r"C:\Users\Chintan Kamani\Desktop\Gemini_Generated_Image_ (1).png"),
]

print("neural available:", neural_available())
for tag, path in IMAGES:
    data = open(path, "rb").read()
    res = imaging.process_image_bytes(data, RemovalSettings())  # AI detect + LaMa
    b, a = res["before"], res["after"]
    print(f"{tag}: mask_px={int((res['mask']>0).sum())}")
    scale = 560 / b.shape[0]
    z = lambda im: cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sep = np.full((z(b).shape[0], 8, 3), 255, np.uint8)
    cv2.imwrite(str(OUT / f"{tag}_ai_result.png"), np.hstack([z(b), sep, z(a)]))
    # zoomed corner
    h, w = b.shape[:2]
    zc = lambda im: cv2.resize(im[int(h*0.78):, int(w*0.6):], None, fx=1.8, fy=1.8,
                               interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT / f"{tag}_ai_corner.png"), np.hstack([zc(b), sep[:zc(b).shape[0]], zc(a)]))
print("wrote *_ai_result.png and *_ai_corner.png")

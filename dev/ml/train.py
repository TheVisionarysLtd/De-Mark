"""Train the sparkle U-Net on synthetic composites; validate on REAL images.

Validation runs the current model on the actual lotus + temple watermarks (whose
true locations are known) and reports how much predicted mask lands ON the mark
vs elsewhere — so we can watch real-world generalisation, not just synth loss.

Run:  python dev/ml/train.py [iters]
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from wmr.neural import build_model, INPUT, ROI_RIGHT, ROI_BOTTOM  # noqa: E402
import dev.ml.synth as synth  # noqa: E402

WEIGHTS = ROOT / "wmr" / "weights" / "sparkle_unet.pt"
WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
VAL_DIR = Path(__file__).resolve().parent / "assets" / "val"
VAL_DIR.mkdir(parents=True, exist_ok=True)

_ASSETS = Path(__file__).resolve().parent / "assets"
REAL = [
    ("lotus", r"C:\Users\Chintan Kamani\Desktop\Gemini_Generated_Image_.png", (787, 1034)),
    ("temple", r"C:\Users\Chintan Kamani\Desktop\Gemini_Generated_Image_ (1).png", (862, 898)),
    ("nb_badge", str(_ASSETS / "nb_slide.png"), (1200, 695)),
]


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return (1 - num / den).mean()


def infer(model, img_bgr, thr=0.5):
    h, w = img_bgr.shape[:2]
    rx, ry = int(w * (1 - ROI_RIGHT)), int(h * (1 - ROI_BOTTOM))
    roi = img_bgr[ry:, rx:]
    rgb = cv2.cvtColor(cv2.resize(roi, (INPUT, INPUT)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb.transpose(2, 0, 1))[None]
    with torch.no_grad():
        p = torch.sigmoid(model(t))[0, 0].numpy()
    p = cv2.resize(p, (roi.shape[1], roi.shape[0]))
    full = np.zeros((h, w), np.uint8)
    full[ry:, rx:] = (p >= thr).astype(np.uint8) * 255
    return full


def validate(model, step):
    """Return a real-image quality score: reward finding each mark, penalise
    false positives elsewhere. Used to keep the BEST checkpoint, not the latest."""
    model.eval()
    score = 0.0
    for tag, path, (tx, ty) in REAL:
        img = cv2.imread(path)
        if img is None:
            continue
        mask = infer(model, img)
        ys, xs = np.nonzero(mask)
        near = int((((xs - tx) ** 2 + (ys - ty) ** 2) < 80 ** 2).sum()) if len(xs) else 0
        total = int(len(xs))
        off = total - near
        # reward COVERAGE of the mark (up to a cap) so removal is complete, not
        # just "found"; penalise false positives elsewhere.
        score += min(near, 900) / 900.0 - 0.0012 * off
        print(f"    [{tag}] mask_px={total} on_mark={near} off_mark={off}", flush=True)
        ov = img.copy(); ov[mask > 0] = (0, 0, 255)
        cv2.imwrite(str(VAL_DIR / f"{tag}_{step:05d}.png"),
                    cv2.addWeighted(ov, 0.5, img, 0.5, 0)[int(img.shape[0]*0.75):, int(img.shape[1]*0.6):])
    model.train()
    return score


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rng = np.random.default_rng(0)
    plates = synth.ensure_plates()
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    model.train()

    t0 = time.time()
    run = 0.0
    best_score = -1e9
    for step in range(1, iters + 1):
        x, y = synth.make_batch(8, rng, plates)
        logits = model(x)
        loss = bce(logits, y) + dice_loss(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        run += loss.item()
        if step % 50 == 0:
            print(f"step {step:5d}/{iters}  loss {run/50:.4f}  "
                  f"{(time.time()-t0)/step:.2f}s/it", flush=True)
            run = 0.0
        if step % 300 == 0 or step == iters:
            sc = validate(model, step)
            torch.save(model.state_dict(), str(WEIGHTS.parent / "sparkle_unet_latest.pt"))
            if sc > best_score:
                best_score = sc
                torch.save(model.state_dict(), str(WEIGHTS))
                print(f"    score {sc:.3f}  NEW BEST -> {WEIGHTS.name}", flush=True)
            else:
                print(f"    score {sc:.3f}  (best {best_score:.3f} kept)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

"""End-to-end self-test on the synthetic samples.

Runs the image and video pipelines and writes visual artefacts to
samples/output/ so results can be eyeballed. Prints diagnostics (mask coverage,
output resolution / fps / audio preservation).

Run:  python dev/selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wmr import imaging, video          # noqa: E402
from wmr.config import RemovalSettings   # noqa: E402
from wmr.media import probe             # noqa: E402

SAMPLES = ROOT / "samples"
OUT = SAMPLES / "output"
OUT.mkdir(exist_ok=True)


def test_image() -> None:
    print("\n=== IMAGE ===")
    data = (SAMPLES / "sample_image.png").read_bytes()
    settings = RemovalSettings(sensitivity=0.55)
    res = imaging.process_image_bytes(data, settings)

    mask_px = int((res["mask"] > 0).sum())
    print(f"mask pixels: {mask_px} ({100*mask_px/res['mask'].size:.3f}% of frame)")
    print(f"output size: {res['size']}  png bytes: {len(res['png'])}")

    cv2.imwrite(str(OUT / "image_before.png"), res["before"])
    cv2.imwrite(str(OUT / "image_after.png"), res["after"])
    cv2.imwrite(str(OUT / "image_overlay.png"), res["overlay"])
    # Side-by-side strip for quick review.
    strip = np.hstack([res["before"], res["overlay"], res["after"]])
    cv2.imwrite(str(OUT / "image_triptych.png"), strip)
    print("wrote image_before/overlay/after + image_triptych.png")


def test_video() -> None:
    print("\n=== VIDEO ===")
    src = SAMPLES / "sample_video.mp4"
    out = OUT / "video_clean.mp4"
    settings = RemovalSettings(sensitivity=0.55, backend="opencv_ns")

    info = video.process_video(str(src), str(out), settings, None)
    print(f"frames: {info['frames']}  size: {info['size']}  fps: {info['fps']:.4g}")

    prev = info["preview"]
    if prev is not None:
        strip = np.hstack([prev["before"], prev["overlay"], prev["after"]])
        cv2.imwrite(str(OUT / "video_frame0_triptych.png"), strip)
        print("wrote video_frame0_triptych.png")

    src_info, out_info = probe(str(src)), probe(str(out))
    print(f"source : {src_info.width}x{src_info.height} fps~{src_info.fps} "
          f"pix_fmt={src_info.pix_fmt} audio={src_info.has_audio}")
    print(f"cleaned: {out_info.width}x{out_info.height} fps~{out_info.fps} "
          f"pix_fmt={out_info.pix_fmt} audio={out_info.has_audio}")
    same_res = (src_info.width, src_info.height) == (out_info.width, out_info.height)
    print(f"resolution preserved: {same_res}")
    print(f"output exists: {out.exists()}  bytes: {out.stat().st_size}")


if __name__ == "__main__":
    test_image()
    test_video()
    print("\nAll artefacts in:", OUT)

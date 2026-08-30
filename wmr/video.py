"""Video processing: streaming decode -> per-frame inpaint -> faithful re-encode.

Pipeline overview:

1. **Static mask** — sample frames across the clip and *vote* per pixel so only
   the persistent logo is masked (``build_static_mask``). One mask is reused for
   every frame, which removes temporal flicker a per-frame mask would cause.
2. **Stream + inpaint** — frames are pulled one at a time from ffmpeg (never the
   whole clip in RAM), the masked corner is inpainted, and frames are pushed
   straight into the encoder.
3. **Fidelity** — exact resolution (``macro_block_size=1``), source fps, source
   pixel format and color metadata are preserved; a final stream-copy mux puts
   every original audio track back with no re-encode.
"""

from __future__ import annotations

from typing import Callable, Optional

import cv2
import imageio_ffmpeg
import numpy as np

from .config import (
    MASK_SAMPLE_FRAMES,
    MASK_VOTE_RATIO,
    RemovalSettings,
    SAFE_PIX_FMTS,
)
from .mask import detect_badge_mask, detect_core_mask, pad_mask, votes_to_mask
from .media import MediaInfo, color_output_params, mux_audio, probe
from .pipeline import _use_neural, _use_sparkle, apply_inpaint, mask_overlay, resolve_roi
from . import files, neural, sparkle

ProgressCb = Optional[Callable[[Optional[float], str], None]]

_MAX_SAMPLE_STRIDE = 20  # cap how far apart mask samples are, bounding the pre-pass


def _open_reader(path: str):
    """Return (frame_generator, meta_dict). meta has 'size' (w,h) and 'fps'."""
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    return reader, meta


def _frame_to_bgr(raw: bytes, w: int, h: int) -> np.ndarray:
    rgb = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_first_frame_bgr(path: str) -> Optional[np.ndarray]:
    """Grab the first frame as BGR for a fast, pre-processing tuning preview."""
    reader, meta = _open_reader(path)
    w, h = meta["size"]
    try:
        raw = next(reader)
    except StopIteration:
        return None
    finally:
        reader.close()
    return _frame_to_bgr(raw, w, h)


def build_static_mask(path: str, settings: RemovalSettings) -> np.ndarray:
    """Vote per-frame ROI detections into one stable full-frame mask."""
    reader, meta = _open_reader(path)
    w, h = meta["size"]
    fps = meta.get("fps") or 30.0
    duration = meta.get("duration") or 0.0
    total = int(round(fps * duration)) if duration else 0

    stride = max(1, total // MASK_SAMPLE_FRAMES) if total else 5
    stride = min(stride, _MAX_SAMPLE_STRIDE)

    roi = resolve_roi(w, h, settings)
    full = np.zeros((h, w), np.uint8)

    # Manual box + force-fill needs no sampling — the whole box is the mask.
    if settings.region_mode == "manual" and settings.force_fill:
        reader.close()
        box = pad_mask(np.full((roi.h, roi.w), 255, np.uint8), settings.padding_px)
        full[roi.y:roi.y1, roi.x:roi.x1] = box
        return full

    # Corner mode: vote the AI sparkle detector + the badge template matcher
    # across sampled frames (the watermark is static). Same detectors the image
    # path uses. Fall back to classical if both find nothing.
    if settings.region_mode == "corner":
        combined = _corner_static_mask(reader, w, h, stride, settings)
        reader.close()
        if combined is not None and combined.any():
            return pad_mask(combined, settings.padding_px)
        reader, _ = _open_reader(path)

    votes = np.zeros((roi.h, roi.w), np.uint16)
    collected = 0
    try:
        for i, raw in enumerate(reader):
            if i % stride:
                continue
            bgr = _frame_to_bgr(raw, w, h)
            roi_bgr = bgr[roi.y:roi.y1, roi.x:roi.x1]
            core = detect_core_mask(roi_bgr, settings.sensitivity)
            votes += (core > 0).astype(np.uint16)
            collected += 1
            if collected >= MASK_SAMPLE_FRAMES:
                break
    finally:
        reader.close()

    roi_mask = votes_to_mask(votes, collected, MASK_VOTE_RATIO, settings.padding_px)
    full[roi.y:roi.y1, roi.x:roi.x1] = roi_mask
    return full


def _corner_static_mask(reader, w: int, h: int, stride: int, settings: RemovalSettings):
    """Build one stable mask for a video's static watermark.

    Sparkle: located by the deterministic glyph matcher on the TEMPORAL AVERAGE
    of sampled frames. A moving background (webgl mesh, dolly, panning scene)
    averages out, leaving the screen-fixed watermark sharp and easy to match,
    whereas per-frame detection drowns in background false positives. As a
    fallback for near-static clips (where the mean equals a normal frame and a
    faint sparkle may not stand out any more than on a still), the matcher is
    also voted across the individual sampled frames. Both paths carry the same
    strict verification as the image path, so a clip with no watermark yields an
    empty mask and nothing is inpainted.

    Badge: template-matched per frame and voted (it shows only when the
    background makes it visible), unioned in.
    """
    use_sparkle = _use_sparkle(settings)
    use_neural = _use_neural(settings)
    acc = np.zeros((h, w, 3), np.float64)
    votes_badge = np.zeros((h, w), np.uint16)
    votes_sparkle = np.zeros((h, w), np.uint16)
    collected = 0
    for i, raw in enumerate(reader):
        if i % stride:
            continue
        bgr = _frame_to_bgr(raw, w, h)
        acc += bgr
        if use_sparkle:
            per = sparkle.locate_sparkle(bgr, settings.sensitivity)
            if per is not None:
                votes_sparkle += (per > 0).astype(np.uint16)
        badge = detect_badge_mask(bgr)
        if badge is not None:
            votes_badge += (badge > 0).astype(np.uint16)
        collected += 1
        if collected >= MASK_SAMPLE_FRAMES:
            break
    if collected == 0:
        return None

    mask = np.zeros((h, w), np.uint8)
    plate = (acc / collected).astype(np.uint8)

    if use_sparkle:
        # Primary: match on the motion-averaged plate (screen-fixed mark stays sharp).
        found = sparkle.locate_sparkle(plate, settings.sensitivity)
        if found is not None and found.any():
            mask = cv2.bitwise_or(mask, found)
        # Fallback: a mark seen in a majority of individual frames is real too.
        elif votes_sparkle.max() >= max(2, int(round(0.5 * collected))):
            voted = np.where(votes_sparkle >= max(2, int(round(0.5 * collected))), 255, 0)
            mask = cv2.bitwise_or(mask, _largest_component(voted.astype(np.uint8)))

    if use_neural and not mask.any():
        predicted = neural.predict_full_mask(plate)
        if predicted is not None and predicted.any():
            comp = _largest_component(predicted)
            if 0 < int((comp > 0).sum()) <= 0.02 * h * w:
                mask = cv2.bitwise_or(mask, comp)

    if int(votes_badge.max()) >= 2:
        badge_mask = np.where(votes_badge >= max(2, int(round(0.3 * collected))), 255, 0)
        mask = cv2.bitwise_or(mask, badge_mask.astype(np.uint8))
    return mask


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected component (the single static watermark)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == biggest, 255, 0).astype(np.uint8)


def _choose_pix_fmt(info: MediaInfo) -> str:
    if info.pix_fmt and info.pix_fmt in SAFE_PIX_FMTS:
        return info.pix_fmt
    return "yuv420p"


def process_video(input_path: str, output_path: str, settings: RemovalSettings,
                  progress_cb: ProgressCb = None) -> dict:
    """Remove the watermark from every frame and write ``output_path``.

    Returns a summary dict including a first-frame before/after/overlay preview.
    """
    info = probe(input_path)
    reader, meta = _open_reader(input_path)
    w, h = meta["size"]
    fps = float(meta.get("fps") or info.fps or 30.0)
    duration = meta.get("duration") or 0.0
    total = int(round(fps * duration)) if duration else 0

    # LaMa only fills the small watermark crop per frame, so it's fast enough
    # (~0.1s/frame) and reconstructs texture far better than OpenCV — use it for
    # "auto". Users can pick OpenCV in Advanced for very long clips.
    inpaint_settings = settings

    if progress_cb:
        progress_cb(0.0, "Analysing watermark across frames…")
    mask = build_static_mask(input_path, settings)

    pix_fmt_out = _choose_pix_fmt(info)
    output_params = ["-crf", str(settings.crf), "-preset", settings.preset]
    output_params += color_output_params(info)

    video_only = files.new_temp_path(".mp4")
    preview = None
    frames_done = 0

    writer = imageio_ffmpeg.write_frames(
        video_only,
        size=(w, h),
        fps=fps,
        pix_fmt_in="rgb24",
        pix_fmt_out=pix_fmt_out,
        codec="libx264",
        quality=None,           # skip library CRF; we supply our own below
        bitrate=None,
        macro_block_size=1,     # keep the exact source resolution
        output_params=output_params,
    )
    writer.send(None)  # prime the generator

    try:
        for i, raw in enumerate(reader):
            bgr = _frame_to_bgr(raw, w, h)
            cleaned = apply_inpaint(bgr, mask, inpaint_settings)

            if i == 0:
                preview = {
                    "before": bgr.copy(),
                    "after": cleaned.copy(),
                    "overlay": mask_overlay(bgr, mask),
                }

            out_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
            writer.send(np.ascontiguousarray(out_rgb).tobytes())
            frames_done += 1

            if progress_cb:
                frac = (i + 1) / total if total else None
                progress_cb(frac, f"Cleaning frame {i + 1}" + (f"/{total}" if total else ""))
    finally:
        writer.close()
        reader.close()

    if progress_cb:
        progress_cb(0.98, "Restoring audio & finalising…")

    muxed = info.has_audio and mux_audio(video_only, input_path, output_path)
    if not muxed:
        _copy_file(video_only, output_path)
    files.safe_unlink(video_only)

    if progress_cb:
        progress_cb(1.0, "Done")

    return {
        "output": output_path,
        "frames": frames_done,
        "size": (w, h),
        "fps": fps,
        "has_audio": bool(muxed),
        "mask": mask,
        "preview": preview,
    }


def _copy_file(src: str, dst: str) -> None:
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        while True:
            chunk = fsrc.read(1024 * 1024)
            if not chunk:
                break
            fdst.write(chunk)

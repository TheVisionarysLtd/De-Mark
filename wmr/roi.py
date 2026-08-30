"""Region-of-Interest geometry.

Watermarks are confined to the bottom-right corner, so the whole pipeline only
ever touches this rectangle — the rest of the frame is guaranteed untouched.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in pixel coordinates (top-left origin)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h


def _clamp_fraction(value: float, lo: float = 0.01, hi: float = 1.0) -> float:
    """Keep a fraction inside a sane range so ROI math never degenerates."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def compute_roi(frame_w: int, frame_h: int, bottom_fraction: float,
                right_fraction: float) -> Rect:
    """Return the bottom-right ROI rectangle for a frame of the given size.

    Args:
        frame_w, frame_h: full frame dimensions in pixels.
        bottom_fraction: portion of the height (from the bottom) to include.
        right_fraction: portion of the width (from the right) to include.
    """
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"Invalid frame size: {frame_w}x{frame_h}")

    bottom_fraction = _clamp_fraction(bottom_fraction)
    right_fraction = _clamp_fraction(right_fraction)

    roi_h = max(1, int(round(frame_h * bottom_fraction)))
    roi_w = max(1, int(round(frame_w * right_fraction)))
    x = frame_w - roi_w
    y = frame_h - roi_h
    return Rect(x=x, y=y, w=roi_w, h=roi_h)


def compute_manual_roi(frame_w: int, frame_h: int, center_x: float, center_y: float,
                       box_w: float, box_h: float) -> Rect:
    """Return a user-placed box (all args as fractions of the frame), clamped in-frame."""
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"Invalid frame size: {frame_w}x{frame_h}")

    w = max(1, int(round(_clamp_fraction(box_w) * frame_w)))
    h = max(1, int(round(_clamp_fraction(box_h) * frame_h)))
    cx = _clamp_fraction(center_x) * frame_w
    cy = _clamp_fraction(center_y) * frame_h

    x = int(round(min(max(cx - w / 2, 0), frame_w - w)))
    y = int(round(min(max(cy - h / 2, 0), frame_h - h)))
    return Rect(x=x, y=y, w=w, h=h)

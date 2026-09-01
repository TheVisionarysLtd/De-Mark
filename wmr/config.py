"""Central configuration constants and the immutable settings object.

Every tunable the pipeline uses lives here so behaviour is predictable and the
UI has a single source of truth for defaults and ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Region of Interest (fraction of the full frame) -------------------------
# The Gemini sparkle and NotebookLM / Gemini Notebook badges both sit in the
# bottom-right corner, so the search window is the bottom band x right column.
ROI_BOTTOM_FRACTION = 0.16  # bottom 16% of the height (marks often sit >12% up)
ROI_RIGHT_FRACTION = 0.28   # rightmost 28% of the width

# --- Mask detection ----------------------------------------------------------
DEFAULT_SENSITIVITY = 0.78   # 0..1 — higher captures fainter / lower-contrast marks.
# Raised 0.55 -> 0.78 (30 Aug 2026): a faint Gemini sparkle over a dark, busy background
# (a Pichwai painting) matched at corr 0.71 / saturation 55 — just past the 0.55 gates —
# so auto-detect wrongly said "nothing". 0.78 catches it with margin and was verified to
# add NO false positives (business card, ornate art, and 21 deck pages all stay clean).
DEFAULT_PADDING_PX = 8       # dilation halo so inpainting overshoots the mark edge
MIN_TOPHAT_KERNEL = 7        # smallest morphological kernel (px), keeps tiny ROIs sane
TOPHAT_KERNEL_RATIO = 0.45   # kernel size as a fraction of the smaller ROI dimension
SPECK_MIN_AREA_RATIO = 0.0006  # drop connected components smaller than this * ROI area
HALO_NEIGHBORHOOD_RATIO = 0.15  # search radius (of ROI) for a watermark's bright glow
HALO_MIN_RADIUS = 11         # minimum halo search radius in px

# --- Inpainting --------------------------------------------------------------
DEFAULT_INPAINT_RADIUS = 4   # OpenCV inpaint neighbourhood radius (px)
INPAINT_EXPAND_PX = 22       # context margin around the mask bbox fed to the inpainter
FEATHER_SIGMA = 2.5          # soft-edge blend width (px) so the fill has no hard seam
VALID_BACKENDS = ("auto", "opencv_ns", "opencv_telea", "lama")

# --- Video -------------------------------------------------------------------
MASK_SAMPLE_FRAMES = 24      # frames sampled to build a flicker-free static mask
MASK_VOTE_RATIO = 0.35       # keep a pixel if flagged in >= this ratio of samples
DEFAULT_CRF = 15             # libx264 quality (lower = better); 15 is ~visually lossless
DEFAULT_PRESET = "slow"      # x264 speed/efficiency trade-off
SAFE_PIX_FMTS = {"yuv420p", "yuv422p", "yuv444p", "yuvj420p", "yuvj422p", "yuvj444p"}

# --- Supported media ---------------------------------------------------------
SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
SUPPORTED_VIDEO_EXT = {".mp4", ".mov"}
SUPPORTED_PDF_EXT = {".pdf"}


@dataclass(frozen=True)
class RemovalSettings:
    """Immutable bundle of every knob the pipeline reads.

    Frozen so a settings object can be used as a cache key and can never be
    mutated underneath an in-flight video job.
    """

    bottom_fraction: float = ROI_BOTTOM_FRACTION
    right_fraction: float = ROI_RIGHT_FRACTION
    sensitivity: float = DEFAULT_SENSITIVITY
    padding_px: int = DEFAULT_PADDING_PX
    inpaint_radius: int = DEFAULT_INPAINT_RADIUS
    backend: str = "auto"
    crf: int = DEFAULT_CRF
    preset: str = DEFAULT_PRESET

    # Detector for corner mode: "auto" (AI net if available, else classical),
    # "neural" (force AI net), or "classic" (force the CV detector).
    detector: str = "auto"

    # Region mode: "corner" (auto bottom-right) or "manual" (user-placed box).
    region_mode: str = "corner"
    center_x: float = 0.90     # manual box centre, fraction of width
    center_y: float = 0.90     # manual box centre, fraction of height
    box_w: float = 0.18        # manual box width, fraction of width
    box_h: float = 0.14        # manual box height, fraction of height
    force_fill: bool = False   # manual: inpaint the whole box (for tricky backgrounds)

    def cache_key(self) -> tuple:
        """Hashable identity used to memoise processed output per input file."""
        return (
            round(self.bottom_fraction, 4),
            round(self.right_fraction, 4),
            round(self.sensitivity, 4),
            self.padding_px,
            self.inpaint_radius,
            self.backend,
            self.crf,
            self.preset,
            self.detector,
            self.region_mode,
            round(self.center_x, 4),
            round(self.center_y, 4),
            round(self.box_w, 4),
            round(self.box_h, 4),
            self.force_fill,
        )

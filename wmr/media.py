"""FFmpeg-facing helpers: binary discovery, metadata probing, audio muxing.

We shell out to the ffmpeg binary bundled with ``imageio-ffmpeg`` so the app has
zero external install requirements. Probing parses ``ffmpeg -i`` stderr because
ffprobe is not bundled; parsing is wrapped so a probe failure never aborts a job.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

import imageio_ffmpeg

# Hide the console window ffmpeg would otherwise flash on Windows.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0


@dataclass(frozen=True)
class MediaInfo:
    """Best-effort source metadata used to preserve fidelity on re-encode."""

    width: int = 0
    height: int = 0
    fps: float = 30.0
    has_audio: bool = False
    pix_fmt: str | None = None
    color_range: str | None = None
    color_primaries: str | None = None
    color_transfer: str | None = None
    color_space: str | None = None
    raw: str = field(default="", repr=False)


def ffmpeg_exe() -> str:
    """Absolute path to the bundled ffmpeg binary."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, creationflags=_NO_WINDOW
    )


def probe(path: str) -> MediaInfo:
    """Extract fidelity-relevant metadata from a media file (best effort)."""
    proc = _run([ffmpeg_exe(), "-hide_banner", "-i", str(path)])
    text = proc.stderr or ""

    video_line = ""
    for line in text.splitlines():
        if "Video:" in line:
            video_line = line
            break

    width = height = 0
    dims = re.search(r",\s*(\d{2,5})x(\d{2,5})", video_line)
    if dims:
        width, height = int(dims.group(1)), int(dims.group(2))

    fps = 30.0
    fps_match = re.search(r"(\d+(?:\.\d+)?)\s+fps", text)
    if fps_match:
        fps = float(fps_match.group(1))

    pix_fmt = None
    pf = re.search(r"Video:\s*[\w]+[^,]*,\s*([a-z][\w]+)", video_line)
    if pf:
        pix_fmt = pf.group(1)

    # Color descriptors live in the parenthetical, e.g. "yuv420p(tv, bt709)".
    paren = re.search(r"\(([^)]*)\)", video_line)
    tokens = [t.strip() for t in paren.group(1).split(",")] if paren else []
    flat = " ".join(tokens)

    color_range = _first(flat, ["tv", "pc", "limited", "full"])
    primaries = _first(flat, ["bt709", "bt2020", "bt470bg", "smpte170m", "smpte240m"])
    transfer = _first(
        flat, ["bt709", "bt2020-10", "smpte2084", "arib-std-b67", "smpte170m", "gamma22"]
    )
    space = _first(
        flat, ["bt709", "bt2020nc", "bt2020c", "smpte170m", "bt470bg", "fcc"]
    )

    return MediaInfo(
        width=width,
        height=height,
        fps=fps,
        has_audio=" Audio:" in text,
        pix_fmt=pix_fmt,
        color_range={"tv": "tv", "limited": "tv", "pc": "pc", "full": "pc"}.get(color_range),
        color_primaries=primaries,
        color_transfer=transfer,
        color_space=space,
        raw=text,
    )


def _first(haystack: str, needles: list[str]) -> str | None:
    """Return the first needle present as a standalone token in haystack."""
    for n in needles:
        if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", haystack):
            return n
    return None


def color_output_params(info: MediaInfo) -> list[str]:
    """Encoder flags that stamp the source's color metadata onto the output."""
    params: list[str] = []
    if info.color_range:
        params += ["-color_range", info.color_range]
    if info.color_primaries:
        params += ["-color_primaries", info.color_primaries]
    if info.color_transfer:
        params += ["-color_trc", info.color_transfer]
    if info.color_space:
        params += ["-colorspace", info.color_space]
    return params


def mux_audio(video_path: str, source_path: str, out_path: str) -> bool:
    """Copy every audio track from ``source_path`` onto the processed video.

    Uses stream copy (``-c copy``) so neither the freshly-encoded video nor the
    original audio is re-encoded — no extra quality loss, all tracks preserved.
    Returns True on success.
    """
    args = [
        ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(source_path),
        "-map", "0:v:0",
        "-map", "1:a?",          # all audio tracks, optional (no error if none)
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = _run(args)
    return proc.returncode == 0

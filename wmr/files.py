"""File-type validation and temp-file helpers.

Kept separate so the UI never trusts a filename without going through these
guards, and so temp-file creation is consistent across the app.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from .config import SUPPORTED_IMAGE_EXT, SUPPORTED_PDF_EXT, SUPPORTED_VIDEO_EXT

_TEMP_PREFIX = "wmr_"


def ext_of(name: str) -> str:
    return Path(name).suffix.lower()


def is_image(name: str) -> bool:
    return ext_of(name) in SUPPORTED_IMAGE_EXT


def is_video(name: str) -> bool:
    return ext_of(name) in SUPPORTED_VIDEO_EXT


def is_pdf(name: str) -> bool:
    return ext_of(name) in SUPPORTED_PDF_EXT


def is_supported(name: str) -> bool:
    return is_image(name) or is_video(name) or is_pdf(name)


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def write_temp(data: bytes, suffix: str) -> str:
    """Persist bytes to a uniquely-named temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=_TEMP_PREFIX)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return path


def new_temp_path(suffix: str) -> str:
    """Reserve a unique temp path (file created empty, handle closed)."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=_TEMP_PREFIX)
    os.close(fd)
    return path


def safe_unlink(*paths: str) -> None:
    """Delete temp files, ignoring any that are already gone or locked."""
    for path in paths:
        if not path:
            continue
        try:
            os.unlink(path)
        except OSError:
            pass

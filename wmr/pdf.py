"""PDF slide-deck watermark removal via PyMuPDF (fitz).

Each page is rendered to an image, the watermark is detected and inpainted, and
only the cleaned watermark PATCH is overlaid back onto the original page — so the
rest of every slide (text, vectors, images) stays exactly as it was. Falls back
gracefully if fitz is unavailable.
"""

from __future__ import annotations

import importlib.util
from typing import Callable, Optional

import cv2
import numpy as np

from .config import RemovalSettings
from .pipeline import apply_inpaint, build_frame_mask

PDF_RENDER_DPI = 150  # render resolution for detection + patch

ProgressCb = Optional[Callable[[Optional[float], str], None]]


def pdf_available() -> bool:
    try:
        return importlib.util.find_spec("fitz") is not None
    except Exception:
        return False


def _page_to_bgr(page, zoom: float) -> np.ndarray:
    import fitz

    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_RGB2BGR)


def first_page_bgr(input_path: str, dpi: int = PDF_RENDER_DPI) -> Optional[np.ndarray]:
    """Render page 1 to a BGR image (for the visual watermark picker), or None."""
    try:
        import fitz
    except Exception:
        return None
    doc = fitz.open(input_path)
    try:
        if doc.page_count == 0:
            return None
        return _page_to_bgr(doc[0], dpi / 72.0)
    finally:
        doc.close()


def process_pdf(input_path: str, output_path: str, settings: RemovalSettings,
                progress_cb: ProgressCb = None) -> dict:
    """Remove watermarks from every page and write a cleaned PDF.

    Returns a summary including a first-cleaned-page before/after preview.
    """
    import fitz

    doc = fitz.open(input_path)
    zoom = PDF_RENDER_DPI / 72.0
    total = doc.page_count
    cleaned_pages = 0
    preview = None

    try:
        for i in range(total):
            page = doc[i]
            bgr = _page_to_bgr(page, zoom)
            mask = build_frame_mask(bgr, settings)

            if mask.any():
                filled = apply_inpaint(bgr, mask, settings)
                ys, xs = np.nonzero(mask)
                pad = 10
                x0 = max(0, int(xs.min()) - pad)
                x1 = min(bgr.shape[1], int(xs.max()) + 1 + pad)
                y0 = max(0, int(ys.min()) - pad)
                y1 = min(bgr.shape[0], int(ys.max()) + 1 + pad)
                patch = filled[y0:y1, x0:x1]

                ok, buf = cv2.imencode(".png", patch)
                if ok:
                    r = page.rect
                    rect = fitz.Rect(r.x0 + x0 / zoom, r.y0 + y0 / zoom,
                                     r.x0 + x1 / zoom, r.y0 + y1 / zoom)
                    page.insert_image(rect, stream=buf.tobytes(), keep_proportion=False,
                                      overlay=True)
                    cleaned_pages += 1
                    if preview is None:
                        preview = {"before": bgr.copy(), "after": filled}

            if progress_cb:
                progress_cb((i + 1) / total, f"Page {i + 1}/{total}")

        doc.save(output_path, garbage=4, deflate=True)
    finally:
        doc.close()

    return {"output": output_path, "pages": total, "cleaned_pages": cleaned_pages,
            "preview": preview}

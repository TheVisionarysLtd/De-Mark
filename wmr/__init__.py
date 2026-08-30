"""Watermark Remover — computer-vision engine for detecting and inpainting
AI-tool branding watermarks (Gemini sparkle, NotebookLM / Gemini Notebook badge)
out of images and videos.

Public engine surface is intentionally small so the Streamlit UI (``app.py``)
and any batch scripts share exactly one implementation.
"""

from .config import RemovalSettings

__all__ = ["RemovalSettings"]
__version__ = "1.0.0"

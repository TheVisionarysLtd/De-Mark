"""Presentation layer for De:Mark — CSS, header, and the before/after slider.

Kept apart from app.py so the styling is easy to iterate on without touching the
flow logic, and apart from wmr/ (the engine) which stays framework-agnostic.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

try:
    from PIL import Image as _PILImage
    from streamlit_drawable_canvas import st_canvas as _st_canvas
    _HAS_CANVAS = True
except Exception:                       # optional dependency — degrade to sliders
    _HAS_CANVAS = False

# streamlit-drawable-canvas 0.9.3 calls streamlit.elements.image.image_to_url to
# serve the canvas background. Newer Streamlit (>=1.x) MOVED that helper to
# streamlit.elements.lib.image_utils and changed its 2nd argument (int width ->
# LayoutConfig), so the old call raises AttributeError and the canvas never
# renders (the same version rot that broke the previous drag component). Bridge
# the old call site to the real, relocated helper — it registers the image with
# Streamlit's media manager and returns a /media/ URL the component can fetch, so
# the background shows and drag-to-box works. Local and Streamlit Cloud run the
# same Streamlit, so verifying locally covers the cloud. If the internal API
# shifts again, fall back to sliders-only rather than offering a broken canvas.
if _HAS_CANVAS:
    try:
        import streamlit.elements.image as _st_image_mod

        if not hasattr(_st_image_mod, "image_to_url"):
            from streamlit.elements.lib.image_utils import WidthBehavior as _WB
            from streamlit.elements.lib.image_utils import image_to_url as _real_img_to_url
            from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig

            def _image_to_url(image, width=None, clamp=False, channels="RGB",
                              output_format="PNG", image_id="", **_kw):
                return _real_img_to_url(image, _LayoutConfig(width=_WB.ORIGINAL),
                                        clamp, channels, output_format, image_id)

            _st_image_mod.image_to_url = _image_to_url
    except Exception:
        _HAS_CANVAS = False       # can't bridge on this Streamlit -> sliders only

ACCENT = "#0A84FF"

# A bare click (no drag) expands into a box at least this fraction of the frame.
_MIN_BOX_FRAC = 0.05

# The Visionarys Ltd branding — De:Mark is built by The Visionarys.
TVL_URL = "https://www.thevisionarys.com/"
_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "tvl-logo.png"
_logo_b64_cache: str | None = None


def _tvl_logo_b64() -> str:
    """Base64 of the Visionarys logo (embedded so it's self-contained), or ''."""
    global _logo_b64_cache
    if _logo_b64_cache is None:
        try:
            _logo_b64_cache = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
        except Exception:
            _logo_b64_cache = ""
    return _logo_b64_cache

# --------------------------------------------------------------------------- #
# Global stylesheet — a calm, Apple-grade light system.
# --------------------------------------------------------------------------- #
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700&display=swap');

:root{
  --surface:#FFFFFF; --glass:rgba(255,255,255,.72); --glass-2:rgba(255,255,255,.5);
  --text:#0E0F24; --text2:#565A76; --text3:#8B90A9;
  --line:#E7E8F3; --radius:24px;
  --accent:#6D5EF6; --accent2:#3B82F6; --accent3:#22D3EE;
  --grad:linear-gradient(115deg,#6D5EF6 0%,#3B82F6 52%,#22D3EE 100%);
  --shadow:0 2px 4px rgba(20,18,60,.04), 0 26px 60px -16px rgba(60,50,140,.22);
  --shadow-sm:0 1px 2px rgba(20,18,60,.05), 0 10px 30px -12px rgba(60,50,140,.16);
  --shadow-glow:0 16px 38px -10px rgba(109,94,246,.48);
}
html, body, [class*="css"], .stApp, button, input, textarea, select{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'SF Pro Display','Segoe UI',sans-serif !important;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
}
.stApp{ background:
  radial-gradient(1100px 620px at 3% -14%, rgba(109,94,246,.26), transparent 58%),
  radial-gradient(1000px 560px at 104% -8%, rgba(59,130,246,.20), transparent 55%),
  radial-gradient(820px 560px at 62% 120%, rgba(34,211,238,.15), transparent 60%),
  radial-gradient(680px 460px at 92% 44%, rgba(244,114,182,.10), transparent 55%),
  linear-gradient(180deg,#FAFBFF 0%, #F0F1F9 100%);
  background-attachment:fixed; }
/* fine grain so the gradient reads as a surface, not a flat wash */
.stApp::before{ content:""; position:fixed; inset:0; z-index:0; pointer-events:none; opacity:.55;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.04'/%3E%3C/svg%3E"); }
[data-testid="stAppViewContainer"]>*{ position:relative; z-index:1; }
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"]{ display:none !important; }
.block-container{ padding-top:1.5rem; padding-bottom:4.5rem; max-width:1080px; }
h1,h2,h3,h4{ color:var(--text); letter-spacing:-.024em; font-weight:700; }
@keyframes dmUp{ from{opacity:0; transform:translateY(16px);} to{opacity:1; transform:none;} }
@keyframes dmFloat{ 0%,100%{transform:translate(0,0) scale(1);} 50%{transform:translate(26px,-18px) scale(1.08);} }
@keyframes dmShine{ to{ background-position:200% center; } }

/* ---- Dark premium sidebar ------------------------------------------------ */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#191A2C 0%, #14152300 0%, #14152A 100%),
             linear-gradient(180deg,#1B1C2F, #131426);
  border-right:1px solid rgba(255,255,255,.05); }
[data-testid="stSidebar"] .block-container{ padding-top:1.5rem; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] li, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] div{ color:#E9EAF4; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] small{ color:#9AA0BC !important; }
[data-testid="stSidebar"] [data-testid="stExpander"]{
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); border-radius:16px; box-shadow:none; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary{ color:#E9EAF4; }
[data-testid="stSidebar"] [data-baseweb="select"]>div{
  background:rgba(255,255,255,.06) !important; border-color:rgba(255,255,255,.12) !important; color:#E9EAF4; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stTickBar"]{ background:transparent; }

/* ---- File uploader ------------------------------------------------------- */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]{
  border:1.5px dashed #C9CBE0; border-radius:var(--radius);
  background:var(--glass); backdrop-filter:blur(12px);
  padding:2.8rem 1.6rem; box-shadow:var(--shadow);
  transition:border-color .25s ease, box-shadow .25s ease, transform .2s ease; }
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover{
  border-color:var(--accent); box-shadow:var(--shadow-glow); transform:translateY(-2px); }
[data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzone"] small{ color:var(--text3) !important; }
[data-testid="stFileUploaderDropzone"] button{
  background:var(--grad) !important; color:#fff !important; border:none !important;
  border-radius:980px !important; padding:.5rem 1.3rem !important; font-weight:600 !important;
  box-shadow:0 8px 20px rgba(76,60,200,.30) !important; }

/* ---- Buttons ------------------------------------------------------------- */
.stButton>button, .stDownloadButton>button{
  border-radius:980px !important; font-weight:600 !important; letter-spacing:-.01em;
  border:1px solid var(--line) !important; background:var(--surface); color:var(--text);
  padding:.6rem 1.25rem !important; box-shadow:var(--shadow-sm);
  transition:transform .12s ease, box-shadow .2s ease, background .2s ease; }
.stButton>button:hover, .stDownloadButton>button:hover{ transform:translateY(-1px); }
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"]{
  background:var(--grad) !important; color:#fff !important; border:none !important;
  box-shadow:0 12px 28px rgba(76,60,200,.34) !important; }
[data-testid="stSidebar"] .stButton>button{
  background:rgba(255,255,255,.06); color:#E9EAF4; border:1px solid rgba(255,255,255,.12) !important; box-shadow:none; }

/* ---- Segmented control (the big Auto / Select-area switch) --------------- */
[data-testid="stSegmentedControl"]{ display:flex !important; width:100% !important;
  gap:.5rem; background:#EAECF5; border:1px solid var(--line); padding:.45rem;
  border-radius:20px; box-shadow:inset 0 1px 3px rgba(30,20,80,.06); }
[data-testid="stSegmentedControl"] > div,
[data-testid="stSegmentedControl"] [role="radiogroup"]{ width:100%; display:flex; gap:.5rem; }
[data-testid="stSegmentedControl"] button, [data-testid="stSegmentedControl"] label{
  flex:1 1 0 !important; min-width:0 !important; justify-content:center !important;
  border:none !important; background:transparent !important; color:var(--text2) !important;
  font-weight:700 !important; font-size:1.1rem !important; border-radius:15px !important;
  padding:.9rem 1rem !important; transition:background .15s ease, box-shadow .2s ease; }
[data-testid="stSegmentedControl"] button p, [data-testid="stSegmentedControl"] label p{
  font-size:1.1rem !important; font-weight:700 !important; }
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"],
[data-testid="stSegmentedControl"] label:has(input:checked){
  background:#fff !important; color:var(--accent) !important; box-shadow:var(--shadow); }

/* ---- Misc accents -------------------------------------------------------- */
[data-baseweb="slider"] [role="slider"]{ background:var(--accent) !important; }
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:16px; background:var(--surface); box-shadow:var(--shadow-sm); }
[data-testid="stExpander"] summary{ font-weight:600; }
[data-testid="stCaptionContainer"], .stCaption{ color:var(--text2); }
[data-testid="stImage"] img{ border-radius:16px; }
hr{ border-color:var(--line); }

/* ---- De:Mark top bar ----------------------------------------------------- */
.dm-top{ display:flex; align-items:center; justify-content:space-between; gap:1rem;
  margin:.1rem 0 1.7rem; animation:dmUp .5s ease both; }
.dm-word{ font-family:'Space Grotesk',Inter,sans-serif; font-size:1.5rem; font-weight:700;
  letter-spacing:-.03em; color:var(--text); }
.dm-word .dm-colon{ background:var(--grad); -webkit-background-clip:text; background-clip:text; color:transparent; }

/* ---- De:Mark hero -------------------------------------------------------- */
.dm-hero{ position:relative; display:flex; flex-direction:row; align-items:center;
  gap:2.6rem; margin:.3rem 0 2.2rem; animation:dmUp .7s ease both; }
.dm-hero-text{ flex:1 1 500px; min-width:0; display:flex; flex-direction:column;
  align-items:flex-start; gap:1.05rem; }
.dm-hero-visual{ flex:0 1 360px; }
@media (max-width:920px){ .dm-hero{ flex-direction:column; align-items:flex-start; }
  .dm-hero-visual{ width:100%; max-width:440px; } }
.dm-aurora{ position:absolute; top:-140px; left:-90px; width:480px; height:360px; z-index:-1;
  background:conic-gradient(from 120deg at 50% 50%, rgba(109,94,246,.5), rgba(59,130,246,.42),
    rgba(34,211,238,.42), rgba(244,114,182,.36), rgba(109,94,246,.5));
  filter:blur(72px); opacity:.6; border-radius:50%; animation:dmFloat 16s ease-in-out infinite; }

/* before/after product preview card (pure CSS, evokes the real slider) */
.dm-shot{ border-radius:20px; overflow:hidden; background:#fff; border:1px solid var(--line);
  box-shadow:var(--shadow); transform:rotate(1.4deg); transition:transform .45s ease; }
.dm-shot:hover{ transform:rotate(0deg) translateY(-4px); }
.dm-shot-bar{ display:flex; align-items:center; gap:.4rem; padding:.6rem .85rem;
  background:#F7F8FC; border-bottom:1px solid var(--line); }
.dm-shot-bar b{ width:9px; height:9px; border-radius:50%; background:#E2E4EF; }
.dm-shot-bar em{ margin-left:auto; font-style:normal; font-weight:700; font-size:.72rem;
  color:var(--text3); font-family:'Space Grotesk',sans-serif; }
.dm-shot-img{ position:relative; height:236px; overflow:hidden;
  background:conic-gradient(from 25deg at 55% 45%,#f0abfc,#a5b4fc,#5eead4,#fde68a,#fca5a5,#f0abfc); }
.dm-shot-wm{ position:absolute; left:50%; bottom:44px; transform:translateX(-50%);
  width:34px; height:34px; display:flex; align-items:center; justify-content:center;
  color:#fff; font-size:1.55rem; text-shadow:0 2px 8px rgba(0,0,0,.3); }
.dm-shot-pill{ position:absolute; left:50%; bottom:12px; transform:translateX(-50%);
  display:inline-flex; align-items:center; gap:.3rem; white-space:nowrap;
  font-size:.62rem; font-weight:600; color:#fff; background:rgba(20,20,30,.38);
  backdrop-filter:blur(3px); border-radius:8px; padding:.2rem .5rem; }
/* the clean (After) half is drawn OVER the mark to erase it on the right side */
.dm-shot-clean{ position:absolute; inset:0; clip-path:inset(0 0 0 50%);
  background:conic-gradient(from 25deg at 55% 45%,#f0abfc,#a5b4fc,#5eead4,#fde68a,#fca5a5,#f0abfc); }
.dm-shot-div{ position:absolute; top:0; bottom:0; left:50%; width:2px; background:rgba(255,255,255,.92);
  box-shadow:0 0 0 1px rgba(0,0,0,.04); }
.dm-shot-div i{ position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
  width:28px; height:28px; border-radius:50%; background:#fff; box-shadow:var(--shadow-sm);
  display:flex; align-items:center; justify-content:center; color:var(--text3); font-size:.7rem; font-style:normal; }
.dm-shot-tags{ display:flex; justify-content:space-between; padding:.55rem .85rem;
  font-size:.68rem; font-weight:700; color:var(--text3); letter-spacing:.04em; }
.dm-shot-tags b{ color:var(--accent2); }
.dm-badge{ display:inline-flex; align-items:center; gap:.5rem; font-size:.8rem; font-weight:600;
  color:var(--text2); background:var(--glass); backdrop-filter:blur(10px); border:1px solid var(--line);
  border-radius:999px; padding:.42rem .95rem; box-shadow:var(--shadow-sm); }
.dm-badge i{ font-style:normal; font-size:.95rem; background:var(--grad);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.dm-h1{ font-family:'Space Grotesk',Inter,sans-serif; font-size:clamp(2.5rem,6vw,3.85rem);
  font-weight:700; line-height:1.03; letter-spacing:-.035em; color:var(--text); margin:0; }
.dm-h1 .dm-grad{ background:linear-gradient(115deg,#6D5EF6,#3B82F6 42%,#22D3EE 72%,#6D5EF6);
  background-size:200% auto; -webkit-background-clip:text; background-clip:text; color:transparent;
  animation:dmShine 6s linear infinite; }
.dm-sub{ font-size:1.1rem; color:var(--text2); max-width:640px; line-height:1.55; margin:0; }
.dm-sub b{ color:var(--text); font-weight:600; }
.dm-chips{ display:flex; gap:.55rem; flex-wrap:wrap; margin-top:.25rem; }
.dm-chip{ display:inline-flex; align-items:center; gap:.5rem; font-size:.8rem; font-weight:600;
  color:var(--text2); background:var(--glass); backdrop-filter:blur(8px); border:1px solid var(--line);
  padding:.42rem .85rem; border-radius:999px; box-shadow:var(--shadow-sm);
  transition:transform .16s ease, box-shadow .22s ease; }
.dm-chip:hover{ transform:translateY(-2px); box-shadow:var(--shadow); }
.dm-chip b{ color:var(--text); }
.dm-chip .dot{ width:8px; height:8px; border-radius:50%; box-shadow:0 0 0 3px rgba(109,94,246,.10); }
.dm-chip .dot.v{ background:#6D5EF6; } .dm-chip .dot.b{ background:#3B82F6; }
.dm-chip .dot.c{ background:#22D3EE; } .dm-chip .dot.g{ background:#34D399; }
.dm-section{ font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase;
  color:var(--text3); margin:1.1rem 0 .7rem; }
.dm-card{ background:var(--glass); backdrop-filter:blur(12px); border:1px solid var(--line);
  border-radius:var(--radius); box-shadow:var(--shadow); padding:1.2rem 1.35rem; }
.dm-foot{ color:var(--text3); font-size:.82rem; text-align:center; margin-top:3rem; display:flex;
  flex-direction:column; align-items:center; gap:.4rem; }
.dm-foot a{ color:var(--text); text-decoration:none; font-weight:700; }
.dm-foot a:hover{ color:var(--accent); }
.dm-foot-brand{ display:inline-flex; align-items:center; gap:.5rem; }
.dm-foot-brand img{ width:20px; height:20px; object-fit:contain; }

/* ---- "Built by The Visionarys" brand link (top-right) ------------------- */
.dm-brandby a{ display:inline-flex; align-items:center; gap:.55rem; text-decoration:none;
  background:var(--glass); backdrop-filter:blur(10px); border:1px solid var(--line);
  border-radius:999px; padding:.34rem .9rem .34rem .5rem; box-shadow:var(--shadow-sm);
  transition:transform .14s ease, box-shadow .22s ease; }
.dm-brandby a:hover{ transform:translateY(-1px); box-shadow:var(--shadow); }
.dm-brandby img{ width:24px; height:24px; object-fit:contain; border-radius:7px; }
.dm-brandby span{ font-size:.8rem; font-weight:500; color:var(--text2); }
.dm-brandby b{ color:var(--text); font-weight:600; }

/* ---- Sidebar brand ------------------------------------------------------- */
.dm-side-brand{ display:flex; align-items:center; gap:.6rem; margin:.2rem 0 1.4rem; }
.dm-side-badge{ width:40px; height:40px; border-radius:13px; background:var(--grad);
  display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; font-size:1.15rem;
  font-family:'Space Grotesk',sans-serif; box-shadow:var(--shadow-glow); }
.dm-side-name{ font-family:'Space Grotesk',sans-serif; font-size:1.2rem; font-weight:700;
  letter-spacing:-.02em; color:#F3F4FB; }
.dm-side-name .c{ background:linear-gradient(115deg,#8AA0FF,#22D3EE);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def wide_mode() -> None:
    """Widen the content area to the full window — used for the Select-area
    picker so the image (and a tiny watermark on it) is as large as possible."""
    st.markdown(
        "<style>.block-container{max-width:100% !important;"
        "padding-left:2.2rem !important;padding-right:2.2rem !important;}</style>",
        unsafe_allow_html=True)


def header() -> None:
    logo = _tvl_logo_b64()
    brand = ""
    if logo:
        brand = (
            f'<div class="dm-brandby"><a href="{TVL_URL}" target="_blank" rel="noopener noreferrer">'
            f'<img src="data:image/png;base64,{logo}" alt="The Visionarys Ltd logo"/>'
            f'<span>Built by <b>The Visionarys Ltd</b></span></a></div>'
        )
    st.markdown(
        f"""
        <div class="dm-top">
          <div class="dm-word">De<span class="dm-colon">:</span>Mark</div>
          {brand}
        </div>
        <div class="dm-hero">
          <div class="dm-aurora"></div>
          <div class="dm-hero-text">
            <span class="dm-badge"><i>✦</i> Open-source · 100% private · runs on your machine</span>
            <div class="dm-h1">Erase AI watermarks.<br><span class="dm-grad">Keep every pixel.</span></div>
            <div class="dm-sub">De:Mark detects and removes the <b>Gemini</b> sparkle and the
            <b>NotebookLM</b> badge from your images, videos and PDF decks. It works on any
            background, and nothing ever leaves your device.</div>
            <div class="dm-chips">
              <span class="dm-chip"><span class="dot v"></span><b>Gemini</b> sparkle</span>
              <span class="dm-chip"><span class="dot b"></span><b>NotebookLM</b> badge</span>
              <span class="dm-chip"><span class="dot c"></span>Image · Video · <b>PDF</b></span>
              <span class="dm-chip"><span class="dot g"></span>Runs <b>locally</b></span>
            </div>
          </div>
          <div class="dm-hero-visual">
            <div class="dm-shot">
              <div class="dm-shot-bar"><b></b><b></b><b></b><em>De:Mark</em></div>
              <div class="dm-shot-img">
                <div class="dm-shot-wm">✦</div>
                <div class="dm-shot-pill">✦ Gemini Notebook</div>
                <div class="dm-shot-clean"></div>
                <div class="dm-shot-div"><i>‹›</i></div>
              </div>
              <div class="dm-shot-tags"><span>Before</span><b>After ✓</b></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    """Small De:Mark mark at the top of the (dark) sidebar."""
    st.sidebar.markdown(
        """
        <div class="dm-side-brand">
          <div class="dm-side-badge">D</div>
          <div class="dm-side-name">De<span class="c">:</span>Mark</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mode_control(picker_ok: bool, key: str = "dm_mode") -> str:
    """Prominent Auto / Select-area switch in the MAIN area.

    Returns "auto" or "pinpoint". Falls back to "auto" when the interactive
    picker isn't installed (the manual path needs it).
    """
    if not picker_ok:
        return "auto"
    auto_lbl, pick_lbl = "✨  Auto detect", "✋  Select area"
    choice = st.segmented_control(
        "Mode", [auto_lbl, pick_lbl], default=auto_lbl, key=key,
        label_visibility="collapsed")
    return "pinpoint" if choice == pick_lbl else "auto"


def section(label: str) -> None:
    st.markdown(f"<div class='dm-section'>{label}</div>", unsafe_allow_html=True)


def footer() -> None:
    logo = _tvl_logo_b64()
    logo_img = f'<img src="data:image/png;base64,{logo}" alt=""/>' if logo else ""
    st.markdown(
        f'<div class="dm-foot">'
        f'<a class="dm-foot-brand" href="{TVL_URL}" target="_blank" rel="noopener noreferrer">'
        f'{logo_img}Built by The Visionarys Ltd</a>'
        f'<div>De:Mark · runs entirely on your machine · '
        f'<a href="{TVL_URL}" target="_blank" rel="noopener noreferrer">thevisionarys.com</a></div>'
        f'</div>',
        unsafe_allow_html=True)


def _b64_jpeg(bgr: np.ndarray, quality: int = 92) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def _display_size(w: int, h: int, max_w: int = 980, max_h: int = 560):
    scale = min(max_w / w, max_h / h, 1.0)
    return int(w * scale), int(h * scale)


def image_compare(before_bgr: np.ndarray, after_bgr: np.ndarray) -> None:
    """A premium draggable before/after comparison slider."""
    h, w = before_bgr.shape[:2]
    disp_w, disp_h = _display_size(w, h)
    # Encode at up to 2x the display size (crisp on hi-dpi) but never upscale, so
    # fine text renders sharp instead of the browser aliasing a huge source image.
    scale = min(1.0, (disp_w * 2) / w)
    if scale < 1.0:
        ew, eh = max(1, int(w * scale)), max(1, int(h * scale))
        before_bgr = cv2.resize(before_bgr, (ew, eh), interpolation=cv2.INTER_AREA)
        after_bgr = cv2.resize(after_bgr, (ew, eh), interpolation=cv2.INTER_AREA)
    before_b64, after_b64 = _b64_jpeg(before_bgr, 95), _b64_jpeg(after_bgr, 95)

    html = f"""
    <div style="display:flex;justify-content:center;font-family:Inter,system-ui,sans-serif;">
      <div id="dmba" style="position:relative;width:{disp_w}px;max-width:100%;aspect-ratio:{disp_w} / {disp_h};
           border-radius:20px;overflow:hidden;user-select:none;touch-action:none;cursor:ew-resize;
           box-shadow:0 1px 2px rgba(0,0,0,.05),0 20px 50px rgba(0,0,0,.12);">
        <img src="data:image/jpeg;base64,{after_b64}" draggable="false"
             style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;"/>
        <div id="dmbefore" style="position:absolute;inset:0;clip-path:inset(0 50% 0 0);">
          <img src="data:image/jpeg;base64,{before_b64}" draggable="false"
               style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;"/>
        </div>
        <div id="dmdiv" style="position:absolute;top:0;bottom:0;left:50%;width:2px;
             background:rgba(255,255,255,.9);box-shadow:0 0 10px rgba(0,0,0,.35);"></div>
        <div id="dmknob" style="position:absolute;top:50%;left:50%;width:44px;height:44px;
             transform:translate(-50%,-50%);border-radius:50%;background:rgba(255,255,255,.95);
             backdrop-filter:blur(6px);box-shadow:0 6px 18px rgba(0,0,0,.28);display:flex;
             align-items:center;justify-content:center;color:#1D1D1F;font-size:16px;">⟺</div>
        <span style="position:absolute;top:14px;left:14px;background:rgba(0,0,0,.55);color:#fff;
             font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;
             padding:4px 10px;border-radius:999px;">Before</span>
        <span style="position:absolute;top:14px;right:14px;background:rgba(10,132,255,.92);color:#fff;
             font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;
             padding:4px 10px;border-radius:999px;">After</span>
      </div>
    </div>
    <script>
      (function(){{
        const ba=document.getElementById('dmba'), before=document.getElementById('dmbefore'),
              div=document.getElementById('dmdiv'), knob=document.getElementById('dmknob');
        let drag=false;
        function set(p){{ p=Math.max(0,Math.min(100,p));
          before.style.clipPath='inset(0 '+(100-p)+'% 0 0)';
          div.style.left=p+'%'; knob.style.left=p+'%'; }}
        function at(e){{ const r=ba.getBoundingClientRect();
          const cx=(e.touches?e.touches[0].clientX:e.clientX); set((cx-r.left)/r.width*100); }}
        ba.addEventListener('pointerdown', e=>{{drag=true; at(e);}});
        window.addEventListener('pointermove', e=>{{ if(drag) at(e); }});
        window.addEventListener('pointerup', ()=>drag=false);
        set(50);
      }})();
    </script>
    """
    components.html(html, height=disp_h + 24)


# --------------------------------------------------------------------------- #
# Visual watermark picker — DRAG a box (desktop). Sliders are the always-on path.
# --------------------------------------------------------------------------- #
def canvas_available() -> bool:
    """True if the drag-a-box canvas is installed AND enabled for this deployment.

    The canvas component (streamlit-drawable-canvas) renders its image background
    on a self-hosted server (local / VPS) but NOT on Streamlit Community Cloud's
    proxied iframe — there the background is fetched yet never drawn, so the canvas
    comes up blank, which is worse than offering no drag at all. Drag is therefore
    opt-in per deployment via the DEMARK_ENABLE_DRAG env var (set in the Docker /
    VPS environment); the hosted free tier leaves it unset and shows the always-
    reliable sliders, which work on phone, desktop, and cloud alike.
    """
    if not _HAS_CANVAS:
        return False
    return os.environ.get("DEMARK_ENABLE_DRAG", "").strip().lower() in ("1", "true", "yes", "on")


def draw_box(image_bgr: np.ndarray, key: str, max_width: int = 680):
    """Let the user DRAG a rectangle over the image to mark the watermark.

    Desktop / mouse oriented (phones stay on the sliders). Returns the most
    recently drawn rectangle as ``(cx, cy, bw, bh)`` fractions of the image, or
    ``None`` if nothing has been drawn yet. Degrades to ``None`` (caller shows
    sliders) if the optional canvas component is unavailable.
    """
    if not _HAS_CANVAS:
        return None

    h, w = image_bgr.shape[:2]
    disp_w = min(max_width, w)
    disp_h = max(1, int(round(disp_w * h / w)))
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    bg = _PILImage.fromarray(cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA))

    try:
        result = _st_canvas(
            fill_color="rgba(255, 90, 10, 0.25)",     # translucent orange fill
            stroke_color="#FF5A0A",
            stroke_width=2,
            background_image=bg,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="rect",
            key=key,
        )
    except Exception:
        return None                                   # canvas failed -> caller keeps sliders
    if result is None or result.json_data is None:
        return None
    rects = [o for o in result.json_data.get("objects", []) if o.get("type") == "rect"]
    if not rects:
        return None

    o = rects[-1]                                  # the most recent box wins
    rw = float(o.get("width", 0)) * float(o.get("scaleX", 1))
    rh = float(o.get("height", 0)) * float(o.get("scaleY", 1))
    if rw < 2 or rh < 2:                            # a stray click, not a box
        return None
    cx = (float(o.get("left", 0)) + rw / 2) / disp_w
    cy = (float(o.get("top", 0)) + rh / 2) / disp_h
    return (
        float(np.clip(cx, 0.0, 1.0)),
        float(np.clip(cy, 0.0, 1.0)),
        float(np.clip(rw / disp_w, _MIN_BOX_FRAC, 1.0)),
        float(np.clip(rh / disp_h, _MIN_BOX_FRAC, 1.0)),
    )

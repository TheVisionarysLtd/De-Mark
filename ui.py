"""Presentation layer for De:Mark — CSS, header, and the before/after slider.

Kept apart from app.py so the styling is easy to iterate on without touching the
flow logic, and apart from wmr/ (the engine) which stays framework-agnostic.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_image_coordinates import streamlit_image_coordinates as _image_picker
    _HAS_PICKER = True
except Exception:                       # optional dependency — degrade to sliders
    _HAS_PICKER = False

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root{
  --surface:#FFFFFF; --text:#14152A; --text2:#6B6F86;
  --line:#ECECF4; --accent:#6C5CE7; --accent2:#3B82F6; --radius:22px;
  --grad:linear-gradient(135deg,#6C5CE7 0%,#3B82F6 100%);
  --shadow:0 1px 2px rgba(23,20,70,.05), 0 20px 50px rgba(40,30,100,.10);
  --shadow-sm:0 1px 2px rgba(23,20,70,.05), 0 8px 24px rgba(40,30,100,.07);
}
html, body, [class*="css"], .stApp, button, input, textarea, select{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'SF Pro Display','Segoe UI',sans-serif !important;
  -webkit-font-smoothing:antialiased;
}
.stApp{ background:
  radial-gradient(1200px 600px at 6% -12%, rgba(108,92,231,.20), transparent 55%),
  radial-gradient(1000px 560px at 110% -8%, rgba(59,130,246,.16), transparent 52%),
  linear-gradient(180deg,#F7F8FD 0%, #EEF0F8 100%); }
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"]{ display:none !important; }
.block-container{ padding-top:2.2rem; padding-bottom:4rem; max-width:1120px; }
h1,h2,h3,h4{ color:var(--text); letter-spacing:-.022em; font-weight:700; }

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
[data-testid="stFileUploaderDropzone"]{
  border:1.5px dashed #C4C6DA; border-radius:var(--radius);
  background:linear-gradient(180deg,#FFFFFF, #FBFBFE);
  padding:2.6rem 1.5rem; box-shadow:var(--shadow);
  transition:border-color .25s ease, box-shadow .25s ease, transform .2s ease; }
[data-testid="stFileUploaderDropzone"]:hover{
  border-color:var(--accent); box-shadow:0 16px 46px rgba(108,92,231,.18); transform:translateY(-1px); }
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

/* ---- De:Mark hero -------------------------------------------------------- */
.dm-hero{ display:flex; flex-direction:column; gap:.6rem; margin:.1rem 0 1.4rem; }
.dm-logo{ font-size:3.1rem; font-weight:900; letter-spacing:-.045em; color:var(--text); line-height:1; }
.dm-logo .c{ background:var(--grad); -webkit-background-clip:text; background-clip:text; color:transparent; }
.dm-tag{ font-size:.74rem; font-weight:800; letter-spacing:.2em; text-transform:uppercase;
  background:var(--grad); -webkit-background-clip:text; background-clip:text; color:transparent; }
.dm-desc{ font-size:1.04rem; color:var(--text2); max-width:660px; line-height:1.55; }
.dm-chips{ display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.35rem; }
.dm-chip{ font-size:.76rem; font-weight:600; color:var(--text2); background:rgba(255,255,255,.75);
  backdrop-filter:blur(6px); border:1px solid var(--line); padding:.3rem .75rem; border-radius:999px;
  box-shadow:var(--shadow-sm); }
.dm-chip b{ color:var(--text); }
.dm-section{ font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase;
  color:var(--text2); margin:.2rem 0 .7rem; }
.dm-card{ background:var(--surface); border:1px solid var(--line); border-radius:var(--radius);
  box-shadow:var(--shadow); padding:1.15rem 1.3rem; }
.dm-foot{ color:#9DA0B5; font-size:.82rem; text-align:center; margin-top:2.6rem; display:flex;
  flex-direction:column; align-items:center; gap:.35rem; }
.dm-foot a{ color:var(--text); text-decoration:none; font-weight:700; }
.dm-foot a:hover{ color:var(--accent); }
.dm-foot-brand{ display:inline-flex; align-items:center; gap:.45rem; }
.dm-foot-brand img{ width:20px; height:20px; object-fit:contain; }

/* ---- "Built by The Visionarys" brand link (top-right of the hero) -------- */
.dm-brandby{ display:flex; justify-content:flex-end; margin:0 0 .1rem; }
.dm-brandby a{ display:inline-flex; align-items:center; gap:.5rem; text-decoration:none;
  background:rgba(255,255,255,.8); backdrop-filter:blur(6px); border:1px solid var(--line);
  border-radius:999px; padding:.32rem .8rem .32rem .5rem; box-shadow:var(--shadow-sm);
  transition:transform .12s ease, box-shadow .2s ease; }
.dm-brandby a:hover{ transform:translateY(-1px); box-shadow:var(--shadow); }
.dm-brandby img{ width:22px; height:22px; object-fit:contain; }
.dm-brandby span{ font-size:.78rem; font-weight:600; color:var(--text2); }
.dm-brandby b{ color:var(--text); }

/* ---- Sidebar brand ------------------------------------------------------- */
.dm-side-brand{ display:flex; align-items:center; gap:.6rem; margin:.2rem 0 1.4rem; }
.dm-side-badge{ width:38px; height:38px; border-radius:12px; background:var(--grad);
  display:flex; align-items:center; justify-content:center; color:#fff; font-weight:900; font-size:1.1rem;
  box-shadow:0 8px 20px rgba(76,60,200,.4); }
.dm-side-name{ font-size:1.15rem; font-weight:800; letter-spacing:-.02em; color:#F3F4FB; }
.dm-side-name .c{ color:#8AA0FF; }
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
        brand +
        """
        <div class="dm-hero">
          <div class="dm-tag">AI Watermark Removal Engine</div>
          <div class="dm-logo">De<span class="c">:</span>Mark</div>
          <div class="dm-desc">A lightweight, open-source computer-vision tool that detects
          and removes watermarks from AI-generated images, videos and PDF decks, seamlessly.</div>
          <div class="dm-chips">
            <span class="dm-chip"><b>Gemini</b> sparkle</span>
            <span class="dm-chip"><b>NotebookLM</b> badge</span>
            <span class="dm-chip">Image · Video · <b>PDF</b></span>
            <span class="dm-chip">Runs <b>locally</b></span>
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
# Visual watermark picker — the manual override
# --------------------------------------------------------------------------- #
def picker_available() -> bool:
    """True if the interactive image component is installed (else use sliders)."""
    return _HAS_PICKER


def pinpoint_box(image_bgr: np.ndarray, key: str, drag: bool = False):
    """Show ``image_bgr`` and capture where the user marked the watermark.

    Two input styles:
    * ``drag=False`` (default, works on **touch screens** too): a single
      tap/click returns just the point — ``(cx, cy, None, None)``; the caller
      draws a box of a chosen size around it.
    * ``drag=True`` (desktop mouse): drag a rectangle — returns
      ``(cx, cy, bw, bh)`` fractions.

    All values are fractions of the shown image, or ``None`` if nothing yet.
    """
    if not _HAS_PICKER:
        return None

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Use the long-standing `use_column_width` API (stable across component and
    # Streamlit versions). The newer `width=`/`cursor=` kwargs are avoided — they
    # can raise on some deployed Streamlit builds. Fall back to a minimal call if
    # even this signature isn't accepted, so the app never hard-crashes.
    try:
        value = _image_picker(rgb, use_column_width="always",
                              click_and_drag=drag, key=key)
    except TypeError:
        value = _image_picker(rgb, key=key)
    if not value:
        return None

    # Coordinates come back in *displayed* pixels; scale to full resolution using
    # the displayed dimensions the component reports alongside them.
    disp_w = float(value.get("width") or w)
    disp_h = float(value.get("height") or h)
    sx, sy = w / disp_w, h / disp_h

    if drag and "x1" in value and "x2" in value:
        x1, x2 = sorted((float(value["x1"]), float(value["x2"])))
        y1, y2 = sorted((float(value["y1"]), float(value["y2"])))
        px1, px2 = x1 * sx, x2 * sx
        py1, py2 = y1 * sy, y2 * sy
        box_w = max(px2 - px1, _MIN_BOX_FRAC * w)
        box_h = max(py2 - py1, _MIN_BOX_FRAC * h)
        return (
            float(np.clip((px1 + px2) / 2.0 / w, 0.0, 1.0)),
            float(np.clip((py1 + py2) / 2.0 / h, 0.0, 1.0)),
            float(np.clip(box_w / w, _MIN_BOX_FRAC, 1.0)),
            float(np.clip(box_h / h, _MIN_BOX_FRAC, 1.0)),
        )

    # Tap / click: just the point. The caller sizes the box.
    px = float(value.get("x", disp_w / 2)) * sx
    py = float(value.get("y", disp_h / 2)) * sy
    return (float(np.clip(px / w, 0.0, 1.0)), float(np.clip(py / h, 0.0, 1.0)), None, None)

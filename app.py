"""De:Mark — AI Watermark Removal Engine (Streamlit UI).

Run with:  streamlit run app.py

Presentation lives in ui.py; the engine lives in the wmr/ package. This file is
flow + state only.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import streamlit as st

import ui
from wmr import feedback, files, imaging, video
from wmr.config import DEFAULT_CRF, RemovalSettings, ROI_BOTTOM_FRACTION, ROI_RIGHT_FRACTION
from wmr.inpaint import lama_available
from wmr.sparkle import sparkle_available
from wmr.pdf import first_page_bgr, pdf_available, process_pdf
from wmr.pipeline import overlay_with_roi

_TYPES = ["png", "jpg", "jpeg", "mp4", "mov"] + (["pdf"] if pdf_available() else [])

st.set_page_config(page_title="De:Mark · AI Watermark Removal", page_icon="🩹", layout="wide")
ui.inject_css()

SAMPLE_DIR = Path(__file__).resolve().parent / "samples"


# --------------------------------------------------------------------------- #
# Sidebar — controls
# --------------------------------------------------------------------------- #
def sidebar_settings() -> RemovalSettings:
    """Secondary settings only. The Auto/Select-area mode lives in the main view
    (see ``ui.mode_control``) so it's never hidden by a collapsed sidebar."""
    ui.sidebar_brand()
    st.sidebar.markdown("<div class='dm-section'>Settings</div>", unsafe_allow_html=True)

    detector = "auto"
    if sparkle_available():
        use_ai = st.sidebar.toggle(
            "Smart detector", value=True,
            help="Matches the fixed Gemini sparkle glyph and NotebookLM badge on "
                 "any background palette, then verifies every hit, so real marks "
                 "are removed and clean images are never touched. Turn off to fall "
                 "back to the classic contrast detector.")
        detector = "auto" if use_ai else "classic"

    with st.sidebar.expander("Search area", expanded=False):
        bottom = st.slider("Bottom band (%)", 4, 45, int(ROI_BOTTOM_FRACTION * 100)) / 100.0
        right = st.slider("Right column (%)", 8, 55, int(ROI_RIGHT_FRACTION * 100)) / 100.0

    with st.sidebar.expander("Advanced", expanded=False):
        sensitivity = st.slider("Detection sensitivity", 0.0, 1.0, 0.55, 0.05)
        padding = st.slider("Edge padding (px)", 0, 20, 8)
        engines = {"Auto (best)": "auto", "OpenCV · Navier-Stokes": "opencv_ns",
                   "OpenCV · Telea": "opencv_telea"}
        if lama_available():
            engines["LaMa · AI fill"] = "lama"
        backend = engines[st.selectbox("Inpainting engine", list(engines))]
        crf = st.slider("Video quality (CRF)", 10, 24, DEFAULT_CRF)

    bits = ["🎯 Smart detector" if sparkle_available() else "○ classic detector",
            "✨ LaMa fill" if lama_available() else "○ OpenCV fill"]
    st.sidebar.markdown(
        f"<div style='margin-top:1.2rem;color:#9AA0BC;font-size:.8rem;'>{' · '.join(bits)}</div>",
        unsafe_allow_html=True)

    return RemovalSettings(
        detector=detector, region_mode="corner", bottom_fraction=bottom, right_fraction=right,
        sensitivity=sensitivity, padding_px=padding, backend=backend, crf=crf)


# --------------------------------------------------------------------------- #
# Image flow
# --------------------------------------------------------------------------- #
def render_image(data: bytes, name: str, settings: RemovalSettings) -> None:
    key = (files.sha1_bytes(data), settings.cache_key())
    if st.session_state.get("img_key") != key:
        with st.spinner("Detecting and removing…"):
            st.session_state["img_result"] = imaging.process_image_bytes(data, settings)
            st.session_state["img_key"] = key
    result = st.session_state["img_result"]

    ui.section("Drag the slider")
    ui.image_compare(result["before"], result["after"])

    left, right = st.columns([3, 1])
    with left:
        with st.expander("Detection detail", expanded=not result["mask"].any()):
            preview, mask = overlay_with_roi(result["before"], settings)
            st.image(preview, channels="BGR", use_container_width=True,
                     caption="Blue box = search region · red = removed pixels")
            if not mask.any():
                st.warning("Nothing detected automatically. Switch to **✋ Select area** "
                           "above and draw a box over the watermark.")
    with right:
        st.download_button("Download PNG", result["png"], type="primary",
                           file_name=f"{Path(name).stem}_demark.png", mime="image/png",
                           use_container_width=True)


# --------------------------------------------------------------------------- #
# Video flow
# --------------------------------------------------------------------------- #
def _ensure_video_input(data: bytes, name: str) -> str:
    file_hash = files.sha1_bytes(data)
    if st.session_state.get("vid_hash") != file_hash:
        files.safe_unlink(st.session_state.get("vid_path", ""), st.session_state.get("vid_out", ""))
        st.session_state["vid_path"] = files.write_temp(data, files.ext_of(name))
        st.session_state["vid_hash"] = file_hash
        for k in ("vid_out", "vid_result_key", "vid_first", "vid_info"):
            st.session_state.pop(k, None)
    return st.session_state["vid_path"]


def render_video(data: bytes, name: str, settings: RemovalSettings) -> None:
    input_path = _ensure_video_input(data, name)
    out_name = f"{Path(name).stem}_demark.mp4"

    col_a, col_b = st.columns(2)
    with col_a:
        ui.section("Detection preview")
        if "vid_first" not in st.session_state:
            st.session_state["vid_first"] = video.read_first_frame_bgr(input_path)
        frame = st.session_state["vid_first"]
        if frame is not None:
            preview, mask = overlay_with_roi(frame, settings)
            st.image(preview, channels="BGR", use_container_width=True)
            if not mask.any():
                st.caption("Nothing on this frame yet. The static mask samples the whole clip.")
    with col_b:
        ui.section("Original")
        st.video(data)

    run = st.button("Remove watermark from full video", type="primary", use_container_width=True)
    result_key = (st.session_state.get("vid_hash"), settings.cache_key())

    if run:
        out_path = files.new_temp_path(".mp4")
        progress = st.progress(0.0)
        status = st.empty()

        def on_progress(frac, msg):
            if frac is not None:
                progress.progress(min(max(frac, 0.0), 1.0))
            status.caption(msg)

        try:
            info = video.process_video(input_path, out_path, settings, on_progress)
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            files.safe_unlink(out_path)
            return
        progress.empty(); status.empty()
        st.session_state.update(vid_out=out_path, vid_result_key=result_key, vid_info=info)

    if st.session_state.get("vid_result_key") == result_key and st.session_state.get("vid_out"):
        info = st.session_state.get("vid_info", {})
        audio = "audio kept" if info.get("has_audio") else "no audio"
        size = info.get("size", ("?", "?"))
        ui.section("Cleaned result")
        st.caption(f"{info.get('frames', 0)} frames · {size[0]}×{size[1]} · "
                   f"{info.get('fps', 0):.3g} fps · {audio}")
        with open(st.session_state["vid_out"], "rb") as fh:
            out_bytes = fh.read()
        st.video(out_bytes)
        st.download_button("Download video", out_bytes, type="primary",
                           file_name=out_name, mime="video/mp4", use_container_width=True)


# --------------------------------------------------------------------------- #
# PDF flow
# --------------------------------------------------------------------------- #
def _ensure_pdf_input(data: bytes, name: str) -> str:
    file_hash = files.sha1_bytes(data)
    if st.session_state.get("pdf_hash") != file_hash:
        files.safe_unlink(st.session_state.get("pdf_path", ""), st.session_state.get("pdf_out", ""))
        st.session_state["pdf_path"] = files.write_temp(data, ".pdf")
        st.session_state["pdf_hash"] = file_hash
        for k in ("pdf_out", "pdf_result_key", "pdf_info"):
            st.session_state.pop(k, None)
    return st.session_state["pdf_path"]


def render_pdf(data: bytes, name: str, settings: RemovalSettings) -> None:
    input_path = _ensure_pdf_input(data, name)
    out_name = f"{Path(name).stem}_demark.pdf"

    st.caption("Removes the watermark from every page; the rest of each slide stays untouched.")
    run = st.button("Remove watermarks from PDF", type="primary", use_container_width=True)
    result_key = (files.sha1_bytes(data), settings.cache_key())

    if run:
        out_path = files.new_temp_path(".pdf")
        progress = st.progress(0.0)
        status = st.empty()

        def on_progress(frac, msg):
            if frac is not None:
                progress.progress(min(max(frac, 0.0), 1.0))
            status.caption(msg)

        try:
            info = process_pdf(input_path, out_path, settings, on_progress)
        except Exception as exc:
            st.error(f"PDF processing failed: {exc}")
            files.safe_unlink(out_path)
            return
        progress.empty(); status.empty()
        st.session_state.update(pdf_out=out_path, pdf_result_key=result_key, pdf_info=info)

    if st.session_state.get("pdf_result_key") == result_key and st.session_state.get("pdf_out"):
        info = st.session_state.get("pdf_info", {})
        ui.section("Cleaned result")
        st.caption(f"{info.get('cleaned_pages', 0)} of {info.get('pages', 0)} pages cleaned")
        prev = info.get("preview")
        if prev is not None:
            ui.image_compare(prev["before"], prev["after"])
        with open(st.session_state["pdf_out"], "rb") as fh:
            out_bytes = fh.read()
        st.download_button("Download cleaned PDF", out_bytes, type="primary",
                           file_name=out_name, mime="application/pdf", use_container_width=True)


# --------------------------------------------------------------------------- #
# Pinpoint flow — visual manual override
# --------------------------------------------------------------------------- #
def _reference_frame(data: bytes, name: str):
    """A still to draw the selection box on: the image, first video frame, or page 1."""
    if files.is_image(name):
        return imaging.decode_image(data)[0]
    if files.is_video(name):
        return video.read_first_frame_bgr(_ensure_video_input(data, name))
    if files.is_pdf(name):
        return first_page_bgr(_ensure_pdf_input(data, name))
    return None


# Corner-zoom shows the bottom-right region (where watermarks sit) so a small
# mark is big and easy to box. These are the crop's top-left offset as fractions.
_ZOOM_X0, _ZOOM_Y0 = 0.50, 0.50

def render_pinpoint(data: bytes, name: str, settings: RemovalSettings) -> None:
    """Let the user mark the watermark on a big image, then remove exactly that box."""
    ui.wide_mode()                          # use the full window width for the picker
    ui.section("Point at the watermark")

    ref = _reference_frame(data, name)
    if ref is None:
        st.error("Couldn't load a preview to mark on.")
        return
    h, w = ref.shape[:2]
    is_pdf = files.is_pdf(name)
    is_video = files.is_video(name)

    if is_pdf:
        st.info("**PDF:** draw the box once on this page — it's removed from the **same "
                "spot on every page**. (Watermarks sit in the same place on each slide.)")
    elif is_video:
        st.info("**Video:** draw the box on this first frame — it's removed from the "
                "same spot in **every frame**.")

    zoom = st.toggle("🔍 Zoom to the bottom-right corner (easier for small marks)", value=True,
                     key="pin_zoom")
    st.caption("**Drag a box** around the watermark (or click it), then press **Remove**. "
               "Everything inside the box is rebuilt from the surrounding pixels.")

    # Show either the whole frame or an enlarged bottom-right crop for precision.
    x0f, y0f = (_ZOOM_X0, _ZOOM_Y0) if zoom else (0.0, 0.0)
    view = ref[int(y0f * h):, int(x0f * w):]
    box = ui.pinpoint_box(view, key=f"pick_{files.sha1_bytes(data)[:10]}_{int(zoom)}")
    if box is None:
        return

    # Map the box (fractions of the shown view) back to full-image fractions.
    cxv, cyv, bwv, bhv = box
    span_x, span_y = 1.0 - x0f, 1.0 - y0f
    cx, cy = x0f + cxv * span_x, y0f + cyv * span_y
    bw, bh = bwv * span_x, bhv * span_y
    manual = replace(settings, region_mode="manual", force_fill=True,
                     center_x=cx, center_y=cy, box_w=bw, box_h=bh)

    px0, py0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
    px1, py1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
    preview = ref.copy()
    cv2.rectangle(preview, (px0, py0), (px1, py1), (255, 90, 10), max(2, w // 300))

    left, right = st.columns([3, 1])
    left.image(preview, channels="BGR", use_container_width=True,
               caption="Selected area (shown on the full image)")
    go = right.button("Remove watermark", type="primary", use_container_width=True)

    go_key = (files.sha1_bytes(data), manual.cache_key())
    if go:
        st.session_state["pin_go"] = go_key
    if st.session_state.get("pin_go") != go_key:
        return                              # wait for the user to confirm the box

    st.write("")
    if files.is_image(name):
        render_image(data, name, manual)
    elif is_video:
        render_video(data, name, manual)
    elif is_pdf:
        render_pdf(data, name, manual)


# --------------------------------------------------------------------------- #
# Disclaimer + one-click "report this file" to The Visionarys
# --------------------------------------------------------------------------- #
def report_widget(data: bytes, name: str) -> None:
    """A disclaimer plus a one-click way to send a tricky file to the team."""
    st.write("")
    st.caption("⚠️ De:Mark cleans most files, but some tricky images or videos may not come "
               "out perfectly. If yours didn't, try **✋ Select area** above — or send it to us "
               "and we'll improve it.")

    sent_key = f"reported_{files.sha1_bytes(data)}"
    with st.expander("📤 Didn't work well? Send this file to The Visionarys"):
        if st.session_state.get(sent_key):
            st.success("Thanks — sent to The Visionarys. We'll take a look. 🙌")
            return
        note = st.text_input(
            "What went wrong? (optional)",
            placeholder="e.g. the watermark on the bottom-right wasn't removed",
            key=f"note_{sent_key}")
        st.caption(f"This sends **{name}** to The Visionarys ({feedback.REPORT_EMAIL}) so we can "
                   "look at it and improve De:Mark. Nothing else is shared.")
        if st.button("Send to The Visionarys", type="primary", key=f"send_{sent_key}"):
            details = f"type={files.ext_of(name)}, size={len(data) / 1e6:.2f} MB"
            with st.spinner("Sending…"):
                ok, attached, err = feedback.send_report(name, data, note, details)
            if ok:
                st.session_state[sent_key] = True
                extra = "" if attached else " (the file was too large to attach, but we got its details)"
                st.success(f"Sent to The Visionarys — thank you!{extra}")
                st.rerun()
            else:
                st.error(f"Couldn't send automatically ({err}). Please email {feedback.REPORT_EMAIL} "
                         "with the file attached and we'll take a look.")


# --------------------------------------------------------------------------- #
# Demo samples
# --------------------------------------------------------------------------- #
def _demo_controls() -> None:
    img, vid = SAMPLE_DIR / "sample_image.png", SAMPLE_DIR / "sample_video.mp4"
    if not (img.exists() or vid.exists()):
        return
    st.caption("No file handy? Try a sample:")
    c1, c2, _ = st.columns([1, 1, 3])
    if img.exists() and c1.button("Sample image", use_container_width=True):
        st.session_state["demo"] = (img.read_bytes(), "sample_image.png"); st.rerun()
    if vid.exists() and c2.button("Sample video", use_container_width=True):
        st.session_state["demo"] = (vid.read_bytes(), "sample_video.mp4"); st.rerun()


def _resolve_source(uploaded):
    if uploaded is not None:
        st.session_state.pop("demo", None)
        return uploaded.getvalue(), uploaded.name
    return st.session_state.get("demo")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ui.header()
    settings = sidebar_settings()

    uploaded = st.file_uploader(
        "Drop an image, video or PDF", type=_TYPES,
        accept_multiple_files=False, label_visibility="collapsed")

    source = _resolve_source(uploaded)
    if source is None:
        formats = "PNG · JPG · MP4 · MOV" + (" · PDF" if pdf_available() else "")
        st.caption(f"{formats}. Processed locally, nothing is uploaded.")
        _demo_controls()
        ui.footer()
        return

    data, name = source
    if uploaded is None:
        note, clear = st.columns([5, 1])
        note.caption(f"Sample: **{name}**")
        if clear.button("Clear", use_container_width=True):
            st.session_state.pop("demo", None); st.rerun()

    # Prominent, always-visible removal mode — never hidden in the sidebar.
    st.write("")
    flow = ui.mode_control(ui.picker_available())
    if ui.picker_available():
        st.caption("**Auto detect** finds the watermark for you. If it misses, pick "
                   "**Select area** and draw a box over the watermark yourself.")

    st.write("")
    supported = files.is_image(name) or files.is_video(name) or files.is_pdf(name)
    if flow == "pinpoint":
        render_pinpoint(data, name, settings)
    elif files.is_image(name):
        render_image(data, name, settings)
    elif files.is_video(name):
        render_video(data, name, settings)
    elif files.is_pdf(name):
        render_pdf(data, name, settings)
    else:
        st.error("Unsupported file type.")

    if supported:
        report_widget(data, name)
    ui.footer()


if __name__ == "__main__":
    main()

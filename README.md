# 🧽 AI Watermark Remover

A local Streamlit web app that detects and cleanly removes AI‑tool **branding
watermarks** — the **Gemini four‑pointed sparkle** and the **NotebookLM /
"Gemini Notebook" badge** in the bottom‑right corner — from your own generated
**images (PNG, JPG)** and **videos (MP4, MOV)**, then rebuilds the background by
inpainting so quality is preserved.

> Intended for cleaning branding badges off content **you generated yourself**.
> It only edits the visible bottom‑right corner logo; it does not defeat
> invisible provenance watermarks (e.g. SynthID), which remain intact.

---

## ✨ Features

- **Drag‑and‑drop** upload for images and videos.
- **AI sparkle detector** — a small neural net (trained on the real Gemini
  sparkle) that finds the mark even on busy, low‑contrast backgrounds; classical
  CV fallback when it's off/unavailable.
- **Automatic detection** confined to the bottom‑right corner (configurable).
- **Contrast‑based masking** (morphological top‑hat/black‑hat + a "bright halo"
  grab) that isolates only the watermark pixels — bright sparkle *and* the
  low‑contrast badge.
- **High‑quality inpainting** — OpenCV Navier‑Stokes/Telea out of the box, with
  an optional **LaMa** (deep‑learning) backend for the cleanest photo fills.
- **Video fidelity**: streamed frame‑by‑frame through ffmpeg keeping **exact
  resolution, frame rate, pixel format, colour metadata**, and **all audio
  tracks** (lossless stream copy).
- **Flicker‑free video**: a single stable mask is *voted* across sampled frames,
  so only the persistent logo is removed.
- **Side‑by‑side before / after** preview, a mask preview for tuning, and an
  instant **download** button.
- Runs **entirely locally** — nothing is uploaded anywhere.

---

## 🚀 Install & run

Requires **Python 3.10+**. No system ffmpeg needed — a binary ships with
`imageio-ffmpeg`.

```bash
cd watermark-remover
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the URL Streamlit prints (default <http://localhost:8501>).

> Already have `opencv-python` installed? That works too — you can skip the
> `opencv-python-headless` line in `requirements.txt`.

### Recommended: LaMa (AI) inpainting

This is what makes a watermark truly **disappear** on textured/photographic
backgrounds — LaMa reconstructs the surrounding texture and colour instead of
smudging. Install in two steps:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps simple-lama-inpainting
```

(`--no-deps` avoids a source build of an old numpy pin; the packages above
satisfy it.) The app detects LaMa automatically and uses it for **Auto** on
images; the ~200 MB model downloads on first use. Works on Python 3.14
(torch 2.13 ships a `cp314` wheel). Without it, the app falls back to OpenCV.
Video keeps OpenCV by default because LaMa is slow per-frame on CPU.

---

## 🧭 How to use

1. Drag an image or video onto the upload zone (or click **Sample image /
   Sample video** to try the bundled demos).
2. Watch the **detection preview** — the red overlay shows exactly what will be
   inpainted.
3. Tune in the sidebar if needed:
   - **Detection region** — *Bottom-right corner (auto)* for the usual case, or
     *Manual box* to place a box anywhere (a mark not in the corner, or one a
     busy background hides from auto-detection).
   - **Detection sensitivity** — raise it to catch a fainter mark, lower it if it
     grabs background.
   - **Edge padding** — grows the mask so the fill overshoots the mark cleanly.
   - **Manual box → "Inpaint the entire box"** — the reliable fallback: draw a
     tight box over the watermark and remove everything inside it.
4. Images clean instantly; for video click **Remove watermark from full video**.
5. Click **Download** to save the cleaned file.

---

## ⚙️ How it works

```
frame ─► ROI crop (bottom-right) ─► mask detection ─► inpaint masked bbox ─► frame
                                    (top-hat/black-hat +          (OpenCV NS/Telea
                                     bright-halo, Otsu-scaled       or LaMa)
                                     threshold, speck cleanup)
```

- **ROI** (`wmr/roi.py`) — only the bottom‑right rectangle is ever touched, so
  the rest of the frame is bit‑for‑bit untouched.
- **AI detector** (`wmr/neural.py`, default) — a small U‑Net trained to segment
  the Gemini sparkle. It was trained on the **real** sparkle (extracted from a
  sample by solving the semi‑transparent overlay for its alpha, `dev/ml/`) then
  composited onto thousands of varied backgrounds, so it finds the mark even on
  busy, low‑contrast scenes (a watercolour) where classical CV can't. Runs on
  CPU; falls back to the classical detector when torch/weights are unavailable or
  it finds nothing. A speck filter drops stray false positives.
- **Classical detector** (`wmr/mask.py`, fallback) — top‑hat/black‑hat contrast +
  a **saturation gate** (keep desaturated grey/white, reject colourful art) +
  **picture‑frame exclusion** + a **peak‑relative threshold** + a 4‑point‑star
  template match. Conservative by design: on a background it can't segment it
  detects nothing rather than removing the wrong thing.
- **Video mask** — per‑frame ROI detections are **voted** (`votes_to_mask`); a
  pixel is kept only if flagged in enough sampled frames, isolating the static
  logo from moving background.
- **Inpaint** (`wmr/inpaint.py`, `wmr/pipeline.py`) — only a padded bounding box
  around the mask is inpainted, which bounds cost per video frame while giving
  the fill local context.
- **Video I/O** (`wmr/video.py`, `wmr/media.py`) — frames are streamed via
  `imageio-ffmpeg`; the encoder is fed exact size (`macro_block_size=1`), source
  fps, a safe/source pixel format, CRF quality, and source colour flags; a final
  `-c copy` mux restores every original audio track with no re‑encode.

---

## 📁 Project structure

```
watermark-remover/
├── app.py                # Streamlit UI (presentation + wiring only)
├── requirements.txt
├── README.md
├── .streamlit/config.toml# 2 GB upload limit, theme
├── wmr/                  # engine (framework-agnostic, reusable)
│   ├── config.py         # all tunables + immutable RemovalSettings
│   ├── roi.py            # bottom-right ROI geometry
│   ├── mask.py           # watermark mask detection + video voting
│   ├── inpaint.py        # OpenCV / LaMa backends
│   ├── pipeline.py       # shared frame detect+inpaint, mask overlay
│   ├── imaging.py        # image decode/encode (+ alpha) & entry point
│   ├── video.py          # streaming video processor
│   ├── media.py          # ffmpeg discovery, probe, audio mux
│   └── files.py          # validation & temp-file helpers
├── samples/
│   ├── make_samples.py   # regenerate synthetic test media
│   └── sample_image.png / sample_video.mp4
└── dev/selftest.py       # headless end-to-end test → samples/output/
```

Regenerate samples / run the headless self‑test:

```bash
python samples/make_samples.py
python dev/selftest.py
```

---

## 📝 Tips & limitations

- The **AI detector** (default when torch is installed) finds the Gemini sparkle
  across backgrounds — including busy, low-contrast scenes the classical detector
  can't segment. Toggle it in the sidebar; it falls back to classical if it finds
  nothing. It's trained on the **sparkle**; the NotebookLM badge is handled by the
  classical detector on clean backgrounds.
- **LaMa** (default fill on images) reconstructs surrounding texture and colour so
  the mark truly disappears; OpenCV is the instant fallback and is great on smooth
  areas but smudges fine texture.
- **Manual box** is the always-works escape hatch for anything the detectors miss
  or a mark that isn't in the corner — position it and tick *Inpaint the entire box*.
- Retrain / extend the detector (e.g. add the badge, more styles) via `dev/ml/`
  (`extract_template.py` → `synth.py` → `train.py`).
- The tool targets the **visible** corner watermark only (not invisible provenance
  marks such as SynthID).

---

## ⚖️ Responsible use

Use this on content **you created or have the right to modify**. Removing the
visible corner badge from your own AI‑generated images/videos is a normal
creative‑workflow need. Don't use it to misrepresent authorship or strip
attribution you're obligated to keep.

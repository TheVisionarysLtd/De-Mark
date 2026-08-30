# De:Mark — containerised for a VPS (CPU-only).
# Build:  docker build -t demark .
# Run:    docker run -d --name demark --restart unless-stopped \
#             -p 8501:8501 -v demark-cache:/root/.cache demark
#
# Python 3.11 is used here (not 3.14) because torch, simple-lama, opencv,
# PyMuPDF and imageio-ffmpeg all ship well-tested wheels for it — no compiling.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=2048 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Runtime libs OpenCV/torch need. ffmpeg is bundled by imageio-ffmpeg, so no
# system ffmpeg is required. curl is only for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Base requirements  2) CPU torch  3) LaMa (its numpy pin must be skipped).
# torch + LaMa power the AI sparkle detector and the seamless palette-adapting
# fill; without them the app still runs on the classical detector + OpenCV.
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-deps simple-lama-inpainting

# App code — the trained weights ship in wmr/weights/. The ~200 MB LaMa model
# downloads to /root/.cache on first use; mount that as a volume (see header)
# so it persists across restarts instead of re-downloading.
COPY . .

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=true"]

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal system deps for audio/video processing + MediaPipe / OpenCV.
RUN apt-get update ; apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
  ; rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY app /app/app
COPY configs /app/configs
COPY scripts /app/scripts
COPY benchmarks /app/benchmarks
COPY presentation /app/presentation
COPY requirements.lock /app/requirements.lock

RUN python -m pip install -U pip ; python -m pip install -e .

EXPOSE 7860

CMD ["python", "app/gradio_app.py"]

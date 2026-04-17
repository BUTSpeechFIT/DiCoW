FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH="/app/DiariZen:/app"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    transformers==4.55.0 \
    gradio==6.4.0 \
    librosa==0.10.2.post1 \
    soundfile==0.13.0 \
    toml==0.10.2 \
    fastapi==0.116.1 \
    uvicorn==0.35.0 \
    python-multipart==0.0.22 \
    aiosqlite==0.20.0

COPY DiariZen /app/DiariZen

RUN pip install --no-cache-dir \
    pyannote.core==5.0.0 \
    pyannote.database==5.1.3 \
    pyannote.metrics==3.2.1 \
    pyannote.pipeline==3.0.1 \
    /app/DiariZen/pyannote-audio

COPY . /app

RUN chmod +x /app/inference.py

EXPOSE 8000

ENTRYPOINT ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

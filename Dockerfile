FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH="/app/DiariZen:/app"

# Install system dependencies (build-essential needed for some packages, ffmpeg for audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install isolated core dependencies matching our successful test matrix
# Doing PyTorch separately for CPU optimizations if no GPU is targeted
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install machine learning and audio packages
# Using specific numpy version to avoid Pyannote compatibility issues (numpy 2.0+ breaks it)
# Using huggingface/transformers exactly at 4.40.1 which we've securely patched
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    transformers==4.55.0 \
    gradio==6.4.0 \
    librosa==0.10.2.post1 \
    soundfile==0.13.0 \
    toml==0.10.2

# We copy DiariZen submodule files in, as they contain the pyannote-audio folder
COPY DiariZen /app/DiariZen

# Install Pyannote components and the local pyannote-audio from the submodule
RUN pip install --no-cache-dir \
    pyannote.core==5.0.0 \
    pyannote.database==5.1.3 \
    pyannote.metrics==3.2.1 \
    pyannote.pipeline==3.0.1 \
    /app/DiariZen/pyannote-audio

# Copy all the remaining project files (including local models, scripts) into the container
COPY . /app

# Final permission fix in case of weird local mappings
RUN chmod +x /app/inference.py

# Entry command for processing
# Since we use shared cache, the HuggingFace cache will automatically be saved to ~/.cache/huggingface on the host mappings
ENTRYPOINT ["python", "inference.py"]
CMD ["--help"]

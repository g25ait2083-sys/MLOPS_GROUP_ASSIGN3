# Task 6 — Inference image for the emotion classifier.
#
# Design choices (explained in the report):
#   * python:3.11-slim  -> small Debian-based base, matches the CI Python version.
#   * CPU-only PyTorch   -> installed from the official CPU wheel index so the image
#                           stays a few hundred MB instead of pulling multi-GB CUDA.
#   * ARG HF_MODEL_NAME  -> the model repo is a build argument with a sensible,
#                           publicly-pullable default, overridable at build time.
#   * Only requirements.txt (the inference deps) is installed — no training stack.
#   * Non-root user      -> the container runs as an unprivileged user.

FROM python:3.11-slim

# --- The model repo id; override with: --build-arg HF_MODEL_NAME=you/your-model
ARG HF_MODEL_NAME=bhadresh-savani/distilbert-base-uncased-emotion
ENV HF_MODEL_NAME=${HF_MODEL_NAME} \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    TRANSFORMERS_VERBOSITY=error

WORKDIR /app

# Install CPU PyTorch first (from the dedicated CPU index), then the remaining
# inference deps. torch is already pinned in requirements.txt, so the second
# install finds it satisfied and only adds transformers / hf_hub / numpy.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Application code only.
COPY src/ ./src/

# Drop privileges and give the runtime user a writable HF cache dir.
RUN useradd --create-home appuser \
 && mkdir -p /app/.hf_cache \
 && chown -R appuser:appuser /app
USER appuser

# INPUT_TEXT is provided at run time:
#   docker run --rm -e INPUT_TEXT="I am so happy!" mlops-a3-emotion:latest
ENTRYPOINT ["python", "src/inference.py"]

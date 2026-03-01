# Fed-Vis Docker Image
#
# Build: docker build -t fedvis:latest .
# Run:   docker run -p 8000:8000 fedvis:latest
# GPU:   docker run --gpus all -p 8000:8000 fedvis:latest
#
# With trained checkpoint:
#   docker run -p 8000:8000 -v ./outputs:/app/outputs fedvis:latest \
#     uvicorn fedvis.api.app:app --host 0.0.0.0 --port 8000

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# python deps — install torch CPU first (smaller image)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# install the rest
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] python-multipart \
    numpy nibabel scipy scikit-image \
    hydra-core omegaconf \
    flwr

# copy source
COPY src/ ./src/
COPY configs/ ./configs/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# default: run FastAPI inference server
CMD ["uvicorn", "fedvis.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

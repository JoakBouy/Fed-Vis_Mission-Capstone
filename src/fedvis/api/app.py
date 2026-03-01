"""Fed-Vis API — FastAPI inference service.

Loads a trained Attention U-Net checkpoint and serves
segmentation predictions over HTTP.

Run:
    uvicorn fedvis.api.app:app --reload --port 8000
    python -m fedvis.api.app --checkpoint outputs/best.pth
"""

import io
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fedvis.models.attention_unet import AttentionUNet3D


# ── schemas ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    device: str

class ModelInfoResponse(BaseModel):
    architecture: str
    parameters: int
    base_features: int
    input_shape: str

class PredictionResponse(BaseModel):
    success: bool
    input_shape: list[int]
    output_shape: list[int]
    inference_time_ms: float
    foreground_voxels: int
    total_voxels: int
    tumor_pct: float

class PredictionRequest(BaseModel):
    volume: list
    threshold: float = 0.5


# ── globals ─────────────────────────────────────────────

model: Optional[AttentionUNet3D] = None
device: str = "cpu"
model_loaded: bool = False
BASE_FEATURES: int = 32


# ── app factory ─────────────────────────────────────────

def create_app(checkpoint_path: Optional[str] = None, base_features: int = 32) -> FastAPI:
    global model, device, model_loaded, BASE_FEATURES
    BASE_FEATURES = base_features

    app = FastAPI(
        title="Fed-Vis Inference API",
        description="3D Medical Image Segmentation with Attention U-Net",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # serve frontend static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.on_event("startup")
    async def startup():
        global model, device, model_loaded
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            model = AttentionUNet3D(
                in_channels=1, out_channels=1, base_features=base_features
            )

            if checkpoint_path and Path(checkpoint_path).exists():
                ckpt = torch.load(checkpoint_path, map_location=device)
                if "model" in ckpt:
                    model.load_state_dict(ckpt["model"])
                elif "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                else:
                    model.load_state_dict(ckpt)

            model.to(device)
            model.eval()
            model_loaded = True
            print(f"Model ready on {device}")
        except Exception as e:
            print(f"Model load failed: {e}")
            model_loaded = False

    return app


app = create_app()


# ── routes ──────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model_loaded,
        device=device,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    if not model_loaded:
        raise HTTPException(503, "Model not loaded")

    n = sum(p.numel() for p in model.parameters())
    return ModelInfoResponse(
        architecture="AttentionUNet3D",
        parameters=n,
        base_features=BASE_FEATURES,
        input_shape="[B, 1, D, H, W]",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    """Run segmentation on an uploaded .npy volume."""
    if not model_loaded:
        raise HTTPException(503, "Model not loaded")

    try:
        data = await file.read()
        volume = np.load(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Failed to read .npy: {e}")

    # add batch/channel dims
    if volume.ndim == 3:
        volume = volume[np.newaxis, np.newaxis, ...]
    elif volume.ndim == 4:
        volume = volume[np.newaxis, ...]
    elif volume.ndim != 5:
        raise HTTPException(400, f"Bad shape: {volume.shape}")

    tensor = torch.from_numpy(volume.astype(np.float32)).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(tensor)
        prob = torch.sigmoid(out)
    elapsed = (time.perf_counter() - t0) * 1000

    mask = (prob > threshold).cpu().numpy().astype(np.uint8)
    fg = int(mask.sum())
    total = int(mask.size)

    return PredictionResponse(
        success=True,
        input_shape=list(volume.shape),
        output_shape=list(mask.shape),
        inference_time_ms=round(elapsed, 2),
        foreground_voxels=fg,
        total_voxels=total,
        tumor_pct=round(fg / total * 100, 2),
    )


@app.post("/predict/mask")
async def predict_mask(file: UploadFile = File(...), threshold: float = 0.5):
    """Run segmentation and return the mask as a downloadable .npy file."""
    if not model_loaded:
        raise HTTPException(503, "Model not loaded")

    try:
        data = await file.read()
        volume = np.load(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Failed to read .npy: {e}")

    if volume.ndim == 3:
        volume = volume[np.newaxis, np.newaxis, ...]
    elif volume.ndim == 4:
        volume = volume[np.newaxis, ...]

    tensor = torch.from_numpy(volume.astype(np.float32)).to(device)

    with torch.no_grad():
        out = model(tensor)
        prob = torch.sigmoid(out)

    mask = (prob > threshold).cpu().numpy().astype(np.uint8)

    buf = io.BytesIO()
    np.save(buf, mask)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=segmentation.npy"},
    )


@app.post("/predict/json", response_model=PredictionResponse)
async def predict_json(req: PredictionRequest):
    """Run inference from JSON (for small test volumes)."""
    if not model_loaded:
        raise HTTPException(503, "Model not loaded")

    try:
        volume = np.array(req.volume, dtype=np.float32)
    except Exception as e:
        raise HTTPException(400, f"Invalid data: {e}")

    if volume.ndim == 3:
        volume = volume[np.newaxis, np.newaxis, ...]

    tensor = torch.from_numpy(volume).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(tensor)
        prob = torch.sigmoid(out)
    elapsed = (time.perf_counter() - t0) * 1000

    mask = (prob > req.threshold).cpu().numpy()
    fg = int(mask.sum())
    total = int(mask.size)

    return PredictionResponse(
        success=True,
        input_shape=list(volume.shape),
        output_shape=list(mask.shape),
        inference_time_ms=round(elapsed, 2),
        foreground_voxels=fg,
        total_voxels=total,
        tumor_pct=round(fg / total * 100, 2),
    )


# ── cli entry ───────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--features", type=int, default=32)
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    app = create_app(checkpoint_path=args.checkpoint, base_features=args.features)
    uvicorn.run(app, host="0.0.0.0", port=args.port)

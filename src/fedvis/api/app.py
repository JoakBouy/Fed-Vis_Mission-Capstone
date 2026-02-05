"""Fed-Vis API - FastAPI inference service."""

from __future__ import annotations

import io
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import RedirectResponse
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError("Install: pip install fastapi uvicorn python-multipart") from e

from fedvis.models import AttentionUNet3D


# Request/Response schemas
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    architecture: str
    input_channels: int
    output_channels: int
    base_features: int
    total_parameters: int
    parameter_breakdown: dict[str, int]
    expected_input_shape: str


class PredictionResponse(BaseModel):
    success: bool
    input_shape: list[int]
    output_shape: list[int]
    inference_time_ms: float
    segmentation_stats: dict[str, Any]


class PredictionRequest(BaseModel):
    volume: list
    threshold: float = Field(0.5)


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Fed-Vis Inference API",
        description="3D Medical Image Segmentation with Attention U-Net",
        version="0.1.0",
        docs_url="/docs",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.model = None
    app.state.device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.model_loaded = False

    @app.on_event("startup")
    async def load_model() -> None:
        try:
            model = AttentionUNet3D(in_channels=1, out_channels=1, base_features=64)

            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=app.state.device)
                model.load_state_dict(checkpoint["model_state_dict"])

            model.to(app.state.device)
            model.eval()
            app.state.model = model
            app.state.model_loaded = True
            print(f"Model loaded on {app.state.device}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            app.state.model_loaded = False

    return app


app = create_app()


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="0.1.0",
        model_loaded=app.state.model_loaded,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    if not app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model: AttentionUNet3D = app.state.model
    params = model.count_parameters()

    return ModelInfoResponse(
        architecture="AttentionUNet3D",
        input_channels=1,
        output_channels=1,
        base_features=64,
        total_parameters=params["total"],
        parameter_breakdown=params,
        expected_input_shape="[B, 1, D, H, W]",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Run segmentation on uploaded .npy volume."""
    if not app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        volume = np.load(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # Add batch/channel dims if needed
    if volume.ndim == 3:
        volume = volume[np.newaxis, np.newaxis, ...]
    elif volume.ndim == 4:
        volume = volume[np.newaxis, ...]
    elif volume.ndim != 5:
        raise HTTPException(status_code=400, detail=f"Invalid shape: {volume.shape}")

    input_tensor = torch.from_numpy(volume.astype(np.float32)).to(app.state.device)

    model: AttentionUNet3D = app.state.model
    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)

    inference_time = (time.perf_counter() - start_time) * 1000

    mask = (probs > 0.5).cpu().numpy()
    stats = {
        "foreground_voxels": int(mask.sum()),
        "total_voxels": int(mask.size),
        "foreground_pct": float(mask.sum() / mask.size * 100),
    }

    return PredictionResponse(
        success=True,
        input_shape=list(volume.shape),
        output_shape=list(mask.shape),
        inference_time_ms=round(inference_time, 2),
        segmentation_stats=stats,
    )


@app.post("/predict/json", response_model=PredictionResponse, tags=["Inference"])
async def predict_json(request: PredictionRequest) -> PredictionResponse:
    """Run inference from JSON (for small test volumes)."""
    if not app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        volume = np.array(request.volume, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}")

    if volume.ndim == 3:
        volume = volume[np.newaxis, np.newaxis, ...]

    input_tensor = torch.from_numpy(volume).to(app.state.device)

    model: AttentionUNet3D = app.state.model
    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)

    inference_time = (time.perf_counter() - start_time) * 1000

    mask = (probs > request.threshold).cpu().numpy()
    stats = {
        "foreground_voxels": int(mask.sum()),
        "total_voxels": int(mask.size),
        "threshold": request.threshold,
    }

    return PredictionResponse(
        success=True,
        input_shape=list(volume.shape),
        output_shape=list(mask.shape),
        inference_time_ms=round(inference_time, 2),
        segmentation_stats=stats,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

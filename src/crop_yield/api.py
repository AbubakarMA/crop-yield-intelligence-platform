"""FastAPI service for crop-yield predictions."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from crop_yield.production import load_production_artifacts


class PredictionRequest(BaseModel):
    """Validated features expected by the crop-yield model."""

    area: str = Field(min_length=1)
    item: str = Field(min_length=1)
    year: int = Field(ge=1990, le=2013)
    average_rainfall_mm_per_year: float = Field(gt=0)
    pesticides_tonnes: float = Field(ge=0)
    average_temperature_c: float = Field(ge=-30, le=50)


class PredictionResponse(BaseModel):
    """Prediction with both source and reader-friendly units."""

    prediction_hg_per_ha: float
    prediction_tonnes_per_ha: float
    model_version: str
    warning: str


@lru_cache(maxsize=1)
def get_artifacts() -> tuple[Any, dict[str, Any]]:
    """Load model artifacts once per service process."""
    model_path = Path(
        os.getenv(
            "MODEL_PATH",
            "artifacts/crop_yield_random_forest.joblib",
        )
    )
    metadata_path = Path(
        os.getenv(
            "MODEL_METADATA_PATH",
            "artifacts/model_metadata.json",
        )
    )
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Model artifacts are unavailable. Run `crop-yield-train` first."
        )
    return load_production_artifacts(model_path, metadata_path)


app = FastAPI(
    title="Crop Yield Intelligence API",
    version="1.0.0",
    description=(
        "Historical crop-yield estimation API. Predictions are not causal "
        "recommendations or long-range forecasts."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    """Report service and artifact readiness."""
    try:
        _, metadata = get_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "model_version": str(metadata["model_version"]),
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    """Expose safe model scope, categories, metrics, and limitations."""
    try:
        _, model_metadata = get_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return model_metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Estimate crop yield for one in-domain historical observation."""
    try:
        model, model_metadata = get_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if request.area not in model_metadata["countries"]:
        raise HTTPException(
            status_code=422,
            detail="Country was not observed during model training.",
        )
    if request.item not in model_metadata["crops"]:
        raise HTTPException(
            status_code=422,
            detail="Crop was not observed during model training.",
        )

    prediction = float(
        model.predict(pd.DataFrame([request.model_dump()]))[0]
    )
    return PredictionResponse(
        prediction_hg_per_ha=prediction,
        prediction_tonnes_per_ha=prediction / 10_000,
        model_version=str(model_metadata["model_version"]),
        warning=(
            "Historical conditional estimate only; do not interpret it as "
            "a causal fertilizer, pesticide, climate, or policy effect."
        ),
    )

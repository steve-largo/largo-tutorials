"""Typed FastAPI inference service exposing /health, /ready, and /predict endpoints."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status

from config import ServingSettings, load_settings
from model import InferencePredictor, TorchSentimentPredictor
from schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    SentimentPrediction,
)


@dataclass
class ServiceTelemetry:
    """Mutable per-application telemetry counters owned by FastAPI app.state."""

    requests_processed: int = 0
    items_processed: int = 0


def _preview_text(text: str, max_chars: int = 100) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def create_app(
    settings: ServingSettings | None = None,
    predictor: InferencePredictor | None = None,
    auto_load_on_startup: bool = True,
) -> FastAPI:
    """Construct a FastAPI application with an injected or configured predictor."""
    cfg = settings or load_settings()
    active_predictor: InferencePredictor = predictor or TorchSentimentPredictor(cfg)
    telemetry = ServiceTelemetry()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if auto_load_on_startup and not active_predictor.is_ready:
            active_predictor.load()
        application.state.settings = cfg
        application.state.predictor = active_predictor
        application.state.telemetry = telemetry
        yield

    application = FastAPI(
        title=cfg.service_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.settings = cfg
    application.state.predictor = active_predictor
    application.state.telemetry = telemetry

    @application.get("/health", response_model=HealthResponse)
    async def get_health(request: Request) -> HealthResponse:
        pred: InferencePredictor = request.app.state.predictor
        stats: ServiceTelemetry = request.app.state.telemetry
        svc_cfg: ServingSettings = request.app.state.settings

        gpu_mb: float | None = None
        if pred.device_name.startswith("cuda") and torch.cuda.is_available():
            gpu_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)

        return HealthResponse(
            status="healthy",
            service_name=svc_cfg.service_name,
            device=pred.device_name,
            runtime=pred.runtime_mode,
            requests_processed=stats.requests_processed,
            items_processed=stats.items_processed,
            gpu_memory_mb=gpu_mb,
        )

    @application.get("/ready", response_model=ReadyResponse)
    async def get_ready(request: Request) -> ReadyResponse:
        pred: InferencePredictor = request.app.state.predictor
        svc_cfg: ServingSettings = request.app.state.settings
        payload = ReadyResponse(
            ready=pred.is_ready,
            model_loaded=pred.is_ready,
            runtime=pred.runtime_mode,
            max_batch_size=svc_cfg.max_batch_size,
        )
        if not pred.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=payload.model_dump(),
            )
        return payload

    @application.post("/predict", response_model=PredictResponse)
    async def post_predict(payload: PredictRequest, request: Request) -> PredictResponse:
        pred: InferencePredictor = request.app.state.predictor
        stats: ServiceTelemetry = request.app.state.telemetry
        svc_cfg: ServingSettings = request.app.state.settings

        if not pred.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model predictor is not ready to serve traffic",
            )
        if len(payload.texts) > svc_cfg.max_batch_size:
            unprocessable_code = getattr(
                status,
                "HTTP_422_UNPROCESSABLE_CONTENT",
                422,
            )
            raise HTTPException(
                status_code=unprocessable_code,
                detail=(
                    f"Batch size {len(payload.texts)} exceeds configured max_batch_size "
                    f"({svc_cfg.max_batch_size})"
                ),
            )

        t0 = time.perf_counter()
        raw_predictions = pred.predict_texts(payload.texts)
        total_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = total_ms / len(payload.texts)

        stats.requests_processed += 1
        stats.items_processed += len(payload.texts)

        formatted = tuple(
            SentimentPrediction(
                text_preview=_preview_text(text),
                sentiment=label,
                confidence=confidence,
                item_latency_ms=round(per_item_ms, 3),
            )
            for text, (label, confidence) in zip(payload.texts, raw_predictions, strict=True)
        )
        return PredictResponse(
            runtime=pred.runtime_mode,
            batch_size=len(payload.texts),
            total_latency_ms=round(total_ms, 3),
            predictions=formatted,
        )

    return application


app = create_app(auto_load_on_startup=True)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, workers=1)

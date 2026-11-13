"""FastAPI TestClient and PyTorch runtime invariant tests for model serving."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import create_app
from benchmark import run_benchmark
from config import ServingSettings
from model import TorchSentimentPredictor
from schemas import ExecutionRuntime, PredictRequest, SentimentLabel


class _DeterministicRulePredictor:
    """Fast injectable test double verifying HTTP routing, readiness gating, and counters."""

    def __init__(self, initially_ready: bool = True) -> None:
        self._ready = initially_ready

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def device_name(self) -> str:
        return "cpu"

    @property
    def runtime_mode(self) -> ExecutionRuntime:
        return "inference_mode_fp32"

    def load(self) -> None:
        self._ready = True

    def predict_texts(self, texts: Sequence[str]) -> list[tuple[SentimentLabel, float]]:
        out: list[tuple[SentimentLabel, float]] = []
        for text in texts:
            if any(word in text.lower() for word in ("great", "fast", "fantastic")):
                out.append(("positive", 0.94))
            else:
                out.append(("negative", 0.88))
        return out


def test_schema_and_settings_validation() -> None:
    with pytest.raises(ValidationError):
        ServingSettings(embed_dim=250, num_heads=4)

    with pytest.raises(ValidationError):
        ServingSettings(device="tpu-pod")

    with pytest.raises(ValidationError):
        PredictRequest.model_validate({"text": "   "})

    with pytest.raises(ValidationError):
        PredictRequest.model_validate({"text": "a", "texts": ["b"]})

    single = PredictRequest.model_validate({"text": "Fast prediction"})
    assert single.texts == ("Fast prediction",)


def test_fastapi_readiness_gating_and_batch_bounds() -> None:
    settings = ServingSettings(max_batch_size=2)
    unready_predictor = _DeterministicRulePredictor(initially_ready=False)
    app_unready = create_app(
        settings=settings,
        predictor=unready_predictor,
        auto_load_on_startup=False,
    )

    with TestClient(app_unready) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"

        ready_resp = client.get("/ready")
        assert ready_resp.status_code == 503

        blocked_pred = client.post("/predict", json={"text": "Fast service"})
        assert blocked_pred.status_code == 503

        unready_predictor.load()
        ready_after = client.get("/ready")
        assert ready_after.status_code == 200
        assert ready_after.json()["ready"] is True

        single_resp = client.post("/predict", json={"text": "Fantastic tail latency!"})
        assert single_resp.status_code == 200
        body = single_resp.json()
        assert body["batch_size"] == 1
        assert body["predictions"][0]["sentiment"] == "positive"
        assert body["predictions"][0]["confidence"] == pytest.approx(0.94)

        batch_resp = client.post(
            "/predict",
            json={"texts": ["Great throughput", "Terrible timeout error"]},
        )
        assert batch_resp.status_code == 200
        assert [p["sentiment"] for p in batch_resp.json()["predictions"]] == [
            "positive",
            "negative",
        ]

        oversized = client.post(
            "/predict",
            json={"texts": ["one", "two", "three"]},
        )
        assert oversized.status_code == 422

        final_health = client.get("/health").json()
        assert final_health["requests_processed"] == 2
        assert final_health["items_processed"] == 3


def test_pytorch_predictor_polarity_and_runtime_logit_parity() -> None:
    base_kwargs = {
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "max_seq_length": 32,
        "warmup_runs": 1,
        "seed": 7,
    }
    eager = TorchSentimentPredictor(ServingSettings(runtime="eager_fp32", **base_kwargs))
    inf_mode = TorchSentimentPredictor(
        ServingSettings(runtime="inference_mode_fp32", **base_kwargs)
    )
    autocast = TorchSentimentPredictor(
        ServingSettings(runtime="autocast_low_precision", **base_kwargs)
    )

    eager.load()
    inf_mode.load()
    autocast.load()

    texts = [
        "Fast accurate reliable service under load",
        "Slow broken crash timeout error",
    ]
    preds = inf_mode.predict_texts(texts)
    assert preds[0][0] == "positive"
    assert preds[1][0] == "negative"
    assert preds[0][1] > 0.6
    assert preds[1][1] > 0.6

    logits_eager = eager.forward_logits(texts)
    logits_inf = inf_mode.forward_logits(texts)
    logits_amp = autocast.forward_logits(texts)

    assert torch.allclose(logits_eager, logits_inf, atol=1e-5)
    assert float((logits_eager - logits_amp).abs().max().item()) < 0.15


def test_end_to_end_fastapi_lifespan_and_benchmark_harness() -> None:
    compact_cfg = ServingSettings(
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        max_seq_length=32,
        warmup_runs=1,
        benchmark_runs=3,
        benchmark_batch_sizes=(1, 4),
        include_torch_compile=False,
    )
    with TestClient(create_app(settings=compact_cfg)) as client:
        assert client.get("/ready").status_code == 200
        resp = client.post("/predict", json={"text": "Fast clean reliable response"})
        assert resp.status_code == 200
        assert resp.json()["predictions"][0]["sentiment"] == "positive"

    report = run_benchmark(settings=compact_cfg)
    assert len(report.rows) == 6
    by_key = {(r.runtime, r.batch_size): r for r in report.rows}
    assert by_key[("eager_fp32", 1)].max_logit_abs_diff == pytest.approx(0.0, abs=1e-6)
    assert by_key[("inference_mode_fp32", 4)].max_logit_abs_diff < 1e-5
    assert by_key[("autocast_low_precision", 4)].items_per_second > 0.0

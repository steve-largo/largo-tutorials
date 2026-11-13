"""PyTorch Transformer sequence classifier and multi-runtime predictor adapters."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Protocol

import torch
from torch import nn

from config import ServingSettings
from schemas import ExecutionRuntime, SentimentLabel

_WORD_RE = re.compile(r"[a-z0-9']+")
_POSITIVE_CUES = (
    "fast",
    "fantastic",
    "great",
    "clean",
    "accurate",
    "reliable",
    "stable",
    "helpful",
    "smooth",
    "excellent",
)
_NEGATIVE_CUES = (
    "slow",
    "terrible",
    "broken",
    "crash",
    "stalled",
    "unstable",
    "error",
    "timeout",
    "noisy",
    "awful",
)


class InferencePredictor(Protocol):
    """Injectable inference backend protocol decoupling HTTP routes from PyTorch internals."""

    @property
    def is_ready(self) -> bool:
        """Whether model weights and optional compiled graphs are warmed up."""

    @property
    def device_name(self) -> str:
        """Target execution device string (e.g. 'cpu' or 'cuda')."""

    @property
    def runtime_mode(self) -> ExecutionRuntime:
        """Configured execution optimization mode."""

    def load(self) -> None:
        """Initialize weights, move tensors to device, and run warmup passes."""

    def predict_texts(self, texts: Sequence[str]) -> list[tuple[SentimentLabel, float]]:
        """Classify a batch of texts into (sentiment_label, probability) pairs."""


class HashedSubwordTokenizer:
    """Deterministic local tokenizer mapping words onto fixed-id vocabulary buckets."""

    PAD_ID = 0
    CLS_ID = 1

    def __init__(self, vocab_size: int = 4096, max_seq_length: int = 64) -> None:
        if vocab_size < 16:
            raise ValueError(f"vocab_size must be >= 16, got {vocab_size}")
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self._usable_buckets = vocab_size - 2

    def token_to_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.lower().encode("utf-8"), digest_size=4).digest()
        bucket = int.from_bytes(digest, byteorder="little") % self._usable_buckets
        return bucket + 2

    def encode_batch(
        self,
        texts: Sequence[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of texts into (input_ids, attention_mask) tensors."""
        if not texts:
            raise ValueError("encode_batch requires at least one text")
        batch_ids: list[list[int]] = []
        batch_masks: list[list[int]] = []

        for text in texts:
            words = _WORD_RE.findall(text.lower())
            ids = [self.CLS_ID] + [
                self.token_to_id(w) for w in words[: self.max_seq_length - 1]
            ]
            real_len = len(ids)
            pad_len = self.max_seq_length - real_len
            batch_ids.append(ids + [self.PAD_ID] * pad_len)
            batch_masks.append([1] * real_len + [0] * pad_len)

        return (
            torch.tensor(batch_ids, dtype=torch.long, device=device),
            torch.tensor(batch_masks, dtype=torch.long, device=device),
        )


class TransformerSentimentClassifier(nn.Module):
    """Pre-norm Transformer encoder sequence classifier for latency and precision benchmarking."""

    def __init__(
        self,
        vocab_size: int = 4096,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_length: int = 64,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        padding_mask = attention_mask == 0
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        mask_float = attention_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def build_seeded_classifier(
    settings: ServingSettings,
    tokenizer: HashedSubwordTokenizer,
) -> TransformerSentimentClassifier:
    """Create a deterministically seeded classifier with calibrated polarity prototypes."""
    torch.manual_seed(settings.seed)
    classifier = TransformerSentimentClassifier(
        vocab_size=settings.vocab_size,
        embed_dim=settings.embed_dim,
        num_heads=settings.num_heads,
        num_layers=settings.num_layers,
        max_seq_length=settings.max_seq_length,
    )
    with torch.no_grad():
        direction = torch.zeros(settings.embed_dim, dtype=torch.float32)
        direction[0] = 3.0
        direction[1] = -3.0
        for word in _POSITIVE_CUES:
            classifier.token_embedding.weight[tokenizer.token_to_id(word)] += direction
        for word in _NEGATIVE_CUES:
            classifier.token_embedding.weight[tokenizer.token_to_id(word)] -= direction

        first_linear = classifier.head[1]
        second_linear = classifier.head[3]
        assert isinstance(first_linear, nn.Linear)
        assert isinstance(second_linear, nn.Linear)
        first_linear.weight.zero_()
        first_linear.bias.zero_()
        first_linear.weight[0, 0] = 2.0
        first_linear.weight[1, 1] = 2.0
        second_linear.weight.zero_()
        second_linear.bias.zero_()
        second_linear.weight[1, 0] = 2.5
        second_linear.weight[0, 1] = 2.5

    classifier.eval()
    return classifier


def resolve_autocast_dtype(device: torch.device) -> torch.dtype:
    """Select FP16 on CUDA/MPS and BF16 on CPU for low-precision autocast execution."""
    if device.type == "cuda":
        return torch.float16
    return torch.bfloat16


def execute_forward(
    callable_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    runtime: ExecutionRuntime,
    device: torch.device,
) -> torch.Tensor:
    """Execute a forward pass under the requested PyTorch runtime context."""
    if runtime == "eager_fp32":
        with torch.no_grad():
            logits = callable_fn(input_ids, attention_mask)
    elif runtime == "inference_mode_fp32" or runtime == "torch_compile":
        with torch.inference_mode():
            logits = callable_fn(input_ids, attention_mask)
    elif runtime == "autocast_low_precision":
        amp_dtype = resolve_autocast_dtype(device)
        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with torch.inference_mode(), amp_ctx:
            logits = callable_fn(input_ids, attention_mask)
    else:
        raise ValueError(f"Unsupported runtime mode: {runtime!r}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    return logits.float()


class TorchSentimentPredictor:
    """Production PyTorch predictor supporting eager, inference_mode, autocast, and compile."""

    _LABELS: tuple[SentimentLabel, SentimentLabel] = ("negative", "positive")

    def __init__(self, settings: ServingSettings) -> None:
        self._settings = settings
        self._device = self._resolve_device(settings.device)
        self._tokenizer = HashedSubwordTokenizer(
            vocab_size=settings.vocab_size,
            max_seq_length=settings.max_seq_length,
        )
        self._base_model: TransformerSentimentClassifier | None = None
        self._forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        self._ready = False

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if requested == "mps" and not torch.backends.mps.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def device_name(self) -> str:
        return str(self._device)

    @property
    def runtime_mode(self) -> ExecutionRuntime:
        return self._settings.runtime

    @property
    def parameter_count(self) -> int:
        if self._base_model is None:
            return 0
        return sum(p.numel() for p in self._base_model.parameters())

    def load(self) -> None:
        """Instantiate model weights, apply optional torch.compile, and run warmup passes."""
        model = build_seeded_classifier(self._settings, self._tokenizer).to(self._device)
        self._base_model = model

        if self._settings.runtime == "torch_compile":
            compiled = torch.compile(model, mode=self._settings.compile_mode)
            self._forward_fn = compiled  # type: ignore[assignment]
        else:
            self._forward_fn = model

        if self._settings.warmup_runs > 0:
            warm_ids, warm_mask = self._tokenizer.encode_batch(
                ["warmup inference request"],
                device=self._device,
            )
            for _ in range(self._settings.warmup_runs):
                execute_forward(
                    self._forward_fn,
                    warm_ids,
                    warm_mask,
                    runtime=self._settings.runtime,
                    device=self._device,
                )

        self._ready = True

    def forward_logits(self, texts: Sequence[str]) -> torch.Tensor:
        """Return raw float32 class logits for a batch of input strings."""
        if not self._ready or self._forward_fn is None:
            raise RuntimeError("Predictor is not loaded; call load() before inference")
        input_ids, attention_mask = self._tokenizer.encode_batch(texts, device=self._device)
        return execute_forward(
            self._forward_fn,
            input_ids,
            attention_mask,
            runtime=self._settings.runtime,
            device=self._device,
        )

    def predict_texts(self, texts: Sequence[str]) -> list[tuple[SentimentLabel, float]]:
        """Classify a batch of texts into (sentiment_label, confidence) pairs."""
        logits = self.forward_logits(texts)
        probs = torch.softmax(logits, dim=-1)
        confidences, pred_indices = torch.max(probs, dim=-1)

        outputs: list[tuple[SentimentLabel, float]] = []
        for idx_tensor, conf_tensor in zip(pred_indices, confidences, strict=True):
            label = self._LABELS[int(idx_tensor.item())]
            outputs.append((label, round(float(conf_tensor.item()), 4)))
        return outputs

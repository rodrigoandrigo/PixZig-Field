from __future__ import annotations

import os

import torch
import torch.nn as nn


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _bool_value(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _force_qwen35_torch_fallback() -> None:
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except ImportError:
        return

    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
    ):
        if hasattr(modeling_qwen3_5, name):
            setattr(modeling_qwen3_5, name, None)
    if hasattr(modeling_qwen3_5, "is_fast_path_available"):
        modeling_qwen3_5.is_fast_path_available = False


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def masked_mean(tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("tokens must be a torch.Tensor")
    if not torch.is_floating_point(tokens):
        raise TypeError("tokens must be a floating point tensor")
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [B, L, D], got {tuple(tokens.shape)}")
    if tokens.shape[0] < 1 or tokens.shape[1] < 1 or tokens.shape[2] < 1:
        raise ValueError("tokens must have non-empty batch, sequence, and feature dimensions")
    if attention_mask is None:
        return tokens.mean(dim=1)
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch.Tensor")
    if attention_mask.shape != tokens.shape[:2]:
        raise ValueError(
            f"attention_mask must have shape {tuple(tokens.shape[:2])}, got {tuple(attention_mask.shape)}"
        )
    mask = attention_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class DeterministicProjection(nn.Module):
    """Fixed projection for frozen text embeddings.

    The text encoder is not part of the training optimizer or checkpoints, so a
    random lazy projection would make conditioning depend on construction order
    or sampling seed. This module creates a deterministic projection from input
    width to output width on first use.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = _positive_int("output_dim", output_dim)
        self.register_buffer("_anchor", torch.empty(0), persistent=False)
        self.register_buffer("_weight", None, persistent=False)

    def _build_weight(self, input_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if input_dim == self.output_dim:
            return torch.empty(0, device=device, dtype=dtype)
        rows = torch.arange(1, input_dim + 1, device=device, dtype=torch.float32).unsqueeze(1)
        cols = torch.arange(1, self.output_dim + 1, device=device, dtype=torch.float32).unsqueeze(0)
        scale = (2.0 / (input_dim + self.output_dim)) ** 0.5
        weight = torch.sin(rows * cols * 0.01337) * scale
        return weight.to(dtype=dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("hidden must be a torch.Tensor")
        if not torch.is_floating_point(hidden):
            raise TypeError("hidden must be a floating point tensor")
        if hidden.ndim < 1 or hidden.shape[-1] < 1:
            raise ValueError("hidden must have a non-empty last dimension")
        input_dim = hidden.shape[-1]
        if input_dim == self.output_dim:
            return hidden
        weight = self._weight
        if (
            weight is None
            or weight.shape != (input_dim, self.output_dim)
            or weight.device != hidden.device
            or weight.dtype != hidden.dtype
        ):
            weight = self._build_weight(input_dim, hidden.device, hidden.dtype)
            self._weight = weight
        return hidden @ weight


class QwenTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-2B-Base",
        output_dim: int | None = None,
        *,
        load_model: bool = False,
        local_files_only: bool = True,
        cache_embeddings: bool = False,
        pooling: str = "mean",
        max_length: int | None = None,
        normalize: bool = False,
        force_torch_fallback: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if output_dim is not None:
            output_dim = _positive_int("output_dim", output_dim)
        _bool_value("load_model", load_model)
        _bool_value("local_files_only", local_files_only)
        _bool_value("cache_embeddings", cache_embeddings)
        _bool_value("normalize", normalize)
        _bool_value("force_torch_fallback", force_torch_fallback)
        if max_length is not None:
            max_length = _positive_int("max_length", max_length)
        force_torch_fallback = force_torch_fallback or _env_flag("PIXZIG_QWEN_TORCH_FALLBACK")
        if pooling not in {"none", "mean"}:
            raise ValueError("pooling must be 'none' or 'mean'")
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self.pooling = pooling
        self.max_length = max_length
        self.normalize = normalize
        self.output_dim = output_dim
        self.force_torch_fallback = force_torch_fallback
        self._cache: dict[tuple[str, ...], torch.Tensor] = {}
        self.tokenizer = None
        self.model = None
        self.proj = DeterministicProjection(output_dim) if output_dim is not None else nn.Identity()

        if load_model:
            if force_torch_fallback:
                _force_qwen35_torch_fallback()
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError("transformers is required to load QwenTextEncoder") from exc
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def _output_device(self, fallback: torch.device | None = None) -> torch.device:
        for parameter in self.parameters():
            return parameter.device
        for buffer in self.buffers():
            return buffer.device
        return torch.device("cpu") if fallback is None else fallback

    def _output_dtype(self, fallback: torch.dtype | None = None) -> torch.dtype:
        for parameter in self.parameters():
            return parameter.dtype
        for buffer in self.buffers():
            if buffer.dtype.is_floating_point:
                return buffer.dtype
        return torch.float32 if fallback is None else fallback

    def clear_cache(self) -> None:
        self._cache.clear()

    def _empty_conditioning(self, batch_size: int) -> torch.Tensor:
        if self.output_dim is None:
            raise RuntimeError("empty text requires output_dim to be configured")
        device = self._output_device()
        dtype = self._output_dtype()
        shape = (batch_size, self.output_dim) if self.pooling == "mean" else (batch_size, 1, self.output_dim)
        return torch.zeros(shape, device=device, dtype=dtype)

    def _project_pool_normalize(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("embeddings must be a torch.Tensor")
        if not torch.is_floating_point(hidden):
            raise TypeError("embeddings must be a floating point tensor")
        if self.pooling == "none":
            if hidden.ndim != 3:
                raise ValueError(f"embeddings must have shape [B, L, D] when pooling='none', got {tuple(hidden.shape)}")
        elif hidden.ndim == 3:
            hidden = masked_mean(hidden, attention_mask)
        elif hidden.ndim != 2:
            raise ValueError(f"embeddings must have shape [B, D] or [B, L, D], got {tuple(hidden.shape)}")
        elif attention_mask is not None:
            raise ValueError("attention_mask can only be used with token embeddings [B, L, D]")
        if hidden.shape[0] < 1 or hidden.shape[-1] < 1:
            raise ValueError("embeddings must have non-empty batch and feature dimensions")
        hidden = hidden.to(device=self._output_device(hidden.device), dtype=self._output_dtype(hidden.dtype))
        hidden = self.proj(hidden)
        if self.normalize:
            hidden = torch.nn.functional.normalize(hidden, dim=-1)
        return hidden

    def _normalize_text_batch(self, text: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        if not isinstance(text, (list, tuple)):
            raise TypeError("text must be a list or tuple of strings")
        if len(text) < 1:
            raise ValueError("text must contain at least one prompt")
        if not all(isinstance(item, str) for item in text):
            raise TypeError("text entries must be strings")
        return tuple(text)

    def forward(
        self,
        text: list[str] | tuple[str, ...] | None = None,
        *,
        embeddings: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text is not None and embeddings is not None:
            raise ValueError("provide either text or precomputed embeddings, not both")
        if embeddings is not None:
            return self._project_pool_normalize(embeddings, attention_mask)
        if text is None:
            raise ValueError("provide either text or precomputed embeddings")

        key = self._normalize_text_batch(text)
        if self.cache_embeddings and key in self._cache:
            cached = self._cache[key]
            return cached.to(device=self._output_device(), dtype=self._output_dtype()).clone()
        if all(not item.strip() for item in text):
            hidden = self._empty_conditioning(len(text))
            if self.cache_embeddings:
                self._cache[key] = hidden.detach().cpu().clone()
            return hidden
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Qwen model is not loaded; pass embeddings or initialize with load_model=True")

        tokenized = self.tokenizer(
            list(text),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        tokenized = {name: value.to(device) for name, value in tokenized.items()}
        with torch.no_grad():
            hidden = self.model(**tokenized).last_hidden_state
        hidden = self._project_pool_normalize(hidden, tokenized.get("attention_mask"))
        if self.cache_embeddings:
            self._cache[key] = hidden.detach().cpu().clone()
        return hidden

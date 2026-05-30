from __future__ import annotations

import math
from numbers import Real

import torch


def _positive_finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{name} must be finite")
    if value_float <= 0:
        raise ValueError(f"{name} must be positive")
    return value_float


def _prepare_t(t: torch.Tensor, eps: float) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a torch.Tensor")
    if t.numel() == 0:
        raise ValueError("t must contain at least one timestep")

    if not torch.is_floating_point(t) or t.dtype in {torch.float16, torch.bfloat16}:
        t = t.float()
    else:
        t = t.reshape(-1)

    if not torch.isfinite(t).all():
        raise ValueError("t must contain only finite values")
    if torch.any((t < 0) | (t > 1)):
        raise ValueError("t values must be in the [0, 1] interval")
    return t.reshape(-1).clamp(eps, 1.0 - eps)


def snr_from_t(t: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    eps = _positive_finite_float("eps", eps)
    if eps >= 0.5:
        raise ValueError("eps must be smaller than 0.5")

    t_clamped = _prepare_t(t, eps)
    alpha = t_clamped
    sigma = 1.0 - t_clamped
    return (alpha * alpha) / (sigma * sigma + eps)


def min_snr_weight(
    t: torch.Tensor,
    gamma: float = 5.0,
    eps: float = 1e-5,
    strategy: str = "min_snr",
) -> torch.Tensor:
    gamma = _positive_finite_float("gamma", gamma)
    snr = snr_from_t(t, eps=eps)
    if strategy == "min_snr":
        weight = torch.minimum(snr, torch.full_like(snr, gamma)) / (snr + eps)
    elif strategy == "soft_min_snr":
        weight = gamma / (snr + gamma + eps)
    else:
        raise ValueError("strategy must be 'min_snr' or 'soft_min_snr'")
    return torch.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)

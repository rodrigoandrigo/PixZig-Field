import math

import torch
import torch.nn as nn


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _prepare_timestep(t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a torch.Tensor")
    if not torch.is_floating_point(t):
        raise TypeError("t must be a floating point tensor")
    if t.numel() < 1:
        raise ValueError("t must contain at least one timestep")
    if not torch.isfinite(t).all():
        raise ValueError("t must contain only finite values")
    return t.reshape(-1).float()


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        frequency_embedding_size = _positive_int("frequency_embedding_size", frequency_embedding_size)
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        dim = _positive_int("dim", dim)
        max_period = _positive_int("max_period", max_period)
        if max_period <= 1:
            raise ValueError("max_period must be greater than 1")
        t = _prepare_timestep(t)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half, 1)
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.sinusoidal_embedding(t, self.frequency_embedding_size)
        first_parameter = next(self.mlp.parameters())
        t_freq = t_freq.to(device=first_parameter.device, dtype=first_parameter.dtype)
        return self.mlp(t_freq)

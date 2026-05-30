import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        if scale.shape != shift.shape:
            raise ValueError(f"scale and shift must have matching shapes, got {tuple(scale.shape)} and {tuple(shift.shape)}")
        if scale.shape[-1] != x.shape[-1]:
            raise ValueError(f"scale/shift last dimension must match x channels: {scale.shape[-1]} != {x.shape[-1]}")
        if scale.ndim > x.ndim:
            raise ValueError("scale and shift cannot have more dimensions than x")
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(-2)
            shift = shift.unsqueeze(-2)
        return x * (1.0 + scale) + shift


class AdaLN(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int | None = None) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        cond_dim = hidden_dim if cond_dim is None else cond_dim
        if cond_dim < 1:
            raise ValueError("cond_dim must be positive")
        bottleneck = min(128, max(32, cond_dim // 4))
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, 2 * hidden_dim),
        )
        self.film = FiLMLayer()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.norm.normalized_shape[0]:
            raise ValueError(f"x last dimension must be {self.norm.normalized_shape[0]}, got {x.shape[-1]}")
        if cond.shape[0] != x.shape[0]:
            raise ValueError(f"cond batch size must match x batch size: {cond.shape[0]} != {x.shape[0]}")
        scale, shift = self.modulation(cond).chunk(2, dim=-1)
        return self.film(self.norm(x), scale, shift)

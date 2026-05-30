import math
from functools import lru_cache
from typing import cast

import torch
import torch.nn as nn


@lru_cache(maxsize=32)
def _local_coords_cached(patch_size: int) -> torch.Tensor:
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    axis = torch.zeros(1) if patch_size == 1 else torch.linspace(-1.0, 1.0, patch_size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def local_patch_coordinates(
    patch_size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    coords = _local_coords_cached(patch_size)
    return coords.to(device=device, dtype=dtype).clone()


class FourierFeatures(nn.Module):
    def __init__(self, num_frequencies: int = 8) -> None:
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        self.num_frequencies = num_frequencies
        frequencies = torch.arange(1, num_frequencies + 1, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.out_dim = 4 * num_frequencies

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim < 1:
            raise ValueError("coords must have at least one dimension")
        if coords.shape[-1] != 2:
            raise ValueError(f"expected coords last dimension to be 2, got {coords.shape[-1]}")
        if not coords.dtype.is_floating_point:
            raise TypeError("coords must be a floating point tensor")
        frequencies = cast(torch.Tensor, self.frequencies)
        freqs = frequencies.to(device=coords.device, dtype=coords.dtype)
        angles = 2.0 * math.pi * coords[..., None] * freqs
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return torch.cat(
            [
                sin[..., 0, :],
                cos[..., 0, :],
                sin[..., 1, :],
                cos[..., 1, :],
            ],
            dim=-1,
        )

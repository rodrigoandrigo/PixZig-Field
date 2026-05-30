from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourier import FourierFeatures, local_patch_coordinates
from .patch import unpatchify


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_grid_size(grid_size: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(grid_size, tuple) or len(grid_size) != 2:
        raise TypeError("grid_size must be a tuple of two positive integers")
    return _positive_int("grid_size[0]", grid_size[0]), _positive_int("grid_size[1]", grid_size[1])


class _PixNerdRMSNorm(nn.Module):
    """RMSNorm used by the original PixNerd decoder blocks."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        _positive_int("hidden_dim", hidden_dim)
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1:
            raise ValueError("x must have at least one dimension")
        if x.shape[-1] != self.weight.shape[0]:
            raise ValueError(f"expected last dimension {self.weight.shape[0]}, got {x.shape[-1]}")
        dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return (x_norm.to(dtype=dtype) * self.weight.to(device=x.device, dtype=dtype))


class _NerfEmbedder(nn.Module):
    """Local pixel-field embedder inspired by PixNerd's NerfEmbedder.

    PixNerd concatenates per-pixel channels with a 2D frequency table before
    projecting into the decoder hidden width. PixZig does not have a decoded
    pixel input at this stage, so it uses a zero pixel canvas plus local Fourier
    coordinates; the generated field is still conditioned by the patch token.
    """

    def __init__(
        self,
        out_channels: int,
        hidden_dim: int,
        fourier_frequencies: int,
    ) -> None:
        super().__init__()
        _positive_int("out_channels", out_channels)
        _positive_int("hidden_dim", hidden_dim)
        _positive_int("fourier_frequencies", fourier_frequencies)
        self.out_channels = out_channels
        self.fourier = FourierFeatures(fourier_frequencies)
        self.embedder = nn.Linear(out_channels + self.fourier.out_dim, hidden_dim)
        self._fourier_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def _local_features(self, patch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        patch_size = _positive_int("patch_size", patch_size)
        key = (patch_size, device, dtype)
        cached = self._fourier_cache.get(key)
        if cached is not None:
            return cached
        coords = local_patch_coordinates(patch_size, device=device, dtype=dtype)
        features = self.fourier(coords)
        zeros = torch.zeros(features.shape[0], self.out_channels, device=device, dtype=dtype)
        features = torch.cat([zeros, features], dim=-1)
        self._fourier_cache[key] = features
        return features

    def forward(self, batch_tokens: int, patch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        batch_tokens = _positive_int("batch_tokens", batch_tokens)
        features = self._local_features(patch_size, device, dtype)
        field = self.embedder(features)
        return field.unsqueeze(0).expand(batch_tokens, -1, -1)


class _FactorizedNerfBlock(nn.Module):
    """Patch-conditioned neural field block close to PixNerd's NerfBlock.

    The original block generates two dense matrices per patch token:
    `[D, D * ratio]` and `[D * ratio, D]`. That is exact but impractical for
    PixZig's large default decoder width. This block keeps the original bmm
    computation and per-patch dynamic weights, while factorizing the dynamic
    component as a low-rank delta over shared base matrices.
    """

    def __init__(
        self,
        cond_dim: int,
        hidden_dim: int,
        *,
        mlp_ratio: int = 2,
        rank: int | None = None,
    ) -> None:
        super().__init__()
        if cond_dim < 1:
            raise ValueError("cond_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if mlp_ratio < 1:
            raise ValueError("mlp_ratio must be positive")
        if rank is not None and rank < 1:
            raise ValueError("rank must be positive when provided")
        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.inner_dim = hidden_dim * mlp_ratio
        self.rank = min(rank or max(4, hidden_dim // 16), hidden_dim, self.inner_dim)

        self.norm = _PixNerdRMSNorm(hidden_dim)
        self.base_fc1 = nn.Parameter(torch.empty(hidden_dim, self.inner_dim))
        self.base_fc2 = nn.Parameter(torch.empty(self.inner_dim, hidden_dim))
        self.param_generator = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, min(512, max(64, cond_dim // 2))),
            nn.SiLU(),
            nn.Linear(
                min(512, max(64, cond_dim // 2)),
                hidden_dim * self.rank
                + self.rank * self.inner_dim
                + self.inner_dim * self.rank
                + self.rank * hidden_dim
                + self.inner_dim
                + hidden_dim,
            ),
        )
        self.residual_gate = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, hidden_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.base_fc1)
        nn.init.xavier_uniform_(self.base_fc2)

    def _dynamic_weights(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if cond.ndim != 2 or cond.shape[-1] != self.param_generator[1].in_features:
            raise ValueError(
                f"cond must have shape [B, {self.param_generator[1].in_features}], got {tuple(cond.shape)}"
            )
        batch = cond.shape[0]
        params = self.param_generator(cond)
        sizes = [
            self.hidden_dim * self.rank,
            self.rank * self.inner_dim,
            self.inner_dim * self.rank,
            self.rank * self.hidden_dim,
            self.inner_dim,
            self.hidden_dim,
        ]
        fc1_left, fc1_right, fc2_left, fc2_right, bias1, bias2 = params.split(sizes, dim=-1)
        fc1_left = fc1_left.view(batch, self.hidden_dim, self.rank)
        fc1_right = fc1_right.view(batch, self.rank, self.inner_dim)
        fc2_left = fc2_left.view(batch, self.inner_dim, self.rank)
        fc2_right = fc2_right.view(batch, self.rank, self.hidden_dim)

        scale = self.rank**-0.5
        fc1 = self.base_fc1.unsqueeze(0) + scale * torch.bmm(fc1_left, fc1_right)
        fc2 = self.base_fc2.unsqueeze(0) + scale * torch.bmm(fc2_left, fc2_right)
        fc1 = F.normalize(fc1, dim=-2)
        return fc1, fc2, bias1, bias2

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.hidden_dim:
            raise ValueError(f"x must have shape [B, N, {self.hidden_dim}], got {tuple(x.shape)}")
        if cond.shape[0] != x.shape[0]:
            raise ValueError(f"cond batch size must match x batch size, got {cond.shape[0]} and {x.shape[0]}")
        residual = x
        fc1, fc2, bias1, bias2 = self._dynamic_weights(cond)
        x = self.norm(x)
        x = torch.bmm(x, fc1) + bias1.unsqueeze(1)
        x = F.silu(x)
        x = torch.bmm(x, fc2) + bias2.unsqueeze(1)
        gate = torch.sigmoid(self.residual_gate(cond)).unsqueeze(1)
        return residual + gate * x


class PixNerdHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        patch_size: int = 16,
        out_channels: int = 3,
        fourier_frequencies: int = 8,
        pixnerd_hidden_dim: int = 768,
        pixnerd_layers: int = 4,
    ) -> None:
        super().__init__()
        _positive_int("hidden_dim", hidden_dim)
        _positive_int("patch_size", patch_size)
        _positive_int("out_channels", out_channels)
        _positive_int("fourier_frequencies", fourier_frequencies)
        _positive_int("pixnerd_hidden_dim", pixnerd_hidden_dim)
        _positive_int("pixnerd_layers", pixnerd_layers)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.x_embedder = _NerfEmbedder(out_channels, pixnerd_hidden_dim, fourier_frequencies)
        self.token_proj = nn.Sequential(
            _PixNerdRMSNorm(hidden_dim),
            nn.Linear(hidden_dim, pixnerd_hidden_dim),
        )
        self.t_proj = nn.Linear(hidden_dim, pixnerd_hidden_dim)
        hyper_rank = min(64, max(4, pixnerd_hidden_dim // 16))
        self.blocks = nn.ModuleList(
            [
                _FactorizedNerfBlock(
                    pixnerd_hidden_dim,
                    pixnerd_hidden_dim,
                    mlp_ratio=2,
                    rank=hyper_rank,
                )
                for _ in range(pixnerd_layers)
            ]
        )
        self.final_norm = _PixNerdRMSNorm(pixnerd_hidden_dim)
        self.final = nn.Linear(pixnerd_hidden_dim, out_channels)
        self._fourier_cache = self.x_embedder._fourier_cache
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def _validate_forward_inputs(
        self,
        hidden_tokens: torch.Tensor,
        t_embed: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> tuple[int, int]:
        if not isinstance(hidden_tokens, torch.Tensor):
            raise TypeError("hidden_tokens must be a torch.Tensor")
        if not torch.is_floating_point(hidden_tokens):
            raise TypeError("hidden_tokens must be a floating point tensor")
        if hidden_tokens.ndim != 3:
            raise ValueError(f"hidden_tokens must have shape [B, N, C], got {tuple(hidden_tokens.shape)}")
        batch_size, num_tokens, hidden_dim = hidden_tokens.shape
        if batch_size < 1:
            raise ValueError("hidden_tokens must include a non-empty batch dimension")
        if num_tokens < 1:
            raise ValueError("hidden_tokens must include at least one token")
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"hidden_tokens last dimension must be {self.hidden_dim}, got {hidden_dim}")

        if not isinstance(t_embed, torch.Tensor):
            raise TypeError("t_embed must be a torch.Tensor")
        if not torch.is_floating_point(t_embed):
            raise TypeError("t_embed must be a floating point tensor")
        if t_embed.ndim != 2:
            raise ValueError(f"t_embed must have shape [B, C], got {tuple(t_embed.shape)}")
        if t_embed.shape[0] not in {1, batch_size}:
            raise ValueError(f"t_embed batch size must be 1 or match hidden_tokens batch size {batch_size}")
        if t_embed.shape[1] != self.hidden_dim:
            raise ValueError(f"t_embed last dimension must be {self.hidden_dim}, got {t_embed.shape[1]}")

        grid_h, grid_w = _validate_grid_size(grid_size)
        expected_tokens = grid_h * grid_w
        if num_tokens != expected_tokens:
            raise ValueError(f"grid_size expects {expected_tokens} tokens, got {num_tokens}")
        return batch_size, num_tokens

    def forward(
        self,
        hidden_tokens: torch.Tensor,
        t_embed: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, num_tokens = self._validate_forward_inputs(hidden_tokens, t_embed, grid_size)
        batch_tokens = batch_size * num_tokens
        cond = self.token_proj(hidden_tokens) + self.t_proj(t_embed).unsqueeze(1)
        cond = F.silu(cond).reshape(batch_tokens, -1)

        field = self.x_embedder(
            batch_tokens,
            self.patch_size,
            hidden_tokens.device,
            hidden_tokens.dtype,
        )
        for block in self.blocks:
            field = block(field, cond)
        pixels = self.final(self.final_norm(field))
        patch_pixels = pixels.transpose(1, 2).reshape(batch_size, num_tokens, -1)
        return unpatchify(patch_pixels, grid_size, self.patch_size, self.out_channels)

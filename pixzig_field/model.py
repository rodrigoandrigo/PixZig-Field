from dataclasses import fields, replace
from typing import Any, cast

import torch
import torch.nn as nn

from .config import PixZigFieldConfig
from .patch import patchify
from .pixnerd_head import PixNerdHead
from .refiner import ResidualConvRefiner
from .timestep import TimestepEmbedder
from .zigma import ZigMaBackbone


def _conditioning_dim(config: PixZigFieldConfig) -> int | None:
    if config.cond_dim is not None:
        return config.cond_dim
    return config.qwen_hidden_dim


class PixZigField(nn.Module):
    def __init__(self, config: PixZigFieldConfig | None = None, **kwargs: object) -> None:
        super().__init__()
        base_config = PixZigFieldConfig() if config is None else config
        if not isinstance(base_config, PixZigFieldConfig):
            raise TypeError("config must be a PixZigFieldConfig")
        valid_fields = {field.name for field in fields(PixZigFieldConfig)}
        unknown = set(kwargs) - valid_fields
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unknown PixZigField config argument(s): {names}")
        replace_config = cast(Any, replace)
        self.config = cast(PixZigFieldConfig, replace_config(base_config, **kwargs))

        patch_dim = self.config.in_channels * self.config.patch_size * self.config.patch_size
        self.patch_embed = nn.Linear(patch_dim, self.config.hidden_dim)
        self.t_embedder = TimestepEmbedder(self.config.hidden_dim)
        cond_dim = _conditioning_dim(self.config)

        self.backbone = ZigMaBackbone(
            hidden_dim=self.config.hidden_dim,
            depth=self.config.depth,
            state_dim=self.config.zigma_state_dim,
            expand=self.config.zigma_expand,
            scan_mode=self.config.scan_mode,
            cond_dim=cond_dim,
            mamba_backend=self.config.mamba_backend,
            mamba_inner_expand=self.config.mamba_inner_expand,
            mamba_head_dim=self.config.mamba_head_dim,
            mamba_num_groups=self.config.mamba_num_groups,
            mamba_conv_kernel=self.config.mamba_conv_kernel,
        )
        self.pixnerd_head = PixNerdHead(
            hidden_dim=self.config.hidden_dim,
            patch_size=self.config.patch_size,
            out_channels=self.config.out_channels,
            fourier_frequencies=self.config.fourier_frequencies,
            pixnerd_hidden_dim=self.config.pixnerd_hidden_dim,
            pixnerd_layers=self.config.pixnerd_layers,
        )
        self.refiner = ResidualConvRefiner(
            channels=self.config.refiner_channels,
            blocks=self.config.refiner_blocks,
            in_channels=self.config.out_channels + (self.config.in_channels if self.config.refiner_use_xt else 0),
            out_channels=self.config.out_channels,
            cond_dim=self.config.hidden_dim if self.config.refiner_conditioning else None,
        )

    def _validate_forward_inputs(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None,
    ) -> None:
        if not isinstance(x_t, torch.Tensor):
            raise TypeError("x_t must be a torch.Tensor")
        if not torch.is_floating_point(x_t):
            raise TypeError("x_t must be a floating point tensor")
        if x_t.ndim != 4:
            raise ValueError(f"x_t must have shape [B, C, H, W], got {tuple(x_t.shape)}")
        batch_size, channels, height, width = x_t.shape
        if batch_size < 1:
            raise ValueError("x_t must include a non-empty batch dimension")
        if channels != self.config.in_channels:
            raise ValueError(f"x_t must have {self.config.in_channels} channels, got {channels}")
        if height < self.config.patch_size or width < self.config.patch_size:
            raise ValueError(
                f"x_t spatial dimensions must be at least patch_size={self.config.patch_size}, "
                f"got H={height}, W={width}"
            )

        if not isinstance(t, torch.Tensor):
            raise TypeError("t must be a torch.Tensor")
        if not torch.is_floating_point(t):
            raise TypeError("t must be a floating point tensor")
        if t.numel() not in {1, batch_size}:
            raise ValueError(f"t must contain either 1 value or one value per batch item, got {t.numel()}")
        if not torch.isfinite(t).all():
            raise ValueError("t must contain only finite values")

        expected_cond_dim = _conditioning_dim(self.config)
        if cond is None:
            return
        if expected_cond_dim is None:
            raise ValueError("cond was provided but cond_dim or qwen_hidden_dim was not configured")
        if not isinstance(cond, torch.Tensor):
            raise TypeError("cond must be a torch.Tensor when provided")
        if not torch.is_floating_point(cond):
            raise TypeError("cond must be a floating point tensor")
        if cond.ndim not in {2, 3}:
            raise ValueError(f"cond must have shape [B, D] or [B, L, D], got {tuple(cond.shape)}")
        if cond.shape[0] not in {1, batch_size}:
            raise ValueError(f"cond batch size must be 1 or match x_t batch size {batch_size}, got {cond.shape[0]}")
        if cond.ndim == 3 and cond.shape[1] < 1:
            raise ValueError("cond sequence length must be non-empty")
        if cond.shape[-1] != expected_cond_dim:
            raise ValueError(f"cond last dimension must be {expected_cond_dim}, got {cond.shape[-1]}")
        if not torch.isfinite(cond).all():
            raise ValueError("cond must contain only finite values")

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_forward_inputs(x_t, t, cond)
        patches, grid_size = patchify(x_t, self.config.patch_size)
        tokens = self.patch_embed(patches)
        t_embed = self.t_embedder(t)
        hidden_tokens = self.backbone(tokens, t_embed, cond=cond, grid_size=grid_size)
        x0_coarse = self.pixnerd_head(hidden_tokens, t_embed, grid_size)
        residual = self.refiner(
            x0_coarse,
            t_embed if self.config.refiner_conditioning else None,
            x_t=x_t if self.config.refiner_use_xt else None,
        )
        return x0_coarse + residual

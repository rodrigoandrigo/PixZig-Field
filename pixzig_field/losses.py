from __future__ import annotations

import warnings
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PixZigLossConfig, PixZigMinSNRConfig
from .flow_matching import velocity_from_x0, velocity_target
from .min_snr import min_snr_weight


def _validate_same_shape(
    left_name: str,
    left: torch.Tensor,
    right_name: str,
    right: torch.Tensor,
) -> None:
    if left.shape != right.shape:
        raise ValueError(
            f"{left_name} and {right_name} must have matching shapes, "
            f"got {tuple(left.shape)} and {tuple(right.shape)}"
        )
    if left.ndim == 0 or left.shape[0] == 0:
        raise ValueError(f"{left_name} and {right_name} must include a non-empty batch dimension")


def _validate_image_pair(
    left_name: str,
    left: torch.Tensor,
    right_name: str,
    right: torch.Tensor,
) -> None:
    _validate_same_shape(left_name, left, right_name, right)
    if left.ndim != 4:
        raise ValueError(
            f"{left_name} and {right_name} must be 4D image tensors [B, C, H, W], "
            f"got {left.ndim}D"
        )
    if min(left.shape[1:]) <= 0:
        raise ValueError(f"{left_name} and {right_name} must have non-empty channel and spatial dimensions")


def _mean_per_batch(loss: torch.Tensor) -> torch.Tensor:
    if loss.ndim == 0 or loss.shape[0] == 0:
        raise ValueError("loss must include a non-empty batch dimension")
    return loss.reshape(loss.shape[0], -1).mean(dim=1)


def flow_mse_loss(v_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_same_shape("v_pred", v_pred, "target", target)
    return _mean_per_batch((v_pred - target) ** 2)


def x0_huber_loss(x0_pred: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    _validate_same_shape("x0_pred", x0_pred, "x0", x0)
    return _mean_per_batch(F.huber_loss(x0_pred, x0, reduction="none"))


def multiscale_huber_loss(
    x0_pred: torch.Tensor,
    x0: torch.Tensor,
    levels: int = 3,
) -> torch.Tensor:
    _validate_image_pair("x0_pred", x0_pred, "x0", x0)
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    losses = [x0_huber_loss(x0_pred, x0)]
    pred_level = x0_pred
    true_level = x0
    for _ in range(max(levels - 1, 0)):
        if min(pred_level.shape[-2:]) < 4:
            break
        pred_level = F.avg_pool2d(pred_level, kernel_size=2, stride=2)
        true_level = F.avg_pool2d(true_level, kernel_size=2, stride=2)
        losses.append(x0_huber_loss(pred_level, true_level))
    return torch.stack(losses, dim=0).mean(dim=0)


def frequency_magnitude_loss(x0_pred: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    _validate_image_pair("x0_pred", x0_pred, "x0", x0)
    pred_fft = torch.fft.rfft2(x0_pred.float(), norm="ortho")
    true_fft = torch.fft.rfft2(x0.float(), norm="ortho")
    pred_mag = torch.log1p(pred_fft.abs())
    true_mag = torch.log1p(true_fft.abs())
    return _mean_per_batch(torch.abs(pred_mag - true_mag)).to(dtype=x0_pred.dtype)


def boundary_gradient_loss(
    x0_pred: torch.Tensor,
    x0: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor:
    _validate_image_pair("x0_pred", x0_pred, "x0", x0)
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    losses: list[torch.Tensor] = []
    height, width = x0_pred.shape[-2:]

    cols = list(range(patch_size, width, patch_size))
    if cols:
        pred_grad = x0_pred[..., :, cols] - x0_pred[..., :, [col - 1 for col in cols]]
        true_grad = x0[..., :, cols] - x0[..., :, [col - 1 for col in cols]]
        losses.append(_mean_per_batch(torch.abs(pred_grad - true_grad)))

    rows = list(range(patch_size, height, patch_size))
    if rows:
        pred_grad = x0_pred[..., rows, :] - x0_pred[..., [row - 1 for row in rows], :]
        true_grad = x0[..., rows, :] - x0[..., [row - 1 for row in rows], :]
        losses.append(_mean_per_batch(torch.abs(pred_grad - true_grad)))

    if not losses:
        return torch.zeros(x0_pred.shape[0], device=x0_pred.device, dtype=x0_pred.dtype)
    return torch.stack(losses, dim=0).mean(dim=0)


def _center_crop_pair(
    x0_pred: torch.Tensor,
    x0: torch.Tensor,
    crop_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_image_pair("x0_pred", x0_pred, "x0", x0)
    if crop_size is None:
        return x0_pred, x0
    if crop_size < 1:
        raise ValueError(f"crop_size must be >= 1, got {crop_size}")
    crop_size = min(int(crop_size), x0_pred.shape[-2], x0_pred.shape[-1])
    if crop_size == x0_pred.shape[-2] == x0_pred.shape[-1]:
        return x0_pred, x0
    top = (x0_pred.shape[-2] - crop_size) // 2
    left = (x0_pred.shape[-1] - crop_size) // 2
    return (
        x0_pred[..., top : top + crop_size, left : left + crop_size],
        x0[..., top : top + crop_size, left : left + crop_size],
    )


class LightweightLPIPSLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor(
            [
                [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("kernel", kernel, persistent=False)

    def forward(self, x0_pred: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        _validate_image_pair("x0_pred", x0_pred, "x0", x0)
        kernel_buffer = cast(torch.Tensor, self.kernel)
        channels = x0_pred.shape[1]
        kernel = kernel_buffer.to(device=x0_pred.device, dtype=x0_pred.dtype).repeat(channels, 1, 1, 1)
        pred_features = F.conv2d(x0_pred, kernel, padding=1, groups=channels)
        true_features = F.conv2d(x0, kernel, padding=1, groups=channels)
        return _mean_per_batch(torch.abs(pred_features - true_features))


class OptionalPerceptualLoss(nn.Module):
    def __init__(self, backend: str = "proxy") -> None:
        super().__init__()
        if backend not in {"proxy", "lpips", "auto"}:
            raise ValueError(f"backend must be one of 'proxy', 'lpips', or 'auto', got {backend!r}")
        self.backend = backend
        self.proxy = LightweightLPIPSLoss()
        self.lpips_model: nn.Module | None = None
        if backend in {"lpips", "auto"}:
            try:
                import lpips  # type: ignore[import-not-found,import-untyped]
            except ImportError:
                if backend == "lpips":
                    warnings.warn("LPIPS backend requested but lpips is not installed; using proxy.", RuntimeWarning)
            else:
                self.lpips_model = lpips.LPIPS(net="vgg").eval()
                for parameter in self.lpips_model.parameters():
                    parameter.requires_grad_(False)

    def forward(self, x0_pred: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        if self.lpips_model is None:
            return self.proxy(x0_pred, x0)
        model = self.lpips_model.to(device=x0_pred.device)
        loss = model(x0_pred.float().clamp(-1, 1), x0.float().clamp(-1, 1), normalize=False)
        return loss.reshape(loss.shape[0], -1).mean(dim=1).to(dtype=x0_pred.dtype)


class PixZigLoss(nn.Module):
    def __init__(
        self,
        loss_config: PixZigLossConfig | None = None,
        min_snr_config: PixZigMinSNRConfig | None = None,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.loss_config = PixZigLossConfig() if loss_config is None else loss_config
        self.min_snr_config = PixZigMinSNRConfig() if min_snr_config is None else min_snr_config
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        self.patch_size = patch_size
        self.perceptual = OptionalPerceptualLoss(self.loss_config.perceptual_backend)
        if self.loss_config.lpips_enabled and self.loss_config.lpips_weight > 0:
            warnings.warn(
                "Using configured perceptual loss backend. Install lpips and set "
                "perceptual_backend='lpips' or 'auto' for exact LPIPS.",
                RuntimeWarning,
                stacklevel=2,
            )

    def forward(
        self,
        *,
        x0_pred: torch.Tensor,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        step: int = 0,
    ) -> dict[str, torch.Tensor]:
        _validate_image_pair("x0_pred", x0_pred, "x0", x0)
        _validate_same_shape("x_t", x_t, "x0", x0)
        _validate_same_shape("noise", noise, "x0", x0)
        v_pred = velocity_from_x0(x_t, x0_pred, t)
        target = velocity_target(x0, noise, t)
        flow = flow_mse_loss(v_pred, target)
        x0_loss = x0_huber_loss(x0_pred, x0)
        multiscale = torch.zeros_like(flow)
        if self.loss_config.multiscale_weight > 0:
            multiscale = multiscale_huber_loss(x0_pred, x0)
        frequency = torch.zeros_like(flow)
        if self.loss_config.frequency_weight > 0:
            frequency = frequency_magnitude_loss(x0_pred, x0)
        boundary = torch.zeros_like(flow)
        if self.loss_config.boundary_weight > 0:
            boundary = boundary_gradient_loss(x0_pred, x0, patch_size=self.patch_size)

        if self.min_snr_config.enabled:
            weights = min_snr_weight(
                t,
                gamma=self.min_snr_config.gamma,
                strategy=self.min_snr_config.strategy,
            ).to(flow)
        else:
            weights = torch.ones_like(flow)

        lpips = torch.zeros_like(flow)
        should_run_lpips = (
            self.loss_config.lpips_enabled
            and self.loss_config.lpips_weight > 0
            and step % max(self.loss_config.lpips_every_n_steps, 1) == 0
        )
        if should_run_lpips:
            perceptual_pred, perceptual_target = _center_crop_pair(
                x0_pred,
                x0,
                self.loss_config.lpips_crop_size,
            )
            lpips = self.perceptual(perceptual_pred, perceptual_target)

        total = (
            self.loss_config.flow_weight * weights * flow
            + self.loss_config.x0_weight * x0_loss
            + self.loss_config.lpips_weight * lpips
            + self.loss_config.multiscale_weight * multiscale
            + self.loss_config.frequency_weight * frequency
            + self.loss_config.boundary_weight * boundary
        )
        return {
            "loss": total.mean(),
            "flow_loss": flow.mean(),
            "x0_loss": x0_loss.mean(),
            "lpips_loss": lpips.mean(),
            "multiscale_loss": multiscale.mean(),
            "frequency_loss": frequency.mean(),
            "boundary_loss": boundary.mean(),
            "min_snr_weight": weights.mean(),
        }

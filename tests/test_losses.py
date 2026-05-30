import torch
import torch.nn as nn
import pytest

from pixzig_field import PixZigField
from pixzig_field.config import PixZigLossConfig
from pixzig_field.flow_matching import make_xt, sample_timesteps, velocity_from_x0, velocity_target, x0_from_velocity
from pixzig_field.losses import (
    LightweightLPIPSLoss,
    OptionalPerceptualLoss,
    PixZigLoss,
    boundary_gradient_loss,
    frequency_magnitude_loss,
    flow_mse_loss,
    multiscale_huber_loss,
    x0_huber_loss,
)
from pixzig_field.min_snr import min_snr_weight


class ShapeRecordingLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shape: tuple[int, ...] | None = None

    def forward(self, x0_pred: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        self.shape = tuple(x0_pred.shape)
        return torch.zeros(x0_pred.shape[0], device=x0_pred.device, dtype=x0_pred.dtype)


def test_velocity_from_x0_matches_target_for_true_x0():
    x0 = torch.randn(2, 3, 16, 16)
    noise = torch.randn_like(x0)
    t = torch.tensor([0.25, 0.75])
    x_t = make_xt(x0, noise, t)

    v_pred = velocity_from_x0(x_t, x0, t)
    v_target = velocity_target(x0, noise, t)

    assert torch.allclose(v_pred, v_target, atol=1e-5)


def test_x0_from_velocity_inverts_flow_state():
    x0 = torch.randn(2, 3, 16, 16)
    noise = torch.randn_like(x0)
    t = torch.tensor([0.2, 0.8])
    x_t = make_xt(x0, noise, t)
    velocity = velocity_target(x0, noise, t)

    restored = x0_from_velocity(x_t, velocity, t)

    assert torch.allclose(restored, x0, atol=1e-5)


def test_timestep_sampling_modes_are_valid():
    for mode in ["uniform", "logit_normal", "cosine", "beta", "stratified"]:
        t = sample_timesteps(32, device="cpu", mode=mode)
        assert t.shape == (32,)
        assert torch.all((t > 0) & (t < 1))


def test_timestep_sampling_rejects_invalid_arguments():
    for kwargs in (
        {"batch_size": 0},
        {"batch_size": 1, "eps": 0.5},
        {"batch_size": 1, "logit_normal_std": 0.0},
        {"batch_size": 1, "beta_alpha": 0.0},
        {"batch_size": 1, "beta_beta": 0.0},
    ):
        try:
            sample_timesteps(device="cpu", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_flow_helpers_validate_shapes_and_accept_scalar_timestep():
    x0 = torch.randn(2, 3, 8, 8)
    noise = torch.randn_like(x0)

    x_t = make_xt(x0, noise, torch.tensor(0.5))
    assert x_t.shape == x0.shape

    bad_t = torch.rand(3)
    for fn, args in (
        (make_xt, (x0, noise, bad_t)),
        (x0_from_velocity, (x0, noise, bad_t)),
        (velocity_from_x0, (x0, noise, bad_t)),
    ):
        try:
            fn(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {fn.__name__}")

    with pytest.raises(ValueError, match="matching shapes"):
        make_xt(x0, noise[:1], torch.rand(2))
    with pytest.raises(ValueError, match="matching shapes"):
        velocity_target(x0, noise[:1])
    with pytest.raises(ValueError, match="matching shapes"):
        x0_from_velocity(x0, noise[:1], torch.rand(2))
    with pytest.raises(ValueError, match="matching shapes"):
        velocity_from_x0(x0, noise[:1], torch.rand(2))


def test_min_snr_no_nan_or_inf():
    t = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    weights = min_snr_weight(t, strategy="soft_min_snr")

    assert weights.shape == t.shape
    assert torch.isfinite(weights).all()


def test_boundary_loss_works():
    x0 = torch.randn(2, 3, 32, 32)
    x0_pred = x0 + 0.1 * torch.randn_like(x0)
    loss = boundary_gradient_loss(x0_pred, x0, patch_size=16)

    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()


def test_multiscale_and_frequency_losses_work():
    x0 = torch.randn(2, 3, 32, 32)
    x0_pred = x0 + 0.1 * torch.randn_like(x0)

    multiscale = multiscale_huber_loss(x0_pred, x0)
    frequency = frequency_magnitude_loss(x0_pred, x0)

    assert multiscale.shape == (2,)
    assert frequency.shape == (2,)
    assert torch.isfinite(multiscale).all()
    assert torch.isfinite(frequency).all()


def test_loss_helpers_reject_invalid_shapes_and_arguments():
    x0 = torch.randn(2, 3, 16, 16)
    x0_pred = x0 + 0.1 * torch.randn_like(x0)

    for fn in (flow_mse_loss, x0_huber_loss):
        with pytest.raises(ValueError, match="matching shapes"):
            fn(x0_pred, x0[:1])

    with pytest.raises(ValueError, match="4D image tensors"):
        frequency_magnitude_loss(x0_pred[0], x0[0])

    with pytest.raises(ValueError, match="levels"):
        multiscale_huber_loss(x0_pred, x0, levels=0)

    with pytest.raises(ValueError, match="patch_size"):
        boundary_gradient_loss(x0_pred, x0, patch_size=0)

    with pytest.raises(ValueError, match="backend"):
        OptionalPerceptualLoss("invalid")

    with pytest.raises(ValueError, match="patch_size"):
        PixZigLoss(patch_size=0)


def test_lightweight_lpips_supports_non_rgb_channels():
    loss_fn = LightweightLPIPSLoss()

    for channels in (1, 3, 4):
        x0 = torch.randn(2, channels, 16, 16)
        x0_pred = x0 + 0.1 * torch.randn_like(x0)
        loss = loss_fn(x0_pred, x0)

        assert loss.shape == (2,)
        assert torch.isfinite(loss).all()


def test_total_loss_scalar_and_backward():
    model = PixZigField(
        image_size=32,
        patch_size=16,
        hidden_dim=64,
        depth=2,
        pixnerd_hidden_dim=48,
        pixnerd_layers=2,
        refiner_channels=8,
        refiner_blocks=1,
    )
    loss_fn = PixZigLoss(PixZigLossConfig(lpips_weight=0.0, lpips_enabled=False), patch_size=16)
    x0 = torch.randn(1, 3, 32, 32)
    noise = torch.randn_like(x0)
    t = sample_timesteps(1, device=x0.device)
    x_t = make_xt(x0, noise, t)
    x0_pred = model(x_t, t)
    loss_dict = loss_fn(x0_pred=x0_pred, x0=x0, x_t=x_t, noise=noise, t=t)

    assert loss_dict["loss"].ndim == 0
    assert torch.isfinite(loss_dict["loss"])
    assert "multiscale_loss" in loss_dict
    assert "frequency_loss" in loss_dict
    loss_dict["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_lpips_crop_size_is_applied_to_perceptual_loss():
    recorder = ShapeRecordingLoss()
    loss_fn = PixZigLoss(
        PixZigLossConfig(
            flow_weight=0.0,
            x0_weight=0.0,
            lpips_weight=1.0,
            lpips_enabled=True,
            lpips_crop_size=8,
            multiscale_weight=0.0,
            frequency_weight=0.0,
            boundary_weight=0.0,
        ),
        patch_size=16,
    )
    loss_fn.perceptual = recorder
    x0 = torch.randn(2, 3, 16, 16)
    noise = torch.randn_like(x0)
    t = torch.full((2,), 0.5)
    x_t = make_xt(x0, noise, t)

    loss_fn(x0_pred=x0, x0=x0, x_t=x_t, noise=noise, t=t)

    assert recorder.shape == (2, 3, 8, 8)

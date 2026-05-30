import pytest
import torch

from pixzig_field.min_snr import min_snr_weight, snr_from_t


def test_snr_from_t_preserves_valid_shape_and_dtype():
    t = torch.tensor([[0.0, 0.5], [1.0, 0.25]], dtype=torch.float64)

    snr = snr_from_t(t)

    assert snr.shape == (4,)
    assert snr.dtype == torch.float64
    assert torch.isfinite(snr).all()


def test_snr_from_t_promotes_low_precision_timesteps():
    t = torch.tensor([0.25, 0.75], dtype=torch.bfloat16)

    snr = snr_from_t(t)

    assert snr.dtype == torch.float32
    assert torch.isfinite(snr).all()


def test_min_snr_weight_rejects_invalid_arguments():
    t = torch.tensor([0.5])

    for kwargs in (
        {"gamma": 0.0},
        {"gamma": float("inf")},
        {"eps": 0.0},
        {"eps": 0.5},
    ):
        with pytest.raises(ValueError):
            min_snr_weight(t, **kwargs)

    with pytest.raises(TypeError, match="gamma"):
        min_snr_weight(t, gamma=True)

    with pytest.raises(ValueError, match="strategy"):
        min_snr_weight(t, strategy="invalid")


def test_snr_from_t_rejects_invalid_timesteps():
    for t in (
        torch.tensor([]),
        torch.tensor([float("nan")]),
        torch.tensor([-0.1]),
        torch.tensor([1.1]),
    ):
        with pytest.raises(ValueError):
            snr_from_t(t)

    with pytest.raises(TypeError, match="torch.Tensor"):
        snr_from_t([0.5])  # type: ignore[arg-type]

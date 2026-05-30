import pytest
import torch

from pixzig_field.film_adaln import AdaLN, FiLMLayer


def test_film_layer_broadcasts_over_middle_dimensions():
    film = FiLMLayer()
    x = torch.ones(2, 3, 4)
    scale = torch.full((2, 4), 2.0)
    shift = torch.full((2, 4), 0.5)

    y = film(x, scale, shift)

    assert y.shape == x.shape
    assert torch.allclose(y, torch.full_like(x, 3.5))


def test_film_layer_broadcasts_image_channels_last_layout():
    film = FiLMLayer()
    x = torch.ones(2, 3, 5, 4)
    scale = torch.full((2, 4), 1.0)
    shift = torch.full((2, 4), -0.25)

    y = film(x, scale, shift)

    assert y.shape == x.shape
    assert torch.allclose(y, torch.full_like(x, 1.75))


def test_film_layer_rejects_incompatible_shapes():
    film = FiLMLayer()

    with pytest.raises(ValueError, match="matching shapes"):
        film(torch.ones(2, 3, 4), torch.ones(2, 4), torch.ones(2, 3))
    with pytest.raises(ValueError, match="last dimension"):
        film(torch.ones(2, 3, 4), torch.ones(2, 5), torch.ones(2, 5))


def test_adaln_preserves_shape_and_supports_token_tensors():
    module = AdaLN(hidden_dim=4, cond_dim=6)
    x = torch.randn(2, 3, 4)
    cond = torch.randn(2, 6)

    y = module(x, cond)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_adaln_rejects_invalid_shapes_and_dims():
    with pytest.raises(ValueError, match="hidden_dim"):
        AdaLN(hidden_dim=0)
    with pytest.raises(ValueError, match="cond_dim"):
        AdaLN(hidden_dim=4, cond_dim=0)

    module = AdaLN(hidden_dim=4, cond_dim=6)
    with pytest.raises(ValueError, match="last dimension"):
        module(torch.randn(2, 3, 5), torch.randn(2, 6))
    with pytest.raises(ValueError, match="batch size"):
        module(torch.randn(2, 3, 4), torch.randn(3, 6))

import pytest
import torch

from pixzig_field.fourier import FourierFeatures, local_patch_coordinates
from pixzig_field.patch import patchify, unpatchify, validate_patchable


def test_patchify_unpatchify_256():
    x = torch.randn(1, 3, 256, 256)
    patches, grid_size = patchify(x, patch_size=16)
    y = unpatchify(patches, grid_size, patch_size=16, out_channels=3)

    assert grid_size == (16, 16)
    assert y.shape == x.shape
    assert torch.allclose(x, y)


def test_patchify_unpatchify_rectangular():
    x = torch.randn(1, 3, 256, 384)
    patches, grid_size = patchify(x, patch_size=16)
    y = unpatchify(patches, grid_size, patch_size=16, out_channels=3)

    assert grid_size == (16, 24)
    assert y.shape == x.shape
    assert torch.allclose(x, y)


def test_patch_size_must_be_positive():
    x = torch.randn(1, 3, 16, 16)
    with pytest.raises(ValueError, match="patch_size"):
        patchify(x, patch_size=0)


def test_patchify_rejects_invalid_input_contract():
    with pytest.raises(TypeError, match="torch.Tensor"):
        patchify([1, 2, 3], patch_size=16)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="integer"):
        patchify(torch.randn(1, 3, 16, 16), patch_size=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty batch"):
        patchify(torch.empty(0, 3, 16, 16), patch_size=16)

    with pytest.raises(ValueError, match="at least one channel"):
        patchify(torch.empty(1, 0, 16, 16), patch_size=16)

    with pytest.raises(ValueError, match="spatial dimensions"):
        validate_patchable(torch.empty(1, 3, 0, 16), patch_size=16)


def test_patchify_rejects_non_divisible_shape():
    x = torch.randn(1, 3, 255, 256)
    with pytest.raises(ValueError, match="divisible"):
        patchify(x, patch_size=16)


def test_unpatchify_rejects_invalid_input_contract():
    patches = torch.randn(1, 4, 3 * 8 * 8)

    with pytest.raises(TypeError, match="torch.Tensor"):
        unpatchify([1, 2, 3], (2, 2), patch_size=8, out_channels=3)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="integer"):
        unpatchify(patches, (2, 2), patch_size=True, out_channels=3)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="out_channels"):
        unpatchify(patches, (2, 2), patch_size=8, out_channels=0)

    with pytest.raises(ValueError, match="shape"):
        unpatchify(torch.randn(4, 3 * 8 * 8), (2, 2), patch_size=8, out_channels=3)

    with pytest.raises(ValueError, match="non-empty batch"):
        unpatchify(torch.empty(0, 4, 3 * 8 * 8), (2, 2), patch_size=8, out_channels=3)

    with pytest.raises(TypeError, match="grid_size"):
        unpatchify(patches, [2, 2], patch_size=8, out_channels=3)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="expected 4 patches"):
        unpatchify(torch.randn(1, 3, 3 * 8 * 8), (2, 2), patch_size=8, out_channels=3)

    with pytest.raises(ValueError, match="patch dim"):
        unpatchify(torch.randn(1, 4, 3 * 8 * 8 + 1), (2, 2), patch_size=8, out_channels=3)


def test_single_pixel_patch_coordinate_is_centered():
    coords = local_patch_coordinates(1)

    assert coords.shape == (1, 2)
    assert torch.allclose(coords, torch.zeros_like(coords))


def test_local_patch_coordinates_cache_is_not_mutable_by_callers():
    coords = local_patch_coordinates(2)
    coords[0, 0] = 123.0

    fresh = local_patch_coordinates(2)

    assert fresh[0, 0] != 123.0


def test_fourier_features_shape_and_validation():
    features = FourierFeatures(num_frequencies=3)
    coords = torch.zeros(5, 2)

    out = features(coords)

    assert out.shape == (5, 12)
    assert torch.isfinite(out).all()

    with pytest.raises(ValueError, match="last dimension"):
        features(torch.zeros(5, 3))
    with pytest.raises(TypeError, match="floating point"):
        features(torch.zeros(5, 2, dtype=torch.long))

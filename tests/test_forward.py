import torch
import pytest

from pixzig_field import PixZigField
from pixzig_field.pixnerd_head import PixNerdHead
from pixzig_field.refiner import ResidualConvRefiner
from pixzig_field.timestep import TimestepEmbedder


def test_forward_rectangular():
    model = PixZigField(
        image_size=256,
        patch_size=16,
        hidden_dim=128,
        depth=2,
        pixnerd_hidden_dim=96,
        pixnerd_layers=2,
        refiner_channels=16,
        refiner_blocks=1,
    )

    x = torch.randn(1, 3, 256, 384)
    t = torch.rand(1)

    y = model(x, t)

    assert y.shape == x.shape


def test_model_rejects_invalid_constructor_arguments():
    with pytest.raises(TypeError, match="PixZigFieldConfig"):
        PixZigField(config=object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="unknown"):
        PixZigField(unknown_field=1)


def test_forward_rejects_invalid_image_inputs():
    model = PixZigField(
        image_size=16,
        patch_size=8,
        hidden_dim=32,
        depth=1,
        pixnerd_hidden_dim=16,
        pixnerd_layers=1,
        refiner_channels=8,
        refiner_blocks=1,
    )

    with pytest.raises(TypeError, match="floating point"):
        model(torch.ones(1, 3, 16, 16, dtype=torch.int64), torch.rand(1))

    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(3, 16, 16), torch.rand(1))

    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(1, 1, 16, 16), torch.rand(1))

    with pytest.raises(ValueError, match="patch_size"):
        model(torch.randn(1, 3, 4, 16), torch.rand(1))


def test_forward_rejects_invalid_timestep_inputs():
    model = PixZigField(
        image_size=16,
        patch_size=8,
        hidden_dim=32,
        depth=1,
        pixnerd_hidden_dim=16,
        pixnerd_layers=1,
        refiner_channels=8,
        refiner_blocks=1,
    )
    x = torch.randn(2, 3, 16, 16)

    with pytest.raises(TypeError, match="floating point"):
        model(x, torch.tensor([1, 2]))

    with pytest.raises(ValueError, match="one value per batch"):
        model(x, torch.rand(3))

    with pytest.raises(ValueError, match="finite"):
        model(x, torch.tensor([float("nan"), 0.5]))


def test_forward_rejects_invalid_conditioning_inputs():
    x = torch.randn(2, 3, 16, 16)
    t = torch.rand(2)
    unconditioned = PixZigField(
        image_size=16,
        patch_size=8,
        hidden_dim=32,
        depth=1,
        pixnerd_hidden_dim=16,
        pixnerd_layers=1,
        refiner_channels=8,
        refiner_blocks=1,
    )

    with pytest.raises(ValueError, match="cond_dim"):
        unconditioned(x, t, cond=torch.randn(2, 4))

    conditioned = PixZigField(
        image_size=16,
        patch_size=8,
        hidden_dim=32,
        depth=1,
        pixnerd_hidden_dim=16,
        pixnerd_layers=1,
        refiner_channels=8,
        refiner_blocks=1,
        cond_dim=4,
    )

    with pytest.raises(ValueError, match="batch size"):
        conditioned(x, t, cond=torch.randn(3, 4))

    with pytest.raises(ValueError, match="sequence length"):
        conditioned(x, t, cond=torch.randn(2, 0, 4))

    with pytest.raises(ValueError, match="last dimension"):
        conditioned(x, t, cond=torch.randn(2, 5))

    with pytest.raises(ValueError, match="finite"):
        conditioned(x, t, cond=torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]))

    assert conditioned(x, torch.tensor(0.5), cond=torch.randn(1, 3, 4)).shape == x.shape


def test_refiner_accepts_timestep_conditioning():
    refiner = ResidualConvRefiner(channels=8, blocks=1, in_channels=6, out_channels=3, cond_dim=16)
    x = torch.randn(2, 3, 16, 16)
    x_t = torch.randn_like(x)
    cond = torch.randn(2, 16)

    y = refiner(x, cond, x_t=x_t)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_refiner_rejects_invalid_constructor_arguments():
    with pytest.raises(TypeError, match="integer"):
        ResidualConvRefiner(channels=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="blocks"):
        ResidualConvRefiner(blocks=-1)

    with pytest.raises(ValueError, match="cond_dim"):
        ResidualConvRefiner(cond_dim=0)


def test_refiner_rejects_invalid_forward_inputs():
    conditioned = ResidualConvRefiner(channels=8, blocks=1, in_channels=6, out_channels=3, cond_dim=16)
    x = torch.randn(2, 3, 16, 16)
    x_t = torch.randn_like(x)
    cond = torch.randn(2, 16)

    with pytest.raises(TypeError, match="floating point"):
        conditioned(torch.ones(2, 3, 16, 16, dtype=torch.long), cond, x_t=x_t)

    with pytest.raises(ValueError, match="shape"):
        conditioned(torch.randn(2, 3, 16), cond, x_t=x_t)

    with pytest.raises(ValueError, match="x0_coarse must have"):
        conditioned(torch.randn(2, 1, 16, 16), cond, x_t=x_t)

    with pytest.raises(ValueError, match="x_t batch size"):
        conditioned(x, cond, x_t=torch.randn(1, 3, 16, 16))

    with pytest.raises(ValueError, match="spatial shape"):
        conditioned(x, cond, x_t=torch.randn(2, 3, 8, 16))

    with pytest.raises(ValueError, match="input channels"):
        conditioned(x, cond, x_t=torch.randn(2, 1, 16, 16))

    with pytest.raises(ValueError, match="cond is required"):
        conditioned(x, x_t=x_t)

    with pytest.raises(ValueError, match="cond batch size"):
        conditioned(x, torch.randn(3, 16), x_t=x_t)

    with pytest.raises(ValueError, match="cond last dimension"):
        conditioned(x, torch.randn(2, 8), x_t=x_t)

    with pytest.raises(ValueError, match="finite"):
        conditioned(x, torch.full((2, 16), float("nan")), x_t=x_t)

    assert conditioned(x, torch.randn(1, 16), x_t=x_t).shape == x.shape

    unconditioned = ResidualConvRefiner(channels=8, blocks=0, in_channels=3, out_channels=3)
    with pytest.raises(ValueError, match="cond_dim"):
        unconditioned(x, cond=torch.randn(2, 16))
    assert unconditioned(x).shape == x.shape


def test_timestep_embedder_matches_module_dtype():
    embedder = TimestepEmbedder(16).to(dtype=torch.float16)

    output = embedder(torch.rand(2))

    assert output.dtype == torch.float16


def test_timestep_embedder_validates_constructor_and_inputs():
    with pytest.raises(ValueError, match="hidden_dim"):
        TimestepEmbedder(0)

    with pytest.raises(TypeError, match="integer"):
        TimestepEmbedder(True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="frequency_embedding_size"):
        TimestepEmbedder(16, frequency_embedding_size=0)

    embedder = TimestepEmbedder(16)

    with pytest.raises(TypeError, match="floating point"):
        embedder(torch.tensor([1, 2]))

    with pytest.raises(ValueError, match="at least one timestep"):
        embedder(torch.empty(0))

    with pytest.raises(ValueError, match="finite"):
        embedder(torch.tensor([float("nan")]))


def test_sinusoidal_embedding_validates_arguments_and_supports_odd_dim():
    embedding = TimestepEmbedder.sinusoidal_embedding(torch.tensor([0.5]), dim=5)

    assert embedding.shape == (1, 5)
    assert embedding.dtype == torch.float32
    assert torch.isfinite(embedding).all()

    with pytest.raises(ValueError, match="dim"):
        TimestepEmbedder.sinusoidal_embedding(torch.tensor([0.5]), dim=0)

    with pytest.raises(ValueError, match="max_period"):
        TimestepEmbedder.sinusoidal_embedding(torch.tensor([0.5]), dim=4, max_period=1)


def test_pixnerd_head_rejects_invalid_constructor_arguments():
    with pytest.raises(ValueError, match="patch_size"):
        PixNerdHead(hidden_dim=32, patch_size=0)

    with pytest.raises(ValueError, match="pixnerd_layers"):
        PixNerdHead(hidden_dim=32, pixnerd_layers=0)

    with pytest.raises(TypeError, match="integer"):
        PixNerdHead(hidden_dim=True)  # type: ignore[arg-type]


def test_pixnerd_head_rejects_invalid_forward_inputs():
    head = PixNerdHead(hidden_dim=32, patch_size=8, pixnerd_hidden_dim=32, pixnerd_layers=1)
    tokens = torch.randn(2, 4, 32)
    t_embed = torch.randn(2, 32)

    with pytest.raises(TypeError, match="floating point"):
        head(torch.ones(2, 4, 32, dtype=torch.long), t_embed, grid_size=(2, 2))

    with pytest.raises(ValueError, match="shape"):
        head(torch.randn(4, 32), t_embed, grid_size=(2, 2))

    with pytest.raises(ValueError, match="at least one token"):
        head(torch.empty(2, 0, 32), t_embed, grid_size=(1, 1))

    with pytest.raises(ValueError, match="last dimension"):
        head(torch.randn(2, 4, 16), t_embed, grid_size=(2, 2))

    with pytest.raises(ValueError, match="t_embed batch size"):
        head(tokens, torch.randn(3, 32), grid_size=(2, 2))

    with pytest.raises(ValueError, match="t_embed last dimension"):
        head(tokens, torch.randn(2, 16), grid_size=(2, 2))

    with pytest.raises(TypeError, match="grid_size"):
        head(tokens, t_embed, grid_size=[2, 2])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="expects 6 tokens"):
        head(tokens, t_embed, grid_size=(2, 3))

    assert head(tokens, torch.randn(1, 32), grid_size=(2, 2)).shape == (2, 3, 16, 16)

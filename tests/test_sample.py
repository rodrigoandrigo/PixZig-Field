import pytest
import torch
from argparse import Namespace

from pixzig_field.config import PixZigFieldConfig, PixZigSampleConfig
from pixzig_field import sample as sample_module
from pixzig_field.sample import generate_sample_image
from pixzig_field.sample import sample_euler


class OracleX0Model(torch.nn.Module):
    def __init__(self, x0: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("x0", x0)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.x0.expand_as(x_t)


class CondX0Model(torch.nn.Module):
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = 0.0 if cond is None else cond.reshape(cond.shape[0], -1).mean(dim=1)
        return value.view(-1, 1, 1, 1).expand_as(x_t)


class ZeroX0Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.zeros_like(x_t) + self.weight


class FailingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise RuntimeError("sample failure")


class DtypeRecordingModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((), dtype=dtype))
        self.seen_x_dtype: torch.dtype | None = None
        self.seen_cond_dtype: torch.dtype | None = None

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.seen_x_dtype = x_t.dtype
        self.seen_cond_dtype = None if cond is None else cond.dtype
        return torch.zeros_like(x_t) + self.weight


def test_sample_euler_reaches_oracle_x0():
    x0 = torch.randn(1, 3, 8, 8)
    noise = torch.randn_like(x0)
    model = OracleX0Model(x0)

    sample = sample_euler(model, noise, n_steps=4)

    assert torch.allclose(sample, x0, atol=1e-5)


def test_sample_heun_reaches_oracle_x0():
    x0 = torch.randn(1, 3, 8, 8)
    noise = torch.randn_like(x0)
    model = OracleX0Model(x0)

    sample = sample_euler(model, noise, n_steps=4, method="heun")

    assert torch.allclose(sample, x0, atol=1e-5)


def test_sample_heun_accepts_cfg_conditioning():
    noise = torch.zeros(1, 3, 8, 8)
    model = CondX0Model()

    sample = sample_euler(
        model,
        noise,
        n_steps=2,
        cond=torch.ones(1, 4),
        negative_cond=torch.zeros(1, 4),
        cfg_scale=2.0,
        method="heun",
    )

    assert sample.shape == noise.shape
    assert torch.isfinite(sample).all()


def test_sample_euler_calls_preview_callback():
    x0 = torch.zeros(1, 3, 8, 8)
    noise = torch.randn_like(x0)
    model = OracleX0Model(x0)
    seen: list[tuple[int, int, torch.Size]] = []

    sample_euler(
        model,
        noise,
        n_steps=3,
        preview_callback=lambda step, total, preview: seen.append((step, total, preview.shape)),
    )

    assert seen == [
        (1, 3, x0.shape),
        (2, 3, x0.shape),
        (3, 3, x0.shape),
    ]


def test_sample_euler_rejects_empty_step_count():
    model = OracleX0Model(torch.zeros(1, 3, 8, 8))
    noise = torch.randn(1, 3, 8, 8)

    with pytest.raises(ValueError, match="n_steps"):
        sample_euler(model, noise, n_steps=0)


def test_sample_euler_rejects_invalid_inputs():
    model = OracleX0Model(torch.zeros(1, 3, 8, 8))
    noise = torch.randn(1, 3, 8, 8)

    with pytest.raises(TypeError, match="n_steps"):
        sample_euler(model, noise, n_steps=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="method"):
        sample_euler(model, noise, n_steps=1, method="bad")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="floating point"):
        sample_euler(model, torch.ones(1, 3, 8, 8, dtype=torch.long), n_steps=1)

    with pytest.raises(ValueError, match="shape"):
        sample_euler(model, torch.randn(3, 8, 8), n_steps=1)

    with pytest.raises(ValueError, match="finite"):
        sample_euler(model, torch.full((1, 3, 8, 8), float("inf")), n_steps=1)

    with pytest.raises(ValueError, match="cfg_scale"):
        sample_euler(model, noise, n_steps=1, cfg_scale=-1.0)

    with pytest.raises(ValueError, match="both be provided"):
        sample_euler(model, noise, n_steps=1, cond=torch.ones(1, 4), cfg_scale=2.0)

    with pytest.raises(ValueError, match="matching non-batch"):
        sample_euler(
            model,
            noise,
            n_steps=1,
            cond=torch.ones(1, 4),
            negative_cond=torch.ones(1, 5),
            cfg_scale=2.0,
        )


def test_to_image_supports_expected_channels_and_rejects_invalid_shape():
    assert sample_module._to_image(torch.zeros(1, 4, 4)).mode == "L"
    assert sample_module._to_image(torch.zeros(3, 4, 4)).mode == "RGB"
    assert sample_module._to_image(torch.zeros(4, 4, 4)).mode == "RGBA"

    with pytest.raises(ValueError, match="1, 3, or 4 channels"):
        sample_module._to_image(torch.zeros(2, 4, 4))

    with pytest.raises(ValueError, match="shape"):
        sample_module._to_image(torch.zeros(1, 3, 4, 4))


def test_generate_sample_image_writes_file(tmp_path):
    model = ZeroX0Model()
    config = PixZigFieldConfig(image_size=8, patch_size=8, hidden_dim=32, depth=1, pixnerd_hidden_dim=16)
    sample_config = PixZigSampleConfig(n_steps=1, method="euler", seed=7)

    output = generate_sample_image(model, config, tmp_path / "sample.png", sample_config=sample_config)

    assert output.exists()


def test_generate_sample_image_restores_training_mode_on_failure(tmp_path):
    model = FailingModel()
    model.train()
    config = PixZigFieldConfig(image_size=8, patch_size=8, hidden_dim=32, depth=1, pixnerd_hidden_dim=16)
    sample_config = PixZigSampleConfig(n_steps=1, method="euler", seed=7)

    with pytest.raises(RuntimeError, match="sample failure"):
        generate_sample_image(model, config, tmp_path / "sample.png", sample_config=sample_config)

    assert model.training


def test_generate_sample_image_uses_model_dtype_and_casts_conditioning(tmp_path):
    model = DtypeRecordingModel(dtype=torch.float64)
    config = PixZigFieldConfig(image_size=8, patch_size=8, hidden_dim=32, depth=1, pixnerd_hidden_dim=16)
    sample_config = PixZigSampleConfig(n_steps=1, method="euler", seed=7)

    output = generate_sample_image(
        model,
        config,
        tmp_path / "sample.png",
        sample_config=sample_config,
        cond=torch.ones(1, 4, dtype=torch.float32),
        negative_cond=torch.zeros(1, 4, dtype=torch.float32),
    )

    assert output.exists()
    assert model.seen_x_dtype == torch.float64
    assert model.seen_cond_dtype == torch.float64


def test_generate_sample_image_rejects_device_mismatch(tmp_path):
    model = ZeroX0Model()
    config = PixZigFieldConfig(image_size=8, patch_size=8, hidden_dim=32, depth=1, pixnerd_hidden_dim=16)

    with pytest.raises(ValueError, match="sampling device"):
        generate_sample_image(model, config, tmp_path / "sample.png", device="meta")


def test_cuda_device_equivalence_accepts_implicit_current_index(monkeypatch):
    monkeypatch.setattr(sample_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sample_module.torch.cuda, "current_device", lambda: 0)

    assert sample_module._devices_equivalent(torch.device("cuda:0"), torch.device("cuda"))
    assert not sample_module._devices_equivalent(torch.device("cuda:1"), torch.device("cuda"))


def test_encode_prompt_uses_checkpoint_cond_dim_override(monkeypatch):
    class DummyEncoder:
        def __init__(self, *args, output_dim: int, **kwargs) -> None:
            self.output_dim = output_dim

        def to(self, device: torch.device) -> "DummyEncoder":
            return self

        def eval(self) -> None:
            return None

        def __call__(self, text: list[str]) -> torch.Tensor:
            return torch.zeros(len(text), self.output_dim)

    monkeypatch.setattr(sample_module, "QwenTextEncoder", DummyEncoder)
    args = Namespace(
        text_encoder="qwen",
        prompt="caption",
        negative_prompt="",
        qwen_model_name="dummy",
        cond_dim=1024,
        allow_qwen_download=False,
        qwen_max_length=None,
        text_encoder_device="cpu",
    )

    cond, negative_cond = sample_module._encode_prompt(args, torch.device("cpu"), cond_dim=7)

    assert cond is not None
    assert negative_cond is not None
    assert cond.shape == (1, 7)
    assert negative_cond.shape == (1, 7)

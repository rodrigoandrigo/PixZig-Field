import torch
import pytest

import nodes as comfy_nodes_module
from nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    PixZigEncodePrompt,
    PixZigModelHandle,
    PixZigModelLoader,
    PixZigQwenLoader,
    PixZigSampler,
    _device,
    _dtype,
    _load_model_config,
    _module_dtype,
    _qwen_choices,
    _qwen_dtype,
    _qwen_force_torch_fallback,
    _resolve_qwen_model_path,
    _resolve_named_path,
    _preview_tuple,
    _to_comfy_image,
)
from pixzig_field.config import PixZigFieldConfig, PixZigSampleConfig, save_config_json
from pixzig_field.model import PixZigField


class FailingSampleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise RuntimeError("sample failed")


def test_comfyui_node_mappings_are_registered():
    assert set(NODE_CLASS_MAPPINGS) == {
        "PixZigModelLoader",
        "PixZigQwenLoader",
        "PixZigEncodePrompt",
        "PixZigSampler",
    }
    assert set(NODE_DISPLAY_NAME_MAPPINGS) == set(NODE_CLASS_MAPPINGS)


def test_comfyui_loaders_expose_precision_controls():
    model_inputs = PixZigModelLoader.INPUT_TYPES()["required"]
    qwen_inputs = PixZigQwenLoader.INPUT_TYPES()["required"]

    assert "precision" in model_inputs
    assert "precision" in qwen_inputs
    assert "qwen_model_name" in qwen_inputs
    assert "qwen_model_path" in PixZigQwenLoader.INPUT_TYPES()["optional"]
    assert _dtype("auto", torch.device("cpu")) == torch.float32
    assert _dtype("auto", torch.device("cpu"), cpu_auto=torch.float16) == torch.float16
    assert _dtype("fp16", torch.device("cpu")) == torch.float16
    assert _module_dtype(torch.nn.Linear(1, 1).to(dtype=torch.float16)) == torch.float16


def test_comfyui_helpers_validate_device_dtype_and_images():
    assert _device("auto").type in {"cpu", "cuda"}
    with pytest.raises(ValueError, match="device"):
        _device("")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            _device("cuda")

    with pytest.raises(TypeError, match="precision"):
        _dtype(123, torch.device("cpu"))  # type: ignore[arg-type]

    image = _to_comfy_image(torch.zeros(1, 3, 4, 4))
    assert image.shape == (1, 4, 4, 3)
    preview_format, preview_image, max_size = _preview_tuple(torch.zeros(1, 3, 4, 4))
    assert preview_format == "JPEG"
    assert preview_image.size == (4, 4)
    assert max_size == 512

    with pytest.raises(ValueError, match="1, 3, or 4 channels"):
        _to_comfy_image(torch.zeros(1, 2, 4, 4))

    with pytest.raises(ValueError, match="shape"):
        _to_comfy_image(torch.zeros(3, 4, 4))


def test_comfyui_qwen_uses_torch_fallback_and_avoids_bf16_on_amd(monkeypatch):
    monkeypatch.setattr("nodes._is_amd_backend", lambda: True)

    assert _qwen_force_torch_fallback(torch.device("cuda"))
    assert _qwen_dtype("auto", torch.device("cuda")) == torch.float16
    assert _qwen_dtype("bf16", torch.device("cuda")) == torch.float16


def test_comfyui_qwen_default_model_is_selectable_by_name():
    choices = _qwen_choices()

    assert choices[0] == "Qwen3.5-2B-Base.safetensors"
    assert _resolve_qwen_model_path("Qwen3.5-2B-Base.safetensors").is_dir()
    assert _resolve_named_path("Qwen3.5-2B-Base", ("pixzig_text_encoders", "text_encoders", "clip")).is_dir()


def test_comfyui_sampler_returns_image_tensor():
    config = PixZigFieldConfig(
        image_size=16,
        hidden_dim=32,
        depth=1,
        pixnerd_hidden_dim=16,
        pixnerd_layers=1,
        refiner_channels=8,
        refiner_blocks=1,
        scan_mode="zigzagN1",
    )
    model = PixZigField(config)
    handle = PixZigModelHandle(model=model, config=config, device=torch.device("cpu"))

    image = PixZigSampler().sample(handle, seed=1, steps=1, method="euler", cfg_scale=1.0)[0]

    assert image.shape == (1, 16, 16, 3)
    assert image.dtype == torch.float32
    assert torch.all((image >= 0) & (image <= 1))


def test_comfyui_sampler_updates_progress_preview(monkeypatch):
    class FakeProgressBar:
        instances: list["FakeProgressBar"] = []

        def __init__(self, total: int) -> None:
            self.total = total
            self.updates: list[tuple[int, int, object]] = []
            FakeProgressBar.instances.append(self)

        def update_absolute(self, value: int, total: int | None = None, preview=None) -> None:
            self.updates.append((value, self.total if total is None else total, preview))

    monkeypatch.setattr(comfy_nodes_module, "comfy_utils", type("FakeComfyUtils", (), {"ProgressBar": FakeProgressBar}))
    config = PixZigFieldConfig(
        image_size=16,
        hidden_dim=32,
        depth=1,
        pixnerd_hidden_dim=16,
        pixnerd_layers=1,
        refiner_channels=8,
        refiner_blocks=1,
        scan_mode="zigzagN1",
    )
    model = PixZigField(config)
    handle = PixZigModelHandle(model=model, config=config, device=torch.device("cpu"))

    PixZigSampler().sample(handle, seed=1, steps=2, method="euler", cfg_scale=1.0, preview_every=1)

    assert len(FakeProgressBar.instances) == 1
    assert [update[0] for update in FakeProgressBar.instances[0].updates] == [1, 2]
    assert all(update[2] is not None for update in FakeProgressBar.instances[0].updates)


def test_comfyui_sampler_restores_training_mode_on_failure():
    config = PixZigFieldConfig(image_size=8, patch_size=8, hidden_dim=32, depth=1, pixnerd_hidden_dim=16)
    model = FailingSampleModel()
    model.train()
    handle = PixZigModelHandle(model=model, config=config, device=torch.device("cpu"))  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="sample failed"):
        PixZigSampler().sample(handle, seed=1, steps=1, method="euler", cfg_scale=1.0)

    assert model.training


def test_comfyui_encode_prompt_rejects_invalid_handle():
    with pytest.raises(TypeError, match="PixZigTextEncoderHandle"):
        PixZigEncodePrompt().encode(object(), "prompt", "")  # type: ignore[arg-type]


def test_comfyui_config_path_accepts_run_config(tmp_path):
    config_path = tmp_path / "run_config.json"
    save_config_json(
        config_path,
        {
            "model": PixZigFieldConfig(image_size=16, hidden_dim=32, depth=1, pixnerd_hidden_dim=16),
            "sample": PixZigSampleConfig(n_steps=1),
        },
    )

    config = _load_model_config("unused.pt", str(config_path))

    assert config.image_size == 16
    assert config.hidden_dim == 32


def test_comfyui_config_path_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="config not found"):
        _load_model_config("unused.pt", str(tmp_path / "missing.json"))

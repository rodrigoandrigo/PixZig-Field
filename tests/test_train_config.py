import subprocess
import sys

import pytest
import torch
from PIL import Image

from pixzig_field import train as train_module
from pixzig_field.config import (
    PixZigCheckpointConfig,
    PixZigDatasetConfig,
    PixZigEMAConfig,
    PixZigFieldConfig,
    PixZigFlowConfig,
    PixZigLossConfig,
    PixZigMinSNRConfig,
    PixZigSampleConfig,
    PixZigTextEncoderConfig,
    PixZigTrainingConfig,
    save_config_json,
)


def test_train_json_config_controls_loss_min_snr_and_disabled_ema(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    Image.new("RGB", (16, 16), (127, 127, 127)).save(data_dir / "sample.png")
    (data_dir / "sample.txt").write_text("caption", encoding="utf-8")

    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "run"
    save_config_json(
        config_path,
        {
            "model": PixZigFieldConfig(
                image_size=16,
                hidden_dim=32,
                depth=1,
                pixnerd_hidden_dim=16,
                pixnerd_layers=1,
                refiner_channels=8,
                refiner_blocks=1,
            ),
            "dataset": PixZigDatasetConfig(image_size=16),
            "flow": PixZigFlowConfig(),
            "loss": PixZigLossConfig(lpips_enabled=False, lpips_weight=0.0, multiscale_weight=0.0, frequency_weight=0.0),
            "min_snr": PixZigMinSNRConfig(enabled=False),
            "ema": PixZigEMAConfig(enabled=False),
            "checkpoint": PixZigCheckpointConfig(
                periodic_enabled=True,
                save_every_steps=1,
                save_safetensors=True,
                safetensors_name="standard-final.safetensors",
                safetensors_source="model",
            ),
            "sample": PixZigSampleConfig(n_steps=1),
            "text_encoder": PixZigTextEncoderConfig(),
            "training": PixZigTrainingConfig(
                image_size=16,
                max_steps=1,
                gradient_accumulation_steps=2,
                precision="fp32",
                autocast=False,
            ),
        },
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pixzig_field.train",
            "--config",
            str(config_path),
            "--data",
            str(data_dir),
            "--output",
            str(output_dir),
        ],
        check=True,
    )

    checkpoint = torch.load(output_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    run_config = torch.load(output_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["configs"]

    assert checkpoint["ema"] is None
    assert checkpoint["step"] == 1
    assert checkpoint["optimizer"]["state"]
    assert run_config["loss"]["_type"] == "PixZigLossConfig"
    assert run_config["min_snr"]["enabled"] is False
    assert (output_dir / "checkpoints" / "step_00000001.pt").exists()
    assert (output_dir / "checkpoints" / "step_00000001.safetensors").exists()
    assert (output_dir / "checkpoints" / "standard-final.safetensors").exists()
    assert (output_dir / "samples" / "final.png").exists()


def test_train_config_section_rejects_malformed_json_section():
    with pytest.raises(TypeError, match="loss"):
        train_module._config_section({"loss": "bad"}, "loss", PixZigLossConfig)


def test_train_checkpoint_safetensors_requires_enabled_ema_for_ema_source():
    checkpoint_config = PixZigCheckpointConfig(save_safetensors=True, safetensors_source="ema")
    ema_config = PixZigEMAConfig(enabled=False)

    with pytest.raises(ValueError, match="EMA"):
        train_module._validate_checkpoint_ema_compatibility(checkpoint_config, ema_config)


def test_train_seed_everything_validates_seed_and_allows_large_seed():
    train_module.seed_everything(2**40)
    assert torch.initial_seed() == 2**40

    with pytest.raises(TypeError, match="integer"):
        train_module.seed_everything(True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-negative"):
        train_module.seed_everything(-1)


def test_train_autocast_and_aux_device_validation():
    with pytest.raises(ValueError, match="precision"):
        train_module._autocast_context(torch.device("cpu"), "bad", True)

    with pytest.raises(TypeError, match="enabled"):
        train_module._autocast_context(torch.device("cpu"), "fp32", "yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="device"):
        train_module._resolve_aux_device("", torch.device("cpu"))

    assert train_module._resolve_aux_device("model", torch.device("cpu")) == torch.device("cpu")


def test_train_progress_formatter_includes_realtime_stats():
    line = train_module._format_training_progress(
        step=5,
        max_steps=10,
        epoch=2,
        loss=1.23456,
        flow_loss=0.5,
        x0_loss=0.25,
        lr=1e-4,
        grad_norm=0.75,
        sec_per_step=0.5,
        images_per_sec=4.0,
        elapsed=12,
        eta=8,
    )

    assert "stage=train" in line
    assert "50.00%" in line
    assert "step=5/10" in line
    assert "epoch=2" in line
    assert "loss=1.23456" in line
    assert "4.00img/s" in line
    assert "eta=8s" in line


def test_training_sample_uses_ema_when_cuda_device_index_is_implicit(monkeypatch, tmp_path):
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(train_module, "_module_device", lambda module: torch.device("cuda:0"))

    live_model = torch.nn.Linear(1, 1)
    ema_model = torch.nn.Linear(1, 1)
    ema = type("DummyEMA", (), {"module": ema_model})()
    chosen: dict[str, torch.nn.Module] = {}

    def fake_generate_sample_image(model, model_config, output, **kwargs):
        chosen["model"] = model
        return output

    monkeypatch.setattr(train_module, "generate_sample_image", fake_generate_sample_image)

    train_module._save_training_sample(
        model=live_model,  # type: ignore[arg-type]
        ema=ema,  # type: ignore[arg-type]
        model_config=PixZigFieldConfig(image_size=16, hidden_dim=32, depth=1, pixnerd_hidden_dim=16),
        sample_config=PixZigSampleConfig(use_ema=True),
        text_encoder=None,
        output=tmp_path / "sample.png",
        device=torch.device("cuda"),
    )

    assert chosen["model"] is ema_model


def test_train_resume_cli_restores_step_optimizer_scheduler_and_ema(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    Image.new("RGB", (16, 16), (127, 127, 127)).save(data_dir / "sample.png")
    (data_dir / "sample.txt").write_text("caption", encoding="utf-8")
    output_dir = tmp_path / "run"

    base_args = [
        sys.executable,
        "-m",
        "pixzig_field.train",
        "--data",
        str(data_dir),
        "--output",
        str(output_dir),
        "--device",
        "cpu",
        "--image-size",
        "16",
        "--patch-size",
        "16",
        "--batch-size",
        "1",
        "--num-workers",
        "0",
        "--gradient-accumulation-steps",
        "1",
        "--precision",
        "fp32",
        "--text-encoder",
        "none",
        "--hidden-dim",
        "32",
        "--depth",
        "1",
        "--zigma-state-dim",
        "8",
        "--zigma-expand",
        "1",
        "--scan-mode",
        "zigzagN1",
        "--mamba-backend",
        "native",
        "--pixnerd-hidden-dim",
        "16",
        "--pixnerd-layers",
        "1",
        "--refiner-channels",
        "8",
        "--refiner-blocks",
        "1",
        "--learning-rate",
        "1e-4",
        "--lr-warmup-steps",
        "0",
        "--sample-n-steps",
        "1",
        "--sample-method",
        "euler",
        "--seed",
        "123",
        "--sample-seed",
        "123",
    ]

    subprocess.run([*base_args, "--max-steps", "1"], check=True)
    first = torch.load(output_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    assert first["step"] == 1
    assert first["scheduler"]["last_epoch"] == 1
    assert first["ema"]["num_updates"] == 1

    subprocess.run(
        [
            *base_args,
            "--max-steps",
            "2",
            "--resume",
            str(output_dir / "checkpoints" / "last.pt"),
        ],
        check=True,
    )
    resumed = torch.load(output_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)

    assert resumed["step"] == 2
    assert resumed["optimizer"] is not None
    assert resumed["scheduler"]["last_epoch"] == 2
    assert resumed["ema"]["num_updates"] == 2

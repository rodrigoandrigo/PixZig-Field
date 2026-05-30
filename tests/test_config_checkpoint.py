import json

import torch

import pytest

from pixzig_field.checkpoint import (
    load_checkpoint_config,
    load_model_weights,
    prune_checkpoints,
    save_checkpoint,
    save_model_safetensors,
)
from pixzig_field.config import (
    PixZigCheckpointConfig,
    PixZigDatasetConfig,
    PixZigEMAConfig,
    PixZigFieldConfig,
    PixZigFlowConfig,
    PixZigLossConfig,
    PixZigSampleConfig,
    PixZigTrainingConfig,
    config_from_json_dict,
    load_field_config_json,
    load_config_json,
    save_config_json,
)


def test_config_roundtrip_uses_json(tmp_path):
    config = PixZigTrainingConfig(max_steps=12, betas=(0.8, 0.9))
    path = tmp_path / "training.json"

    save_config_json(path, config)
    loaded = load_config_json(path, PixZigTrainingConfig)

    assert isinstance(loaded, PixZigTrainingConfig)
    assert loaded.max_steps == 12
    assert loaded.betas == (0.8, 0.9)
    assert json.loads(path.read_text(encoding="utf-8"))["_type"] == "PixZigTrainingConfig"


def test_configs_reject_invalid_values():
    with pytest.raises(ValueError, match="patch_size"):
        PixZigFieldConfig(patch_size=0)
    with pytest.raises(ValueError, match="divisible"):
        PixZigFieldConfig(image_size=18, patch_size=16)
    with pytest.raises(ValueError, match="caption_extension"):
        PixZigDatasetConfig(caption_extension=".")
    with pytest.raises(ValueError, match="timestep_eps"):
        PixZigFlowConfig(timestep_eps=0.5)
    with pytest.raises(ValueError, match="learning_rate"):
        PixZigTrainingConfig(learning_rate=float("nan"))
    with pytest.raises(ValueError, match="lpips_every_n_steps"):
        PixZigLossConfig(lpips_every_n_steps=0)
    with pytest.raises(ValueError, match="decay"):
        PixZigEMAConfig(decay=1.1)
    with pytest.raises(ValueError, match="output_dir"):
        PixZigCheckpointConfig(output_dir="")
    with pytest.raises(ValueError, match="precision"):
        PixZigTrainingConfig(precision="tf32")


def test_config_from_json_rejects_unknown_fields_and_normalizes_betas():
    loaded = config_from_json_dict(
        {"_type": "PixZigTrainingConfig", "betas": [0.8, 0.9]},
        PixZigTrainingConfig,
    )

    assert isinstance(loaded, PixZigTrainingConfig)
    assert loaded.betas == (0.8, 0.9)

    with pytest.raises(ValueError, match="unknown PixZigTrainingConfig"):
        config_from_json_dict(
            {"_type": "PixZigTrainingConfig", "max_step": 12},
            PixZigTrainingConfig,
        )
    with pytest.raises(ValueError, match="unknown config type"):
        config_from_json_dict({"_type": "MissingConfig"})
    with pytest.raises(ValueError, match="unknown config type"):
        config_from_json_dict({}, "MissingConfig")


def test_checkpoint_stores_json_compatible_config_and_sidecar(tmp_path):
    model = torch.nn.Linear(2, 2)
    model_config = PixZigFieldConfig(image_size=16, hidden_dim=32, depth=1, pixnerd_hidden_dim=16)
    run_configs = {"model": model_config, "sample": PixZigSampleConfig(n_steps=2)}
    path = tmp_path / "last.pt"

    save_checkpoint(path, model=model, config=model_config, configs=run_configs, step=3, epoch=1)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    loaded_config = load_checkpoint_config(path)

    assert isinstance(raw["config"], dict)
    assert raw["config"]["_type"] == "PixZigFieldConfig"
    assert isinstance(loaded_config, PixZigFieldConfig)
    assert loaded_config.image_size == 16
    assert (tmp_path / "last.pt.json").exists()


def test_load_checkpoint_config_falls_back_to_run_configs_model(tmp_path):
    path = tmp_path / "last.pt"
    model_config = PixZigFieldConfig(image_size=16, hidden_dim=32, depth=1, pixnerd_hidden_dim=16)
    save_checkpoint(path, model=torch.nn.Linear(2, 2), configs={"model": model_config}, save_json_sidecar=False)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    raw["config"] = None
    torch.save(raw, path)

    loaded_config = load_checkpoint_config(path)

    assert isinstance(loaded_config, PixZigFieldConfig)
    assert loaded_config.image_size == 16


def test_load_field_config_accepts_run_config_model_section(tmp_path):
    path = tmp_path / "run_config.json"
    save_config_json(
        path,
        {
            "model": PixZigFieldConfig(image_size=16, hidden_dim=32, depth=1, pixnerd_hidden_dim=16),
            "sample": PixZigSampleConfig(n_steps=2),
        },
    )

    loaded = load_field_config_json(path)

    assert loaded.image_size == 16
    assert loaded.hidden_dim == 32


def test_load_field_config_rejects_unrelated_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"sample": {"n_steps": 2}}', encoding="utf-8")

    with pytest.raises(TypeError, match="PixZigFieldConfig"):
        load_field_config_json(path)


def test_checkpoint_restores_scheduler_state(tmp_path):
    source_model = torch.nn.Linear(2, 2)
    target_model = torch.nn.Linear(2, 2)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-3)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=1e-3)
    source_scheduler = torch.optim.lr_scheduler.LambdaLR(source_optimizer, lambda step: 0.5**step)
    target_scheduler = torch.optim.lr_scheduler.LambdaLR(target_optimizer, lambda step: 0.5**step)
    source_optimizer.step()
    source_scheduler.step()
    path = tmp_path / "last.pt"

    save_checkpoint(
        path,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        step=1,
    )
    raw = torch.load(path, map_location="cpu", weights_only=False)
    from pixzig_field.checkpoint import load_checkpoint

    load_checkpoint(
        path,
        model=target_model,
        optimizer=target_optimizer,
        scheduler=target_scheduler,
    )

    assert raw["scheduler"] is not None
    assert target_scheduler.state_dict()["last_epoch"] == source_scheduler.state_dict()["last_epoch"]
    assert target_optimizer.param_groups[0]["lr"] == source_optimizer.param_groups[0]["lr"]


def test_save_model_safetensors_writes_tensor_file(tmp_path):
    from safetensors.torch import load_file

    model = torch.nn.Linear(2, 2)
    path = tmp_path / "model.safetensors"

    model_config = PixZigFieldConfig(image_size=16, hidden_dim=32, depth=1, pixnerd_hidden_dim=16)
    save_model_safetensors(path, model=model, source="model", metadata={"step": "1"}, config=model_config, step=1)
    loaded = load_file(str(path))
    loaded_config = load_checkpoint_config(path)

    assert path.exists()
    assert (tmp_path / "model.safetensors.json").exists()
    assert set(loaded) == set(model.state_dict())
    assert isinstance(loaded_config, PixZigFieldConfig)
    assert loaded_config.image_size == 16


def test_save_model_safetensors_writes_sidecar_from_run_configs(tmp_path):
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "model.safetensors"
    model_config = PixZigFieldConfig(image_size=16, hidden_dim=32, depth=1, pixnerd_hidden_dim=16)

    save_model_safetensors(path, model=model, source="model", configs={"model": model_config})
    loaded_config = load_checkpoint_config(path)

    assert (tmp_path / "model.safetensors.json").exists()
    assert isinstance(loaded_config, PixZigFieldConfig)
    assert loaded_config.image_size == 16


def test_save_model_safetensors_rejects_wrong_suffix(tmp_path):
    model = torch.nn.Linear(2, 2)

    with pytest.raises(ValueError, match=".safetensors"):
        save_model_safetensors(tmp_path / "model.pt", model=model, source="model")


def test_save_model_safetensors_rejects_missing_ema_source(tmp_path):
    model = torch.nn.Linear(2, 2)

    with pytest.raises(ValueError, match="ema"):
        save_model_safetensors(tmp_path / "model.safetensors", model=model, source="ema")


def test_load_model_weights_accepts_safetensors(tmp_path):
    source = torch.nn.Linear(2, 2)
    target = torch.nn.Linear(2, 2)
    path = tmp_path / "model.safetensors"

    save_model_safetensors(path, model=source, source="model")
    load_model_weights(path, target)

    for source_param, target_param in zip(source.parameters(), target.parameters()):
        assert torch.allclose(source_param, target_param)


def test_prune_checkpoints_removes_json_sidecars(tmp_path):
    model = torch.nn.Linear(2, 2)
    for step in [1, 2, 3]:
        save_checkpoint(tmp_path / f"step_{step:08d}.pt", model=model, step=step, config=PixZigFieldConfig())
        (tmp_path / f"step_{step:08d}.safetensors").write_text("weights", encoding="utf-8")
        (tmp_path / f"step_{step:08d}.safetensors.json").write_text("{}", encoding="utf-8")

    prune_checkpoints(tmp_path, keep_last=1)

    assert not (tmp_path / "step_00000001.pt").exists()
    assert not (tmp_path / "step_00000001.pt.json").exists()
    assert not (tmp_path / "step_00000001.safetensors").exists()
    assert not (tmp_path / "step_00000001.safetensors.json").exists()
    assert not (tmp_path / "step_00000002.pt").exists()
    assert not (tmp_path / "step_00000002.pt.json").exists()
    assert not (tmp_path / "step_00000002.safetensors").exists()
    assert not (tmp_path / "step_00000002.safetensors.json").exists()
    assert (tmp_path / "step_00000003.pt").exists()
    assert (tmp_path / "step_00000003.pt.json").exists()
    assert (tmp_path / "step_00000003.safetensors").exists()
    assert (tmp_path / "step_00000003.safetensors.json").exists()


def test_prune_checkpoints_keep_zero_removes_all_periodic_checkpoints(tmp_path):
    model = torch.nn.Linear(2, 2)
    for step in [1, 2]:
        save_checkpoint(tmp_path / f"step_{step:08d}.pt", model=model, step=step, config=PixZigFieldConfig())
        (tmp_path / f"step_{step:08d}.safetensors").write_text("weights", encoding="utf-8")
        (tmp_path / f"step_{step:08d}.safetensors.json").write_text("{}", encoding="utf-8")

    prune_checkpoints(tmp_path, keep_last=0)

    assert not list(tmp_path.glob("step_*.pt"))
    assert not list(tmp_path.glob("step_*.pt.json"))
    assert not list(tmp_path.glob("step_*.safetensors"))
    assert not list(tmp_path.glob("step_*.safetensors.json"))

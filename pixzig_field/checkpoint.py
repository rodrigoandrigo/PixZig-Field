from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import config_from_json_dict, config_to_json_dict, save_config_json
from .ema import EMAModel


def _checkpoint_metadata(config: Any, step: int, epoch: int, configs: Any = None) -> dict[str, Any]:
    return {
        "config": config_to_json_dict(config) if config is not None else None,
        "configs": config_to_json_dict(configs) if configs is not None else None,
        "step": int(step),
        "epoch": int(epoch),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    ema: EMAModel | None = None,
    scaler: Any = None,
    config: Any = None,
    configs: Any = None,
    step: int = 0,
    epoch: int = 0,
    save_json_sidecar: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _checkpoint_metadata(config, step, epoch, configs)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            **metadata,
        },
        path,
    )
    if save_json_sidecar:
        save_config_json(path.with_suffix(path.suffix + ".json"), metadata)
    return path


def _safetensors_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone().contiguous()
        for name, tensor in module.state_dict().items()
        if torch.is_tensor(tensor)
    }


def save_model_safetensors(
    path: str | Path,
    *,
    model: nn.Module,
    ema: EMAModel | None = None,
    source: str = "ema",
    metadata: dict[str, str] | None = None,
    config: Any = None,
    configs: Any = None,
    step: int = 0,
    epoch: int = 0,
    save_json_sidecar: bool = True,
) -> Path:
    if source not in {"model", "ema"}:
        raise ValueError("source must be 'model' or 'ema'")
    if source == "ema" and ema is None:
        raise ValueError("source='ema' was requested but ema is not available")
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("safetensors path must end with '.safetensors'")
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError("safetensors is required to save .safetensors model weights") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    module = ema.module if source == "ema" and ema is not None else model
    save_file(
        _safetensors_state_dict(module),
        str(path),
        metadata={**(metadata or {}), "source": source},
    )
    if save_json_sidecar and (config is not None or configs is not None):
        save_config_json(path.with_suffix(path.suffix + ".json"), _checkpoint_metadata(config, step, epoch, configs))
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    ema: EMAModel | None = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if ema is not None and checkpoint.get("ema") is not None:
        ema.load_state_dict(checkpoint["ema"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def load_checkpoint_config(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    path = Path(path)
    if path.suffix == ".safetensors":
        sidecar = path.with_suffix(path.suffix + ".json")
        if not sidecar.exists():
            return None
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        config = metadata.get("config") if isinstance(metadata, dict) else None
        if isinstance(config, dict):
            return config_from_json_dict(config)
        configs = metadata.get("configs") if isinstance(metadata, dict) else None
        if isinstance(configs, dict):
            model_config = configs.get("model")
            if isinstance(model_config, dict):
                return config_from_json_dict(model_config)
        return None
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = checkpoint.get("config")
    if isinstance(config, dict):
        return config_from_json_dict(config)
    configs = checkpoint.get("configs")
    if isinstance(configs, dict):
        model_config = configs.get("model")
        if isinstance(model_config, dict):
            return config_from_json_dict(model_config)
    return config


def load_model_weights(
    path: str | Path,
    model: nn.Module,
    *,
    use_ema: bool = False,
    map_location: str | torch.device = "cpu",
) -> None:
    if Path(path).suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required to load .safetensors model weights") from exc
        state_dict = load_file(str(path), device=str(map_location))
        model.load_state_dict(state_dict)
        return
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if use_ema and checkpoint.get("ema") is not None:
        state_dict = checkpoint["ema"]["model"]
    else:
        state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)


def prune_checkpoints(output_dir: str | Path, keep_last: int = 3) -> None:
    paths = sorted(Path(output_dir).glob("step_*.pt"))
    stale_paths = paths if keep_last <= 0 else paths[:-keep_last]
    for path in stale_paths:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".json").unlink(missing_ok=True)
        safetensors_path = path.with_suffix(".safetensors")
        safetensors_path.unlink(missing_ok=True)
        safetensors_path.with_suffix(safetensors_path.suffix + ".json").unlink(missing_ok=True)

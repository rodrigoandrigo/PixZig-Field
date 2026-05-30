from __future__ import annotations

import json
import math
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, cast


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


@dataclass
class PixZigFieldConfig:
    image_size: int = 512
    patch_size: int = 16
    in_channels: int = 3
    out_channels: int = 3
    prediction_type: str = "x0"
    hidden_dim: int = 3072
    depth: int = 28
    zigma_state_dim: int = 128
    zigma_expand: int = 2
    scan_mode: str = "zigzagN16"
    mamba_backend: str = "native"
    mamba_inner_expand: int = 1
    mamba_head_dim: int = 64
    mamba_num_groups: int = 1
    mamba_conv_kernel: int = 4
    fourier_frequencies: int = 8
    pixnerd_hidden_dim: int = 768
    pixnerd_layers: int = 4
    refiner_channels: int = 96
    refiner_blocks: int = 3
    refiner_conditioning: bool = True
    refiner_use_xt: bool = True
    cond_dim: int | None = None
    qwen_hidden_dim: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "image_size",
            "patch_size",
            "in_channels",
            "out_channels",
            "hidden_dim",
            "depth",
            "zigma_state_dim",
            "zigma_expand",
            "mamba_inner_expand",
            "mamba_head_dim",
            "mamba_num_groups",
            "mamba_conv_kernel",
            "fourier_frequencies",
            "pixnerd_hidden_dim",
            "pixnerd_layers",
            "refiner_channels",
            "refiner_blocks",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("refiner_conditioning", "refiner_use_xt"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.mamba_backend not in {"native", "external", "auto"}:
            raise ValueError("mamba_backend must be 'native', 'external', or 'auto'")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.prediction_type != "x0":
            raise ValueError("PixZigField currently supports only prediction_type='x0'")
        if self.cond_dim is not None and self.cond_dim < 1:
            raise ValueError("cond_dim must be positive when provided")
        if self.qwen_hidden_dim is not None and self.qwen_hidden_dim < 1:
            raise ValueError("qwen_hidden_dim must be positive when provided")


@dataclass
class PixZigDatasetConfig:
    image_size: int = 512
    captions_file: str | None = None
    caption_extension: str = ".txt"
    require_captions: bool = True
    recursive: bool = True
    crop_mode: str = "center"
    horizontal_flip_p: float = 0.0
    return_metadata: bool = True

    def __post_init__(self) -> None:
        if self.image_size < 1:
            raise ValueError("image_size must be positive")
        if not self.caption_extension.startswith("."):
            self.caption_extension = f".{self.caption_extension}"
        if self.caption_extension == ".":
            raise ValueError("caption_extension must include a suffix after '.'")
        for name in ("require_captions", "recursive", "return_metadata"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.crop_mode not in {"center", "random", "pad"}:
            raise ValueError("crop_mode must be 'center', 'random', or 'pad'")
        _require_finite("horizontal_flip_p", self.horizontal_flip_p)
        if not 0.0 <= self.horizontal_flip_p <= 1.0:
            raise ValueError("horizontal_flip_p must be in [0, 1]")


@dataclass
class PixZigLossConfig:
    flow_weight: float = 1.0
    x0_weight: float = 0.05
    lpips_weight: float = 0.02
    multiscale_weight: float = 0.02
    frequency_weight: float = 0.01
    boundary_weight: float = 0.01
    lpips_enabled: bool = True
    perceptual_backend: str = "proxy"
    lpips_every_n_steps: int = 1
    lpips_crop_size: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "flow_weight",
            "x0_weight",
            "lpips_weight",
            "multiscale_weight",
            "frequency_weight",
            "boundary_weight",
        ):
            value = getattr(self, name)
            _require_finite(name, value)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.lpips_enabled, bool):
            raise TypeError("lpips_enabled must be bool")
        if self.perceptual_backend not in {"proxy", "lpips", "auto"}:
            raise ValueError("perceptual_backend must be 'proxy', 'lpips', or 'auto'")
        if self.lpips_every_n_steps < 1:
            raise ValueError("lpips_every_n_steps must be at least 1")
        if self.lpips_crop_size is not None and self.lpips_crop_size < 1:
            raise ValueError("lpips_crop_size must be positive when provided")


@dataclass
class PixZigMinSNRConfig:
    enabled: bool = True
    gamma: float = 5.0
    strategy: str = "min_snr"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")
        _require_finite("gamma", self.gamma)
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.strategy not in {"min_snr", "soft_min_snr"}:
            raise ValueError("strategy must be 'min_snr' or 'soft_min_snr'")


@dataclass
class PixZigFlowConfig:
    prediction_type: str = "x0"
    timestep_sampling: str = "uniform"
    timestep_eps: float = 1e-5
    logit_normal_std: float = 1.0
    beta_alpha: float = 2.0
    beta_beta: float = 2.0

    def __post_init__(self) -> None:
        if self.prediction_type != "x0":
            raise ValueError("PixZigField currently supports only prediction_type='x0'")
        if self.timestep_sampling not in {"uniform", "logit_normal", "cosine", "beta", "stratified"}:
            raise ValueError("timestep_sampling must be 'uniform', 'logit_normal', 'cosine', 'beta', or 'stratified'")
        for name in ("timestep_eps", "logit_normal_std", "beta_alpha", "beta_beta"):
            _require_finite(name, getattr(self, name))
        if not 0.0 <= self.timestep_eps < 0.5:
            raise ValueError("timestep_eps must be in [0, 0.5)")
        if self.logit_normal_std <= 0:
            raise ValueError("logit_normal_std must be positive")
        if self.beta_alpha <= 0 or self.beta_beta <= 0:
            raise ValueError("beta_alpha and beta_beta must be positive")


@dataclass
class PixZigEMAConfig:
    enabled: bool = True
    decay: float = 0.9999
    update_every: int = 1
    warmup_steps: int = 0
    min_decay: float = 0.0
    device: str = "model"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")
        for name in ("decay", "min_decay"):
            _require_finite(name, getattr(self, name))
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if self.update_every < 1:
            raise ValueError("update_every must be at least 1")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 <= self.min_decay <= self.decay:
            raise ValueError("min_decay must be in [0, decay]")
        if not self.device.strip():
            raise ValueError("device must not be empty")


@dataclass
class PixZigCheckpointConfig:
    output_dir: str = "checkpoints"
    periodic_enabled: bool = False
    save_every_steps: int = 1000
    keep_last: int = 3
    final_name: str = "last.pt"
    save_json_sidecar: bool = True
    save_safetensors: bool = False
    safetensors_name: str = "model.safetensors"
    safetensors_source: str = "ema"

    def __post_init__(self) -> None:
        if not self.output_dir.strip():
            raise ValueError("output_dir must not be empty")
        if not self.final_name.strip():
            raise ValueError("final_name must not be empty")
        for name in ("periodic_enabled", "save_json_sidecar", "save_safetensors"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.save_every_steps < 1:
            raise ValueError("save_every_steps must be at least 1")
        if self.keep_last < 0:
            raise ValueError("keep_last must be non-negative")
        if self.safetensors_source not in {"model", "ema"}:
            raise ValueError("safetensors_source must be 'model' or 'ema'")
        if not self.safetensors_name.endswith(".safetensors"):
            raise ValueError("safetensors_name must end with '.safetensors'")


@dataclass
class PixZigSampleConfig:
    periodic_enabled: bool = False
    sample_every_steps: int = 1000
    n_steps: int = 20
    method: str = "heun"
    output_dir: str = "samples"
    final_name: str = "final.png"
    prompt: str = ""
    negative_prompt: str = ""
    cfg_scale: float = 1.0
    seed: int = 1234
    use_ema: bool = True

    def __post_init__(self) -> None:
        if not self.output_dir.strip():
            raise ValueError("output_dir must not be empty")
        if not self.final_name.strip():
            raise ValueError("final_name must not be empty")
        for name in ("periodic_enabled", "use_ema"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.sample_every_steps < 1:
            raise ValueError("sample_every_steps must be at least 1")
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if self.method not in {"euler", "heun"}:
            raise ValueError("method must be 'euler' or 'heun'")
        _require_finite("cfg_scale", self.cfg_scale)
        if self.cfg_scale < 0:
            raise ValueError("cfg_scale must be non-negative")


@dataclass
class PixZigTextEncoderConfig:
    kind: str = "none"
    qwen_model_name: str = "Qwen/Qwen3.5-2B-Base"
    allow_qwen_download: bool = False
    cache_embeddings: bool = False
    pooling: str = "mean"
    cond_dim: int = 1024
    max_length: int | None = None
    device: str = "model"

    def __post_init__(self) -> None:
        for name in ("allow_qwen_download", "cache_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.kind not in {"none", "qwen"}:
            raise ValueError("kind must be 'none' or 'qwen'")
        if self.pooling not in {"none", "mean"}:
            raise ValueError("pooling must be 'none' or 'mean'")
        if self.cond_dim < 1:
            raise ValueError("cond_dim must be positive")
        if self.max_length is not None and self.max_length < 1:
            raise ValueError("max_length must be positive when provided")
        if not self.device.strip():
            raise ValueError("device must not be empty")


@dataclass
class PixZigTrainingConfig:
    image_size: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 100000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    lr_warmup_steps: int = 1000
    min_lr_ratio: float = 0.1
    num_workers: int = 0
    log_every_steps: int = 10
    precision: str = "bf16"
    autocast: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        self.betas = tuple(self.betas)
        if self.image_size < 1:
            raise ValueError("image_size must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        for name in ("learning_rate", "weight_decay", "eps", "max_grad_norm", "min_lr_ratio"):
            _require_finite(name, getattr(self, name))
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if len(self.betas) != 2 or not all(math.isfinite(float(beta)) and 0.0 <= beta < 1.0 for beta in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be non-negative")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.log_every_steps < 1:
            raise ValueError("log_every_steps must be at least 1")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be 'fp32', 'fp16', or 'bf16'")
        if not isinstance(self.autocast, bool):
            raise TypeError("autocast must be bool")


ConfigT = TypeVar("ConfigT")

CONFIG_TYPES: dict[str, type[Any]] = {
    cls.__name__: cls
    for cls in (
        PixZigFieldConfig,
        PixZigDatasetConfig,
        PixZigLossConfig,
        PixZigMinSNRConfig,
        PixZigFlowConfig,
        PixZigEMAConfig,
        PixZigCheckpointConfig,
        PixZigSampleConfig,
        PixZigTextEncoderConfig,
        PixZigTrainingConfig,
    )
}


def _jsonify(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        data = {key: _jsonify(item) for key, item in asdict(value).items()}
        data["_type"] = type(value).__name__
        return data
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    return value


def config_to_json_dict(config: Any) -> dict[str, Any]:
    data = _jsonify(config)
    if not isinstance(data, dict):
        raise TypeError("config must serialize to a JSON object")
    return data


def config_from_json_dict(data: dict[str, Any], config_type: type[ConfigT] | str | None = None) -> ConfigT | dict[str, Any]:
    if not isinstance(data, dict):
        raise TypeError("JSON config data must be an object")
    type_name = data.get("_type")
    cls: type[Any] | None
    if config_type is None:
        cls = CONFIG_TYPES.get(type_name) if isinstance(type_name, str) else None
        if isinstance(type_name, str) and cls is None:
            raise ValueError(f"unknown config type: {type_name}")
    elif isinstance(config_type, str):
        try:
            cls = CONFIG_TYPES[config_type]
        except KeyError as exc:
            raise ValueError(f"unknown config type: {config_type}") from exc
    else:
        cls = cast(type[Any], config_type)
    if cls is None:
        return data
    valid_field_names = {field.name for field in fields(cls)}
    unknown = set(data) - valid_field_names - {"_type"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {cls.__name__} config field(s): {names}")
    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in data:
            continue
        value = data[field.name]
        default = field.default
        if default is not MISSING and isinstance(default, tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[field.name] = value
    return cls(**kwargs)


def save_config_json(path: str | Path, config: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonify(config), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_config_json(path: str | Path, config_type: type[ConfigT] | str | None = None) -> ConfigT | dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if config_type is not None or (isinstance(data, dict) and "_type" in data):
        return config_from_json_dict(data, config_type)
    return data


def load_field_config_json(path: str | Path) -> PixZigFieldConfig:
    data = load_config_json(path)
    if isinstance(data, PixZigFieldConfig):
        return data
    if isinstance(data, dict):
        if data.get("_type") == "PixZigFieldConfig":
            config = config_from_json_dict(data, PixZigFieldConfig)
            if isinstance(config, PixZigFieldConfig):
                return config
        model = data.get("model")
        if isinstance(model, dict):
            config = config_from_json_dict(model, PixZigFieldConfig)
            if isinstance(config, PixZigFieldConfig):
                return config
    raise TypeError("config JSON must be a PixZigFieldConfig or a run config with a model section")

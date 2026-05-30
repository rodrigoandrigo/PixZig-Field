from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal, cast

import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # ComfyUI is available only when this package is loaded as a custom node.
    import comfy.model_management as comfy_model_management
    import comfy.utils as comfy_utils
    import folder_paths
except ImportError:  # pragma: no cover - local unit tests run without ComfyUI.
    comfy_model_management = None
    comfy_utils = None
    folder_paths = None

from pixzig_field.checkpoint import load_checkpoint_config, load_model_weights
from pixzig_field.config import PixZigFieldConfig, PixZigSampleConfig, load_field_config_json
from pixzig_field.model import PixZigField
from pixzig_field.sample import sample_euler
from pixzig_field.text_encoder import QwenTextEncoder


_PIXZIG_ROOT = _ROOT
_PROJECTS_ROOT = _PIXZIG_ROOT.parent
_DEFAULT_QWEN_DIR_NAME = "Qwen3.5-2B-Base"
_DEFAULT_QWEN_DISPLAY_NAME = "Qwen3.5-2B-Base.safetensors"
_DEFAULT_QWEN_CANDIDATES = (
    _PROJECTS_ROOT / _DEFAULT_QWEN_DIR_NAME,
    Path.home() / "Dev1" / "Projetos" / _DEFAULT_QWEN_DIR_NAME,
    Path.home() / "Notebooks" / "ComfyUI" / "models" / "text_encoders" / _DEFAULT_QWEN_DIR_NAME,
    Path.home() / "Notebooks" / "ComfyUI" / "models" / "clip" / _DEFAULT_QWEN_DIR_NAME,
)
_DEFAULT_QWEN_PATH = next((path for path in _DEFAULT_QWEN_CANDIDATES if path.exists()), _DEFAULT_QWEN_CANDIDATES[0])
_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".safetensors")
_CONFIG_SUFFIXES = (".json",)
_QWEN_FILE_SUFFIXES = (".safetensors", ".bin")


@dataclass
class PixZigModelHandle:
    model: torch.nn.Module
    config: PixZigFieldConfig
    device: torch.device


@dataclass
class PixZigTextEncoderHandle:
    encoder: QwenTextEncoder
    cond_dim: int
    device: torch.device


def _register_model_folder_paths() -> None:
    if folder_paths is None or not hasattr(folder_paths, "add_model_folder_path"):
        return
    folder_paths.add_model_folder_path("pixzig_checkpoints", str(_PIXZIG_ROOT / "runs"))
    folder_paths.add_model_folder_path("pixzig_configs", str(_PIXZIG_ROOT))
    for path in (_DEFAULT_QWEN_PATH.parent, _PROJECTS_ROOT):
        if path.exists():
            folder_paths.add_model_folder_path("pixzig_text_encoders", str(path))


def _preferred_first(options: list[str], preferred: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for name in preferred:
        if name in options and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in options:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _filename_list(*categories: str, suffixes: tuple[str, ...] | None = None, include_none: bool = False) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    if include_none:
        names.append("")
        seen.add("")
    if folder_paths is not None:
        for category in categories:
            try:
                category_names = folder_paths.get_filename_list(category)
            except Exception:
                continue
            for name in category_names:
                if suffixes is not None and Path(name).suffix not in suffixes:
                    continue
                if name not in seen:
                    names.append(name)
                    seen.add(name)
    return names or [""]


def _folder_path_roots(*categories: str) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    if folder_paths is not None:
        for category in categories:
            try:
                category_roots = folder_paths.get_folder_paths(category)
            except Exception:
                continue
            for root in category_roots:
                path = Path(root).expanduser()
                if path not in seen:
                    roots.append(path)
                    seen.add(path)
    for path in (_DEFAULT_QWEN_PATH.parent, _PROJECTS_ROOT):
        if path.exists() and path not in seen:
            roots.append(path)
            seen.add(path)
    return roots


def _is_hf_text_encoder_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists() and (
        (path / "tokenizer.json").exists() or (path / "tokenizer_config.json").exists()
    )


def _directory_list(*categories: str, preferred: list[str] | None = None) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for root in _folder_path_roots(*categories):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if not _is_hf_text_encoder_dir(child):
                continue
            if child.name not in seen:
                names.append(child.name)
                seen.add(child.name)
    if preferred is not None:
        names = _preferred_first(names, preferred)
    if not names and _DEFAULT_QWEN_PATH.exists():
        names = [_DEFAULT_QWEN_PATH.name]
    return names or [_DEFAULT_QWEN_DIR_NAME]


def _resolve_named_path(value: str, categories: tuple[str, ...], *, required: bool = True) -> Path | None:
    if not isinstance(value, str):
        raise TypeError("path value must be a string")
    value = value.strip()
    if not value:
        if required:
            raise ValueError("path value is required")
        return None

    path = Path(value).expanduser()
    if path.exists():
        return path

    for root in _folder_path_roots(*categories):
        candidate = root / value
        if candidate.exists():
            return candidate

    if folder_paths is not None:
        for category in categories:
            try:
                resolved = Path(folder_paths.get_full_path_or_raise(category, value))
            except Exception:
                continue
            if resolved.exists():
                return resolved

    raise FileNotFoundError(f"path not found: {value}")


def _device(name: str) -> torch.device:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("device must be a non-empty string")
    if name == "auto":
        if comfy_model_management is not None:
            return torch.device(comfy_model_management.get_torch_device())
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but torch.cuda.is_available() is false")
    return device


def _dtype(name: str, device: torch.device, *, cpu_auto: torch.dtype = torch.float32) -> torch.dtype:
    if not isinstance(name, str):
        raise TypeError("precision must be a string")
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else cpu_auto
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError("precision must be 'auto', 'fp32', 'fp16', or 'bf16'")


def _is_amd_backend() -> bool:
    if getattr(torch.version, "hip", None):
        return True
    if comfy_model_management is not None and hasattr(comfy_model_management, "is_amd"):
        try:
            return bool(comfy_model_management.is_amd())
        except Exception:
            return False
    return False


def _qwen_force_torch_fallback(device: torch.device) -> bool:
    return device.type == "cpu" or (device.type == "cuda" and _is_amd_backend())


def _qwen_dtype(name: str, device: torch.device) -> torch.dtype:
    dtype = _dtype(name, device, cpu_auto=torch.float16)
    if device.type == "cuda" and _is_amd_backend() and dtype is torch.bfloat16:
        return torch.float16
    return dtype


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    for parameter in module.parameters():
        if parameter.dtype.is_floating_point:
            return parameter.dtype
    for buffer in module.buffers():
        if buffer.dtype.is_floating_point:
            return buffer.dtype
    return torch.float32


def _load_model_config(checkpoint_path: str | Path, config_path: str | Path | None = None) -> PixZigFieldConfig:
    if config_path is not None and str(config_path).strip():
        config_path_obj = Path(config_path).expanduser()
        if not config_path_obj.exists():
            raise FileNotFoundError(f"config not found: {config_path_obj}")
        config = load_field_config_json(config_path_obj)
    else:
        config = load_checkpoint_config(checkpoint_path)
    if not isinstance(config, PixZigFieldConfig):
        raise ValueError("PixZig model config was not found. Provide a checkpoint sidecar JSON or set config_path.")
    return config


def _to_comfy_image(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not torch.is_floating_point(x):
        raise TypeError("x must be a floating point tensor")
    if x.ndim != 4:
        raise ValueError(f"x must have shape [B, C, H, W], got {tuple(x.shape)}")
    batch_size, channels, height, width = x.shape
    if batch_size < 1:
        raise ValueError("x must include a non-empty batch dimension")
    if channels not in {1, 3, 4}:
        raise ValueError(f"x must have 1, 3, or 4 channels, got {channels}")
    if height < 1 or width < 1:
        raise ValueError(f"x spatial dimensions must be positive, got H={height}, W={width}")
    x = x.detach().clamp(-1, 1)
    x = (x + 1.0) * 0.5
    return x.permute(0, 2, 3, 1).contiguous().cpu()


def _to_preview_image(x: torch.Tensor) -> Image.Image:
    if x.ndim != 4:
        raise ValueError(f"x must have shape [B, C, H, W], got {tuple(x.shape)}")
    channels = x.shape[1]
    if channels not in {1, 3, 4}:
        raise ValueError(f"x must have 1, 3, or 4 channels, got {channels}")
    image = x[:1].detach().clamp(-1, 1)
    image = ((image[0] + 1.0) * 0.5 * 255.0).to(device="cpu", dtype=torch.uint8)
    if channels == 1:
        return Image.fromarray(image[0].numpy())
    return Image.fromarray(image.permute(1, 2, 0).numpy())


def _preview_tuple(x: torch.Tensor) -> tuple[str, Image.Image, int]:
    return ("JPEG", _to_preview_image(x), 512)


def _checkpoint_choices() -> list[str]:
    choices = _filename_list("pixzig_checkpoints", "checkpoints", "diffusion_models", suffixes=_CHECKPOINT_SUFFIXES)
    return _preferred_first(choices, ["last.pt", "pixzig-standard.safetensors"])


def _config_choices() -> list[str]:
    return _filename_list("pixzig_configs", "pixzig_checkpoints", "checkpoints", suffixes=_CONFIG_SUFFIXES, include_none=True)


def _qwen_choices() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for root in _folder_path_roots("pixzig_text_encoders", "text_encoders", "clip"):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if child.name.startswith("."):
                continue
            if _is_hf_text_encoder_dir(child):
                display_name = f"{child.name}.safetensors"
            elif child.is_file() and child.suffix in _QWEN_FILE_SUFFIXES:
                display_name = child.name
            else:
                continue
            if display_name not in seen:
                names.append(display_name)
                seen.add(display_name)
    if _DEFAULT_QWEN_PATH.exists() and _DEFAULT_QWEN_DISPLAY_NAME not in seen:
        names.insert(0, _DEFAULT_QWEN_DISPLAY_NAME)
    names = _preferred_first(names, [_DEFAULT_QWEN_DISPLAY_NAME])
    return names or [_DEFAULT_QWEN_DISPLAY_NAME]


def _resolve_qwen_model_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("qwen model name is required")
    value = value.strip()
    path = Path(value).expanduser()
    if _is_hf_text_encoder_dir(path):
        return path
    if path.is_file() and path.suffix in _QWEN_FILE_SUFFIXES and _is_hf_text_encoder_dir(path.parent):
        return path.parent

    candidate_names = [value]
    if Path(value).suffix in _QWEN_FILE_SUFFIXES:
        candidate_names.append(Path(value).stem)
    else:
        candidate_names.append(f"{value}.safetensors")

    for root in _folder_path_roots("pixzig_text_encoders", "text_encoders", "clip"):
        for name in candidate_names:
            candidate = root / name
            if _is_hf_text_encoder_dir(candidate):
                return candidate
            if candidate.is_file() and candidate.suffix in _QWEN_FILE_SUFFIXES and _is_hf_text_encoder_dir(candidate.parent):
                return candidate.parent
        stem_candidate = root / Path(value).stem
        if _is_hf_text_encoder_dir(stem_candidate):
            return stem_candidate

    if value == _DEFAULT_QWEN_DISPLAY_NAME and _is_hf_text_encoder_dir(_DEFAULT_QWEN_PATH):
        return _DEFAULT_QWEN_PATH
    raise FileNotFoundError(f"Qwen model not found or missing config/tokenizer files: {value}")


class PixZigModelLoader:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "checkpoint_name": (_checkpoint_choices(),),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
                "use_ema": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "checkpoint_path": ("STRING", {"default": ""}),
                "config_name": (_config_choices(),),
                "config_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("PIXZIG_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads a PixZig Field checkpoint from ComfyUI model folders or an explicit filesystem path."

    def load(
        self,
        checkpoint_name: str,
        device: str = "auto",
        precision: str = "auto",
        use_ema: bool = True,
        checkpoint_path: str = "",
        config_name: str = "",
        config_path: str = "",
    ) -> tuple[PixZigModelHandle]:
        if not isinstance(use_ema, bool):
            raise TypeError("use_ema must be bool")
        checkpoint_value = checkpoint_path.strip() or checkpoint_name
        checkpoint = _resolve_named_path(checkpoint_value, ("pixzig_checkpoints", "checkpoints", "diffusion_models"))
        assert checkpoint is not None

        config_value = config_path.strip() or config_name.strip()
        config = _load_model_config(
            checkpoint,
            _resolve_named_path(config_value, ("pixzig_configs", "pixzig_checkpoints", "checkpoints"), required=False),
        )
        target_device = _device(device)
        target_dtype = _dtype(precision, target_device)

        model = PixZigField(config)
        load_model_weights(checkpoint, model, use_ema=use_ema, map_location="cpu")
        model.to(device=target_device, dtype=target_dtype)
        model.eval()
        return (PixZigModelHandle(model=model, config=config, device=target_device),)


class PixZigQwenLoader:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "qwen_model_name": (_qwen_choices(),),
                "cond_dim": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),
                "device": (["cpu", "cuda", "auto"], {"default": "cpu"}),
                "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
                "cache_embeddings": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "qwen_model_path": ("STRING", {"default": ""}),
                "max_length": ("INT", {"default": 128, "min": 0, "max": 8192, "step": 1}),
            },
        }

    RETURN_TYPES = ("PIXZIG_TEXT_ENCODER",)
    RETURN_NAMES = ("text_encoder",)
    FUNCTION = "load"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads the local Qwen text encoder used for PixZig conditioning."

    def load(
        self,
        qwen_model_name: str,
        cond_dim: int,
        device: str = "cpu",
        precision: str = "auto",
        cache_embeddings: bool = True,
        qwen_model_path: str = "",
        max_length: int = 128,
    ) -> tuple[PixZigTextEncoderHandle]:
        model_value = qwen_model_path.strip() or qwen_model_name
        model_path = _resolve_qwen_model_path(model_value)
        if isinstance(cond_dim, bool) or cond_dim < 1:
            raise ValueError("cond_dim must be positive")
        if not isinstance(cache_embeddings, bool):
            raise TypeError("cache_embeddings must be bool")
        target_device = _device(device)
        encoder = QwenTextEncoder(
            model_name=str(model_path),
            output_dim=cond_dim,
            load_model=True,
            local_files_only=True,
            cache_embeddings=cache_embeddings,
            pooling="mean",
            max_length=max_length or None,
            force_torch_fallback=_qwen_force_torch_fallback(target_device),
        ).to(device=target_device, dtype=_qwen_dtype(precision, target_device))
        encoder.eval()
        return (PixZigTextEncoderHandle(encoder=encoder, cond_dim=cond_dim, device=target_device),)


class PixZigEncodePrompt:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "text_encoder": ("PIXZIG_TEXT_ENCODER",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("PIXZIG_CONDITIONING", "PIXZIG_CONDITIONING")
    RETURN_NAMES = ("cond", "negative_cond")
    FUNCTION = "encode"
    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes positive and negative PixZig prompts."

    def encode(
        self,
        text_encoder: PixZigTextEncoderHandle,
        prompt: str,
        negative_prompt: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(text_encoder, PixZigTextEncoderHandle):
            raise TypeError("text_encoder must be a PixZigTextEncoderHandle")
        with torch.no_grad():
            cond = text_encoder.encoder([prompt]).to(text_encoder.device)
            negative_cond = text_encoder.encoder([negative_prompt]).to(text_encoder.device)
        return (cond, negative_cond)


class PixZigSampler:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "model": ("PIXZIG_MODEL",),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 2**63 - 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000, "step": 1}),
                "method": (["euler", "heun"], {"default": "euler"}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "preview_every": ("INT", {"default": 1, "min": 0, "max": 1000, "step": 1}),
            },
            "optional": {
                "cond": ("PIXZIG_CONDITIONING",),
                "negative_cond": ("PIXZIG_CONDITIONING",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = "Runs the PixZig Euler or Heun sampler and returns a ComfyUI IMAGE tensor."

    def sample(
        self,
        model: PixZigModelHandle,
        seed: int,
        steps: int,
        method: str,
        cfg_scale: float,
        preview_every: int = 1,
        cond: torch.Tensor | None = None,
        negative_cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        if not isinstance(model, PixZigModelHandle):
            raise TypeError("model must be a PixZigModelHandle")
        if model.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("model handle points to CUDA but CUDA is not available")
        if isinstance(preview_every, bool) or not isinstance(preview_every, int):
            raise TypeError("preview_every must be an integer")
        if preview_every < 0:
            raise ValueError("preview_every must be non-negative")

        model_dtype = _module_dtype(model.model)
        generator = torch.Generator(device=model.device)
        generator.manual_seed(int(seed))
        noise = torch.randn(
            1,
            model.config.in_channels,
            model.config.image_size,
            model.config.image_size,
            device=model.device,
            dtype=model_dtype,
            generator=generator,
        )
        cond = cond.to(device=model.device, dtype=model_dtype) if cond is not None else None
        negative_cond = negative_cond.to(device=model.device, dtype=model_dtype) if negative_cond is not None else None
        sample_config = PixZigSampleConfig(n_steps=steps, method=method, cfg_scale=cfg_scale, seed=int(seed))
        progress_bar = comfy_utils.ProgressBar(sample_config.n_steps) if comfy_utils is not None else None

        def preview_callback(step: int, total_steps: int, preview: torch.Tensor) -> None:
            if progress_bar is None:
                return
            preview_payload = None
            if preview_every > 0 and (step == 1 or step == total_steps or step % preview_every == 0):
                preview_payload = _preview_tuple(preview)
            progress_bar.update_absolute(step, total_steps, preview_payload)

        was_training = model.model.training
        model.model.eval()
        try:
            with torch.no_grad():
                image = sample_euler(
                    model.model,
                    noise,
                    sample_config.n_steps,
                    cond=cond,
                    negative_cond=negative_cond,
                    cfg_scale=sample_config.cfg_scale,
                    method=cast(Literal["euler", "heun"], sample_config.method),
                    preview_callback=preview_callback,
                )
        finally:
            if was_training:
                model.model.train()
        return (_to_comfy_image(image),)


_register_model_folder_paths()

NODE_CLASS_MAPPINGS = {
    "PixZigModelLoader": PixZigModelLoader,
    "PixZigQwenLoader": PixZigQwenLoader,
    "PixZigEncodePrompt": PixZigEncodePrompt,
    "PixZigSampler": PixZigSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixZigModelLoader": "Load PixZig Field",
    "PixZigQwenLoader": "Load PixZig Qwen",
    "PixZigEncodePrompt": "PixZig Encode Prompt",
    "PixZigSampler": "PixZig Sampler",
}

from __future__ import annotations

import argparse
import logging
import math
import time
from numbers import Real
from pathlib import Path
from typing import Callable, Literal, cast

import torch
import torch.nn as nn
from PIL import Image

from .checkpoint import load_checkpoint_config, load_model_weights
from .config import PixZigFieldConfig, PixZigSampleConfig, load_field_config_json
from .flow_matching import velocity_from_x0
from .model import PixZigField
from .text_encoder import QwenTextEncoder
from .utils import format_duration


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _sample_log_interval(n_steps: int) -> int:
    if n_steps <= 10:
        return 1
    if n_steps <= 50:
        return 5
    return max(n_steps // 10, 1)


def _module_device(module: nn.Module) -> torch.device:
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return torch.device("cpu")


def _module_dtype(module: nn.Module) -> torch.dtype:
    for tensor in list(module.parameters()) + list(module.buffers()):
        if torch.is_floating_point(tensor):
            return tensor.dtype
    return torch.float32


def _module_has_state(module: nn.Module) -> bool:
    return next(module.parameters(), None) is not None or next(module.buffers(), None) is not None


def _resolve_implicit_cuda_index(device: torch.device) -> torch.device:
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _devices_equivalent(left: torch.device, right: torch.device) -> bool:
    return _resolve_implicit_cuda_index(left) == _resolve_implicit_cuda_index(right)


def _validate_cfg_scale(cfg_scale: float) -> float:
    if isinstance(cfg_scale, bool) or not isinstance(cfg_scale, Real):
        raise TypeError("cfg_scale must be a real number")
    cfg_scale = float(cfg_scale)
    if not math.isfinite(cfg_scale):
        raise ValueError("cfg_scale must be finite")
    if cfg_scale < 0:
        raise ValueError("cfg_scale must be non-negative")
    return cfg_scale


def _validate_conditioning(
    name: str,
    cond: torch.Tensor | None,
    batch_size: int,
) -> None:
    if cond is None:
        return
    if not isinstance(cond, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(cond):
        raise TypeError(f"{name} must be a floating point tensor")
    if cond.ndim not in {2, 3}:
        raise ValueError(f"{name} must have shape [B, D] or [B, L, D], got {tuple(cond.shape)}")
    if cond.shape[0] not in {1, batch_size}:
        raise ValueError(f"{name} batch size must be 1 or match noise batch size {batch_size}, got {cond.shape[0]}")
    if cond.ndim == 3 and cond.shape[1] < 1:
        raise ValueError(f"{name} sequence length must be non-empty")
    if cond.shape[-1] < 1:
        raise ValueError(f"{name} last dimension must be non-empty")
    if not torch.isfinite(cond).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_sampler_inputs(
    noise: torch.Tensor,
    n_steps: int,
    cond: torch.Tensor | None,
    negative_cond: torch.Tensor | None,
    cfg_scale: float,
    method: str,
) -> float:
    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError("n_steps must be an integer")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if method not in {"euler", "heun"}:
        raise ValueError("method must be 'euler' or 'heun'")
    cfg_scale = _validate_cfg_scale(cfg_scale)
    if not isinstance(noise, torch.Tensor):
        raise TypeError("noise must be a torch.Tensor")
    if not torch.is_floating_point(noise):
        raise TypeError("noise must be a floating point tensor")
    if noise.ndim != 4:
        raise ValueError(f"noise must have shape [B, C, H, W], got {tuple(noise.shape)}")
    batch_size, channels, height, width = noise.shape
    if batch_size < 1:
        raise ValueError("noise must include a non-empty batch dimension")
    if channels < 1:
        raise ValueError("noise must include at least one channel")
    if height < 1 or width < 1:
        raise ValueError(f"noise spatial dimensions must be positive, got H={height}, W={width}")
    if not torch.isfinite(noise).all():
        raise ValueError("noise must contain only finite values")

    _validate_conditioning("cond", cond, batch_size)
    _validate_conditioning("negative_cond", negative_cond, batch_size)
    if cfg_scale != 1.0 and (cond is None) != (negative_cond is None):
        raise ValueError("cond and negative_cond must both be provided when cfg_scale is not 1.0")
    if cond is not None and negative_cond is not None:
        if cond.ndim != negative_cond.ndim or cond.shape[1:] != negative_cond.shape[1:]:
            raise ValueError(
                "cond and negative_cond must have matching non-batch dimensions, "
                f"got {tuple(cond.shape)} and {tuple(negative_cond.shape)}"
            )
    return cfg_scale


def _to_image(x: torch.Tensor) -> Image.Image:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 3:
        raise ValueError(f"x must have shape [C, H, W], got {tuple(x.shape)}")
    channels, height, width = x.shape
    if channels not in {1, 3, 4}:
        raise ValueError(f"x must have 1, 3, or 4 channels, got {channels}")
    if height < 1 or width < 1:
        raise ValueError(f"x spatial dimensions must be positive, got H={height}, W={width}")
    x = x.detach().clamp(-1, 1)
    x = ((x + 1.0) * 0.5 * 255.0).to(torch.uint8)
    if channels == 1:
        return Image.fromarray(x[0].cpu().numpy())
    return Image.fromarray(x.permute(1, 2, 0).cpu().numpy())


def save_image_tensor(x: torch.Tensor, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _to_image(x).save(output)
    return output


def _cfg_x0(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor | None,
    negative_cond: torch.Tensor | None,
    cfg_scale: float,
) -> torch.Tensor:
    if cond is None or negative_cond is None or cfg_scale == 1.0:
        return model(x, t, cond=cond)
    x0_uncond = model(x, t, cond=negative_cond)
    x0_cond = model(x, t, cond=cond)
    return x0_uncond + cfg_scale * (x0_cond - x0_uncond)


@torch.no_grad()
def sample_euler(
    model: nn.Module,
    noise: torch.Tensor,
    n_steps: int,
    *,
    cond: torch.Tensor | None = None,
    negative_cond: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
    method: Literal["euler", "heun"] = "euler",
    progress_callback: Callable[[int, int, float], None] | None = None,
    preview_callback: Callable[[int, int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    cfg_scale = _validate_sampler_inputs(noise, n_steps, cond, negative_cond, cfg_scale, method)
    x = noise
    dt = 1.0 / n_steps
    timesteps = torch.arange(n_steps, device=x.device, dtype=x.dtype) * dt
    started_at = time.perf_counter()
    for index, t_value in enumerate(timesteps, start=1):
        t = t_value.expand(x.shape[0])
        x0_pred = _cfg_x0(model, x, t, cond, negative_cond, cfg_scale)
        velocity = velocity_from_x0(x, x0_pred, t)
        if method == "heun":
            x_next = x + dt * velocity
            if index == n_steps:
                x = x_next
                if progress_callback is not None or preview_callback is not None:
                    _synchronize_device(x.device)
                    if preview_callback is not None:
                        preview_callback(index, n_steps, x0_pred)
                    if progress_callback is not None:
                        progress_callback(index, n_steps, time.perf_counter() - started_at)
                continue
            t_next = t + dt
            x0_next = _cfg_x0(model, x_next, t_next, cond, negative_cond, cfg_scale)
            velocity_next = velocity_from_x0(x_next, x0_next, t_next)
            x = x + 0.5 * dt * (velocity + velocity_next)
        else:
            x = x + dt * velocity
        if progress_callback is not None or preview_callback is not None:
            _synchronize_device(x.device)
            if preview_callback is not None:
                preview_callback(index, n_steps, x0_pred)
            if progress_callback is not None:
                progress_callback(index, n_steps, time.perf_counter() - started_at)
    return x


def _load_checkpoint_config(path: str | None) -> PixZigFieldConfig | None:
    if path is None:
        return None
    config = load_checkpoint_config(path)
    return config if isinstance(config, PixZigFieldConfig) else None


def _resolve_aux_device(value: str, model_device: torch.device) -> torch.device:
    if value in {"model", "auto"}:
        return model_device
    return torch.device(value)


def encode_prompt(
    *,
    text_encoder_kind: str,
    prompt: str,
    negative_prompt: str | None,
    qwen_model_name: str,
    cond_dim: int,
    allow_qwen_download: bool,
    device: torch.device,
    text_encoder_device: torch.device | None = None,
    cache_embeddings: bool = True,
    pooling: str = "mean",
    max_length: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if text_encoder_kind == "none":
        return None, None
    encoder_device = text_encoder_device if text_encoder_device is not None else device
    encoder = QwenTextEncoder(
        model_name=qwen_model_name,
        output_dim=cond_dim,
        load_model=True,
        local_files_only=not allow_qwen_download,
        cache_embeddings=cache_embeddings,
        pooling=pooling,
        max_length=max_length,
        force_torch_fallback=encoder_device.type == "cpu",
    ).to(encoder_device)
    encoder.eval()
    cond = encoder([prompt]).to(device)
    negative_cond = encoder([negative_prompt or ""]).to(device)
    return cond, negative_cond


def _encode_prompt(
    args: argparse.Namespace,
    device: torch.device,
    *,
    cond_dim: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    return encode_prompt(
        text_encoder_kind=args.text_encoder,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        qwen_model_name=args.qwen_model_name,
        cond_dim=cond_dim,
        allow_qwen_download=args.allow_qwen_download,
        device=device,
        text_encoder_device=_resolve_aux_device(args.text_encoder_device, device),
        cache_embeddings=True,
        max_length=args.qwen_max_length,
    )


@torch.no_grad()
def generate_sample_image(
    model: nn.Module,
    model_config: PixZigFieldConfig,
    output: str | Path,
    *,
    sample_config: PixZigSampleConfig | None = None,
    cond: torch.Tensor | None = None,
    negative_cond: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> Path:
    sample_config = PixZigSampleConfig() if sample_config is None else sample_config
    if not isinstance(sample_config, PixZigSampleConfig):
        raise TypeError("sample_config must be a PixZigSampleConfig")
    if not isinstance(model_config, PixZigFieldConfig):
        raise TypeError("model_config must be a PixZigFieldConfig")
    model_device = _resolve_implicit_cuda_index(torch.device(device)) if device is not None else _module_device(model)
    model_dtype = _module_dtype(model)
    module_device = _module_device(model)
    if not _devices_equivalent(module_device, model_device) and _module_has_state(model):
        raise ValueError(f"model is on {module_device}, but sampling device is {model_device}")
    output = Path(output)
    interval = _sample_log_interval(sample_config.n_steps)
    logging.info(
        "sample started: output=%s method=%s steps=%d image_size=%d seed=%d",
        output,
        sample_config.method,
        sample_config.n_steps,
        model_config.image_size,
        sample_config.seed,
    )
    generator = torch.Generator(device=model_device)
    generator.manual_seed(sample_config.seed)
    noise = torch.randn(
        1,
        model_config.in_channels,
        model_config.image_size,
        model_config.image_size,
        device=model_device,
        dtype=model_dtype,
        generator=generator,
    )
    if cond is not None:
        cond = cond.to(device=model_device, dtype=model_dtype)
    if negative_cond is not None:
        negative_cond = negative_cond.to(device=model_device, dtype=model_dtype)
    was_training = model.training
    model.eval()
    _synchronize_device(model_device)
    started_at = time.perf_counter()

    def log_progress(current_step: int, total_steps: int, elapsed: float) -> None:
        if current_step != 1 and current_step != total_steps and current_step % interval != 0:
            return
        sec_per_step = elapsed / max(current_step, 1)
        eta = max(total_steps - current_step, 0) * sec_per_step
        logging.info(
            "sample progress: step=%d/%d elapsed=%s eta=%s sec_per_step=%.3f",
            current_step,
            total_steps,
            format_duration(elapsed),
            format_duration(eta),
            sec_per_step,
        )

    try:
        image = sample_euler(
            model,
            noise,
            sample_config.n_steps,
            cond=cond,
            negative_cond=negative_cond,
            cfg_scale=sample_config.cfg_scale,
            method=cast(Literal["euler", "heun"], sample_config.method),
            progress_callback=log_progress,
        )[0]
        _synchronize_device(model_device)
    finally:
        if was_training:
            model.train()
    saved_path = save_image_tensor(image, output)
    elapsed = time.perf_counter() - started_at
    logging.info(
        "sample completed: output=%s elapsed=%s seconds=%.3f",
        saved_path,
        format_duration(elapsed),
        elapsed,
    )
    return saved_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample PixZig Field with a simple Euler flow sampler.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON PixZigFieldConfig.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="sample.png")
    parser.add_argument("--image-size", type=int, default=PixZigFieldConfig.image_size)
    parser.add_argument("--patch-size", type=int, default=PixZigFieldConfig.patch_size)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--method", choices=["euler", "heun"], default="heun")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--text-encoder", choices=["none", "qwen"], default="none")
    parser.add_argument("--qwen-model-name", type=str, default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--allow-qwen-download", action="store_true")
    parser.add_argument("--qwen-max-length", type=int, default=None)
    parser.add_argument("--text-encoder-device", type=str, default="cpu")
    parser.add_argument("--cond-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=PixZigFieldConfig.hidden_dim)
    parser.add_argument("--depth", type=int, default=PixZigFieldConfig.depth)
    parser.add_argument("--zigma-state-dim", type=int, default=PixZigFieldConfig.zigma_state_dim)
    parser.add_argument("--zigma-expand", type=int, default=PixZigFieldConfig.zigma_expand)
    parser.add_argument("--scan-mode", type=str, default=PixZigFieldConfig.scan_mode)
    parser.add_argument(
        "--mamba-backend",
        choices=["native", "external", "auto"],
        default=PixZigFieldConfig.mamba_backend,
    )
    parser.add_argument("--mamba-inner-expand", type=int, default=PixZigFieldConfig.mamba_inner_expand)
    parser.add_argument("--mamba-head-dim", type=int, default=PixZigFieldConfig.mamba_head_dim)
    parser.add_argument("--mamba-num-groups", type=int, default=PixZigFieldConfig.mamba_num_groups)
    parser.add_argument("--mamba-conv-kernel", type=int, default=PixZigFieldConfig.mamba_conv_kernel)
    parser.add_argument("--pixnerd-hidden-dim", type=int, default=PixZigFieldConfig.pixnerd_hidden_dim)
    parser.add_argument("--pixnerd-layers", type=int, default=PixZigFieldConfig.pixnerd_layers)
    parser.add_argument("--refiner-channels", type=int, default=PixZigFieldConfig.refiner_channels)
    parser.add_argument("--refiner-blocks", type=int, default=PixZigFieldConfig.refiner_blocks)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint_config = _load_checkpoint_config(args.checkpoint)
    if checkpoint_config is not None:
        model_config = checkpoint_config
        if args.text_encoder != "none" and model_config.cond_dim is None:
            raise ValueError("checkpoint config has cond_dim=None; it was not trained for text conditioning")
    elif args.config is not None:
        model_config = load_field_config_json(args.config)
    else:
        model_config = PixZigFieldConfig(
            image_size=args.image_size,
            patch_size=args.patch_size,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            zigma_state_dim=args.zigma_state_dim,
            zigma_expand=args.zigma_expand,
            scan_mode=args.scan_mode,
            mamba_backend=args.mamba_backend,
            mamba_inner_expand=args.mamba_inner_expand,
            mamba_head_dim=args.mamba_head_dim,
            mamba_num_groups=args.mamba_num_groups,
            mamba_conv_kernel=args.mamba_conv_kernel,
            pixnerd_hidden_dim=args.pixnerd_hidden_dim,
            pixnerd_layers=args.pixnerd_layers,
            refiner_channels=args.refiner_channels,
            refiner_blocks=args.refiner_blocks,
            cond_dim=args.cond_dim if args.text_encoder != "none" else None,
        )
    prompt_cond_dim = model_config.cond_dim if model_config.cond_dim is not None else args.cond_dim
    model = PixZigField(model_config).to(device)
    if args.checkpoint is not None:
        load_model_weights(args.checkpoint, model, use_ema=True, map_location="cpu")
    model.eval()
    cond, negative_cond = _encode_prompt(args, device, cond_dim=prompt_cond_dim)

    sample_config = PixZigSampleConfig(
        n_steps=args.n_steps,
        method=args.method,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )
    generate_sample_image(
        model,
        model_config,
        args.output,
        sample_config=sample_config,
        cond=cond,
        negative_cond=negative_cond,
        device=device,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, prune_checkpoints, save_checkpoint, save_model_safetensors
from .config import (
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
    config_from_json_dict,
    load_config_json,
    save_config_json,
)
from .dataset import PixZigImageTextDataset, collate_fn
from .ema import EMAModel
from .flow_matching import make_xt, sample_timesteps
from .losses import PixZigLoss
from .model import PixZigField
from .sample import _devices_equivalent, generate_sample_image
from .text_encoder import QwenTextEncoder
from .utils import count_parameters, format_duration


def _progress_bar(step: int, total_steps: int, width: int = 24) -> str:
    total_steps = max(int(total_steps), 1)
    step = min(max(int(step), 0), total_steps)
    filled = round(width * step / total_steps)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _format_training_progress(
    *,
    step: int,
    max_steps: int,
    epoch: int,
    loss: float,
    flow_loss: float,
    x0_loss: float,
    lr: float,
    grad_norm: float,
    sec_per_step: float,
    images_per_sec: float,
    elapsed: float,
    eta: float,
) -> str:
    percent = 100.0 * min(max(step, 0), max(max_steps, 1)) / max(max_steps, 1)
    return (
        f"stage=train {_progress_bar(step, max_steps)} {percent:6.2f}% "
        f"step={step}/{max_steps} epoch={epoch} "
        f"loss={loss:.5f} flow={flow_loss:.5f} x0={x0_loss:.5f} "
        f"lr={lr:.2e} grad={grad_norm:.3f} "
        f"{sec_per_step:.3f}s/step {images_per_sec:.2f}img/s "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
    )


def _write_live_progress(message: str) -> None:
    if sys.stderr.isatty():
        sys.stderr.write("\r\033[K" + message)
        sys.stderr.flush()


def _finish_live_progress() -> None:
    if sys.stderr.isatty():
        sys.stderr.write("\n")
        sys.stderr.flush()


def _autocast_context(device: torch.device, precision: str, enabled: bool):
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be 'fp32', 'fp16', or 'bf16'")
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be bool")
    if not enabled or precision == "fp32":
        return nullcontext()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return torch.autocast(device_type=device.type, dtype=dtype)


def _resolve_aux_device(value: str, model_device: torch.device) -> torch.device:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("device value must be a non-empty string")
    if value in {"model", "auto"}:
        return model_device
    return torch.device(value)


def _module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_model_config(args: argparse.Namespace) -> PixZigFieldConfig:
    cond_dim = args.cond_dim if args.text_encoder != "none" else None
    return PixZigFieldConfig(
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
        cond_dim=cond_dim,
    )


def build_text_encoder(config: PixZigTextEncoderConfig, device: torch.device) -> QwenTextEncoder | None:
    if config.kind == "none":
        return None
    encoder_device = _resolve_aux_device(config.device, device)
    encoder = QwenTextEncoder(
        model_name=config.qwen_model_name,
        output_dim=config.cond_dim,
        load_model=True,
        local_files_only=not config.allow_qwen_download,
        cache_embeddings=config.cache_embeddings,
        pooling=config.pooling,
        max_length=config.max_length,
        force_torch_fallback=encoder_device.type == "cpu",
    ).to(encoder_device)
    encoder.eval()
    return encoder


def build_lr_scheduler(optimizer: torch.optim.Optimizer, config: PixZigTrainingConfig):
    warmup_steps = max(config.lr_warmup_steps, 0)
    total_steps = max(config.max_steps, 1)
    min_ratio = max(min(config.min_lr_ratio, 1.0), 0.0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        decay_span = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_span, 0.0), 1.0)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _maybe_load_json_config(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    config = load_config_json(path)
    if not isinstance(config, dict):
        raise TypeError("train --config must point to a JSON object")
    return config


def _config_section(json_config: dict[str, object], name: str, config_type: type[object]) -> object | None:
    value = json_config.get(name)
    if isinstance(value, dict):
        return config_from_json_dict(value, config_type)
    if name in json_config:
        raise TypeError(f"config section '{name}' must be a JSON object")
    return None


def _validate_checkpoint_ema_compatibility(
    checkpoint_config: PixZigCheckpointConfig,
    ema_config: PixZigEMAConfig,
) -> None:
    if (
        checkpoint_config.save_safetensors
        and checkpoint_config.safetensors_source == "ema"
        and not ema_config.enabled
    ):
        raise ValueError("safetensors_source='ema' requires EMA to be enabled")


def _run_configs(
    *,
    model: PixZigFieldConfig,
    dataset: PixZigDatasetConfig,
    flow: PixZigFlowConfig,
    loss: PixZigLossConfig,
    min_snr: PixZigMinSNRConfig,
    ema: PixZigEMAConfig,
    checkpoint: PixZigCheckpointConfig,
    sample: PixZigSampleConfig,
    text_encoder: PixZigTextEncoderConfig,
    training: PixZigTrainingConfig,
) -> dict[str, object]:
    return {
        "model": model,
        "dataset": dataset,
        "flow": flow,
        "loss": loss,
        "min_snr": min_snr,
        "ema": ema,
        "checkpoint": checkpoint,
        "sample": sample,
        "text_encoder": text_encoder,
        "training": training,
    }


@torch.no_grad()
def _save_training_sample(
    *,
    model: PixZigField,
    ema: EMAModel | None,
    model_config: PixZigFieldConfig,
    sample_config: PixZigSampleConfig,
    text_encoder: QwenTextEncoder | None,
    output: Path,
    device: torch.device,
) -> Path:
    cond = None
    negative_cond = None
    if text_encoder is not None:
        cond = text_encoder([sample_config.prompt]).to(device)
        negative_cond = text_encoder([sample_config.negative_prompt]).to(device)
    sample_model = ema.module if sample_config.use_ema and ema is not None else model
    if sample_model is not model and not _devices_equivalent(_module_device(sample_model), device):
        logging.warning("EMA is on %s; using live model for sample on %s", _module_device(sample_model), device)
        sample_model = model
    return generate_sample_image(
        sample_model,
        model_config,
        output,
        sample_config=sample_config,
        cond=cond,
        negative_cond=negative_cond,
        device=device,
    )


def _periodic_safetensors_name(step: int) -> str:
    return f"step_{step:08d}.safetensors"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PixZig Field on image-text data.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON run config.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--captions", type=str, default=None)
    parser.add_argument("--caption-extension", type=str, default=PixZigDatasetConfig.caption_extension)
    parser.add_argument("--allow-empty-text", action="store_true")
    parser.add_argument("--non-recursive-data", action="store_true")
    parser.add_argument("--crop-mode", choices=["center", "random", "pad"], default=PixZigDatasetConfig.crop_mode)
    parser.add_argument("--horizontal-flip-p", type=float, default=PixZigDatasetConfig.horizontal_flip_p)
    parser.add_argument("--output", type=str, default="./runs/pixzig-field")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image-size", type=int, default=PixZigFieldConfig.image_size)
    parser.add_argument("--patch-size", type=int, default=PixZigFieldConfig.patch_size)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=PixZigTrainingConfig.num_workers)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--timestep-sampling",
        choices=["uniform", "logit_normal", "cosine", "beta", "stratified"],
        default=PixZigFlowConfig.timestep_sampling,
    )
    parser.add_argument("--timestep-eps", type=float, default=PixZigFlowConfig.timestep_eps)
    parser.add_argument("--logit-normal-std", type=float, default=PixZigFlowConfig.logit_normal_std)
    parser.add_argument("--beta-alpha", type=float, default=PixZigFlowConfig.beta_alpha)
    parser.add_argument("--beta-beta", type=float, default=PixZigFlowConfig.beta_beta)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--text-encoder", choices=["none", "qwen"], default="none")
    parser.add_argument("--qwen-model-name", type=str, default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--allow-qwen-download", action="store_true")
    parser.add_argument("--qwen-cache-embeddings", action="store_true")
    parser.add_argument("--qwen-max-length", type=int, default=None)
    parser.add_argument("--text-encoder-device", type=str, default=PixZigTextEncoderConfig.device)
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
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-warmup-steps", type=int, default=PixZigTrainingConfig.lr_warmup_steps)
    parser.add_argument("--min-lr-ratio", type=float, default=PixZigTrainingConfig.min_lr_ratio)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=PixZigEMAConfig.decay)
    parser.add_argument("--ema-update-every", type=int, default=PixZigEMAConfig.update_every)
    parser.add_argument("--ema-warmup-steps", type=int, default=PixZigEMAConfig.warmup_steps)
    parser.add_argument("--ema-device", type=str, default=PixZigEMAConfig.device)
    parser.add_argument("--periodic-checkpoints", action="store_true")
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--save-safetensors", action="store_true")
    parser.add_argument("--safetensors-name", type=str, default=PixZigCheckpointConfig.safetensors_name)
    parser.add_argument(
        "--safetensors-source",
        choices=["model", "ema"],
        default=PixZigCheckpointConfig.safetensors_source,
    )
    parser.add_argument("--periodic-samples", action="store_true")
    parser.add_argument("--sample-every-steps", type=int, default=PixZigSampleConfig.sample_every_steps)
    parser.add_argument("--sample-n-steps", type=int, default=PixZigSampleConfig.n_steps)
    parser.add_argument("--sample-method", choices=["euler", "heun"], default=PixZigSampleConfig.method)
    parser.add_argument("--sample-prompt", type=str, default=PixZigSampleConfig.prompt)
    parser.add_argument("--sample-negative-prompt", type=str, default=PixZigSampleConfig.negative_prompt)
    parser.add_argument("--sample-cfg-scale", type=float, default=PixZigSampleConfig.cfg_scale)
    parser.add_argument("--sample-seed", type=int, default=PixZigSampleConfig.seed)
    parser.add_argument("--no-sample-ema", action="store_true")
    parser.add_argument("--log-every-steps", type=int, default=PixZigTrainingConfig.log_every_steps)
    parser.add_argument("--seed", type=int, default=PixZigTrainingConfig.seed)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)
    json_config = _maybe_load_json_config(args.config)
    checkpoint_config = PixZigCheckpointConfig(
        periodic_enabled=args.periodic_checkpoints or args.save_every_steps is not None,
        save_every_steps=args.save_every_steps or PixZigCheckpointConfig.save_every_steps,
        save_safetensors=args.save_safetensors,
        safetensors_name=args.safetensors_name,
        safetensors_source=args.safetensors_source,
    )
    sample_config = PixZigSampleConfig(
        periodic_enabled=args.periodic_samples,
        sample_every_steps=args.sample_every_steps,
        n_steps=args.sample_n_steps,
        method=args.sample_method,
        prompt=args.sample_prompt,
        negative_prompt=args.sample_negative_prompt,
        cfg_scale=args.sample_cfg_scale,
        seed=args.sample_seed,
        use_ema=not args.no_sample_ema,
    )
    dataset_config = PixZigDatasetConfig(
        image_size=args.image_size,
        captions_file=args.captions,
        caption_extension=args.caption_extension,
        require_captions=not args.allow_empty_text,
        recursive=not args.non_recursive_data,
        crop_mode=args.crop_mode,
        horizontal_flip_p=args.horizontal_flip_p,
    )
    flow_config = PixZigFlowConfig(
        timestep_sampling=args.timestep_sampling,
        timestep_eps=args.timestep_eps,
        logit_normal_std=args.logit_normal_std,
        beta_alpha=args.beta_alpha,
        beta_beta=args.beta_beta,
    )
    loss_config = PixZigLossConfig()
    min_snr_config = PixZigMinSNRConfig()
    ema_config = PixZigEMAConfig(
        decay=args.ema_decay,
        update_every=args.ema_update_every,
        warmup_steps=args.ema_warmup_steps,
        device=args.ema_device,
    )
    text_encoder_config = PixZigTextEncoderConfig(
        kind=args.text_encoder,
        qwen_model_name=args.qwen_model_name,
        allow_qwen_download=args.allow_qwen_download,
        cache_embeddings=args.qwen_cache_embeddings,
        cond_dim=args.cond_dim,
        max_length=args.qwen_max_length,
        device=args.text_encoder_device,
    )
    training_config = PixZigTrainingConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_warmup_steps=args.lr_warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        num_workers=args.num_workers,
        log_every_steps=args.log_every_steps,
        precision=args.precision,
        autocast=not args.no_autocast,
        seed=args.seed,
    )
    if json_config:
        loaded_dataset = _config_section(json_config, "dataset", PixZigDatasetConfig)
        loaded_flow = _config_section(json_config, "flow", PixZigFlowConfig)
        loaded_loss = _config_section(json_config, "loss", PixZigLossConfig)
        loaded_min_snr = _config_section(json_config, "min_snr", PixZigMinSNRConfig)
        loaded_ema = _config_section(json_config, "ema", PixZigEMAConfig)
        loaded_checkpoint = _config_section(json_config, "checkpoint", PixZigCheckpointConfig)
        loaded_sample = _config_section(json_config, "sample", PixZigSampleConfig)
        loaded_text_encoder = _config_section(json_config, "text_encoder", PixZigTextEncoderConfig)
        loaded_training = _config_section(json_config, "training", PixZigTrainingConfig)
        if isinstance(loaded_dataset, PixZigDatasetConfig):
            dataset_config = loaded_dataset
        if isinstance(loaded_flow, PixZigFlowConfig):
            flow_config = loaded_flow
        if isinstance(loaded_loss, PixZigLossConfig):
            loss_config = loaded_loss
        if isinstance(loaded_min_snr, PixZigMinSNRConfig):
            min_snr_config = loaded_min_snr
        if isinstance(loaded_ema, PixZigEMAConfig):
            ema_config = loaded_ema
        if isinstance(loaded_checkpoint, PixZigCheckpointConfig):
            checkpoint_config = loaded_checkpoint
        if isinstance(loaded_sample, PixZigSampleConfig):
            sample_config = loaded_sample
        if isinstance(loaded_text_encoder, PixZigTextEncoderConfig):
            text_encoder_config = loaded_text_encoder
        if isinstance(loaded_training, PixZigTrainingConfig):
            training_config = loaded_training
        if isinstance(loaded_dataset, PixZigDatasetConfig) and isinstance(loaded_training, PixZigTrainingConfig):
            if dataset_config.image_size != training_config.image_size:
                raise ValueError("dataset.image_size and training.image_size must match")
        elif isinstance(loaded_dataset, PixZigDatasetConfig):
            training_config = replace(training_config, image_size=dataset_config.image_size)
        elif isinstance(loaded_training, PixZigTrainingConfig):
            dataset_config = replace(dataset_config, image_size=training_config.image_size)
        logging.info("loaded JSON config %s", args.config)
    _validate_checkpoint_ema_compatibility(checkpoint_config, ema_config)
    seed_everything(training_config.seed)
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(training_config.seed)
    checkpoint_dir = Path(args.output) / checkpoint_config.output_dir
    sample_dir = Path(args.output) / sample_config.output_dir

    dataset = PixZigImageTextDataset(
        args.data,
        image_size=dataset_config.image_size,
        captions_file=dataset_config.captions_file,
        allow_empty_text=not dataset_config.require_captions,
        caption_extension=dataset_config.caption_extension,
        recursive=dataset_config.recursive,
        crop_mode=dataset_config.crop_mode,
        horizontal_flip_p=dataset_config.horizontal_flip_p,
        return_metadata=dataset_config.return_metadata,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=training_config.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=training_config.num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=dataloader_generator,
    )
    loaded_model = _config_section(json_config, "model", PixZigFieldConfig) if json_config else None
    model_config = loaded_model if isinstance(loaded_model, PixZigFieldConfig) else build_model_config(args)
    if not isinstance(loaded_model, PixZigFieldConfig):
        model_config = replace(model_config, image_size=dataset_config.image_size)
    elif model_config.image_size != dataset_config.image_size:
        raise ValueError("model.image_size and dataset.image_size must match")
    if text_encoder_config.kind != "none":
        model_config = replace(model_config, cond_dim=text_encoder_config.cond_dim)
    model = PixZigField(model_config)
    ema_device = _resolve_aux_device(ema_config.device, device)
    ema = (
        EMAModel(
            model,
            decay=ema_config.decay,
            update_every=ema_config.update_every,
            warmup_steps=ema_config.warmup_steps,
            min_decay=ema_config.min_decay,
            device=ema_device,
        )
        if ema_config.enabled
        else None
    )
    model.to(device)
    text_encoder = build_text_encoder(text_encoder_config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        betas=training_config.betas,
        eps=training_config.eps,
    )
    scheduler = build_lr_scheduler(optimizer, training_config)
    scaler_enabled = device.type == "cuda" and training_config.precision == "fp16" and training_config.autocast
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    loss_fn = PixZigLoss(loss_config=loss_config, min_snr_config=min_snr_config, patch_size=model_config.patch_size)
    run_configs = _run_configs(
        model=model_config,
        dataset=dataset_config,
        flow=flow_config,
        loss=loss_config,
        min_snr=min_snr_config,
        ema=ema_config,
        checkpoint=checkpoint_config,
        sample=sample_config,
        text_encoder=text_encoder_config,
        training=training_config,
    )
    save_config_json(Path(args.output) / "run_config.json", run_configs)
    logging.info("trainable parameters: %s", f"{count_parameters(model):,}")

    step = 0
    epoch = 0
    micro_step = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            scaler=scaler if scaler_enabled else None,
            map_location=device,
        )
        step = int(checkpoint.get("step", 0))
        epoch = int(checkpoint.get("epoch", 0))
        logging.info("resumed checkpoint %s at step=%d epoch=%d", args.resume, step, epoch)

    start = time.time()
    start_step = step
    model.train()
    optimizer.zero_grad(set_to_none=True)

    accumulation_steps = max(training_config.gradient_accumulation_steps, 1)
    while step < training_config.max_steps:
        for batch in dataloader:
            x0 = batch["image"].to(device)
            t = sample_timesteps(
                x0.shape[0],
                device=device,
                eps=flow_config.timestep_eps,
                mode=flow_config.timestep_sampling,
                logit_normal_std=flow_config.logit_normal_std,
                beta_alpha=flow_config.beta_alpha,
                beta_beta=flow_config.beta_beta,
            )
            noise = torch.randn_like(x0)
            x_t = make_xt(x0, noise, t)
            cond = None
            if text_encoder is not None:
                with torch.no_grad():
                    cond = text_encoder(batch["text"]).to(device)

            with _autocast_context(device, training_config.precision, training_config.autocast):
                x0_pred = model(x_t, t, cond=cond)
                loss_dict = loss_fn(x0_pred=x0_pred, x0=x0, x_t=x_t, noise=noise, t=t, step=step)
                loss = loss_dict["loss"] / accumulation_steps
            if scaler_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_step += 1
            if micro_step % accumulation_steps != 0:
                continue

            if scaler_enabled:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
            optimizer_stepped = True
            if scaler_enabled:
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= scale_before_step
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if not optimizer_stepped:
                logging.warning("skipped optimizer step because GradScaler detected fp16 overflow")
                continue

            scheduler.step()
            if ema is not None:
                ema.update(model)

            step += 1

            elapsed = max(time.time() - start, 1e-6)
            completed_since_start = max(step - start_step, 1)
            sec_per_step = elapsed / completed_since_start
            remaining_steps = max(training_config.max_steps - step, 0)
            eta = remaining_steps * sec_per_step
            images_per_step = training_config.batch_size * accumulation_steps
            images_per_sec = images_per_step / max(sec_per_step, 1e-12)
            loss_value = loss_dict["loss"].item()
            flow_loss_value = loss_dict["flow_loss"].item()
            x0_loss_value = loss_dict["x0_loss"].item()
            grad_norm_value = float(grad_norm)
            current_lr = optimizer.param_groups[0]["lr"]
            _write_live_progress(
                _format_training_progress(
                    step=step,
                    max_steps=training_config.max_steps,
                    epoch=epoch,
                    loss=loss_value,
                    flow_loss=flow_loss_value,
                    x0_loss=x0_loss_value,
                    lr=current_lr,
                    grad_norm=grad_norm_value,
                    sec_per_step=sec_per_step,
                    images_per_sec=images_per_sec,
                    elapsed=elapsed,
                    eta=eta,
                )
            )

            if step % max(training_config.log_every_steps, 1) == 0:
                _finish_live_progress()
                logging.info(
                    "stage=train progress=%.2f%% step=%d/%d epoch=%d loss=%.5f flow=%.5f x0=%.5f lpips=%.5f ms=%.5f freq=%.5f boundary=%.5f snr_w=%.3f lr=%.2e grad_norm=%.3f sec_per_step=%.3f images_per_sec=%.2f elapsed=%s eta=%s",
                    100.0 * step / max(training_config.max_steps, 1),
                    step,
                    training_config.max_steps,
                    epoch,
                    loss_value,
                    flow_loss_value,
                    x0_loss_value,
                    loss_dict["lpips_loss"].item(),
                    loss_dict["multiscale_loss"].item(),
                    loss_dict["frequency_loss"].item(),
                    loss_dict["boundary_loss"].item(),
                    loss_dict["min_snr_weight"].item(),
                    current_lr,
                    grad_norm_value,
                    sec_per_step,
                    images_per_sec,
                    format_duration(elapsed),
                    format_duration(eta),
                )

            if checkpoint_config.periodic_enabled and step > 0 and step % checkpoint_config.save_every_steps == 0:
                _finish_live_progress()
                periodic_checkpoint_path = checkpoint_dir / f"step_{step:08d}.pt"
                save_checkpoint(
                    periodic_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=ema,
                    scaler=scaler if scaler_enabled else None,
                    config=model_config,
                    configs=run_configs,
                    step=step,
                    epoch=epoch,
                    save_json_sidecar=checkpoint_config.save_json_sidecar,
                )
                logging.info("saved periodic checkpoint: %s", periodic_checkpoint_path)
                if checkpoint_config.save_safetensors:
                    periodic_safetensors_path = checkpoint_dir / _periodic_safetensors_name(step)
                    save_model_safetensors(
                        periodic_safetensors_path,
                        model=model,
                        ema=ema,
                        source=checkpoint_config.safetensors_source,
                        metadata={
                            "step": str(step),
                            "epoch": str(epoch),
                            "prediction_type": model_config.prediction_type,
                            "checkpoint": "periodic",
                        },
                        config=model_config,
                        configs=run_configs,
                        step=step,
                        epoch=epoch,
                        save_json_sidecar=checkpoint_config.save_json_sidecar,
                    )
                    logging.info("saved periodic safetensors: %s", periodic_safetensors_path)
                prune_checkpoints(checkpoint_dir, keep_last=checkpoint_config.keep_last)

            if sample_config.periodic_enabled and step > 0 and step % sample_config.sample_every_steps == 0:
                _finish_live_progress()
                sample_path = _save_training_sample(
                    model=model,
                    ema=ema,
                    model_config=model_config,
                    sample_config=sample_config,
                    text_encoder=text_encoder,
                    output=sample_dir / f"step_{step:08d}.png",
                    device=device,
                )
                logging.info("saved periodic sample: %s", sample_path)

            if step >= training_config.max_steps:
                break
        epoch += 1

    _finish_live_progress()
    final_checkpoint_path = checkpoint_dir / checkpoint_config.final_name
    save_checkpoint(
        final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        scaler=scaler if scaler_enabled else None,
        config=model_config,
        configs=run_configs,
        step=step,
        epoch=epoch,
        save_json_sidecar=checkpoint_config.save_json_sidecar,
    )
    logging.info("saved final checkpoint: %s", final_checkpoint_path)
    if checkpoint_config.save_safetensors:
        final_safetensors_path = checkpoint_dir / checkpoint_config.safetensors_name
        save_model_safetensors(
            final_safetensors_path,
            model=model,
            ema=ema,
            source=checkpoint_config.safetensors_source,
            metadata={
                "step": str(step),
                "epoch": str(epoch),
                "prediction_type": model_config.prediction_type,
            },
            config=model_config,
            configs=run_configs,
            step=step,
            epoch=epoch,
            save_json_sidecar=checkpoint_config.save_json_sidecar,
        )
        logging.info("saved final safetensors: %s", final_safetensors_path)
    final_sample_path = _save_training_sample(
        model=model,
        ema=ema,
        model_config=model_config,
        sample_config=sample_config,
        text_encoder=text_encoder,
        output=sample_dir / sample_config.final_name,
        device=device,
    )
    total_elapsed = time.time() - start
    logging.info("saved final sample: %s", final_sample_path)
    logging.info("training completed in %s", format_duration(total_elapsed))


if __name__ == "__main__":
    main()

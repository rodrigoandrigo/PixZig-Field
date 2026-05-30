from __future__ import annotations

import copy
import math
from typing import Any, Mapping, cast

import torch
import torch.nn as nn


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


class EMAModel:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        *,
        update_every: int = 1,
        warmup_steps: int = 0,
        min_decay: float = 0.0,
        device: str | torch.device | None = None,
    ) -> None:
        if not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be finite and in [0, 1]")
        if update_every < 1:
            raise ValueError("update_every must be at least 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not math.isfinite(min_decay) or not 0.0 <= min_decay <= decay:
            raise ValueError("min_decay must be finite and in [0, decay]")
        self.decay = decay
        self.update_every = update_every
        self.warmup_steps = warmup_steps
        self.min_decay = min_decay
        self.num_updates = 0
        self.num_update_calls = 0
        self.module = copy.deepcopy(model).eval()
        if device is not None:
            self.module.to(device)
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    def _current_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        progress = min(self.num_updates / self.warmup_steps, 1.0)
        return self.min_decay + (self.decay - self.min_decay) * progress

    @torch.no_grad()
    def update(self, model: nn.Module) -> bool:
        self.num_update_calls += 1
        if self.num_update_calls % self.update_every != 0:
            return False
        self.num_updates += 1
        decay = self._current_decay()
        ema_params = dict(self.module.named_parameters())
        for name, parameter in model.named_parameters():
            if name in ema_params:
                source = parameter.detach().to(device=ema_params[name].device, dtype=ema_params[name].dtype)
                ema_params[name].mul_(decay).add_(source, alpha=1.0 - decay)

        ema_buffers = dict(self.module.named_buffers())
        for name, buffer in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].copy_(buffer.to(device=ema_buffers[name].device, dtype=ema_buffers[name].dtype))
        return True

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.module.state_dict())

    def to(self, *args: Any, **kwargs: Any) -> "EMAModel":
        self.module.to(*args, **kwargs)
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "update_every": self.update_every,
            "warmup_steps": self.warmup_steps,
            "min_decay": self.min_decay,
            "num_updates": self.num_updates,
            "num_update_calls": self.num_update_calls,
            "device": str(_module_device(self.module)),
            "model": self.module.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self.update_every = int(state_dict.get("update_every", self.update_every))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.min_decay = float(state_dict.get("min_decay", self.min_decay))
        if not math.isfinite(self.decay) or not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be finite and in [0, 1]")
        if self.update_every < 1:
            raise ValueError("update_every must be at least 1")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not math.isfinite(self.min_decay) or not 0.0 <= self.min_decay <= self.decay:
            raise ValueError("min_decay must be finite and in [0, decay]")
        self.num_updates = int(state_dict.get("num_updates", self.num_updates))
        self.num_update_calls = int(state_dict.get("num_update_calls", self.num_updates * self.update_every))
        if self.num_updates < 0 or self.num_update_calls < 0:
            raise ValueError("EMA update counters must be non-negative")
        self.module.load_state_dict(cast(Mapping[str, Any], state_dict["model"]))

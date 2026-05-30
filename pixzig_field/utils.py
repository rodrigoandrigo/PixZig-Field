import math
from numbers import Real

import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def print_parameter_count(model: nn.Module) -> None:
    count = count_parameters(model)
    print(f"Trainable parameters: {count:,}")


def format_duration(seconds: float) -> str:
    if isinstance(seconds, bool) or not isinstance(seconds, Real):
        raise TypeError("seconds must be a real number")
    if not math.isfinite(float(seconds)):
        raise ValueError("seconds must be finite")
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"

import torch
import torch.nn as nn


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_image_tensor(name: str, value: torch.Tensor) -> tuple[int, int, int, int]:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating point tensor")
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape [B, C, H, W], got {tuple(value.shape)}")
    batch_size, channels, height, width = value.shape
    if batch_size < 1:
        raise ValueError(f"{name} must include a non-empty batch dimension")
    if channels < 1:
        raise ValueError(f"{name} must include at least one channel")
    if height < 1 or width < 1:
        raise ValueError(f"{name} spatial dimensions must be positive, got H={height}, W={width}")
    return batch_size, channels, height, width


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, cond_dim: int | None = None) -> None:
        super().__init__()
        channels = _positive_int("channels", channels)
        kernel_size = _positive_int("kernel_size", kernel_size)
        if cond_dim is not None:
            cond_dim = _positive_int("cond_dim", cond_dim)
        self.channels = channels
        self.cond_dim = cond_dim
        padding = kernel_size // 2
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.cond_proj = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, 4 * channels),
            )
            if cond_dim is not None
            else None
        )

    def _film(self, x: torch.Tensor, params: torch.Tensor | None, offset: int) -> torch.Tensor:
        if params is None:
            return x
        scale, shift = params[:, offset : offset + x.shape[1]], params[:, offset + x.shape[1] : offset + 2 * x.shape[1]]
        return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, channels, _, _ = _validate_image_tensor("x", x)
        if channels != self.channels:
            raise ValueError(f"x must have {self.channels} channels, got {channels}")
        if self.cond_proj is not None:
            if cond is None:
                raise ValueError("cond is required when cond_dim is configured")
            if not isinstance(cond, torch.Tensor):
                raise TypeError("cond must be a torch.Tensor")
            if not torch.is_floating_point(cond):
                raise TypeError("cond must be a floating point tensor")
            if cond.ndim != 2:
                raise ValueError(f"cond must have shape [B, D], got {tuple(cond.shape)}")
            if cond.shape[0] not in {1, batch_size}:
                raise ValueError(f"cond batch size must be 1 or match x batch size {batch_size}, got {cond.shape[0]}")
            if cond.shape[1] != self.cond_dim:
                raise ValueError(f"cond last dimension must be {self.cond_dim}, got {cond.shape[1]}")
            if not torch.isfinite(cond).all():
                raise ValueError("cond must contain only finite values")
        params = self.cond_proj(cond) if self.cond_proj is not None and cond is not None else None
        h = self._film(self.norm1(x), params, 0)
        h = self.conv1(torch.nn.functional.silu(h))
        h = self._film(self.norm2(h), params, 2 * x.shape[1])
        h = self.conv2(torch.nn.functional.silu(h))
        return x + h


class ResidualConvRefiner(nn.Module):
    def __init__(
        self,
        channels: int = 128,
        blocks: int = 3,
        in_channels: int = 3,
        out_channels: int = 3,
        kernel_size: int = 3,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        channels = _positive_int("channels", channels)
        blocks = _non_negative_int("blocks", blocks)
        in_channels = _positive_int("in_channels", in_channels)
        out_channels = _positive_int("out_channels", out_channels)
        kernel_size = _positive_int("kernel_size", kernel_size)
        if cond_dim is not None:
            cond_dim = _positive_int("cond_dim", cond_dim)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        padding = kernel_size // 2
        self.in_proj = nn.Conv2d(in_channels, channels, kernel_size, padding=padding)
        self.blocks = nn.ModuleList(_ResidualConvBlock(channels, kernel_size, cond_dim=cond_dim) for _ in range(blocks))
        self.out_norm = nn.GroupNorm(1, channels)
        self.out_proj = nn.Conv2d(channels, out_channels, kernel_size, padding=padding)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _validate_forward_inputs(
        self,
        x0_coarse: torch.Tensor,
        cond: torch.Tensor | None,
        x_t: torch.Tensor | None,
    ) -> None:
        batch_size, x0_channels, height, width = _validate_image_tensor("x0_coarse", x0_coarse)
        if x0_channels != self.out_channels:
            raise ValueError(f"x0_coarse must have {self.out_channels} channels, got {x0_channels}")

        total_channels = x0_channels
        if x_t is not None:
            x_t_batch, x_t_channels, x_t_height, x_t_width = _validate_image_tensor("x_t", x_t)
            if x_t_batch != batch_size:
                raise ValueError(f"x_t batch size must match x0_coarse batch size {batch_size}, got {x_t_batch}")
            if (x_t_height, x_t_width) != (height, width):
                raise ValueError(
                    f"x_t spatial shape must match x0_coarse, got {(x_t_height, x_t_width)} and {(height, width)}"
                )
            total_channels += x_t_channels

        if total_channels != self.in_channels:
            raise ValueError(f"refiner expected {self.in_channels} input channels, got {total_channels}")

        if self.cond_dim is None:
            if cond is not None:
                raise ValueError("cond was provided but cond_dim was not configured")
            return
        if cond is None:
            raise ValueError("cond is required when cond_dim is configured")
        if not isinstance(cond, torch.Tensor):
            raise TypeError("cond must be a torch.Tensor")
        if not torch.is_floating_point(cond):
            raise TypeError("cond must be a floating point tensor")
        if cond.ndim != 2:
            raise ValueError(f"cond must have shape [B, D], got {tuple(cond.shape)}")
        if cond.shape[0] not in {1, batch_size}:
            raise ValueError(
                f"cond batch size must be 1 or match x0_coarse batch size {batch_size}, got {cond.shape[0]}"
            )
        if cond.shape[1] != self.cond_dim:
            raise ValueError(f"cond last dimension must be {self.cond_dim}, got {cond.shape[1]}")
        if not torch.isfinite(cond).all():
            raise ValueError("cond must contain only finite values")

    def forward(
        self,
        x0_coarse: torch.Tensor,
        cond: torch.Tensor | None = None,
        x_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_forward_inputs(x0_coarse, cond, x_t)
        inputs = torch.cat([x0_coarse, x_t], dim=1) if x_t is not None else x0_coarse
        hidden = torch.nn.functional.silu(self.in_proj(inputs))
        for block in self.blocks:
            hidden = block(hidden, cond)
        hidden = torch.nn.functional.silu(self.out_norm(hidden))
        return self.out_proj(hidden)

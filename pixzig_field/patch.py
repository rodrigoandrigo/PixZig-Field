import torch
import torch.nn.functional as F


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_grid_size(grid_size: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(grid_size, tuple) or len(grid_size) != 2:
        raise TypeError("grid_size must be a tuple of two positive integers")
    grid_h = _positive_int("grid_size[0]", grid_size[0])
    grid_w = _positive_int("grid_size[1]", grid_size[1])
    return grid_h, grid_w


def validate_patchable(x: torch.Tensor, patch_size: int) -> tuple[int, int]:
    patch_size = _positive_int("patch_size", patch_size)
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 4:
        raise ValueError(f"expected x to have shape [B, C, H, W], got {tuple(x.shape)}")
    batch_size, channels, height, width = x.shape
    if batch_size < 1:
        raise ValueError("x must include a non-empty batch dimension")
    if channels < 1:
        raise ValueError("x must include at least one channel")
    if height < 1 or width < 1:
        raise ValueError(f"x spatial dimensions must be positive, got H={height}, W={width}")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"height and width must be divisible by patch_size={patch_size}; "
            f"got H={height}, W={width}"
        )
    return height // patch_size, width // patch_size


def patchify(x: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    grid_h, grid_w = validate_patchable(x, patch_size)
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    patches = patches.transpose(1, 2).contiguous()
    return patches, (grid_h, grid_w)


def unpatchify(
    patches: torch.Tensor,
    grid_size: tuple[int, int],
    patch_size: int,
    out_channels: int,
) -> torch.Tensor:
    patch_size = _positive_int("patch_size", patch_size)
    out_channels = _positive_int("out_channels", out_channels)
    if not isinstance(patches, torch.Tensor):
        raise TypeError("patches must be a torch.Tensor")
    if patches.ndim != 3:
        raise ValueError(
            f"expected patches to have shape [B, N, patch_dim], got {tuple(patches.shape)}"
        )
    if patches.shape[0] < 1:
        raise ValueError("patches must include a non-empty batch dimension")
    if patches.shape[1] < 1:
        raise ValueError("patches must include at least one patch token")
    if patches.shape[2] < 1:
        raise ValueError("patches must include a non-empty patch dimension")

    grid_h, grid_w = _validate_grid_size(grid_size)
    expected_tokens = grid_h * grid_w
    expected_dim = out_channels * patch_size * patch_size
    if patches.shape[1] != expected_tokens:
        raise ValueError(f"expected {expected_tokens} patches, got {patches.shape[1]}")
    if patches.shape[2] != expected_dim:
        raise ValueError(f"expected patch dim {expected_dim}, got {patches.shape[2]}")

    height, width = grid_h * patch_size, grid_w * patch_size
    patches = patches.transpose(1, 2).contiguous()
    return F.fold(patches, output_size=(height, width), kernel_size=patch_size, stride=patch_size)

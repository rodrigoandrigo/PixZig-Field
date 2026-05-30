from __future__ import annotations

import re
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_scan_mode(mode: str) -> int:
    if not isinstance(mode, str):
        raise TypeError("mode must be a string")
    match = re.fullmatch(r"zigzagN(\d+)", mode)
    if match is None:
        raise ValueError(f"unsupported scan mode: {mode}")
    count = int(match.group(1))
    if count < 1:
        raise ValueError("zigzag scan count must be positive")
    return count


def _index(row: int, col: int, grid_w: int) -> int:
    return row * grid_w + col


def _row_serpentine(
    grid_h: int,
    grid_w: int,
    *,
    top_down: bool,
    left_first: bool,
) -> list[int]:
    rows = range(grid_h) if top_down else range(grid_h - 1, -1, -1)
    path: list[int] = []
    for offset, row in enumerate(rows):
        forward = left_first if offset % 2 == 0 else not left_first
        cols = range(grid_w) if forward else range(grid_w - 1, -1, -1)
        path.extend(_index(row, col, grid_w) for col in cols)
    return path


def _col_serpentine(
    grid_h: int,
    grid_w: int,
    *,
    left_right: bool,
    top_first: bool,
) -> list[int]:
    cols = range(grid_w) if left_right else range(grid_w - 1, -1, -1)
    path: list[int] = []
    for offset, col in enumerate(cols):
        forward = top_first if offset % 2 == 0 else not top_first
        rows = range(grid_h) if forward else range(grid_h - 1, -1, -1)
        path.extend(_index(row, col, grid_w) for row in rows)
    return path


def _diagonal_serpentine(
    grid_h: int,
    grid_w: int,
    *,
    reverse_diagonals: bool,
    flip_inside: bool,
    anti: bool,
) -> list[int]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for row in range(grid_h):
        for col in range(grid_w):
            key = row + (grid_w - 1 - col if anti else col)
            groups.setdefault(key, []).append((row, col))

    keys = sorted(groups.keys(), reverse=reverse_diagonals)
    path: list[int] = []
    for offset, key in enumerate(keys):
        coords = sorted(groups[key])
        if (offset % 2 == 1) ^ flip_inside:
            coords.reverse()
        path.extend(_index(row, col, grid_w) for row, col in coords)
    return path


def invert_indices(indices: torch.Tensor) -> torch.Tensor:
    if not isinstance(indices, torch.Tensor):
        raise TypeError("indices must be a torch.Tensor")
    if indices.dtype != torch.long:
        raise TypeError("indices must have dtype torch.long")
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1D, got {tuple(indices.shape)}")
    if indices.numel() < 1:
        raise ValueError("indices must be non-empty")
    if torch.any((indices < 0) | (indices >= indices.numel())):
        raise ValueError("indices values must be in [0, len(indices))")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("indices must be a permutation without duplicates")
    inverse = torch.empty_like(indices)
    inverse[indices] = torch.arange(indices.numel(), device=indices.device)
    return inverse


@lru_cache(maxsize=128)
def _get_zigzag_indices_cached(grid_h: int, grid_w: int, mode: str) -> tuple[torch.Tensor, ...]:
    return tuple(_get_zigzag_indices_uncached(grid_h, grid_w, mode))


def get_zigzag_indices(
    grid_h: int,
    grid_w: int,
    mode: str = "zigzagN16",
) -> list[torch.Tensor]:
    grid_h = _positive_int("grid_h", grid_h)
    grid_w = _positive_int("grid_w", grid_w)
    _validate_scan_mode(mode)
    return [path.clone() for path in _get_zigzag_indices_cached(grid_h, grid_w, mode)]


def _get_zigzag_indices_uncached(
    grid_h: int,
    grid_w: int,
    mode: str,
) -> list[torch.Tensor]:
    count = _validate_scan_mode(mode)

    candidates = [
        _row_serpentine(grid_h, grid_w, top_down=True, left_first=True),
        _row_serpentine(grid_h, grid_w, top_down=True, left_first=False),
        _row_serpentine(grid_h, grid_w, top_down=False, left_first=True),
        _row_serpentine(grid_h, grid_w, top_down=False, left_first=False),
        _col_serpentine(grid_h, grid_w, left_right=True, top_first=True),
        _col_serpentine(grid_h, grid_w, left_right=True, top_first=False),
        _col_serpentine(grid_h, grid_w, left_right=False, top_first=True),
        _col_serpentine(grid_h, grid_w, left_right=False, top_first=False),
        _diagonal_serpentine(grid_h, grid_w, reverse_diagonals=False, flip_inside=False, anti=False),
        _diagonal_serpentine(grid_h, grid_w, reverse_diagonals=False, flip_inside=True, anti=False),
        _diagonal_serpentine(grid_h, grid_w, reverse_diagonals=True, flip_inside=False, anti=False),
        _diagonal_serpentine(grid_h, grid_w, reverse_diagonals=True, flip_inside=True, anti=False),
        _diagonal_serpentine(grid_h, grid_w, reverse_diagonals=False, flip_inside=False, anti=True),
        _diagonal_serpentine(grid_h, grid_w, reverse_diagonals=False, flip_inside=True, anti=True),
        _diagonal_serpentine(grid_h, grid_w, reverse_diagonals=True, flip_inside=False, anti=True),
        _diagonal_serpentine(grid_h, grid_w, reverse_diagonals=True, flip_inside=True, anti=True),
    ]

    num_tokens = grid_h * grid_w
    result: list[torch.Tensor] = []
    for path in candidates:
        tensor = torch.tensor(path, dtype=torch.long)
        if tensor.numel() != num_tokens:
            raise RuntimeError("zigzag path has invalid length")
        if torch.unique(tensor).numel() != num_tokens:
            raise RuntimeError("zigzag path is not a permutation")
        result.append(tensor)

    while len(result) < count:
        result.extend([path.flip(0) for path in result])
    return result[:count]


def _choose_head_dim(inner_dim: int, preferred_head_dim: int = 64) -> int:
    inner_dim = _positive_int("inner_dim", inner_dim)
    preferred_head_dim = _positive_int("preferred_head_dim", preferred_head_dim)
    head_dim = min(preferred_head_dim, inner_dim)
    while inner_dim % head_dim != 0:
        head_dim -= 1
    return head_dim


class _RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        dim = _positive_int("dim", dim)
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if x.shape != gate.shape:
            raise ValueError(f"x and gate must have matching shapes, got {tuple(x.shape)} and {tuple(gate.shape)}")
        if x.ndim < 1 or x.shape[-1] != self.weight.shape[0]:
            raise ValueError(f"x last dimension must be {self.weight.shape[0]}")
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        gated = normed.to(dtype=dtype) * F.silu(gate)
        return gated * self.weight.to(device=x.device, dtype=dtype)


class _NativeMambaMixer(nn.Module):
    """Portable Mamba mixer with causal conv and selective recurrent SSM.

    The equations mirror the non-fused Mamba/Mamba-2 path from the reference
    repos, but run in plain PyTorch so ROCm/CPU environments do not require
    Triton, TileLang, causal-conv1d, or mamba_ssm wheels. Unlike the previous
    lightweight block, the recurrent state is per inner channel and state
    dimension, with input-dependent B, C, dt and learned A/D terms.
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 128,
        *,
        inner_expand: int = 1,
        head_dim: int = 64,
        num_groups: int = 1,
        conv_kernel: int = 4,
    ) -> None:
        super().__init__()
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        state_dim = _positive_int("state_dim", state_dim)
        inner_expand = _positive_int("inner_expand", inner_expand)
        head_dim = _positive_int("head_dim", head_dim)
        num_groups = _positive_int("num_groups", num_groups)
        conv_kernel = _positive_int("conv_kernel", conv_kernel)

        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.inner_expand = inner_expand
        self.d_inner = hidden_dim * inner_expand
        self.head_dim = _choose_head_dim(self.d_inner, preferred_head_dim=head_dim)
        self.num_heads = self.d_inner // self.head_dim
        self.num_groups = min(num_groups, self.num_heads)
        while self.num_heads % self.num_groups != 0:
            self.num_groups -= 1
        self.conv_kernel = conv_kernel

        in_proj_dim = 2 * self.d_inner + 2 * self.num_groups * state_dim + self.num_heads
        self.in_proj = nn.Linear(hidden_dim, in_proj_dim, bias=False)
        conv_dim = self.d_inner + 2 * self.num_groups * state_dim
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=conv_dim,
        )
        self.norm = _RMSNormGated(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, hidden_dim, bias=False)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.num_heads + 1, dtype=torch.float32)))
        self.dt_bias = nn.Parameter(torch.full((self.num_heads,), -4.6))
        self.D = nn.Parameter(torch.ones(self.num_heads))
        head_to_group = torch.div(
            torch.arange(self.num_heads, dtype=torch.long) * self.num_groups,
            self.num_heads,
            rounding_mode="floor",
        )
        self.register_buffer("head_to_group", head_to_group, persistent=False)

    def _selective_scan(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        b_param: torch.Tensor,
        c_param: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        batch, seqlen, num_heads, head_dim = x.shape
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError("invalid Mamba head layout")

        scan_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x_scan = x.to(scan_dtype)
        b_scan = b_param.to(scan_dtype)
        c_scan = c_param.to(scan_dtype)
        dt_scan = F.softplus(dt.to(scan_dtype) + self.dt_bias.to(scan_dtype))
        a = -torch.exp(self.A_log.to(scan_dtype)).view(1, self.num_heads, 1, 1)
        d = self.D.to(scan_dtype).view(1, self.num_heads, 1)
        head_to_group = self.head_to_group.to(device=x.device)

        state = torch.zeros(
            batch,
            self.num_heads,
            self.head_dim,
            self.state_dim,
            device=x.device,
            dtype=scan_dtype,
        )
        outputs = x_scan.new_empty(batch, seqlen, self.num_heads, self.head_dim)
        for step in range(seqlen):
            dt_step = dt_scan[:, step]
            b_step = b_scan[:, step].index_select(1, head_to_group)
            c_step = c_scan[:, step].index_select(1, head_to_group)
            x_step = x_scan[:, step]

            decay = torch.exp(a * dt_step[:, :, None, None])
            drive = x_step.unsqueeze(-1) * b_step[:, :, None, :] * dt_step[:, :, None, None]
            state = state * decay + drive
            y_step = (state * c_step[:, :, None, :]).sum(dim=-1)
            y_step = y_step + d * x_step
            outputs[:, step] = y_step

        y = outputs.reshape(batch, seqlen, self.d_inner).to(dtype=x.dtype)
        return self.out_proj(self.norm(y, z))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if not torch.is_floating_point(tokens):
            raise TypeError("tokens must be a floating point tensor")
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B, N, C], got {tuple(tokens.shape)}")
        if tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("tokens must have non-empty batch and sequence dimensions")
        if tokens.shape[2] != self.hidden_dim:
            raise ValueError(f"tokens last dimension must be {self.hidden_dim}, got {tokens.shape[2]}")
        batch, seqlen, _ = tokens.shape
        projected = self.in_proj(tokens)
        z, xbc, dt = torch.split(
            projected,
            [self.d_inner, self.d_inner + 2 * self.num_groups * self.state_dim, self.num_heads],
            dim=-1,
        )
        xbc = self.conv1d(xbc.transpose(1, 2))[..., :seqlen].transpose(1, 2)
        xbc = F.silu(xbc)
        x, b_param, c_param = torch.split(
            xbc,
            [self.d_inner, self.num_groups * self.state_dim, self.num_groups * self.state_dim],
            dim=-1,
        )
        x_heads = x.reshape(batch, seqlen, self.num_heads, self.head_dim)
        b_heads = b_param.reshape(batch, seqlen, self.num_groups, self.state_dim)
        c_heads = c_param.reshape(batch, seqlen, self.num_groups, self.state_dim)
        return self._selective_scan(x_heads, z, b_heads, c_heads, dt)


class _ExternalMambaMixer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        *,
        inner_expand: int,
        head_dim: int,
        num_groups: int,
        conv_kernel: int,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm.modules.mamba2_simple import Mamba2Simple
        except Exception as exc:  # pragma: no cover - depends on optional native build
            raise ImportError(
                "mamba_backend='external' requires an installed mamba_ssm build. "
                "Use mamba_backend='native' for the portable PyTorch backend."
            ) from exc

        inner_dim = hidden_dim * inner_expand
        chosen_head_dim = _choose_head_dim(inner_dim, preferred_head_dim=head_dim)
        chosen_groups = min(num_groups, inner_dim // chosen_head_dim)
        while (inner_dim // chosen_head_dim) % chosen_groups != 0:
            chosen_groups -= 1
        self.impl = Mamba2Simple(
            d_model=hidden_dim,
            d_state=state_dim,
            d_conv=conv_kernel,
            expand=inner_expand,
            headdim=chosen_head_dim,
            ngroups=chosen_groups,
            use_mem_eff_path=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.impl(tokens)


def _build_mamba_mixer(
    *,
    hidden_dim: int,
    state_dim: int,
    backend: str,
    inner_expand: int,
    head_dim: int,
    num_groups: int,
    conv_kernel: int,
) -> nn.Module:
    if backend == "native":
        return _NativeMambaMixer(
            hidden_dim,
            state_dim,
            inner_expand=inner_expand,
            head_dim=head_dim,
            num_groups=num_groups,
            conv_kernel=conv_kernel,
        )
    if backend == "external":
        return _ExternalMambaMixer(
            hidden_dim,
            state_dim,
            inner_expand=inner_expand,
            head_dim=head_dim,
            num_groups=num_groups,
            conv_kernel=conv_kernel,
        )
    if backend == "auto":
        try:
            return _ExternalMambaMixer(
                hidden_dim,
                state_dim,
                inner_expand=inner_expand,
                head_dim=head_dim,
                num_groups=num_groups,
                conv_kernel=conv_kernel,
            )
        except ImportError:
            return _NativeMambaMixer(
                hidden_dim,
                state_dim,
                inner_expand=inner_expand,
                head_dim=head_dim,
                num_groups=num_groups,
                conv_kernel=conv_kernel,
            )
    raise ValueError("backend must be 'native', 'external', or 'auto'")


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if shift.shape != scale.shape:
        raise ValueError(f"shift and scale must have matching shapes, got {tuple(shift.shape)} and {tuple(scale.shape)}")
    if x.ndim != 3 or shift.ndim != 2 or shift.shape[0] not in {1, x.shape[0]} or x.shape[2] != shift.shape[1]:
        raise ValueError(
            f"expected x [B, N, C] and shift/scale [B, C], got {tuple(x.shape)} and {tuple(shift.shape)}"
        )
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _ZigZagMambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        *,
        backend: str = "native",
        inner_expand: int = 1,
        head_dim: int = 64,
        num_groups: int = 1,
        conv_kernel: int = 4,
    ) -> None:
        super().__init__()
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        state_dim = _positive_int("state_dim", state_dim)
        self.state_dim = state_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 3 * hidden_dim))
        self.mixer = _build_mamba_mixer(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            backend=backend,
            inner_expand=inner_expand,
            head_dim=head_dim,
            num_groups=num_groups,
            conv_kernel=conv_kernel,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        cond: torch.Tensor,
        indices: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        if indices.ndim != 1 or inverse_indices.ndim != 1:
            raise ValueError("indices and inverse_indices must be 1D")
        if indices.numel() != tokens.shape[1] or inverse_indices.numel() != tokens.shape[1]:
            raise ValueError("indices and inverse_indices must match token sequence length")
        scanned = tokens.index_select(1, indices)
        shift, scale, gate = self.adaLN_modulation(cond).chunk(3, dim=-1)
        mixed = self.mixer(_modulate(self.norm(scanned), shift, scale))
        scanned = scanned + gate.unsqueeze(1) * mixed
        return scanned.index_select(1, inverse_indices)


class ZigMaBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        state_dim: int = 128,
        expand: int = 2,
        scan_mode: str = "zigzagN16",
        cond_dim: int | None = None,
        mamba_backend: str = "native",
        mamba_inner_expand: int = 1,
        mamba_head_dim: int = 64,
        mamba_num_groups: int = 1,
        mamba_conv_kernel: int = 4,
    ) -> None:
        super().__init__()
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        depth = _positive_int("depth", depth)
        state_dim = _positive_int("state_dim", state_dim)
        expand = _positive_int("expand", expand)
        _validate_scan_mode(scan_mode)
        if cond_dim is not None:
            cond_dim = _positive_int("cond_dim", cond_dim)
        if mamba_backend not in {"native", "external", "auto"}:
            raise ValueError("mamba_backend must be 'native', 'external', or 'auto'")
        mamba_inner_expand = _positive_int("mamba_inner_expand", mamba_inner_expand)
        mamba_head_dim = _positive_int("mamba_head_dim", mamba_head_dim)
        mamba_num_groups = _positive_int("mamba_num_groups", mamba_num_groups)
        mamba_conv_kernel = _positive_int("mamba_conv_kernel", mamba_conv_kernel)
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.scan_mode = scan_mode
        self.cond_proj = nn.Linear(cond_dim, hidden_dim) if cond_dim is not None else None
        self.blocks = nn.ModuleList(
            [
                _ZigZagMambaBlock(
                    hidden_dim,
                    state_dim=state_dim * expand,
                    backend=mamba_backend,
                    inner_expand=mamba_inner_expand,
                    head_dim=mamba_head_dim,
                    num_groups=mamba_num_groups,
                    conv_kernel=mamba_conv_kernel,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self._path_cache: dict[tuple[int, int, str, torch.device], tuple[list[torch.Tensor], list[torch.Tensor]]] = {}

    def _get_device_paths(
        self,
        grid_size: tuple[int, int],
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        grid_h, grid_w = grid_size
        _positive_int("grid_size[0]", grid_h)
        _positive_int("grid_size[1]", grid_w)
        key = (grid_size[0], grid_size[1], self.scan_mode, device)
        cached = self._path_cache.get(key)
        if cached is not None:
            return cached
        paths = get_zigzag_indices(*grid_size, mode=self.scan_mode)
        device_paths = [path.to(device=device) for path in paths]
        inverse_paths = [invert_indices(path) for path in device_paths]
        cached = (device_paths, inverse_paths)
        self._path_cache[key] = cached
        return cached

    def forward(
        self,
        tokens: torch.Tensor,
        t_embed: torch.Tensor,
        cond: torch.Tensor | None = None,
        grid_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if not torch.is_floating_point(tokens):
            raise TypeError("tokens must be a floating point tensor")
        if tokens.ndim != 3:
            raise ValueError(f"expected tokens [B, N, C], got {tuple(tokens.shape)}")
        batch_size, num_tokens, hidden_dim = tokens.shape
        if batch_size < 1 or num_tokens < 1:
            raise ValueError("tokens must have non-empty batch and sequence dimensions")
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"tokens last dimension must be {self.hidden_dim}, got {hidden_dim}")
        if not isinstance(t_embed, torch.Tensor):
            raise TypeError("t_embed must be a torch.Tensor")
        if not torch.is_floating_point(t_embed):
            raise TypeError("t_embed must be a floating point tensor")
        if t_embed.ndim != 2:
            raise ValueError(f"t_embed must have shape [B, C], got {tuple(t_embed.shape)}")
        if t_embed.shape[0] not in {1, batch_size}:
            raise ValueError(f"t_embed batch size must be 1 or match tokens batch size {batch_size}")
        if t_embed.shape[1] != self.hidden_dim:
            raise ValueError(f"t_embed last dimension must be {self.hidden_dim}, got {t_embed.shape[1]}")
        if grid_size is None:
            side = int(tokens.shape[1] ** 0.5)
            if side * side != tokens.shape[1]:
                raise ValueError("grid_size is required for non-square token grids")
            grid_size = (side, side)
        else:
            if not isinstance(grid_size, tuple) or len(grid_size) != 2:
                raise TypeError("grid_size must be a tuple of two positive integers")
            grid_h = _positive_int("grid_size[0]", grid_size[0])
            grid_w = _positive_int("grid_size[1]", grid_size[1])
            if grid_h * grid_w != num_tokens:
                raise ValueError(f"grid_size expects {grid_h * grid_w} tokens, got {num_tokens}")

        cond_vec = t_embed
        if cond is not None:
            if self.cond_proj is None:
                raise ValueError("cond was provided but cond_dim was not configured")
            if not isinstance(cond, torch.Tensor):
                raise TypeError("cond must be a torch.Tensor")
            if not torch.is_floating_point(cond):
                raise TypeError("cond must be a floating point tensor")
            if cond.ndim not in {2, 3}:
                raise ValueError(f"cond must have shape [B, D] or [B, L, D], got {tuple(cond.shape)}")
            if cond.shape[0] not in {1, batch_size}:
                raise ValueError(f"cond batch size must be 1 or match tokens batch size {batch_size}, got {cond.shape[0]}")
            if cond.ndim == 3 and cond.shape[1] < 1:
                raise ValueError("cond sequence length must be non-empty")
            if cond.shape[-1] != self.cond_proj.in_features:
                raise ValueError(f"cond last dimension must be {self.cond_proj.in_features}, got {cond.shape[-1]}")
            if cond.ndim == 3:
                cond = cond.mean(dim=1)
            cond_vec = cond_vec + self.cond_proj(cond)

        device_paths, inverse_paths = self._get_device_paths(grid_size, tokens.device)

        hidden = tokens
        for layer_idx, block in enumerate(self.blocks):
            path_idx = layer_idx % len(device_paths)
            hidden = block(hidden, cond_vec, device_paths[path_idx], inverse_paths[path_idx])
        return self.final_norm(hidden)

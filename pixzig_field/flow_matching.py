import torch


def _expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if t.ndim == 0:
        t = t.expand(x.shape[0])
    if t.ndim != 1:
        raise ValueError(f"t must be a scalar or 1D tensor, got shape {tuple(t.shape)}")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if t.shape[0] != x.shape[0]:
        raise ValueError(f"t batch size must match x batch size: {t.shape[0]} != {x.shape[0]}")
    return t.reshape(t.shape[0], *([1] * (x.ndim - 1))).to(device=x.device, dtype=x.dtype)


def sample_timesteps(
    batch_size: int,
    device: torch.device | str,
    eps: float = 1e-5,
    mode: str = "uniform",
    logit_normal_std: float = 1.0,
    beta_alpha: float = 2.0,
    beta_beta: float = 2.0,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0.0 <= eps < 0.5:
        raise ValueError("eps must be in [0, 0.5)")
    if logit_normal_std <= 0:
        raise ValueError("logit_normal_std must be positive")
    if beta_alpha <= 0 or beta_beta <= 0:
        raise ValueError("beta_alpha and beta_beta must be positive")

    if mode == "uniform":
        t = torch.rand(batch_size, device=device)
    elif mode == "logit_normal":
        logits = torch.randn(batch_size, device=device) * logit_normal_std
        t = torch.sigmoid(logits)
    elif mode == "cosine":
        u = torch.rand(batch_size, device=device)
        t = 0.5 - 0.5 * torch.cos(torch.pi * u)
    elif mode == "beta":
        dist = torch.distributions.Beta(
            torch.tensor(beta_alpha, device=device),
            torch.tensor(beta_beta, device=device),
        )
        t = dist.sample((batch_size,))
    elif mode == "stratified":
        bins = (torch.arange(batch_size, device=device, dtype=torch.float32) + torch.rand(batch_size, device=device))
        t = bins / max(batch_size, 1)
        t = t[torch.randperm(batch_size, device=device)]
    else:
        raise ValueError("mode must be 'uniform', 'logit_normal', 'cosine', 'beta', or 'stratified'")
    return t.clamp(eps, 1.0 - eps)


def make_xt(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if x0.shape != noise.shape:
        raise ValueError(f"x0 and noise must have matching shapes, got {tuple(x0.shape)} and {tuple(noise.shape)}")
    t_expanded = _expand_t(t, x0)
    return (1.0 - t_expanded) * noise + t_expanded * x0


def velocity_target(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
    if x0.shape != noise.shape:
        raise ValueError(f"x0 and noise must have matching shapes, got {tuple(x0.shape)} and {tuple(noise.shape)}")
    return x0 - noise


def x0_from_velocity(x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if x_t.shape != v.shape:
        raise ValueError(f"x_t and v must have matching shapes, got {tuple(x_t.shape)} and {tuple(v.shape)}")
    t_expanded = _expand_t(t, x_t)
    return x_t + (1.0 - t_expanded) * v


def velocity_from_x0(
    x_t: torch.Tensor,
    x0_pred: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    if x_t.shape != x0_pred.shape:
        raise ValueError(f"x_t and x0_pred must have matching shapes, got {tuple(x_t.shape)} and {tuple(x0_pred.shape)}")
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    one_minus_t = (1.0 - t).clamp(min=eps, max=1.0)
    one_minus_t_expanded = _expand_t(one_minus_t, x_t)
    return (x0_pred - x_t) / one_minus_t_expanded

from __future__ import annotations

import torch
from torch.nn import functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(pred.float(), target.float())
    return 20.0 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10.0 * torch.log10(mse + eps)


def ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, window_size: int = 11) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    padding = window_size // 2
    channels = pred.shape[1]
    weight = torch.ones(channels, 1, window_size, window_size, device=pred.device) / (window_size * window_size)
    mu_x = F.conv2d(pred, weight, padding=padding, groups=channels)
    mu_y = F.conv2d(target, weight, padding=padding, groups=channels)
    sigma_x = F.conv2d(pred * pred, weight, padding=padding, groups=channels) - mu_x.square()
    sigma_y = F.conv2d(target * target, weight, padding=padding, groups=channels) - mu_y.square()
    sigma_xy = F.conv2d(pred * target, weight, padding=padding, groups=channels) - mu_x * mu_y
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean()


def lpips_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    try:
        import lpips
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install with `pip install -e .[metrics]` to compute LPIPS.") from exc
    model = lpips.LPIPS(net="alex").to(pred.device)
    pred_norm = pred.float() * 2.0 - 1.0
    target_norm = target.float() * 2.0 - 1.0
    return model(pred_norm, target_norm).mean()


def trajectory_confusion_matrix(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Build M in R^{T x T} for point-light trajectory precision analysis.

    `pred[i]` is compared against every `target[j]` by per-image MSE. Lower is
    better. The diagonal is the intended trajectory alignment.
    """
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError("pred and target must have shape [T, C, H, W]")
    if pred.shape[1:] != target.shape[1:]:
        raise ValueError(f"Shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    diff = pred[:, None].float() - target[None].float()
    return diff.square().mean(dim=(2, 3, 4))


def precision_scores(confusion: torch.Tensor) -> dict[str, float]:
    """Return diagonal error A and off-diagonal sensitivity B/A."""
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be square")
    diagonal = torch.diagonal(confusion)
    a = diagonal.mean()
    off_diag = confusion[~torch.eye(confusion.shape[0], dtype=torch.bool, device=confusion.device)]
    b = off_diag.mean()
    return {
        "diagonal_error": float(a.detach().cpu()),
        "off_diagonal_error": float(b.detach().cpu()),
        "sensitivity_b_over_a": float((b / (a + 1e-8)).detach().cpu()),
    }

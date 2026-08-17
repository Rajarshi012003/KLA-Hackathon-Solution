from __future__ import annotations

import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1) Charbonnier loss  (robust L1-like, edge-preserving)
def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred {tuple(pred.shape)} and target {tuple(target.shape)} "
            "must match."
        )
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


class CharbonnierLoss(nn.Module):
    """Module wrapper around :func:`charbonnier_loss` (weighted anchor)."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return charbonnier_loss(pred, target, eps=self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"

def _gaussian_window(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = (
        torch.arange(window_size, device=device, dtype=dtype)
        - (window_size - 1) / 2.0
    )
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()                     # normalize 1-D kernel to sum == 1
    _1d = g.view(1, -1)                 # (1, w)
    _2d = _1d.t() @ _1d                 # (w, w) outer product (unit-sum)
    return _2d.view(1, 1, window_size, window_size)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")
    if pred.shape != target.shape:
        raise ValueError("pred/target shape mismatch in SSIM.")

    channel = pred.size(1)
    window = (
        _gaussian_window(window_size, sigma, pred.device, pred.dtype)
        .expand(channel, 1, window_size, window_size)
        .contiguous()
    )
    pad = window_size // 2

    # Local statistics via grouped convolution (one window per channel).
    mu1 = F.conv2d(pred, window, padding=pad, groups=channel)
    mu2 = F.conv2d(target, window, padding=pad, groups=channel)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channel) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.02 * data_range) ** 2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return (numerator / denominator).mean()


class SSIMLoss(nn.Module):
    def __init__(
        self,
        data_range: float = 1.0,
        window_size: int = 11,
        sigma: float = 1.5,
    ) -> None:
        super().__init__()
        self.data_range = data_range
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(
            pred,
            target,
            data_range=self.data_range,
            window_size=self.window_size,
            sigma=self.sigma,
        )

    def extra_repr(self) -> str:
        return (f"data_range={self.data_range}, window_size={self.window_size}, "
                f"sigma={self.sigma}")

class LPIPSLoss(nn.Module):
    def __init__(self, net: str = "vgg", grayscale_input: bool = True) -> None:
        super().__init__()
        self.net = net
        self.grayscale_input = grayscale_input
        self.lpips_model: Optional[nn.Module] = None  # lazy submodule
        self._init_error: Optional[str] = None
        self._warned_missing = False

    def _ensure_model(self, device: torch.device) -> Optional[nn.Module]:
        """Create (once) and move the frozen LPIPS backbone onto ``device``."""
        if self.lpips_model is None:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(
                    pretrained=True, net=self.net, verbose=False
                )
            except Exception as exc: 
                self._init_error = str(exc)
                self.lpips_model = None
                return None

        if next(self.lpips_model.parameters()).device != device:
            self.lpips_model = self.lpips_model.to(device)

        self.lpips_model.eval()
        for p in self.lpips_model.parameters():
            p.requires_grad_(False)
        return self.lpips_model

    def _to_vgg_input(self, t: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) in [0,1] -> (B,3,H,W) in [-1,1] for the VGG backbone."""
        if self.grayscale_input:
            t = t.repeat(1, 3, 1, 1)       # broadcast grayscale -> RGB
        return t * 2.0 - 1.0               # [0, 1] -> [-1, 1]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("pred/target shape mismatch in LPIPS.")

        model = self._ensure_model(pred.device)
        if model is None:
            if not self._warned_missing:
                self._warned_missing = True
                warnings.warn(
                    "LPIPS unavailable; returning 0 loss. "
                    f"Error: {self._init_error}"
                )
            return pred.sum() * 0.0

        x0 = self._to_vgg_input(pred)      # (B, 3, H, W) in [-1, 1]
        x1 = self._to_vgg_input(target)
        val = model(x0, x1)                # (B, 1, 1, 1) -> mean -> scalar
        return val.mean()

    def extra_repr(self) -> str:
        return f"net={self.net}, grayscale_input={self.grayscale_input}"

class CompositeLoss(nn.Module):
    def __init__(
        self,
        lambda_char: float = 1.0,
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.01,
        charbonnier_eps: float = 1e-3,
        ssim_data_range: float = 1.0,
        ssim_window_size: int = 11,
        ssim_sigma: float = 1.5,
        lpips_net: str = "vgg",
        use_lpips: bool = True,
        grayscale_input: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_char = lambda_char
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips

        self.charbonnier = CharbonnierLoss(eps=charbonnier_eps)
        self.ssim_loss = SSIMLoss(
            data_range=ssim_data_range,
            window_size=ssim_window_size,
            sigma=ssim_sigma,
        )
        self.lpips = (
            LPIPSLoss(net=lpips_net, grayscale_input=grayscale_input)
            if use_lpips
            else None
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the total weighted loss as a scalar tensor."""
        return self.breakdown(pred, target)["total"]

    def breakdown(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        lpips_val = (
            self.lpips(pred, target) if self.lpips is not None else pred.sum() * 0.0
        )
        terms = {
            "charbonnier": self.lambda_char * self.charbonnier(pred, target),
            "ssim": self.lambda_ssim * self.ssim_loss(pred, target),
            "lpips": self.lambda_lpips * lpips_val,
        }
        terms["total"] = sum(terms.values())
        return terms

    def extra_repr(self) -> str:
        return (f"lambda_char={self.lambda_char}, lambda_ssim={self.lambda_ssim}, "
                f"lambda_lpips={self.lambda_lpips}")

def build_composite_loss(**kwargs) -> CompositeLoss:
    """Return a fully-configured :class:`CompositeLoss`."""
    return CompositeLoss(**kwargs)

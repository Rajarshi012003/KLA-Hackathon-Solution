"""
train.py
========
1. AdamW optimizer + CosineAnnealingLR scheduler.
2. Automatic Mixed Precision (AMP) via ``torch.amp.autocast('cuda')`` and
   ``torch.cuda.amp.GradScaler()`` (T4/half-precision, speed).
3. ``clip_grad_norm_`` BEFORE the optimizer step (unbounded speckle spikes).
4. CRITICAL clamping rule: training loss on RAW (unclamped) output so gradients
   flow freely; in the ``torch.no_grad()`` VALIDATION loop we clamp
   ``out = torch.clamp(out, 0.0, 1.0)`` before computing PSNR/SSIM.
5. Epoch-averaged PSNR/SSIM; save ``best_model.pt`` ONLY when
   ``score = PSNR + SSIM * 100`` improves.

Run:
    python train.py --epochs 100 --batch_size 32 --lr 1e-4 \
        --lr_dir train/train/NoisyLR --gt_dir train/train/GT
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from dataset import build_dataloaders
from loss import build_composite_loss, ssim as ssim_fn
from model import NAFNetSR


def set_seed(seed: int) -> None:
    """Seed every random source for a deterministic, reproducible run."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio in dB computed per image and averaged over the batch.
    ``pred`` and ``target`` must have shape (B, C, H, W).
    """
    mse_per_img = ((pred - target) ** 2).flatten(start_dim=1).mean(dim=1)
    eps = 1e-10
    psnr_per_img = 10.0 * torch.log10((data_range ** 2) / (mse_per_img + eps))
    psnr_per_img = torch.clamp(psnr_per_img, max=100.0)
    return float(psnr_per_img.mean().item())


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Structural Similarity in ``[0, 1]`` (reuse our dependency-free loss impl)."""
    with torch.no_grad():
        return float(ssim_fn(pred, target, data_range=data_range).item())


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model. Returns epoch-averaged (psnr_db, ssim).
    """
    model.eval()
    psnr_acc = 0.0
    ssim_acc = 0.0
    n_total = 0

    with torch.no_grad():
        for lr, gt in val_loader:
            lr = lr.to(device)
            gt = gt.to(device)

            out = model(lr)                       # (B, 1, 2H, 2W)
            out = torch.clamp(out, 0.0, 1.0)
            out = out.detach()
            gt = gt.detach()

            b = lr.size(0)
            psnr_acc += compute_psnr(out, gt) * b
            ssim_acc += compute_ssim(out, gt) * b
            n_total += b

    if n_total == 0:
        return 0.0, 0.0
    return psnr_acc / n_total, ssim_acc / n_total


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
) -> float:
    """Run a single training epoch. Returns the mean loss."""
    model.train()
    loss_acc = 0.0
    n_total = 0

    for lr, gt in train_loader:
        lr = lr.to(device)
        gt = gt.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(lr)                  # RAW (unclamped in train mode)
                loss_val = criterion(out, gt)
            scaler.scale(loss_val).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(lr)
            loss_val = criterion(out, gt)
            loss_val.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        b = lr.size(0)
        loss_acc += float(loss_val.detach().item()) * b
        n_total += b

    return loss_acc / max(n_total, 1)


def save_checkpoint(path: str, model: nn.Module) -> None:
    """Best-model weights only (keeps checkpoints tiny for storage)."""
    torch.save(model.state_dict(), path)


def save_last(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_score: float,
) -> None:
    """Snapshot for resuming mid-training."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
        },
        path,
    )


def load_last(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> Tuple[int, float]:
    """Load a resume snapshot; returns (last_epoch, best_score)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    epoch = int(ckpt.get("epoch", -1))
    best_score = float(ckpt.get("best_score", float("-inf")))
    return epoch, best_score


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SR-NAFNet for KLA 2026")
    p.add_argument("--lr_dir", default="train/NoisyLR")
    p.add_argument("--gt_dir", default="train/GT")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--resume", default=None, help="path to checkpoint_last.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--lr_patch_size", type=int, default=64)
    p.add_argument("--patches_per_image", type=int, default=1)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--lambda_char", type=float, default=1.0)
    p.add_argument("--lambda_ssim", type=float, default=0.2)
    p.add_argument("--lambda_lpips", type=float, default=0.01)
    p.add_argument("--use_lpips", action="store_true", default=True)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[setup] device={device}  AMP={use_amp}  workers={args.num_workers}")

    train_loader, val_loader, (train_files, val_files) = build_dataloaders(
        lr_dir=args.lr_dir,
        gt_dir=args.gt_dir,
        val_split=args.val_split,
        seed=args.seed,
        scale=args.scale,
        lr_patch_size=args.lr_patch_size,
        patches_per_image=args.patches_per_image,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
    )
    print(f"[data] train={len(train_files)} val={len(val_files)} "
          f"train_steps={len(train_loader)}")

    #  model / loss / optimizer / scheduler 
    model = NAFNetSR(
        in_ch=1, out_ch=1, width=args.width,
        num_blocks=args.num_blocks, scale=args.scale,
    ).to(device)
    print(f"[model] params={model.num_params:,}")

    criterion = build_composite_loss(
        lambda_char=args.lambda_char,
        lambda_ssim=args.lambda_ssim,
        lambda_lpips=args.lambda_lpips,
        use_lpips=args.use_lpips,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # (optional)
    start_epoch, best_score = 0, float("-inf")
    if args.resume:
        start_epoch, best_score = load_last(
            args.resume, model, optimizer, scheduler, scaler, device
        )
        start_epoch += 1
        print(f"[resume] continuing from epoch {start_epoch} (best={best_score:.3f})")

    # training 
    last_path = os.path.join(args.out_dir, "last.pt")
    best_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model, criterion=criterion, train_loader=train_loader,
            optimizer=optimizer, scaler=scaler, device=device,
            grad_clip=args.grad_clip, use_amp=use_amp,
        )

        psnr_v, ssim_v = validate(model, val_loader, device)
        score = psnr_v + 100.0 * ssim_v
        lr_now = float(optimizer.param_groups[0]["lr"])

        improved = bool(score > best_score)
        if improved:
            best_score = score
            save_checkpoint(best_path, model)

        save_last(last_path, model, optimizer, scheduler, scaler, epoch, best_score)
        scheduler.step()

        print(
            f"[epoch {epoch + 1:3d}/{args.epochs}] loss={train_loss:.5f} "
            f"PSNR={psnr_v:6.3f} SSIM={ssim_v:5.4f} score={score:6.3f} "
            f"lr={lr_now:.2e} time={time.time() - t0:5.1f}s "
            f"{'*best*' if improved else ''}"
        )

    print(f"[done] best PSNR + 100*SSIM = {best_score:.3f}  "
          f"saved -> {best_path}")


if __name__ == "__main__":
    main()

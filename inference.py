from __future__ import annotations

import argparse
import os
import time
import numpy as np
import torch

from model import NAFNetSR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KLA Inference Pipeline")
    p.add_argument("--input_dir", type=str, required=True, help="Path to degraded LR .npy directory")
    p.add_argument("--output_dir", type=str, required=True, help="Path to save restored .npy images")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model weights")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    model = NAFNetSR(
        in_ch=1,
        out_ch=1,
        width=args.width,
        num_blocks=args.num_blocks,
        scale=args.scale,
    ).to(device)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    filenames = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".npy")])
    if not filenames:
        raise ValueError(f"No .npy files found in {args.input_dir}")

    print(f"[Inference] Found {len(filenames)} images. Running on {device}...")
    t0 = time.time()

    with torch.no_grad():
        for f in filenames:
            in_path = os.path.join(args.input_dir, f)
            out_path = os.path.join(args.output_dir, f)

            lr_arr = np.ascontiguousarray(np.load(in_path), dtype=np.float32)
            lr_tensor = torch.from_numpy(lr_arr).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

            sr_tensor = model(lr_tensor)
            sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)

            sr_arr = sr_tensor.squeeze().cpu().numpy().astype(np.float32)

            np.save(out_path, sr_arr)

    elapsed = time.time() - t0
    fps = len(filenames) / max(elapsed, 1e-5)
    print(f"[Inference Complete] Processed {len(filenames)} images in {elapsed:.2f}s ({fps:.1f} img/s).")


if __name__ == "__main__":
    main()

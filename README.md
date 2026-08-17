# KLA-Hackathon-Solution
### AI-Based Restoration of Degraded Images for Semiconductor Inspection

**Hackathon 2026 — Organized as part of SEMICON India**
**Problem Statement:** KLA — *AI-Based Restoration of Degraded Images for Semiconductor Inspection*

An end-to-end, reproducible deep-learning pipeline that takes a **degraded** (noisy,
low-resolution) semiconductor inspection image and reconstructs a **clean** image at the
**original ground-truth (GT) resolution** — simultaneously performing **denoising** and
**super-resolution**.

---

## Table of Contents

- [1. Problem Statement](#1-problem-statement)
- [2. Input / Output Contract & Assumptions](#2-input--output-contract--assumptions)
- [3. Dataset](#3-dataset)
- [4. Strict Data Rules](#4-strict-data-rules)
- [5. Evaluation Criteria](#5-evaluation-criteria)
- [6. Technical Strategy](#6-technical-strategy)
- [7. Repository Structure](#7-repository-structure)
- [8. Environment Setup & Installation](#8-environment-setup--installation)
- [9. Standalone Inference](#9-standalone-inference)
- [10. Reproducible Training](#10-reproducible-training)
- [11. Results & Metric Summary](#11-results--metric-summary)
- [12. Key Design Decisions & Constraints](#12-key-design-decisions--constraints)
- [13. Deliverables Checklist](#13-deliverables-checklist)
- [14. Reproducibility](#14-reproducibility)

---

## 1. Problem Statement

Inspection systems in semiconductor manufacturing capture **degraded** images. The goal is to
restore a clean image at the original GT resolution from a degraded input.

Each input image suffers from **three** degradation mechanisms, applied in an **undisclosed order**:

1. **Speckle noise** (multiplicative)
2. **Additive Gaussian noise**
3. **Downsampling** (loss of spatial resolution)

Because degradations are applied in an unknown order, the model must learn a **blind**
restoration that is robust to the combined degradation distribution rather than assuming a
fixed degradation pipeline.

**The restoration is a joint task:** denoising (speckle + Gaussian) **and** super-resolution
(upsampling) in a single forward pass.

---

## 2. Input / Output Contract & Assumptions

To comply with the strict dataset rules and evaluation parameters, our pipeline adheres to the
following data contract:

- **Input Data (NoisyLR):** The pipeline expects single-channel `.npy` arrays. We assume the
  input values may extend outside the `[0, 1]` range due to speckle and Gaussian noise.
  **Inputs are intentionally never clipped or normalized prior to the forward pass.**
- **Output Data (Restored):** The pipeline outputs single-channel `.npy` arrays upscaled to the
  ground-truth resolution. Outputs are explicitly clamped to `[0, 1]` just before saving, as
  evaluators will score the images exactly as saved.
- **File Naming:** The inference script preserves the exact file naming and format of the input
  directory.

---

## 3. Dataset

> **Verified by direct inspection of the shipped data.** The data is stored as **NumPy
> `.npy` arrays** (`float32`, single-channel 2D grayscale), *not* as PNG/JPEG images.

| Split | Path | Count | Array Shape | dtype | Observed Range |
|-------|------|-------|-------------|-------|----------------|
| **Train GT** | `train/GT/` | 3200 | `(256, 256)` | `float32` | `[0.0, 1.0]` (strict) |
| **Train NoisyLR** | `train/NoisyLR/` | 3200 | `(128, 128)` | `float32` | extends **outside** `[0,1]`, e.g. `[-0.082, 1.72]` |
| **Test NoisyLR** | `NoisyLR/` | 400 | `(128, 128)` | `float32` | extends outside `[0,1]` |

**Key observations:**

- **Scale factor = 2×.** NoisyLR is `128×128`, GT is `256×256`. The model must upsample by
  **2×** to recover GT resolution. (The brief notes eval images are *approximately* `256×256`
  or `512×512`, so the pipeline keeps the scale factor configurable for a possible 4× case.)
- Files are paired **by filename** (`000000.npy` in `train/GT/` ↔ `000000.npy` in `train/NoisyLR/`).
- GT is strictly normalized to `[0, 1]`.
- NoisyLR **intentionally exceeds** `[0, 1]` (both `< 0` and `> 1`) due to speckle + Gaussian
  noise. This is expected and must be preserved.

---

## 4. Strict Data Rules

These rules are **mandatory** and drive the whole design:

1. **GT ∈ [0, 1].** Ground-truth targets are strictly normalized to `[0, 1]`.
2. **NoisyLR is unbounded.** Degraded inputs may be `< 0` or `> 1`. **Do NOT clip or
   normalize** the inputs before feeding them to the network. Feed raw `float32` values as-is.
3. **Super-resolution is required.** Because of the downsampling degradation, the model must
   include an **upsampling mechanism** (e.g. `PixelShuffle`) to restore original dimensions
   (2×: `128→256`).
4. **Clip only at the very end.** Evaluators score images exactly as saved. Apply explicit
   clipping (`torch.clamp(output, 0.0, 1.0)`) **only as the final step** of the forward pass —
   never on inputs, never mid-network.

---

## 5. Evaluation Criteria

Evaluated on a **hidden test set** (both **in-distribution** and **out-of-distribution** content):

- **Restoration quality** — internal KLA combination of **PSNR**, **SSIM**, and **LPIPS**.
- **Throughput / latency** — end-to-end inference time on a common **NVIDIA H100** GPU.
  This measurement **includes the full path**: disk read → preprocessing → CPU→GPU transfer →
  model execution → GPU→CPU transfer → disk write.

> **Implication:** the inference script must be optimized for *total wall-clock time*, not just
> forward-pass FLOPs. I/O and host↔device transfers count.

---

## 6. Technical Strategy

### Compute environment (development constraint)

- Developed on **free-tier GPUs** (Kaggle / Google Colab **NVIDIA T4**, 16 GB VRAM).
- **Cannot train on full `512×512` images** → use **patch-based training**.

### Training approach

- **Paired random patch cropping**: e.g. a `64×64` NoisyLR patch ↔ corresponding `128×128`
  GT patch (2× scale). Crop coordinates are shared/aligned between LR and HR.
- **Batch size:** 16–32.
- **Mixed precision:** `torch.cuda.amp` (autocast + GradScaler) for speed & memory on T4.
- **Robust to OOD:** augment carefully; avoid overfitting to the training degradation level.

### Architecture

- Lightweight, high-capacity restoration network — **SR-NAFNet** (Super-Resolution Nonlinear
  Activation Free Network), a lightweight, fast CNN-Transformer-free backbone.
- Must be **exceptionally fast** to maximize the H100 throughput score.
- Upsampling head via **PixelShuffle** for the 2× (configurable) scale.

### Loss function

Composite loss to reconstruct repeating semiconductor grid structures **without hallucination**:

- **Charbonnier loss** (robust L1) — pixel fidelity / PSNR-oriented.
- **SSIM loss** — structural similarity.
- **LPIPS loss** — perceptual quality (used during training as a soft perceptual regularizer).

---

## 7. Repository Structure

This repository follows the official **KLA recommended structure**:

```text
repository/
 ├── README.md                      # This documentation
 ├── requirements.txt               # Environment dependencies
 ├── train.py                       # Reproducible training script
 ├── inference.py                   # Standalone inference script
 ├── model.py                       # SR-NAFNet architecture
 ├── dataset.py                     # RAM-cached dataloader & augmentations
 ├── loss.py                        # Composite loss (Charbonnier + SSIM + LPIPS)
 ├── weights/
 │   └── best_model.pt              # Final submitted checkpoint
 └── solution_presentation.pptx     # Phase 1 slide deck
```
---

## 8. Environment Setup & Installation

Our code is designed to run efficiently on an NVIDIA GPU (developed and tested on NVIDIA T4;
fully compatible with NVIDIA H100).

**Step 1 — Create a virtual environment (optional but recommended):**

```bash
python -m venv kla_env
source kla_env/bin/activate  # On Windows: kla_env\Scripts\activate
```

**Step 2 — Install dependencies** (exact versions are pinned in `requirements.txt`):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Current `requirements.txt`:

```
torch
torchvision
numpy
opencv-python
einops
lpips
matplotlib
```

> **Note:** Data is `.npy`, so `numpy` handles all I/O. `opencv-python` is available for any
> image export/visualization; `lpips` is used for the perceptual loss/metric during validation.

---

## 9. Standalone Inference

The inference script is **fully standalone** and does not require evaluators to edit source
code or local paths. End-to-end runtime is optimized, tracking disk reading, pre/post-
processing, CPU-to-GPU transfer, and saving.

```bash
python inference.py \
    --input_dir NoisyLR \
    --output_dir results \
    --checkpoint weights/best_model.pt \
    --device cuda
```

The script loads every degraded image from `--input_dir`, restores it at GT resolution
(2× upscale, outputs clamped to `[0, 1]` only at the very end), and saves each restored `.npy`
array to `--output_dir`, **preserving the exact input file names**. It supports batch
processing if GPU memory permits, but defaults to optimized single-image continuous
processing for absolute VRAM safety.

---

## 10. Reproducible Training

To reproduce our submitted checkpoint, run the `train.py` script.

**Key training features:**

- **RAM Caching:** the entire `.npy` dataset is cached into memory at initialization to
  eliminate disk I/O bottlenecks.
- **Automatic Mixed Precision (AMP):** utilises FP16 (autocast + GradScaler, enabled
  automatically when CUDA is available) to maximise throughput without sacrificing gradient
  stability.
- **Composite Loss:** balances pixel fidelity (Charbonnier), structural similarity (SSIM),
  and perceptual quality (LPIPS).

```bash
python train.py \
    --lr_dir train/NoisyLR \
    --gt_dir train/GT \
    --out_dir weights/ \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --scale 2
```

**Key hyperparameters (all exposed as CLI flags):**

| Flag | Default | Meaning |
|------|---------|---------|
| `--width` | `32` | SR-NAFNet base channel width |
| `--num_blocks` | `4` | number of NAFNet encoder/decoder blocks |
| `--scale` | `2` | super-resolution scale factor |
| `--lr_patch_size` | `64` | LR patch size (GT patch = ×2) |
| `--patches_per_image` | `1` | random patches cropped per image per epoch |
| `--val_split` | `0.1` | fraction of training files held out for validation |
| `--seed` | `42` | random seed (data split, crops, training) |
| `--lr` | `1e-4` | AdamW base learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--grad_clip` | `1.0` | global gradient-norm clipping (speckle robustness) |
| `--lambda_char / --lambda_ssim / --lambda_lpips` | `1.0 / 0.2 / 0.01` | composite-loss weights |

Training can be resumed from a snapshot with `--resume <path/to/checkpoint_last.pt>`.

---

## 11. Results & Metric Summary

Our model was evaluated on a **held-out validation split** to prevent training leakage. We
tracked the three mandatory metrics used by KLA:

- **PSNR:** ~26.97 dB
- **SSIM:** ~0.676
- **LPIPS:** ~0.082

| Item | Value |
|------|-------|
| **Hardware** | NVIDIA T4 (16 GB VRAM) |
| **Throughput** | ~181 images/second (end-to-end, including disk I/O) |

Visual examples of restored structural integrity and failure analyses are detailed in the
included `solution_presentation.pptx`.

---

## 12. Key Design Decisions & Constraints

| Decision | Rationale |
|----------|-----------|
| **Feed raw NoisyLR (no clipping/normalization)** | Strict rule — inputs are intentionally unbounded; clipping would destroy signal. |
| **Clamp to `[0,1]` only at final output** | Evaluators score exactly what is saved; GT lives in `[0,1]`. |
| **Patch-based training (64 LR → 128 HR)** | T4 16 GB VRAM cannot fit full 512×512 training; matches 2× scale. |
| **PixelShuffle upsampling head** | Efficient, artifact-free 2× super-resolution. |
| **NAFNet-style lightweight backbone** | Maximizes H100 throughput score (latency is scored end-to-end). |
| **Mixed precision (`torch.cuda.amp`)** | Speed + memory headroom on free-tier T4. |
| **Charbonnier + SSIM + LPIPS** | Balances PSNR/SSIM/LPIPS; recovers periodic grids without hallucination. |
| **Blind restoration (no assumed degradation order)** | Degradations applied in undisclosed order → model must generalize. |
| **Optimize full inference path (I/O + transfer + model)** | Latency scoring includes disk read, H2D/D2H transfer, and disk write. |

> **Optional loss note:** Focal Frequency Loss (FFL) was investigated to further recover
> high-frequency periodic patterns.

---

## 13. Deliverables Checklist

- [x] **`inference.py`** — standalone, accepts `--input_dir` and `--output_dir`.
- [x] **`train.py`** — fully reproducible training.
- [x] **`requirements.txt`** — dependencies.
- [x] **`README.md`** — this document, with execution commands.
- [x] **Weights** — `weights/best_model.pt` produced by `train.py`.
- [x] **Solution presentation** — `solution_presentation.pptx` (Phase 1 slide deck).

---

## 14. Reproducibility

- Fixed random seeds for data splitting, augmentation, and training.
- All hyperparameters exposed via CLI flags / config.
- Deterministic patch sampling option for exact rerun parity.

---

*Built for Hackathon 2026 — SEMICON India · KLA Problem Statement.*

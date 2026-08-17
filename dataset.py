from __future__ import annotations
import os
import random
from typing import List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_generator(seed: int) -> torch.Generator:
    """A torch.Generator with a fixed seed for deterministic shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def split_filenames(
    filenames: Sequence[str],
    val_split: float,
    seed: int,
) -> Tuple[List[str], List[str]]:

    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")
    files = sorted(filenames)
    indices = list(range(len(files)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    
    n_val = int(round(len(files) * val_split))
    val_idx = set(indices[:n_val])
    
    train_files = [f for i, f in enumerate(files) if i not in val_idx]
    val_files = [f for i, f in enumerate(files) if i in val_idx]
    
    return train_files, val_files

class RestorationDataset(Dataset):
    def __init__(
        self,
        lr_dir: str,
        gt_dir: str,
        file_list: Sequence[str],
        scale: int = 2,
        lr_patch_size: Optional[int] = None,
        training: bool = True,
        patches_per_image: int = 1,
        check_nan: bool = True,
    ) -> None:
        super().__init__()
        
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")
        if lr_patch_size is not None and lr_patch_size < 1:
            raise ValueError(f"lr_patch_size must be >= 1, got {lr_patch_size}")
            
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.file_list = list(file_list)
        self.scale = int(scale)
        self.lr_patch_size = lr_patch_size
        self.training = training
        self.patches_per_image = int(patches_per_image)
        self.check_nan = check_nan
        
        if not self.file_list:
            raise ValueError("file_list is empty: nothing to train/validate on.")
            
        self._lr_cache: List[np.ndarray] = []
        self._gt_cache: List[np.ndarray] = []
        
        for name in self.file_list:
            lr_path = os.path.join(lr_dir, name)
            gt_path = os.path.join(gt_dir, name)
            
            if not (os.path.isfile(lr_path) and os.path.isfile(gt_path)):
                raise FileNotFoundError(f"missing paired file: {name}")
                
            lr = np.ascontiguousarray(np.load(lr_path), dtype=np.float32)
            gt = np.ascontiguousarray(np.load(gt_path), dtype=np.float32)
            
            self._validate_pair(lr, gt, name)
            if check_nan:
                self._check_nan_inf(lr, gt, name)
                
            self._lr_cache.append(lr)
            self._gt_cache.append(gt)
            
        self._mem_bytes = sum(a.nbytes for a in self._lr_cache) + sum(
            a.nbytes for a in self._gt_cache
        )

    def _validate_pair(self, lr: np.ndarray, gt: np.ndarray, name: str) -> None:
        """Assert spatial consistency and the scale relationship for {name}."""
        if lr.ndim != 2 or gt.ndim != 2:
            raise ValueError(
                f"{name}: expected 2D grayscale arrays, got LR{lr.ndim}D, GT{gt.ndim}D"
            )
            
        lr_h, lr_w = lr.shape
        gt_h, gt_w = gt.shape
        
        if (gt_h, gt_w) != (lr_h * self.scale, lr_w * self.scale):
            raise ValueError(
                f"{name}: GT {gt.shape} != LR {lr.shape} * scale {self.scale} "
                f"(expected {lr_h * self.scale, lr_w * self.scale})"
            )

    @staticmethod
    def _check_nan_inf(lr: np.ndarray, gt: np.ndarray, name: str) -> None:
        """Raise if either array contains NaN/Inf (would poison AMP training)."""
        if not np.isfinite(lr).all() or not np.isfinite(gt).all():
            raise ValueError(
                f"{name}: found NaN or Inf in cached array. "
                "Refusing to train on corrupt data."
            )

    def _paired_random_crop(
        self, lr: np.ndarray, gt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        ps = self.lr_patch_size
        lr_h, lr_w = lr.shape
        
        assert lr_h >= ps and lr_w >= ps, (
            f"LR image {lr.shape} smaller than patch {ps}; lower lr_patch_size."
        )
        s = self.scale
        y = int(torch.randint(0, lr_h - ps + 1, (1,)).item())
        x = int(torch.randint(0, lr_w - ps + 1, (1,)).item())
        lr_patch = np.ascontiguousarray(lr[y : y + ps, x : x + ps])
        
        gt_patch = np.ascontiguousarray(
            gt[y * s : (y + ps) * s, x * s : (x + ps) * s]
        )
        
        return lr_patch, gt_patch

    @staticmethod
    def _d4_augment(
        lr: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
       
        if torch.rand(1).item() < 0.5:
            lr = torch.flip(lr, dims=[-1])
            gt = torch.flip(gt, dims=[-1])
            
        if torch.rand(1).item() < 0.5:
            lr = torch.flip(lr, dims=[-2])
            gt = torch.flip(gt, dims=[-2])
            
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            lr = torch.rot90(lr, k, dims=[-2, -1])
            gt = torch.rot90(gt, k, dims=[-2, -1])
            
        return lr.contiguous(), gt.contiguous()

    def __len__(self) -> int:
        """Samples per epoch = images * virtual patches per image."""
        return len(self.file_list) * self.patches_per_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a ``(lr, gt)`` tuple of ``(1, H, W)`` float32 tensors."""
        img_idx = idx % len(self.file_list)  # wrap when patches_per_image > 1
        lr = self._lr_cache[img_idx]
        gt = self._gt_cache[img_idx]
        
        if self.training and self.lr_patch_size is not None:
            # Random paired crop on the LR grid -> exact 2x-aligned GT crop.
            lr, gt = self._paired_random_crop(lr, gt)
            
        # (H, W) -> (1, H, W). RAW float values: NO clipping / normalization.
        lr_t = torch.from_numpy(lr).unsqueeze(0)   # shape (1, lr_h, lr_w)
        gt_t = torch.from_numpy(gt).unsqueeze(0)   # shape (1, gt_h, gt_w)
        
        if self.training:
            lr_t, gt_t = self._d4_augment(lr_t, gt_t)
            
        return lr_t, gt_t

def build_dataloaders(
    lr_dir: str,
    gt_dir: str,
    val_split: float = 0.1,
    seed: int = 42,
    scale: int = 2,
    lr_patch_size: int = 64,
    patches_per_image: int = 1,
    batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader, Tuple[List[str], List[str]]]:
    
    all_names = sorted(os.listdir(lr_dir))
    all_names = [n for n in all_names if n.endswith(".npy")]
    
    if not all_names:
        raise ValueError(f"no .npy files found in {lr_dir}")
        
    train_files, val_files = split_filenames(all_names, val_split, seed)
    
    if not val_files:
        raise ValueError("validation split is empty; lower val_split or check data.")
        
    train_ds = RestorationDataset(
        lr_dir=lr_dir,
        gt_dir=gt_dir,
        file_list=train_files,
        scale=scale,
        lr_patch_size=lr_patch_size,
        training=True,
        patches_per_image=patches_per_image,
    )
    
    val_ds = RestorationDataset(
        lr_dir=lr_dir,
        gt_dir=gt_dir,
        file_list=val_files,
        scale=scale,
        lr_patch_size=None,  # full image
        training=False,
    )
    
    gen = make_generator(seed)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=gen,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size or batch_size,
        shuffle=False,  # deterministic order for reproducible metric logging
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    return train_loader, val_loader, (train_files, val_files)

# Example
if __name__ == "__main__":
    tr_loader, va_loader, (tr_names, va_names) = build_dataloaders(
        lr_dir="train/train/NoisyLR",
        gt_dir="train/train/GT",
        val_split=0.1,
        seed=42,
        scale=2,
        lr_patch_size=64,     
        batch_size=32,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"train images: {len(tr_names)}, val images: {len(va_names)}")
    
    lr, gt = next(iter(tr_loader))
    print(f"train LR batch: {tuple(lr.shape)}  (dtype {lr.dtype})")
    print(f"train GT batch: {tuple(gt.shape)}  (dtype {gt.dtype})")
    print(f"   LR value range after dataloader: [{lr.min().item():.4f}, {lr.max().item():.4f}] "
          f"(unbounded, NOT clipped)")
          
    lr, gt = next(iter(va_loader))
    print(f"val  LR batch: {tuple(lr.shape)}  (full image, no crop)")
    print(f"val  GT batch: {tuple(gt.shape)}")
    print(f"   GT value range: [{gt.min().item():.4f}, {gt.max().item():.4f}] "
          f"(within [0,1])")

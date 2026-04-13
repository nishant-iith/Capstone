"""
PyTorch Dataset and DataLoader for paired H&E staining data.

Each sample is a (unstained, stained) tensor pair, both in [-1, 1] range,
resized to 256×256 pixels — the standard Pix2Pix input size.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import random


# ── Transforms ─────────────────────────────────────────────────────────────────

def _to_tensor_minus1_1(img: Image.Image, size: int = 256) -> torch.Tensor:
    """
    Resize → [0,1] tensor → [-1,1] tensor.
    """
    img = img.resize((size, size), Image.BILINEAR)
    t = TF.to_tensor(img)          # [C,H,W] in [0,1]
    t = t * 2.0 - 1.0              # [C,H,W] in [-1,1]
    return t


def _paired_augment(unstained_img, stained_img):
    """
    Apply the same random horizontal/vertical flip to both images.
    Only used during training; call before converting to tensors.
    """
    if random.random() > 0.5:
        unstained_img = TF.hflip(unstained_img)
        stained_img   = TF.hflip(stained_img)
    if random.random() > 0.5:
        unstained_img = TF.vflip(unstained_img)
        stained_img   = TF.vflip(stained_img)
    return unstained_img, stained_img


# ── Dataset ────────────────────────────────────────────────────────────────────

class StainingDataset(Dataset):
    """
    Paired dataset of unstained → stained tissue patches.

    Args:
        df:        DataFrame with columns [unstained, stained] containing file paths.
                   A column named 'id' is optional but useful for downstream logging.
        augment:   If True, apply random horizontal/vertical flips (use for training).
        img_size:  Resize target (default 256 for Pix2Pix).

    Returns per __getitem__:
        (unstained_tensor, stained_tensor) both [3, img_size, img_size] in [-1, 1]
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False, img_size: int = 256):
        self.df       = df.reset_index(drop=True)
        self.augment  = augment
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        unstained_img = Image.open(row["unstained"]).convert("RGB")
        stained_img   = Image.open(row["stained"]).convert("RGB")

        if self.augment:
            unstained_img, stained_img = _paired_augment(unstained_img, stained_img)

        unstained_t = _to_tensor_minus1_1(unstained_img, self.img_size)
        stained_t   = _to_tensor_minus1_1(stained_img,   self.img_size)

        return unstained_t, stained_t


# ── DataLoader factory ─────────────────────────────────────────────────────────

def get_dataloader(
    df: pd.DataFrame,
    batch_size: int = 4,
    shuffle: bool = True,
    augment: bool = False,
    num_workers: int = 2,
    img_size: int = 256,
) -> DataLoader:
    """
    Build a DataLoader from a pairs DataFrame.

    Args:
        df:          DataFrame with [unstained, stained] columns.
        batch_size:  Samples per batch (default 4, fits T4/P100 VRAM at 256px).
        shuffle:     Whether to shuffle each epoch (True for train, False for val/test).
        augment:     Apply random flips (True for train only).
        num_workers: Parallel data loading workers.
        img_size:    Resize target in pixels.

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = StainingDataset(df, augment=augment, img_size=img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,   # keeps batch sizes uniform; avoids BN issues
    )

"""
Train v16: Attention U-Net + MultiScale Discriminator on full-size (1024×1024) images.
Uses top-1000 registered pairs from registered_pairs_all.csv (mean SSIM ≈ 0.6).
Combines v11's successful full-size approach with v13's architectural improvements.

Loss: WGAN-GP(10) + L1(100) + Sobel(20) + Percept(10) [no HED to avoid divergence].
Data: top-1000 pairs from registration pipeline (SSIM filtered).
Target SSIM: 0.74-0.76 (better than v14's patch-based 0.7080).
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
from skimage import io
import warnings
import signal
import sys
warnings.filterwarnings('ignore')

BEST_CKPT = {"path": None, "ssim": 0.0}

def graceful_shutdown(signum, frame):
    print(f"\n\n  {'='*90}")
    print(f"  GRACEFUL SHUTDOWN")
    if BEST_CKPT["path"]:
        print(f"  Best checkpoint: {BEST_CKPT['path']}")
        print(f"  Best SSIM: {BEST_CKPT['ssim']:.4f}")
    else:
        print(f"  No checkpoint saved")
    print(f"  {'='*90}\n")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_shutdown)

from src.models.gan import UNetGenerator, MultiScaleDiscriminator
from src.training.lightning_module_v16 import GANModuleV16

# ── Config ────────────────────────────────────────────────────────────────────
REGISTERED_CSV = "data/processed/registered_pairs_all.csv"
CHECKPOINT_DIR = "checkpoints/v16"
LOG_DIR = "logs/v16"
MODEL_PATH = "models/v16_model.pth"

BATCH_SIZE = 20  # Batch size tuned for stability
NUM_WORKERS = 0  # Disable workers (caching in main process is better)
MAX_EPOCHS = 200
LEARNING_RATE = 5e-5  # Lower for warm-start fine-tuning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss weights
LAMBDA_L1 = 100
LAMBDA_GP = 10
LAMBDA_STRUCT = 20
LAMBDA_PERCEPT = 10
# Note: No HED loss (caused v15 divergence)

W = 90  # Width for terminal output


# ── Dataset ────────────────────────────────────────────────────────────────────
class FullSizeDataset(Dataset):
    """Load full-size (1024×1024) registered image pairs from CSV into RAM."""

    def __init__(self, csv_file, top_n=1000):
        self.df = pd.read_csv(csv_file)
        # Take top-n pairs by SSIM (highest quality)
        self.df = self.df.head(top_n).reset_index(drop=True)

        print(f"Caching {len(self.df)} full-size image pairs into RAM...")
        self.stained_imgs = []
        self.unstained_imgs = []

        for idx, row in self.df.iterrows():
            s_img = io.imread(row["stained"]).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
            u_img = io.imread(row["unstained"]).astype(np.float32) / 127.5 - 1.0  # [-1, 1]

            # Convert to tensor (C, H, W)
            if len(s_img.shape) == 3:
                s_t = torch.from_numpy(s_img).permute(2, 0, 1)
                u_t = torch.from_numpy(u_img).permute(2, 0, 1)
            else:
                # Grayscale -> expand to 3 channels
                s_t = torch.from_numpy(np.stack([s_img]*3, axis=0))
                u_t = torch.from_numpy(np.stack([u_img]*3, axis=0))

            self.stained_imgs.append(s_t)
            self.unstained_imgs.append(u_t)

            if (idx + 1) % 100 == 0:
                print(f"  Loaded {idx + 1}/{len(self.df)} images...", end='\r')

        print(f"\nRAM cache complete. Dataset: {len(self.df)} pairs")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.unstained_imgs[idx], self.stained_imgs[idx]


# ── Main Trainer ────────────────────────────────────────────────────────────────
def main():
    # GPU optimizations
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Load dataset
    dataset = FullSizeDataset(REGISTERED_CSV, top_n=1000)

    # Train/val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Create models
    gen = UNetGenerator()
    disc = MultiScaleDiscriminator()

    # Warm-start: Load v14 weights if available
    v14_path = "models/v14_model.pth"
    if os.path.exists(v14_path):
        print(f"  Loading v14 warm-start weights: {v14_path}")
        v14_state = torch.load(v14_path, map_location=DEVICE)
        # Filter keys that match gen architecture
        gen_keys = gen.state_dict().keys()
        matching_keys = {k: v for k, v in v14_state.items() if k in gen_keys}
        if matching_keys:
            gen.load_state_dict(matching_keys, strict=False)
            print(f"  ✓ Loaded {len(matching_keys)} matching layers from v14")
        else:
            print(f"  ✗ No matching layers found, training from scratch")

    # Create Lightning module
    module = GANModuleV16(
        gen=gen,
        disc=disc,
        lambda_l1=LAMBDA_L1,
        lambda_gp=LAMBDA_GP,
        lambda_struct=LAMBDA_STRUCT,
        lambda_percept=LAMBDA_PERCEPT,
        lr=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
    )

    # Callbacks
    ckpt_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="v16-{epoch:03d}-{val_ssim:.4f}",
        monitor="val_ssim",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor="val_ssim",
        patience=30,
        mode="max",
        min_delta=0.002,
    )

    # Print training config
    print()
    print(f"  {'='*W}")
    print(f"  {'V16: ATTENTION U-NET + MULTISCALE DISC | FULL-SIZE (1024×1024)':^{W}}")
    print(f"  {'='*W}")
    print(f"  Architecture       Attention U-Net + MultiScale Disc")
    print(f"  Loss               WGAN-GP(10) + L1(100) + Sobel(20) + Percept(10)")
    print(f"  Optimizer          Adam  β=(0.0, 0.9)  lr={LEARNING_RATE}")
    print(f"  LR Schedule        CosineAnnealingLR {LEARNING_RATE}→1e-6")
    print(f"  Device             {str(DEVICE).upper()}")
    print(f"  Batch Size         {BATCH_SIZE}  |  Workers: {NUM_WORKERS}")
    print(f"  Images (Full-sz)   Top-1000 registered pairs (mean SSIM ≈0.6)")
    print(f"  Train/Val Split    {train_size}/{val_size}")
    print(f"  Max Epochs         {MAX_EPOCHS}  |  Target: SSIM > 0.74")
    print(f"  Checkpoint Dir     {CHECKPOINT_DIR}")
    print(f"  {'='*W}")
    print()

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_callback, early_stop],
        default_root_dir=LOG_DIR,
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_model_summary=False,
        precision="16-mixed",  # Mixed precision for 2x speed
    )

    # Train
    trainer.fit(module, train_loader, val_loader)

    # Save best model
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        best_ckpt = trainer.checkpoint_callback.best_model_path
        best_ssim = trainer.checkpoint_callback.best_model_score
        BEST_CKPT["path"] = best_ckpt
        BEST_CKPT["ssim"] = float(best_ssim) if best_ssim else 0.0

        print(f"\n  {'='*90}")
        print(f"  ★ Training Complete")
        print(f"  Best checkpoint: {best_ckpt}")
        print(f"  Best SSIM: {BEST_CKPT['ssim']:.4f}")
        print(f"  Loading best model...")
        module = GANModuleV16.load_from_checkpoint(
            best_ckpt, gen=gen, disc=disc
        )
        torch.save(module.gen.state_dict(), MODEL_PATH)
        print(f"  ★ Generator saved: {MODEL_PATH}")
        print(f"  {'='*90}\n")


if __name__ == "__main__":
    main()

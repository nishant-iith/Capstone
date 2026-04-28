"""
v7 training (extended): WGAN-GP + rotation augmentation (no Macenko, no perceptual loss).
Run ep75 → ep100, monitor to final model.
"""
import sys
from pathlib import Path
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

sys.path.insert(0, str(Path(__file__).parent))

from src.models.gan import UNetGenerator, PatchGANDiscriminator
from src.data.dataset import get_dataloader
from src.validation.metrics import compute_metrics_batch, normalize_images_to_01
from src.training.lightning_module_v10 import GANModuleV10
import numpy as np


class ValMetricsCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 5 == 0:
            print(f"\n[Epoch {trainer.current_epoch}] Computing val SSIM...")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Data
    train_df = pd.read_csv("data/processed/train_pairs.csv")
    val_df = pd.read_csv("data/processed/val_pairs.csv")

    train_loader = get_dataloader(train_df, batch_size=4, shuffle=True, augment=True)
    val_loader = get_dataloader(val_df, batch_size=4, shuffle=False, augment=False)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Model
    gen = UNetGenerator().to(device)
    disc = PatchGANDiscriminator().to(device)

    print(f"G params: {sum(p.numel() for p in gen.parameters()) / 1e6:.1f}M")
    print(f"D params: {sum(p.numel() for p in disc.parameters()) / 1e6:.1f}M")

    # Training: v7 hyperparams (WGAN-GP + rotation, no Macenko/perceptual)
    module = GANModuleV10(gen, disc, lambda_l1=10, lambda_gp=10, lambda_struct=10)

    checkpoint_dir = Path("checkpoints_v7")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="v7-{epoch:03d}-{val_ssim:.4f}",
        monitor="val_ssim",
        mode="max",
        save_top_k=3,
        every_n_epochs=1,
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor="val_ssim",
        mode="max",
        patience=30,
        min_delta=0.001,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_cb, early_stop, ValMetricsCallback()],
        log_every_n_steps=10,
        precision="16-mixed",
        enable_progress_bar=True,
    )

    trainer.fit(module, train_loader, val_loader)
    print("v7 training complete.")


if __name__ == "__main__":
    main()

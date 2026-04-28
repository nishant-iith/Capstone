"""
Pix2Pix training on REGISTERED histology pairs.
Using WGAN-GP + L1 + Structural loss.
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
from src.training.lightning_module_v10 import GANModuleV10

class PrintMetricsCallback(pl.Callback):
    """Custom callback to print SSIM and PSNR clearly at every validation epoch."""
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_ssim" in metrics:
            ssim = metrics["val_ssim"].item()
            psnr = metrics["val_psnr"].item()
            print(f"\n>>> [Epoch {trainer.current_epoch}] Validation SSIM: {ssim:.4f} | Validation PSNR: {psnr:.2f} dB")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==========================================")
    print(f" TRAINING TRACK: REGISTERED PIX2PIX ")
    print(f" Device: {device.upper()}")
    print(f"==========================================")

    # Data
    # Use the newly registered pairs
    df = pd.read_csv("data/processed/registered_pairs.csv")
    
    # Shuffle and split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # Optimized for 6GB VRAM
    train_loader = get_dataloader(train_df, batch_size=8, shuffle=True, augment=True, num_workers=4)
    val_loader = get_dataloader(val_df, batch_size=8, shuffle=False, augment=False, num_workers=4)

    print(f"Registered Train: {len(train_df)} pairs")
    print(f"Registered Val:   {len(val_df)} pairs")

    # Model
    gen = UNetGenerator().to(device)
    disc = PatchGANDiscriminator().to(device)

    # v10 Module (WGAN-GP + L1)
    # We increase lambda_l1 to 100 as standard for Pix2Pix to focus on color reconstruction
    module = GANModuleV10(gen, disc, lambda_l1=100, lambda_gp=10, lambda_struct=10)

    checkpoint_dir = Path("checkpoints_registered_v1")
    checkpoint_dir.mkdir(exist_ok=True)

    # Callback 1: Checkpointing
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="reg-{epoch:03d}-{val_ssim:.4f}",
        monitor="val_ssim",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    # Callback 2: Early Stopping
    # Stop if SSIM doesn't improve by 0.001 for 30 epochs
    early_stop_cb = EarlyStopping(
        monitor="val_ssim",
        patience=30,
        min_delta=0.001,
        mode="max",
        verbose=True
    )

    # Callback 3: Visible Printing
    print_cb = PrintMetricsCallback()

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb, print_cb],
        log_every_n_steps=10,
        precision="16-mixed" if device == "cuda" else "bf16-mixed",
    )

    print("\nStarting training... Look for '>>>' lines for SSIM scores.\n")
    trainer.fit(module, train_loader, val_loader)

if __name__ == "__main__":
    main()

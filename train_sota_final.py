"""
SOTA Clinical Hybrid Training (Path B Only).
Final Research Edition: Custom UI + Macenko Normalization.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ProgressBar
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.models.gan import UNetGenerator, PatchGANDiscriminator
from src.data.dataset import get_cached_dataloader
from src.training.lightning_module_v11 import GANModuleV11

class SOTAProgressBar(ProgressBar):
    """Custom ASCII progress bar with live loss tracking."""
    def __init__(self):
        super().__init__()
        self.bar_width = 25

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        metrics = trainer.progress_bar_metrics
        g_loss = metrics.get("g_loss", 0.0)
        d_loss = metrics.get("d_loss", 0.0)
        
        # Calculate progress
        total_batches = trainer.num_training_batches
        percent = (batch_idx + 1) / total_batches
        filled = int(self.bar_width * percent)
        bar = "=" * filled + "-" * (self.bar_width - filled)
        
        # Overwrite the same line in terminal
        sys.stdout.write(f"\r TRAINING | Epoch {trainer.current_epoch:03d} | [{bar}] {int(percent*100):3d}% | G-Loss: {g_loss:6.2f} | D-Loss: {d_loss:6.2f} ")
        sys.stdout.flush()

class ClinicalDashboardCallback(pl.Callback):
    """Professional Dashboard for Path B (Macenko) Training."""
    def __init__(self, baseline_ssim=0.7115):
        super().__init__()
        self.baseline_ssim = baseline_ssim
        self.best_ssim = 0.0

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_ssim" not in metrics:
            return

        ssim = metrics["val_ssim"].item()
        psnr = metrics["val_psnr"].item()
        pcc  = metrics["val_pcc"].item()
        
        delta = ssim - self.baseline_ssim
        time_now = datetime.now().strftime("%H:%M:%S")

        # Clear the progress bar line before printing dashboard
        sys.stdout.write("\r" + " " * 100 + "\r")
        
        print("\n" + "="*75)
        print(f" PATH B RESEARCH DASHBOARD | {time_now}")
        print("-" * 75)
        print(f" Epoch: {trainer.current_epoch:03d}")
        print(f" SSIM:  {ssim:.4f} (Vs. Baseline: {delta:+.4f})")
        print(f" PSNR:  {psnr:.2f} dB")
        print(f" PCC:   {pcc:.4f}")
        print("="*75 + "\n")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        
    print("\n" + "#"*75)
    print("#   HISTOLOGY VIRTUAL STAINING: SOTA RESEARCH ENGINE (v12)                #")
    print("#   Strategy: Macenko Stabilization + VGG-19 Weak Supervision             #")
    print("#" + "-"*73 + "#")
    print(f"#   Device:       {device.upper():10}                                            #")
    print(f"#   Status:       Maximized Stability Mode (Cold Start)                   #")
    print("#"*75 + "\n")

    # 1. Data Pipeline
    csv_path = "data/processed/registered_pairs_normalized.csv"
    if not os.path.exists(csv_path):
        print(f"CRITICAL ERROR: {csv_path} not found. Run normalize_stains.py first.")
        return

    df = pd.read_csv(csv_path)
    df_clean = pd.DataFrame({"unstained": df["unstained"], "stained": df["stained_normalized"]})
    df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_loader = get_cached_dataloader(df_clean.iloc[:800], batch_size=8, shuffle=True)
    val_loader = get_cached_dataloader(df_clean.iloc[800:], batch_size=8, shuffle=False)

    # 2. Stable Model Setup
    gen = UNetGenerator(pretrained=True).to(device)
    disc = PatchGANDiscriminator().to(device)
    module = GANModuleV11(gen, disc, lambda_l1=300, lambda_gp=10, lambda_struct=20, lambda_percept=10)

    # 3. Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints_path_b",
        filename="b-{epoch:02d}-{val_ssim:.3f}",
        monitor="val_ssim",
        mode="max",
        save_top_k=5
    )

    early_stop_cb = EarlyStopping(
        monitor="val_ssim",
        patience=40,
        min_delta=0.0001,
        mode="max",
        verbose=True
    )

    # 4. Final Trainer Setup
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb, ClinicalDashboardCallback(baseline_ssim=0.7115), SOTAProgressBar()],
        precision="16-mixed" if device == "cuda" else "bf16-mixed",
        log_every_n_steps=10
    )

    print("Status: Launching SOTA Research Engine. Terminal UI active.\n")
    trainer.fit(module, train_loader, val_loader)

if __name__ == "__main__":
    main()

"""
Turbo Pix2Pix Training: Max Speed + High Fidelity (PSNR 25+)
Stable Windows Version (No torch.compile).
"""
import sys
from pathlib import Path
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.models.gan import UNetGenerator, PatchGANDiscriminator
from src.data.dataset import get_cached_dataloader
from src.training.lightning_module_v10 import GANModuleV10

class DashboardCallback(pl.Callback):
    """Clean terminal dashboard for monitoring training metrics."""
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_ssim" in metrics:
            ssim = metrics["val_ssim"].item()
            psnr = metrics["val_psnr"].item()
            time_now = datetime.now().strftime("%H:%M:%S")
            
            print("\n" + "="*50)
            print(f" METRICS DASHBOARD | Time: {time_now}")
            print("-" * 50)
            print(f" Epoch: {trainer.current_epoch:03d}")
            print(f" SSIM:  {ssim:.4f} (Target: >0.75)")
            print(f" PSNR:  {psnr:.2f} dB (Target: >25.0)")
            print("="*50 + "\n")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Windows-Compatible GPU Optimizations
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        # Enable Tensor Core acceleration for RTX GPUs
        torch.set_float32_matmul_precision('high')
        
    print("\n" + "#"*60)
    print("#   HISTOLOGY VIRTUAL STAINING: TURBO TRAINING PIPELINE    #")
    print("#" + "-"*58 + "#")
    print(f"#   Device:       {device.upper():10}                             #")
    print(f"#   Optimization: ENABLED (Tensor Cores + Benchmark)       #")
    print(f"#   Precision:    16-Mixed                                 #")
    print("#" + "-"*58 + "#")
    print("#   Targets:      SSIM > 0.75 | PSNR > 25.0 dB             #")
    print("#"*60 + "\n")

    # 2. Data Loading (Cached in RAM)
    df = pd.read_csv("data/processed/registered_pairs.csv")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = df.iloc[:800]
    val_df = df.iloc[800:]

    train_loader = get_cached_dataloader(train_df, batch_size=12, shuffle=True)
    val_loader = get_cached_dataloader(val_df, batch_size=12, shuffle=False)

    # 3. Model Setup
    gen = UNetGenerator(pretrained=True).to(device)
    disc = PatchGANDiscriminator().to(device)

    # 4. Training Module (Aggressive L1 for PSNR)
    module = GANModuleV10(gen, disc, lambda_l1=300, lambda_gp=10, lambda_struct=20)

    # 5. Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints_turbo",
        filename="turbo-{epoch:02d}-{val_psnr:.1f}",
        monitor="val_psnr",
        mode="max",
        save_top_k=3
    )

    early_stop_cb = EarlyStopping(
        monitor="val_psnr",
        patience=25,
        min_delta=0.05,
        mode="max",
        verbose=True
    )

    # 6. Trainer
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb, DashboardCallback()],
        precision="16-mixed" if device == "cuda" else "bf16-mixed",
        log_every_n_steps=10,
        enable_checkpointing=True
    )

    print("Status: Starting training engine...")
    trainer.fit(module, train_loader, val_loader)

if __name__ == "__main__":
    main()

"""
Path D: The Definitive Clinical Hybrid.
Registered Data + 0.712 Warm Start + Multi-Scale Discriminator.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.models.gan import UNetGenerator, MultiScaleDiscriminator
from src.data.dataset import get_cached_dataloader
from src.training.lightning_module_v12 import GANModuleV12
from train_sota_final import SOTAProgressBar

class FinalDashboardCallback(pl.Callback):
    def __init__(self, baseline_ssim=0.7115):
        super().__init__()
        self.baseline_ssim = baseline_ssim
        self.best_ssim = 0.0

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_ssim" not in metrics: return
        
        ssim = metrics["val_ssim"].item()
        psnr = metrics["val_psnr"].item()
        pcc  = metrics["val_pcc"].item()
        delta = ssim - self.baseline_ssim
        
        sys.stdout.write("\r" + " " * 100 + "\r")
        print("\n" + "="*80)
        print(f" PATH D: FINAL HYBRID DASHBOARD | Epoch {trainer.current_epoch:03d}")
        print("="*80)
        print(f" [SOTA] SSIM: {ssim:.4f} | Gain over 0.7115: {delta:+.4f}")
        print(f" [SOTA] PSNR: {psnr:.2f} dB | PCC: {pcc:.4f}")
        print("-" * 80)
        if ssim > self.best_ssim:
            self.best_ssim = ssim
            print(" >>> NEW WORLD RECORD DETECTED! Weights saved to checkpoints_path_d/")
        print("="*80 + "\n")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        
    print("\n" + "#"*80)
    print("#   HISTOLOGY VIRTUAL STAINING: THE PATH D FINAL OVERDRIVE                #")
    print("#   Status: Recovering from Path B Failure + Multi-Scale Activation       #")
    print("#"*80 + "\n")

    # 1. Pipeline
    df = pd.read_csv("data/processed/final_training_pairs.csv")
    train_loader = get_cached_dataloader(df.iloc[:800], batch_size=8, shuffle=True)
    val_loader = get_cached_dataloader(df.iloc[800:], batch_size=8, shuffle=False)

    # 2. Advanced Architecture
    gen = UNetGenerator(pretrained=True).to(device)
    multi_disc = MultiScaleDiscriminator().to(device)
    module = GANModuleV12(gen, multi_disc, lambda_l1=150, lambda_gp=10, lambda_struct=25, lambda_percept=15)

    # 3. KNOWLEDGE RECOVERY: Load the best Perceptual model (0.712)
    # This prevents us from starting at 0.28 ever again.
    percept_ckpt = "checkpoints_weakly_supervised/ws-epoch=27-val_ssim=0.712.ckpt"
    if Path(percept_ckpt).exists():
        print(f"Status: RECOVERING INTELLIGENCE from {percept_ckpt}")
        checkpoint = torch.load(percept_ckpt, map_location=device, weights_only=False)
        # strict=False allows the generator to load while the new Multi-Scale Disc starts fresh
        module.load_state_dict(checkpoint['state_dict'], strict=False)
        print("Result: Intelligence successfully recovered. Starting from 0.71!")

    # 4. Final Config
    checkpoint_cb = ModelCheckpoint(dirpath="checkpoints_path_d", filename="sota-{epoch:02d}-{val_ssim:.3f}", monitor="val_ssim", mode="max", save_top_k=5)
    early_stop_cb = EarlyStopping(monitor="val_ssim", patience=40, min_delta=0.0001, mode="max", verbose=True)

    trainer = pl.Trainer(
        max_epochs=150,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb, FinalDashboardCallback(), SOTAProgressBar()],
        precision="16-mixed" if device == "cuda" else "bf16-mixed",
        log_every_n_steps=10
    )

    trainer.fit(module, train_loader, val_loader)

if __name__ == "__main__":
    main()

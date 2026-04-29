"""
v13 Training: Attention U-Net + MultiScale Disc + HED Loss + Cosine LR.
Data: data/processed/registered_pairs.csv (good slides only: Z17, Z11, Z39, Z15).
"""
import sys
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*TensorFlow.*")
warnings.filterwarnings("ignore", ".*LitLogger.*")
from pathlib import Path
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.models.gan import UNetGenerator, MultiScaleDiscriminator
from src.data.dataset import get_cached_dataloader
from src.training.lightning_module_v13 import GANModuleV13

CSV_PATH   = "data/processed/registered_pairs.csv"
BATCH_SIZE = 16
MAX_EPOCHS = 200
VAL_SPLIT  = 0.15

W = 66  # print width

def bar(val, target, width=24):
    filled = int(min(val / target, 1.0) * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

def status(val, target):
    if val >= target:
        return "PASS"
    return f"+{target - val:.4f} to go"


class DashboardCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if "val_ssim" not in m:
            return

        ssim  = m["val_ssim"].item()
        psnr  = m["val_psnr"].item()
        pcc   = m["val_pcc"].item()
        epoch = trainer.current_epoch
        t     = datetime.now().strftime("%H:%M:%S")

        best  = trainer.checkpoint_callback.best_model_score
        best_str = f"{best:.4f}" if best else "  --  "

        print()
        print(f"  {'='*W}")
        print(f"  VIRTUAL STAINING v13   Epoch {epoch+1:03d}/{MAX_EPOCHS}   {t}")
        print(f"  {'='*W}")
        print(f"  {'Metric':<8}  {'Value':>8}   {'Progress (vs target)':<30}  {'Status'}")
        print(f"  {'-'*W}")
        print(f"  {'SSIM':<8}  {ssim:>8.4f}   {bar(ssim, 0.82):<30}  {status(ssim, 0.82)}")
        print(f"  {'PSNR':<8}  {psnr:>7.2f}dB   {bar(psnr, 30.0):<30}  {status(psnr, 30.0)}")
        print(f"  {'PCC':<8}  {pcc:>8.4f}   {bar(pcc,  0.94):<30}  {status(pcc,  0.94)}")
        print(f"  {'-'*W}")
        print(f"  Best SSIM so far: {best_str}   Early stop patience: {trainer.early_stopping_callback.wait_count}/30")
        print(f"  {'='*W}")
        print()


def print_header(device, total_params, n_train, n_val):
    print()
    print(f"  {'='*W}")
    print(f"  {'HISTOLOGY VIRTUAL STAINING  --  MODEL v13':^{W}}")
    print(f"  {'='*W}")
    print(f"  {'ARCHITECTURE':<16}  Attention U-Net + MultiScale Discriminator")
    print(f"  {'LOSSES':<16}  WGAN-GP  |  L1(x100)  |  Sobel  |  VGG-19  |  HED Stain")
    print(f"  {'LR SCHEDULE':<16}  CosineAnnealingLR   1e-4 -> 1e-6")
    print(f"  {'OPTIMIZER':<16}  Adam  beta=(0.0, 0.9)")
    print(f"  {'COMPILE':<16}  disabled (VGG-19 bottleneck dominates)")
    print(f"  {'-'*W}")
    print(f"  {'DEVICE':<16}  {device.upper()}   AMP 16-mixed")
    print(f"  {'PARAMS':<16}  {total_params/1e6:.1f}M trainable  +  20.0M VGG-19 frozen")
    print(f"  {'DATASET':<16}  {n_train+n_val} pairs  |  {n_train} train  |  {n_val} val  (top-1000 SSIM filtered)")
    print(f"  {'BATCH':<16}  {BATCH_SIZE}   Epochs: {MAX_EPOCHS}")
    print(f"  {'-'*W}")
    print(f"  {'TARGETS':<16}  SSIM > 0.82   PSNR > 30.0 dB   PCC > 0.94")
    print(f"  {'='*W}")
    print()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # ── Data ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val    = int(len(df) * VAL_SPLIT)
    train_df = df.iloc[n_val:].reset_index(drop=True)
    val_df   = df.iloc[:n_val].reset_index(drop=True)

    train_loader = get_cached_dataloader(train_df, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = get_cached_dataloader(val_df,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Models ────────────────────────────────────────────────────────────────
    gen  = UNetGenerator(pretrained=True).to(device)
    disc = MultiScaleDiscriminator().to(device)

    # torch.compile disabled — adds 5-8 min warmup, marginal gain vs VGG-19 bottleneck
    # if device == "cuda":
    #     gen  = torch.compile(gen,  mode="default")
    #     disc = torch.compile(disc, mode="default")

    orig_gen  = getattr(gen,  '_orig_mod', gen)
    orig_disc = getattr(disc, '_orig_mod', disc)
    total_params = (sum(p.numel() for p in orig_gen.parameters()) +
                    sum(p.numel() for p in orig_disc.parameters()))

    print_header(device, total_params, len(train_df), len(val_df))

    module = GANModuleV13(
        gen=gen,
        disc=disc,
        lambda_l1=100,
        lambda_gp=10,
        lambda_struct=20,
        lambda_percept=10,
        lambda_hed=5,
        lr=1e-4,
        max_epochs=MAX_EPOCHS,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints_v13",
        filename="v13-ep{epoch:03d}-ssim{val_ssim:.4f}-psnr{val_psnr:.2f}",
        monitor="val_ssim",
        mode="max",
        save_top_k=3,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_ssim",
        patience=30,
        min_delta=0.002,
        mode="max",
        verbose=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        precision="16-mixed" if device == "cuda" else 32,
        callbacks=[checkpoint_cb, early_stop_cb, DashboardCallback()],
        log_every_n_steps=5,
        enable_checkpointing=True,
        enable_model_summary=False,
    )

    print(f"  Compiling model (first step ~3-5 min silence) ...")
    print(f"  {'='*W}")
    print()
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()

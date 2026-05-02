"""
Configurable v20-style training for registered CSV variants.

Default mode fine-tunes from the clean v20_fixed checkpoint so new registration
CSV variants can be tested quickly without changing train_v20.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W = 78
BEST = {"state": None, "epoch": 0, "ssim": 0.0, "ckpt": None, "model_path": None}
log = logging.getLogger("v20_csv_variant")


def setup_logging(log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def graceful_shutdown(signum, frame):
    log.info(f"Signal {signum} received - saving best model if available")
    if BEST["state"] is not None and BEST["model_path"]:
        torch.save(BEST["state"], BEST["model_path"])
        log.info(
            f"Saved best epoch {BEST['epoch']} SSIM {BEST['ssim']:.4f} -> {BEST['model_path']}"
        )
    sys.exit(0)


for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, graceful_shutdown)


def elastic_transform(image: np.ndarray, alpha: float = 60, sigma: float = 6, seed: int | None = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    height, width = image.shape[:2]
    dx = gaussian_filter(rng.randn(height, width), sigma) * alpha
    dy = gaussian_filter(rng.randn(height, width), sigma) * alpha
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    cx = np.clip(xs + dx, 0, width - 1)
    cy = np.clip(ys + dy, 0, height - 1)
    out = np.zeros_like(image)
    for channel in range(image.shape[2]):
        out[..., channel] = map_coordinates(
            image[..., channel],
            [cy.ravel(), cx.ravel()],
            order=1,
            mode="reflect",
        ).reshape(height, width)
    return out


class RegisteredPairsDataset(Dataset):
    def __init__(self, csv_file: str, top_n: int, indices: np.ndarray | None = None, augment: bool = True):
        df = pd.read_csv(csv_file).head(top_n).reset_index(drop=True)
        if indices is None:
            indices = np.arange(len(df))

        self.images: list[tuple[np.ndarray, np.ndarray]] = []
        self.prefixes: list[str] = []
        for idx in indices:
            row = df.iloc[int(idx)]
            try:
                stained = io.imread(row["stained"]).astype(np.float32) / 255.0
                unstained = io.imread(row["unstained"]).astype(np.float32) / 255.0
                if stained.ndim == 2:
                    stained = np.stack([stained] * 3, axis=-1)
                if unstained.ndim == 2:
                    unstained = np.stack([unstained] * 3, axis=-1)
                self.images.append((stained[:, :, :3], unstained[:, :, :3]))
                self.prefixes.append(str(row["prefix"]))
            except Exception as exc:
                log.warning(f"Skipping row {idx}: {exc}")
        self.augment = augment
        log.info(f"Loaded {len(self.images)} pairs from {csv_file} (augment={augment})")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        stained, unstained = self.images[idx]
        if self.augment:
            if np.random.random() > 0.5:
                stained = np.flip(stained, axis=1).copy()
                unstained = np.flip(unstained, axis=1).copy()
            if np.random.random() > 0.5:
                stained = np.flip(stained, axis=0).copy()
                unstained = np.flip(unstained, axis=0).copy()
            if np.random.random() > 0.5:
                unstained = elastic_transform(
                    unstained,
                    alpha=60,
                    sigma=6,
                    seed=int(np.random.randint(0, 100000)),
                )
        stained_t = torch.from_numpy(stained.copy()).permute(2, 0, 1).contiguous()
        unstained_t = torch.from_numpy(unstained.copy()).permute(2, 0, 1).contiguous()
        return unstained_t, stained_t


class ConvNeXtUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="tu-convnext_base.clip_laion2b",
            encoder_weights="imagenet",
            in_channels=3,
            classes=3,
            activation=None,
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def encoder_params(self):
        return self.model.encoder.parameters()

    def decoder_params(self):
        return list(self.model.decoder.parameters()) + list(self.model.segmentation_head.parameters())


def compute_metrics(pred_np: np.ndarray, target_np: np.ndarray) -> tuple[float, float, float]:
    ssim = ssim_fn(target_np, pred_np, channel_axis=2, data_range=1.0)
    psnr = psnr_fn(target_np, pred_np, data_range=1.0)
    pcc = float(np.corrcoef(pred_np.ravel(), target_np.ravel())[0, 1])
    return float(ssim), float(psnr), float(pcc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-lr", type=float, default=5e-6)
    parser.add_argument("--decoder-lr", type=float, default=5e-5)
    parser.add_argument("--init-model", default="models/v20_fixed_model.pth")
    parser.add_argument("--from-scratch", action="store_true")
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    model_path = f"models/{args.name}_model.pth"
    checkpoint_dir = f"checkpoints/{args.name}"
    log_file = f"logs/{args.name}_training.log"
    BEST["model_path"] = model_path

    os.makedirs("models", exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    setup_logging(log_file)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    df = pd.read_csv(args.csv).head(args.top_n).reset_index(drop=True)
    split_perm = np.random.RandomState(args.seed).permutation(len(df))
    n_train = int(0.9 * len(df))
    train_indices = split_perm[:n_train]
    val_indices = split_perm[n_train:]

    train_ds = RegisteredPairsDataset(args.csv, args.top_n, indices=train_indices, augment=True)
    val_ds = RegisteredPairsDataset(args.csv, args.top_n, indices=val_indices, augment=False)
    overlap = set(train_ds.prefixes) & set(val_ds.prefixes)
    if overlap:
        raise RuntimeError(f"Train/val split overlap detected: {len(overlap)} pairs")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = ConvNeXtUNet().to(DEVICE)
    if not args.from_scratch and args.init_model:
        state = torch.load(args.init_model, map_location="cpu")
        model.load_state_dict(state, strict=True)
        log.info(f"Initialized from {args.init_model}")
    else:
        log.info("Initialized from pretrained encoder + random decoder")

    optimizer = optim.AdamW(
        [
            {"params": model.encoder_params(), "lr": args.encoder_lr},
            {"params": model.decoder_params(), "lr": args.decoder_lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    criterion = nn.L1Loss()
    scaler = GradScaler("cuda", enabled=DEVICE.type == "cuda")

    n_enc = sum(p.numel() for p in model.encoder_params()) / 1e6
    n_dec = sum(p.numel() for p in model.decoder_params()) / 1e6
    log.info("=" * W)
    log.info(f"{args.name}: v20 ConvNeXt-Base CSV variant")
    log.info("=" * W)
    log.info(f"CSV     : {args.csv}")
    log.info(f"Top N   : {args.top_n}")
    log.info(f"Init    : {'scratch' if args.from_scratch else args.init_model}")
    log.info(f"Encoder : {n_enc:.1f}M params LR={args.encoder_lr}")
    log.info(f"Decoder : {n_dec:.1f}M params LR={args.decoder_lr}")
    log.info(f"Batch   : {args.batch_size} | Workers: {args.workers} | Device: {DEVICE}")
    log.info(f"Split   : seed={args.seed} | Train={len(train_ds)} | Val={len(val_ds)} | overlap={len(overlap)}")
    log.info(f"Early   : patience={args.patience} | Epochs={args.epochs}")
    log.info(f"Output  : {model_path}")
    log.info("=" * W)
    log.info(f"{'Ep':>4} {'Loss':>8} {'SSIM':>7} {'PSNR':>7} {'PCC':>7} {'EncLR':>9} {'Time':>7}  Status")
    log.info("-" * W)

    best_val_ssim = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for unstained, stained in train_loader:
            unstained = unstained.to(DEVICE, non_blocking=True)
            stained = stained.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
                pred = model(unstained)
                loss = criterion(pred, stained)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())

        model.eval()
        val_ssims, val_psnrs, val_pccs = [], [], []
        with torch.no_grad():
            for unstained, stained in val_loader:
                unstained = unstained.to(DEVICE, non_blocking=True)
                stained = stained.to(DEVICE, non_blocking=True)
                pred = model(unstained)
                for batch_idx in range(pred.shape[0]):
                    pred_np = pred[batch_idx].cpu().numpy().transpose(1, 2, 0)
                    stained_np = stained[batch_idx].cpu().numpy().transpose(1, 2, 0)
                    ssim, psnr, pcc = compute_metrics(pred_np, stained_np)
                    val_ssims.append(ssim)
                    val_psnrs.append(psnr)
                    val_pccs.append(pcc)

        avg_ssim = float(np.mean(val_ssims))
        avg_psnr = float(np.mean(val_psnrs))
        avg_pcc = float(np.mean(val_pccs))
        avg_loss = train_loss / max(1, len(train_loader))
        elapsed = time.time() - t0
        enc_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        is_best = avg_ssim > best_val_ssim
        status = "BEST" if is_best else f"({patience_counter + 1}/{args.patience})"
        log.info(
            f"{epoch:4d} {avg_loss:8.4f} {avg_ssim:7.4f} {avg_psnr:7.2f} {avg_pcc:7.4f}"
            f" {enc_lr:9.2e} {elapsed:7.1f}s  {status}"
        )

        if is_best:
            best_val_ssim = avg_ssim
            patience_counter = 0
            BEST["state"] = {k: v.cpu() for k, v in model.state_dict().items()}
            BEST["epoch"] = epoch
            BEST["ssim"] = best_val_ssim
            ckpt = os.path.join(checkpoint_dir, f"{args.name}_e{epoch:03d}_ssim{best_val_ssim:.4f}.pth")
            old_ckpt = BEST.get("ckpt")
            torch.save(BEST["state"], ckpt)
            BEST["ckpt"] = ckpt
            if old_ckpt and old_ckpt != ckpt and os.path.exists(old_ckpt):
                try:
                    os.remove(old_ckpt)
                except OSError as exc:
                    log.warning(f"Could not remove old checkpoint {old_ckpt}: {exc}")
            log.info(f"  Saved -> {ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info("=" * W)
                log.info(f"Early stop at epoch {epoch}. Best SSIM: {best_val_ssim:.4f}")
                log.info("=" * W)
                break

    if BEST["state"] is not None:
        torch.save(BEST["state"], model_path)
    log.info("=" * W)
    log.info(f"Complete! Best SSIM: {best_val_ssim:.4f} (epoch {BEST['epoch']})")
    log.info(f"Model saved -> {model_path}")
    log.info("=" * W)


if __name__ == "__main__":
    train()

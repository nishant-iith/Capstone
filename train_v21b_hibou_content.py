"""
v21B: Hibou-B warm-start on content-quality CLAHE top-1000.

This is the controlled follow-up to v22A:
  - same content-quality CSV and split discipline as v22A
  - same Hibou-B frozen-feature architecture as v21A
  - warm-start from the completed v21A checkpoint/model

The goal is to test whether the histology-pretrained Hibou feature path benefits
from the improved registration/data selection, without changing loss or decoder
design at the same time.
"""

import argparse
import logging
import os
import signal
import sys
import time
import warnings

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import train_v21a_hibou_b as v21a

warnings.filterwarnings("ignore")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST = {"state": None, "epoch": 0, "ssim": 0.0, "ckpt": None, "model_path": None}
LOG = logging.getLogger("v21b_hibou_content")
W = 88


def configure_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
        handler.close()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    v21a.log = LOG


def graceful_shutdown(signum, frame):
    LOG.info(f"Signal {signum} received - saving best model if available")
    if BEST["state"] is not None and BEST["model_path"]:
        torch.save(BEST["state"], BEST["model_path"])
        LOG.info(
            f"Saved best epoch {BEST['epoch']} SSIM {BEST['ssim']:.4f} -> {BEST['model_path']}"
        )
    sys.exit(0)


for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, graceful_shutdown)


def build_split(csv_file, top_n, seed):
    df = pd.read_csv(csv_file).head(top_n).reset_index(drop=True)
    split_perm = np.random.RandomState(seed).permutation(len(df))
    n_train = int(0.9 * len(df))
    return split_perm[:n_train], split_perm[n_train:]


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    model_path = f"models/{args.name}_model.pth"
    checkpoint_dir = f"checkpoints/{args.name}"
    log_file = f"logs/{args.name}_training.log"
    configure_logging(log_file)
    BEST["model_path"] = model_path

    os.makedirs("models", exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_indices, val_indices = build_split(args.csv, args.top_n, args.seed)
    train_ds = v21a.RegisteredPairsDataset(args.csv, args.top_n, train_indices, augment=True)
    val_ds = v21a.RegisteredPairsDataset(args.csv, args.top_n, val_indices, augment=False)
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

    model = v21a.HibouBInputSkipUNet(args.hf_model_id).to(DEVICE)
    if args.init_model:
        state = torch.load(args.init_model, map_location="cpu")
        model.load_state_dict(state, strict=True)
        LOG.info(f"Initialized from {args.init_model}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.decoder_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    criterion = nn.L1Loss()
    scaler = GradScaler("cuda", enabled=DEVICE.type == "cuda")

    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    LOG.info("=" * W)
    LOG.info("v21B: frozen Hibou-B warm-start on content-quality CLAHE top-1000")
    LOG.info("=" * W)
    LOG.info(f"CSV     : {args.csv}")
    LOG.info(f"Top N   : {args.top_n}")
    LOG.info(f"Init    : {args.init_model}")
    LOG.info(f"Hibou   : {args.hf_model_id}, local cached load, frozen")
    LOG.info(f"Params  : total={n_total:.1f}M, trainable={n_trainable:.1f}M")
    LOG.info(f"Split   : seed={args.seed} | Train={len(train_ds)} | Val={len(val_ds)} | overlap={len(overlap)}")
    LOG.info(f"Batch   : {args.batch_size} | Workers: {args.workers} | Device: {DEVICE}")
    LOG.info(f"LR      : decoder/input pyramid={args.decoder_lr}")
    LOG.info(f"Early   : patience={args.patience} | Epochs={args.epochs}")
    LOG.info(f"Output  : {model_path}")
    LOG.info("=" * W)
    LOG.info(f"{'Ep':>4} {'Loss':>8} {'SSIM':>7} {'PSNR':>7} {'PCC':>7} {'LR':>9} {'Time':>7}  Status")
    LOG.info("-" * W)

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
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_ssims, val_psnrs, val_pccs = [], [], []
        with torch.no_grad():
            for unstained, stained in val_loader:
                unstained = unstained.to(DEVICE, non_blocking=True)
                stained = stained.to(DEVICE, non_blocking=True)
                pred = model(unstained)
                for bidx in range(pred.shape[0]):
                    pred_np = pred[bidx].float().cpu().numpy().transpose(1, 2, 0)
                    stained_np = stained[bidx].float().cpu().numpy().transpose(1, 2, 0)
                    ssim, psnr, pcc = v21a.compute_metrics(pred_np, stained_np)
                    val_ssims.append(ssim)
                    val_psnrs.append(psnr)
                    val_pccs.append(pcc)

        avg_loss = train_loss / len(train_loader)
        avg_ssim = float(np.mean(val_ssims))
        avg_psnr = float(np.mean(val_psnrs))
        avg_pcc = float(np.mean(val_pccs))
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        is_best = avg_ssim > best_val_ssim
        status = "BEST" if is_best else f"({patience_counter + 1}/{args.patience})"
        LOG.info(
            f"{epoch:4d} {avg_loss:8.4f} {avg_ssim:7.4f} "
            f"{avg_psnr:7.2f} {avg_pcc:7.4f} {lr:9.2e} {elapsed:7.1f}s  {status}"
        )

        if is_best:
            best_val_ssim = avg_ssim
            patience_counter = 0
            BEST["state"] = {k: v.cpu() for k, v in model.state_dict().items()}
            BEST["epoch"] = epoch
            BEST["ssim"] = best_val_ssim
            ckpt = os.path.join(
                checkpoint_dir, f"{args.name}_e{epoch:03d}_ssim{best_val_ssim:.4f}.pth"
            )
            old_ckpt = BEST.get("ckpt")
            torch.save(BEST["state"], ckpt)
            BEST["ckpt"] = ckpt
            if old_ckpt and old_ckpt != ckpt and os.path.exists(old_ckpt):
                try:
                    os.remove(old_ckpt)
                except OSError as exc:
                    LOG.warning(f"Could not remove old checkpoint {old_ckpt}: {exc}")
            LOG.info(f"Saved -> {ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                LOG.info("=" * W)
                LOG.info(f"Early stop at epoch {epoch}. Best SSIM: {best_val_ssim:.4f}")
                LOG.info("=" * W)
                break

    if BEST["state"] is not None:
        torch.save(BEST["state"], model_path)
    LOG.info("=" * W)
    LOG.info(f"Complete. Best SSIM: {best_val_ssim:.4f} (epoch {BEST['epoch']})")
    LOG.info(f"Model saved -> {model_path}")
    LOG.info("=" * W)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/processed/content_quality_csvs/content_quality_minrgb0.50_positive_top1000.csv",
    )
    parser.add_argument("--name", default="v21b_hibou_content_quality_ft")
    parser.add_argument("--init-model", default="models/v21a_hibou_b_model.pth")
    parser.add_argument("--hf-model-id", default="histai/hibou-b")
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--decoder-lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

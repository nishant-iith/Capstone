"""
v20: ConvNeXt-Base (LAION-2B) + Elastic Augmentation + L1
==========================================================
Architecture : ConvNeXt-Base encoder pretrained on LAION-2B CLIP + UNet decoder
Loss         : L1 only (stable, proven by v19b)
New          : elastic deformation augmentation + differential LR
Metrics      : SSIM, PSNR, PCC logged every epoch
"""

import os
import sys
import time
import signal
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from skimage import io
from skimage.metrics import (
    structural_similarity as ssim_fn,
    peak_signal_noise_ratio as psnr_fn,
)
from scipy.ndimage import map_coordinates, gaussian_filter
import segmentation_models_pytorch as smp
import warnings
import logging

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────
REGISTERED_CSV  = "data/processed/registered_pairs_all.csv"
MODEL_PATH      = "models/v20_fixed_model.pth"
CHECKPOINT_DIR  = "checkpoints/v20_fixed"
LOG_FILE        = "logs/v20_fixed_training.log"

SEED        = 42
TOP_N       = 1000
BATCH_SIZE  = 4          # ConvNeXt-Base ~90M params
ENCODER_LR  = 1e-5       # pretrained encoder — slow
DECODER_LR  = 1e-4       # fresh decoder — normal
EPOCHS      = 80
NUM_WORKERS = 4
PATIENCE    = 15
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W           = 70         # log print width

# ── GPU priority ───────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
try:
    import ctypes
    ctypes.cdll.LoadLibrary("libcuda.so")
except Exception:
    pass

# ── Logging ────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("v20")

# ── Graceful shutdown ──────────────────────────────────────────────────
BEST = {"state": None, "epoch": 0, "ssim": 0.0, "ckpt": None}

def graceful_shutdown(signum, frame):
    log.info(f"Signal {signum} received — saving best model...")
    if BEST["state"] is not None:
        torch.save(BEST["state"], MODEL_PATH)
        log.info(f"Saved best (ep {BEST['epoch']}, SSIM {BEST['ssim']:.4f}) → {MODEL_PATH}")
    sys.exit(0)

for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, graceful_shutdown)


# ── Elastic deformation ────────────────────────────────────────────────
def elastic_transform(image, alpha=60, sigma=6, seed=None):
    """Random elastic deformation to simulate registration residuals."""
    rng   = np.random.RandomState(seed)
    H, W_ = image.shape[:2]
    dx    = gaussian_filter(rng.randn(H, W_), sigma) * alpha
    dy    = gaussian_filter(rng.randn(H, W_), sigma) * alpha
    xs, ys = np.meshgrid(np.arange(W_), np.arange(H))
    cx    = np.clip(xs + dx, 0, W_ - 1)
    cy    = np.clip(ys + dy, 0, H  - 1)
    out   = np.zeros_like(image)
    for c in range(image.shape[2]):
        out[..., c] = map_coordinates(
            image[..., c], [cy.ravel(), cx.ravel()], order=1, mode="reflect"
        ).reshape(H, W_)
    return out


# ── Dataset ────────────────────────────────────────────────────────────
class RegisteredPairsDataset(Dataset):
    def __init__(self, csv_file, top_n, indices=None, augment=True):
        df = pd.read_csv(csv_file).head(top_n).reset_index(drop=True)
        if indices is None:
            indices = np.arange(len(df))
        self.images = []
        self.prefixes = []
        for idx in indices:
            row = df.iloc[idx]
            try:
                s = io.imread(row["stained"]).astype(np.float32)  / 255.0
                u = io.imread(row["unstained"]).astype(np.float32) / 255.0
                if s.ndim == 2: s = np.stack([s] * 3, axis=-1)
                if u.ndim == 2: u = np.stack([u] * 3, axis=-1)
                self.images.append((s, u))
                self.prefixes.append(row["prefix"])
            except Exception:
                continue
        self.augment = augment
        log.info(f"Loaded {len(self.images)} pairs (augment={augment})")

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        s, u = self.images[idx]
        if self.augment:
            if np.random.random() > 0.5:
                s = np.flip(s, axis=1).copy(); u = np.flip(u, axis=1).copy()
            if np.random.random() > 0.5:
                s = np.flip(s, axis=0).copy(); u = np.flip(u, axis=0).copy()
            if np.random.random() > 0.5:
                seed = np.random.randint(0, 100000)
                u = elastic_transform(u, alpha=60, sigma=6, seed=seed)
        s = torch.from_numpy(s.copy()).permute(2, 0, 1).contiguous()
        u = torch.from_numpy(u.copy()).permute(2, 0, 1).contiguous()
        return u, s


# ── Model ──────────────────────────────────────────────────────────────
class ConvNeXtUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name    = "tu-convnext_base.clip_laion2b",
            encoder_weights = "imagenet",
            in_channels     = 3,
            classes         = 3,
            activation      = None,
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def encoder_params(self):
        return self.model.encoder.parameters()

    def decoder_params(self):
        return (
            list(self.model.decoder.parameters())
            + list(self.model.segmentation_head.parameters())
        )


# ── Metrics ────────────────────────────────────────────────────────────
def compute_metrics(pred_np, target_np):
    """
    pred_np, target_np: float32 [H, W, 3] in [0, 1].
    Returns (ssim, psnr, pcc).
    """
    ssim = ssim_fn(target_np, pred_np, channel_axis=2, data_range=1.0)
    psnr = psnr_fn(target_np, pred_np, data_range=1.0)
    pcc  = float(np.corrcoef(pred_np.ravel(), target_np.ravel())[0, 1])
    return ssim, psnr, pcc


# ── Training ───────────────────────────────────────────────────────────
def train_v20():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

    # Data: one shared deterministic split. Previous v20 runs created train and
    # validation datasets from independent shuffles, which leaked validation
    # images into training.
    df = pd.read_csv(REGISTERED_CSV).head(TOP_N).reset_index(drop=True)
    split_perm = np.random.RandomState(SEED).permutation(len(df))
    n_train = int(0.9 * len(df))
    train_indices = split_perm[:n_train]
    val_indices = split_perm[n_train:]

    train_ds = RegisteredPairsDataset(
        REGISTERED_CSV, TOP_N, indices=train_indices, augment=True
    )
    val_ds = RegisteredPairsDataset(
        REGISTERED_CSV, TOP_N, indices=val_indices, augment=False
    )
    overlap = set(train_ds.prefixes) & set(val_ds.prefixes)
    if overlap:
        raise RuntimeError(f"Train/val split overlap detected: {len(overlap)} pairs")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    # Model
    model = ConvNeXtUNet().to(DEVICE)

    # Differential LR — pretrained encoder slow, fresh decoder fast
    optimizer = optim.AdamW(
        [
            {"params": model.encoder_params(), "lr": ENCODER_LR},
            {"params": model.decoder_params(), "lr": DECODER_LR},
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-7
    )
    criterion = nn.L1Loss()
    scaler    = GradScaler("cuda")

    n_enc = sum(p.numel() for p in model.encoder_params()) / 1e6
    n_dec = sum(p.numel() for p in model.decoder_params()) / 1e6

    log.info("=" * W)
    log.info("v20: ConvNeXt-Base LAION-2B + Elastic Aug + L1")
    log.info("=" * W)
    log.info(f"Encoder : convnext_base.clip_laion2b  ({n_enc:.1f}M params, LR={ENCODER_LR})")
    log.info(f"Decoder : UNet decoder+head           ({n_dec:.1f}M params, LR={DECODER_LR})")
    log.info(f"Elastic : alpha=60, sigma=6, p=0.5 on unstained only")
    log.info(f"Batch   : {BATCH_SIZE}  |  Device: {DEVICE}  |  Epochs: {EPOCHS}")
    log.info(f"Split   : seed={SEED}  |  overlap={len(overlap)}")
    log.info(f"Train   : {len(train_ds)}  |  Val: {len(val_ds)}")
    log.info(f"Early   : patience={PATIENCE}")
    log.info("=" * W)
    log.info(f"{'Ep':>4} {'Loss':>8} {'SSIM':>7} {'PSNR':>7} {'PCC':>7} {'EncLR':>9} {'Time':>7}  Status")
    log.info("-" * W)

    best_val_ssim   = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for u, s in train_loader:
            u, s = u.to(DEVICE), s.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                pred = model(u)
                loss = criterion(pred, s)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_ssims, val_psnrs, val_pccs = [], [], []

        with torch.no_grad():
            for u, s in val_loader:
                u, s = u.to(DEVICE), s.to(DEVICE)
                pred = model(u)
                for b in range(pred.shape[0]):
                    p_np = pred[b].cpu().numpy().transpose(1, 2, 0)
                    s_np = s[b].cpu().numpy().transpose(1, 2, 0)
                    ssim, psnr, pcc = compute_metrics(p_np, s_np)
                    val_ssims.append(ssim)
                    val_psnrs.append(psnr)
                    val_pccs.append(pcc)

        avg_ssim  = float(np.mean(val_ssims))
        avg_psnr  = float(np.mean(val_psnrs))
        avg_pcc   = float(np.mean(val_pccs))
        avg_loss  = train_loss / len(train_loader)
        elapsed   = time.time() - t0
        enc_lr    = optimizer.param_groups[0]["lr"]
        scheduler.step()

        is_best = avg_ssim > best_val_ssim
        status  = "★ BEST" if is_best else f"({patience_counter + 1}/{PATIENCE})"

        log.info(
            f"{epoch:4d} {avg_loss:8.4f} {avg_ssim:7.4f} {avg_psnr:7.2f} {avg_pcc:7.4f}"
            f" {enc_lr:9.2e} {elapsed:7.1f}s  {status}"
        )

        if is_best:
            best_val_ssim    = avg_ssim
            patience_counter = 0
            BEST["state"]    = {k: v.cpu() for k, v in model.state_dict().items()}
            BEST["epoch"]    = epoch
            BEST["ssim"]     = best_val_ssim
            ckpt = os.path.join(CHECKPOINT_DIR, f"v20_fixed_e{epoch:03d}_ssim{best_val_ssim:.4f}.pth")
            old_ckpt = BEST.get("ckpt")
            torch.save(BEST["state"], ckpt)
            BEST["ckpt"] = ckpt
            if old_ckpt and old_ckpt != ckpt and os.path.exists(old_ckpt):
                try:
                    os.remove(old_ckpt)
                except OSError as exc:
                    log.warning(f"Could not remove old checkpoint {old_ckpt}: {exc}")
            log.info(f"  ★ Saved → {ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info("=" * W)
                log.info(f"Early stop at epoch {epoch}. Best SSIM: {best_val_ssim:.4f}")
                log.info("=" * W)
                break

    # ── Final save ─────────────────────────────────────────────────────
    if BEST["state"] is not None:
        torch.save(BEST["state"], MODEL_PATH)
    log.info("=" * W)
    log.info(f"Complete!  Best SSIM: {best_val_ssim:.4f}  (epoch {BEST['epoch']})")
    log.info(f"Model saved → {MODEL_PATH}")
    log.info("=" * W)


if __name__ == "__main__":
    train_v20()

"""
v21A-B: Hibou-B frozen pathology features + high-resolution input-skip decoder.

Purpose:
- First real v21A encoder ablation after v20_fixed.
- Keep the v20_fixed clean 900/100 split, L1-only loss, and elastic augmentation.
- Use cached Hibou-B weights through transformers; do not require a token at runtime
  once the model has been downloaded.

Important:
This is not a direct smp.Unet encoder swap. Hibou-B is a DINOv2/ViT model with
patch tokens, so the model uses frozen 224px Hibou tokens as semantic context and
adds a shallow high-resolution input pyramid to preserve 1024px detail.
"""

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
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

warnings.filterwarnings("ignore")


# Config
REGISTERED_CSV = "data/processed/registered_pairs_all.csv"
HF_MODEL_ID = "histai/hibou-b"
MODEL_PATH = "models/v21a_hibou_b_model.pth"
CHECKPOINT_DIR = "checkpoints/v21a_hibou_b"
LOG_FILE = "logs/v21a_hibou_b_training.log"

SEED = 42
TOP_N = 1000
BATCH_SIZE = 2
EPOCHS = 120
NUM_WORKERS = 4
DECODER_LR = 1e-4
PATIENCE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W = 82

HIBOU_MEAN = torch.tensor([0.7068, 0.5755, 0.7220]).view(1, 3, 1, 1)
HIBOU_STD = torch.tensor([0.1950, 0.2316, 0.1816]).view(1, 3, 1, 1)


os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("v21a_hibou_b")

BEST = {"state": None, "epoch": 0, "ssim": 0.0, "ckpt": None}


def graceful_shutdown(signum, frame):
    log.info(f"Signal {signum} received - saving best model if available")
    if BEST["state"] is not None:
        torch.save(BEST["state"], MODEL_PATH)
        log.info(f"Saved best model to {MODEL_PATH}")
    sys.exit(0)


for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, graceful_shutdown)


def elastic_transform(image, alpha=60, sigma=6, seed=None):
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
    def __init__(self, csv_file, top_n, indices, augment):
        df = pd.read_csv(csv_file).head(top_n).reset_index(drop=True)
        self.images = []
        self.prefixes = []
        for idx in indices:
            row = df.iloc[int(idx)]
            stained = io.imread(row["stained"]).astype(np.float32) / 255.0
            unstained = io.imread(row["unstained"]).astype(np.float32) / 255.0
            if stained.ndim == 2:
                stained = np.stack([stained] * 3, axis=-1)
            if unstained.ndim == 2:
                unstained = np.stack([unstained] * 3, axis=-1)
            self.images.append((stained, unstained))
            self.prefixes.append(row["prefix"])
        self.augment = augment
        log.info(f"Loaded {len(self.images)} pairs (augment={augment})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        stained, unstained = self.images[idx]
        if self.augment:
            if np.random.random() > 0.5:
                stained = np.flip(stained, axis=1).copy()
                unstained = np.flip(unstained, axis=1).copy()
            if np.random.random() > 0.5:
                stained = np.flip(stained, axis=0).copy()
                unstained = np.flip(unstained, axis=0).copy()
            if np.random.random() > 0.5:
                seed = np.random.randint(0, 100000)
                unstained = elastic_transform(unstained, alpha=60, sigma=6, seed=seed)

        stained = torch.from_numpy(stained.copy()).permute(2, 0, 1).contiguous()
        unstained = torch.from_numpy(unstained.copy()).permute(2, 0, 1).contiguous()
        return unstained, stained


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.down(x)


class UpFuseBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse = conv_block(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class InputPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1024 = conv_block(3, 24)
        self.s512 = DownBlock(24, 32)
        self.s256 = DownBlock(32, 64)
        self.s128 = DownBlock(64, 96)
        self.s64 = DownBlock(96, 128)
        self.s32 = DownBlock(128, 160)

    def forward(self, x):
        s1024 = self.s1024(x)
        s512 = self.s512(s1024)
        s256 = self.s256(s512)
        s128 = self.s128(s256)
        s64 = self.s64(s128)
        s32 = self.s32(s64)
        return {
            "s1024": s1024,
            "s512": s512,
            "s256": s256,
            "s128": s128,
            "s64": s64,
            "s32": s32,
        }


class HibouBInputSkipUNet(nn.Module):
    def __init__(self, hf_model_id):
        super().__init__()
        self.hibou = AutoModel.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            local_files_only=True,
        )
        for param in self.hibou.parameters():
            param.requires_grad = False
        self.hibou.eval()

        hidden = int(self.hibou.config.hidden_size)
        self.num_register_tokens = int(getattr(self.hibou.config, "num_register_tokens", 4))

        self.input_pyramid = InputPyramid()
        self.hibou_proj = nn.Sequential(
            nn.Conv2d(hidden, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.up32 = UpFuseBlock(512, 160, 256)
        self.up64 = UpFuseBlock(256, 128, 192)
        self.up128 = UpFuseBlock(192, 96, 128)
        self.up256 = UpFuseBlock(128, 64, 96)
        self.up512 = UpFuseBlock(96, 32, 64)
        self.up1024 = UpFuseBlock(64, 24, 32)
        self.head = nn.Conv2d(32, 3, 1)

    def train(self, mode=True):
        super().train(mode)
        self.hibou.eval()
        return self

    def encode_hibou(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = HIBOU_MEAN.to(device=x.device, dtype=x.dtype)
        std = HIBOU_STD.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        with torch.no_grad():
            out = self.hibou(pixel_values=x, return_dict=True)
        tokens = out.last_hidden_state[:, 1 + self.num_register_tokens :, :]
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise RuntimeError(f"Unexpected Hibou token count: {tokens.shape[1]}")
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)

    def forward(self, x):
        skips = self.input_pyramid(x)
        h = self.hibou_proj(self.encode_hibou(x))
        x = self.up32(h, skips["s32"])
        x = self.up64(x, skips["s64"])
        x = self.up128(x, skips["s128"])
        x = self.up256(x, skips["s256"])
        x = self.up512(x, skips["s512"])
        x = self.up1024(x, skips["s1024"])
        return torch.sigmoid(self.head(x))


def compute_metrics(pred_np, target_np):
    ssim = ssim_fn(target_np, pred_np, channel_axis=2, data_range=1.0)
    psnr = psnr_fn(target_np, pred_np, data_range=1.0)
    pcc = float(np.corrcoef(pred_np.ravel(), target_np.ravel())[0, 1])
    return ssim, psnr, pcc


def build_split():
    df = pd.read_csv(REGISTERED_CSV).head(TOP_N).reset_index(drop=True)
    split_perm = np.random.RandomState(SEED).permutation(len(df))
    n_train = int(0.9 * len(df))
    return split_perm[:n_train], split_perm[n_train:]


def train_v21a():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

    train_indices, val_indices = build_split()
    train_ds = RegisteredPairsDataset(REGISTERED_CSV, TOP_N, train_indices, augment=True)
    val_ds = RegisteredPairsDataset(REGISTERED_CSV, TOP_N, val_indices, augment=False)
    overlap = set(train_ds.prefixes) & set(val_ds.prefixes)
    if overlap:
        raise RuntimeError(f"Train/val split overlap detected: {len(overlap)} pairs")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    model = HibouBInputSkipUNet(HF_MODEL_ID).to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=DECODER_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    criterion = nn.L1Loss()
    scaler = GradScaler("cuda", enabled=DEVICE.type == "cuda")

    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    log.info("=" * W)
    log.info("v21A-B: frozen Hibou-B + high-resolution input-skip decoder")
    log.info("=" * W)
    log.info(f"Hibou   : {HF_MODEL_ID}, cached local load, frozen")
    log.info(f"Params  : total={n_total:.1f}M, trainable={n_trainable:.1f}M")
    log.info(f"Data    : train={len(train_ds)} val={len(val_ds)} overlap={len(overlap)}")
    log.info(f"Batch   : {BATCH_SIZE} | Device: {DEVICE} | Epochs: {EPOCHS}")
    log.info(f"LR      : decoder/input pyramid={DECODER_LR}")
    log.info(f"Early   : patience={PATIENCE}")
    log.info("=" * W)
    log.info(f"{'Ep':>4} {'Loss':>8} {'SSIM':>7} {'PSNR':>7} {'PCC':>7} {'LR':>9} {'Time':>7}  Status")
    log.info("-" * W)

    best_val_ssim = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for unstained, stained in train_loader:
            unstained = unstained.to(DEVICE)
            stained = stained.to(DEVICE)
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
                unstained = unstained.to(DEVICE)
                stained = stained.to(DEVICE)
                pred = model(unstained)
                for bidx in range(pred.shape[0]):
                    pred_np = pred[bidx].cpu().numpy().transpose(1, 2, 0)
                    stained_np = stained[bidx].cpu().numpy().transpose(1, 2, 0)
                    ssim, psnr, pcc = compute_metrics(pred_np, stained_np)
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
        status = "BEST" if is_best else f"({patience_counter + 1}/{PATIENCE})"
        log.info(
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
                CHECKPOINT_DIR,
                f"v21a_hibou_b_e{epoch:03d}_ssim{best_val_ssim:.4f}.pth",
            )
            old_ckpt = BEST.get("ckpt")
            torch.save(BEST["state"], ckpt)
            BEST["ckpt"] = ckpt
            if old_ckpt and old_ckpt != ckpt and os.path.exists(old_ckpt):
                try:
                    os.remove(old_ckpt)
                except OSError as exc:
                    log.warning(f"Could not remove old checkpoint {old_ckpt}: {exc}")
            log.info(f"Saved -> {ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info("=" * W)
                log.info(f"Early stop at epoch {epoch}. Best SSIM: {best_val_ssim:.4f}")
                log.info("=" * W)
                break

    if BEST["state"] is not None:
        torch.save(BEST["state"], MODEL_PATH)
    log.info("=" * W)
    log.info(f"Complete. Best SSIM: {best_val_ssim:.4f} (epoch {BEST['epoch']})")
    log.info(f"Model saved -> {MODEL_PATH}")
    log.info("=" * W)


if __name__ == "__main__":
    train_v21a()

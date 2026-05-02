"""
v21A smoke test: Hibou-B frozen feature encoder + lightweight decoder.

This is a compatibility/resource smoke test, not a final v21 architecture.
It verifies that gated Hibou-B can load, produce spatial patch tokens, train
through a decoder, and evaluate on the same clean v20 split without overlap.
"""

import logging
import os
import signal
import sys
import time
import warnings

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
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# Config
REGISTERED_CSV = "data/processed/registered_pairs_all.csv"
HF_MODEL_ID = "histai/hibou-b"
MODEL_PATH = "models/v21a_hibou_b_smoke_model.pth"
CHECKPOINT_DIR = "checkpoints/v21a_hibou_b_smoke"
LOG_FILE = "logs/v21a_hibou_b_smoke.log"

SEED = 42
TOP_N = 1000
TRAIN_LIMIT = 180
VAL_LIMIT = 40
BATCH_SIZE = 2
EPOCHS = 1
NUM_WORKERS = 2
DECODER_LR = 1e-4
PATIENCE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W = 74

HIBOU_MEAN = torch.tensor([0.7068, 0.5755, 0.7220]).view(1, 3, 1, 1)
HIBOU_STD = torch.tensor([0.1950, 0.2316, 0.1816]).view(1, 3, 1, 1)


os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("v21a_smoke")

BEST = {"state": None, "epoch": 0, "ssim": 0.0}


def graceful_shutdown(signum, frame):
    log.info(f"Signal {signum} received - saving best smoke model if available")
    if BEST["state"] is not None:
        torch.save(BEST["state"], MODEL_PATH)
        log.info(f"Saved best smoke model to {MODEL_PATH}")
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


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HibouSmokeUNet(nn.Module):
    def __init__(self, hf_model_id, token):
        super().__init__()
        self.hibou = AutoModel.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            token=token,
        )
        for param in self.hibou.parameters():
            param.requires_grad = False
        self.hibou.eval()
        hidden = int(self.hibou.config.hidden_size)
        self.num_register_tokens = int(getattr(self.hibou.config, "num_register_tokens", 4))

        self.proj = nn.Sequential(
            nn.Conv2d(hidden, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.dec1 = UpBlock(512, 256)  # 16 -> 32
        self.dec2 = UpBlock(256, 128)  # 32 -> 64
        self.dec3 = UpBlock(128, 64)   # 64 -> 128
        self.dec4 = UpBlock(64, 32)    # 128 -> 256
        self.dec5 = UpBlock(32, 16)    # 256 -> 512
        self.dec6 = UpBlock(16, 16)    # 512 -> 1024
        self.head = nn.Conv2d(16, 3, 1)

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
        feat = self.encode_hibou(x)
        x = self.proj(feat)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        return torch.sigmoid(self.head(x))


def compute_metrics(pred_np, target_np):
    ssim = ssim_fn(target_np, pred_np, channel_axis=2, data_range=1.0)
    psnr = psnr_fn(target_np, pred_np, data_range=1.0)
    pcc = float(np.corrcoef(pred_np.ravel(), target_np.ravel())[0, 1])
    return ssim, psnr, pcc


def train_smoke():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN for gated Hibou access")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

    df = pd.read_csv(REGISTERED_CSV).head(TOP_N).reset_index(drop=True)
    split_perm = np.random.RandomState(SEED).permutation(len(df))
    n_train = int(0.9 * len(df))
    train_indices = split_perm[:n_train][:TRAIN_LIMIT]
    val_indices = split_perm[n_train:][:VAL_LIMIT]

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

    model = HibouSmokeUNet(HF_MODEL_ID, token).to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=DECODER_LR, weight_decay=1e-4)
    criterion = nn.L1Loss()
    scaler = GradScaler("cuda", enabled=DEVICE.type == "cuda")

    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    log.info("=" * W)
    log.info("v21A smoke: frozen Hibou-B tokens + lightweight decoder")
    log.info("=" * W)
    log.info(f"Hibou   : {HF_MODEL_ID} ({n_total:.1f}M total params)")
    log.info(f"Train   : decoder only ({n_trainable:.1f}M trainable params)")
    log.info(f"Data    : train={len(train_ds)} val={len(val_ds)} overlap={len(overlap)}")
    log.info(f"Batch   : {BATCH_SIZE} | Device: {DEVICE} | Epochs: {EPOCHS}")
    log.info(f"LR      : decoder={DECODER_LR}")
    log.info("=" * W)
    log.info(f"{'Ep':>4} {'Loss':>8} {'SSIM':>7} {'PSNR':>7} {'PCC':>7} {'Time':>7}")
    log.info("-" * W)

    best_val_ssim = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.hibou.eval()
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
        log.info(
            f"{epoch:4d} {avg_loss:8.4f} {avg_ssim:7.4f} "
            f"{avg_psnr:7.2f} {avg_pcc:7.4f} {elapsed:7.1f}s"
        )

        if avg_ssim > best_val_ssim:
            best_val_ssim = avg_ssim
            patience_counter = 0
            BEST["state"] = {k: v.cpu() for k, v in model.state_dict().items()}
            BEST["epoch"] = epoch
            BEST["ssim"] = best_val_ssim
            ckpt = os.path.join(
                CHECKPOINT_DIR,
                f"v21a_hibou_b_smoke_e{epoch:03d}_ssim{best_val_ssim:.4f}.pth",
            )
            torch.save(BEST["state"], ckpt)
            log.info(f"Saved smoke checkpoint -> {ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if BEST["state"] is not None:
        torch.save(BEST["state"], MODEL_PATH)
    log.info("=" * W)
    log.info(f"Smoke complete. Best SSIM: {best_val_ssim:.4f} (epoch {BEST['epoch']})")
    log.info(f"Model saved -> {MODEL_PATH}")
    log.info("=" * W)


if __name__ == "__main__":
    train_smoke()

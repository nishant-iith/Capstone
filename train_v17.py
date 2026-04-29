"""
v17 (rev2): 5-level Simple U-Net + L1+SSIM on full-size 1024×1024.
NO warm-start (v14 4-level → v17 5-level scale mismatch broke prior run).
Kaiming init. Seeded split. Augment train-only (val pristine).
Target: 0.73-0.76 SSIM.
"""

import os
import time
import signal
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from skimage import io
from skimage.metrics import structural_similarity as ssim_fn
from pytorch_msssim import SSIM as SSIM_Module
import warnings
import logging
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
REGISTERED_CSV  = "data/processed/registered_pairs_all.csv"
MODEL_PATH      = "models/v17_model.pth"
SEED            = 42
CHECKPOINT_DIR  = "checkpoints/v17"
VALID_OUT_DIR   = "val_results/v17"
LOG_FILE        = "logs/v17_training.log"

TOP_N           = 1000
BATCH_SIZE      = 20
LEARNING_RATE   = 5e-4
EPOCHS          = 150
NUM_WORKERS     = 0
PATCH_SIZE      = 1024
PATIENCE        = 10       # Early stop after 10 bad epochs
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W               = 90

LAMBDA_L1   = 0.5
LAMBDA_SSIM = 0.5

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("v17")

# ── Process Protection ─────────────────────────────────────────────────────────
try:
    os.nice(-10)  # High CPU priority
except PermissionError:
    pass

# ── Graceful Shutdown (SIGINT, SIGTERM, SIGHUP, SIGQUIT) ──────────────────────
BEST_MODEL_STATE = {"state": None, "epoch": 0, "ssim": 0.0}

def graceful_shutdown(signum, frame):
    sig_names = {2: "SIGINT", 15: "SIGTERM", 1: "SIGHUP", 3: "SIGQUIT"}
    log.info(f"{'='*W}")
    log.info(f"GRACEFUL SHUTDOWN (signal {sig_names.get(signum, signum)})")
    if BEST_MODEL_STATE["state"] is not None:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(BEST_MODEL_STATE["state"], MODEL_PATH)
        log.info(f"Best model saved → Epoch {BEST_MODEL_STATE['epoch']}  SSIM {BEST_MODEL_STATE['ssim']:.4f}")
        log.info(f"Path: {MODEL_PATH}")
    else:
        log.info("No model saved (training incomplete)")
    log.info(f"{'='*W}")
    sys.exit(0)

for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
    signal.signal(sig, graceful_shutdown)

# ── Model (same as v14) ────────────────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = conv_block(512, 1024)
        self.pool5 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(1024, 2048)
        self.up5   = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec5  = conv_block(2048, 1024)
        self.up4   = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4  = conv_block(1024, 512)
        self.up3   = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3  = conv_block(512, 256)
        self.up2   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2  = conv_block(256, 128)
        self.up1   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1  = conv_block(128, 64)
        self.final = nn.Conv2d(64, 3, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))
        b  = self.bottleneck(self.pool5(e5))
        d5 = self.dec5(torch.cat([self.up5(b),  e5], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.sigmoid(self.final(d1))

# ── Dataset (RAM cached, shared cache between train/val) ──────────────────────
class _SharedCache:
    def __init__(self, csv_file, top_n):
        df = pd.read_csv(csv_file).head(top_n).reset_index(drop=True)
        print(f"  Caching {len(df)} full-size pairs into RAM...")
        self.s_imgs, self.u_imgs = [], []
        for idx, row in df.iterrows():
            s = io.imread(row["stained"]).astype(np.float32)  / 255.0
            u = io.imread(row["unstained"]).astype(np.float32) / 255.0
            if len(s.shape) == 2:
                s = np.stack([s]*3, axis=-1)
                u = np.stack([u]*3, axis=-1)
            self.s_imgs.append(torch.from_numpy(s).permute(2, 0, 1).contiguous())
            self.u_imgs.append(torch.from_numpy(u).permute(2, 0, 1).contiguous())
            if (idx + 1) % 200 == 0:
                print(f"    Loaded {idx+1}/{len(df)}...", end='\r')
        print(f"\n  RAM cache complete: {len(df)} pairs")
        self.n = len(df)


class FullSizeDataset(Dataset):
    def __init__(self, cache, indices, augment):
        self.cache = cache
        self.indices = list(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        u = self.cache.u_imgs[idx]
        s = self.cache.s_imgs[idx]
        if self.augment:
            if np.random.random() > 0.5:
                u = torch.flip(u, [2]); s = torch.flip(s, [2])
            if np.random.random() > 0.5:
                u = torch.flip(u, [1]); s = torch.flip(s, [1])
        return u, s

# ── Combined L1 + SSIM Loss ────────────────────────────────────────────────────
class L1SSIMLoss(nn.Module):
    def __init__(self, lambda_l1=0.5, lambda_ssim=0.5):
        super().__init__()
        self.lambda_l1   = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.l1   = nn.L1Loss()
        self.ssim = SSIM_Module(data_range=1.0, size_average=True, channel=3)

    def forward(self, pred, target):
        l1_loss   = self.l1(pred, target)
        ssim_loss = 1.0 - self.ssim(pred, target)  # Maximize SSIM = minimize (1 - SSIM)
        return self.lambda_l1 * l1_loss + self.lambda_ssim * ssim_loss

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VALID_OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Cache once, then build train/val with disjoint indices and per-split augment
    cache = _SharedCache(REGISTERED_CSV, top_n=TOP_N)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(cache.n)
    n_val = int(round(0.1 * cache.n))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    log.info(f"Split (seed={SEED}): train={len(train_idx)}  val={len(val_idx)}")

    train_ds = FullSizeDataset(cache, train_idx, augment=True)
    val_ds   = FullSizeDataset(cache, val_idx,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model — resume from best v17 checkpoint if available, else fresh Kaiming init
    model = UNet().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    criterion = L1SSIMLoss(lambda_l1=LAMBDA_L1, lambda_ssim=LAMBDA_SSIM)
    scaler    = GradScaler('cuda')

    print()
    print(f"  {'='*W}")
    print(f"  {'V17: SIMPLE U-NET | L1+SSIM LOSS | FULL-SIZE (1024×1024)':^{W}}")
    print(f"  {'='*W}")
    print(f"  Architecture       5-level U-Net (encoder-decoder, sigmoid output)")
    print(f"  Loss               L1({LAMBDA_L1}) + SSIM({LAMBDA_SSIM}) — directly optimize metric")
    print(f"  Optimizer          Adam  lr={LEARNING_RATE}")
    print(f"  LR Schedule        CosineAnnealingLR {LEARNING_RATE}→1e-7")
    print(f"  Device             {str(DEVICE).upper()}")
    print(f"  Batch Size         {BATCH_SIZE}  |  Workers: {NUM_WORKERS}")
    print(f"  Images (Full-sz)   Top-{TOP_N} pairs (mean SSIM ≈0.61)")
    print(f"  Train/Val Split    {len(train_idx)}/{len(val_idx)}  (seed={SEED})")
    print(f"  Max Epochs         {EPOCHS}  |  Target: SSIM > 0.73")
    print(f"  Warm-start         disabled (Kaiming init from scratch)")
    print(f"  {'='*W}")
    print()

    best_val_ssim = 0.0
    patience_counter = 0
    START_EPOCH = 1

    resume_ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]) if os.path.isdir(CHECKPOINT_DIR) else []
    if resume_ckpts:
        resume_file = resume_ckpts[-1]
        resume_path = os.path.join(CHECKPOINT_DIR, resume_file)
        model.load_state_dict(torch.load(resume_path, map_location=DEVICE))
        # parse epoch and ssim from filename: v17_epochNNN_ssimX.XXXX.pth
        import re
        m = re.search(r'epoch(\d+)_ssim([\d]+\.[\d]+)', resume_file)
        if m:
            START_EPOCH = int(m.group(1)) + 1
            best_val_ssim = float(m.group(2))
        log.info(f"Resumed: {resume_path}  |  Start epoch: {START_EPOCH}  |  Best SSIM: {best_val_ssim:.4f}")
    else:
        log.info("No checkpoint found — Kaiming init from scratch")

    history = []

    log.info(f"{'='*W}")
    log.info("TRAINING START")
    log.info(f"PID: {os.getpid()}  |  Device: {DEVICE}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    log.info(f"{'='*W}")

    for epoch in range(START_EPOCH, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for i, (u, s) in enumerate(train_loader):
            u, s = u.to(DEVICE), s.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                pred = model(u)
                loss = criterion(pred, s)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if (i + 1) % 5 == 0 or i == 0:
                pct = (i + 1) / len(train_loader) * 100
                bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
                print(f"  Epoch {epoch:03d} [{bar}] {i+1:>3}/{len(train_loader)}  Loss: {loss.item():.4f}", end='\r')

        # Validation
        model.eval()
        val_ssims = []
        with torch.no_grad():
            for u, s in val_loader:
                u, s = u.to(DEVICE), s.to(DEVICE)
                pred = model(u)
                for b in range(pred.shape[0]):
                    p_np = pred[b].cpu().float().numpy().transpose(1, 2, 0)
                    s_np = s[b].cpu().float().numpy().transpose(1, 2, 0)
                    score = ssim_fn(p_np, s_np, channel_axis=2, data_range=1.0)
                    val_ssims.append(score)

        avg_val_ssim = float(np.mean(val_ssims))
        avg_train_loss = train_loss / len(train_loader)
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        gpu_mem = torch.cuda.memory_allocated() / 1e9
        status = "★ BEST" if avg_val_ssim > best_val_ssim else f"  (patience {patience_counter+1}/{PATIENCE})"

        log.info(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"Val SSIM: {avg_val_ssim:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"GPU: {gpu_mem:.1f}GB | "
            f"Time: {elapsed:.1f}s | {status}"
        )

        history.append({"epoch": epoch, "loss": avg_train_loss, "ssim": avg_val_ssim})

        if avg_val_ssim > best_val_ssim:
            best_val_ssim = avg_val_ssim
            patience_counter = 0
            state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(state, MODEL_PATH)
            BEST_MODEL_STATE["state"] = state
            BEST_MODEL_STATE["epoch"] = epoch
            BEST_MODEL_STATE["ssim"]  = best_val_ssim
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"v17_epoch{epoch:03d}_ssim{best_val_ssim:.4f}.pth")
            torch.save(state, ckpt_path)
            log.info(f"  → Checkpoint saved: {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info(f"{'='*W}")
                log.info(f"EARLY STOPPING — patience {PATIENCE} exceeded at epoch {epoch}")
                log.info(f"Best SSIM: {best_val_ssim:.4f}  (Epoch {BEST_MODEL_STATE['epoch']})")
                log.info(f"{'='*W}")
                break

        torch.cuda.empty_cache()

    log.info(f"{'='*W}")
    log.info(f"TRAINING COMPLETE | Best Val SSIM: {best_val_ssim:.4f} | Model: {MODEL_PATH}")
    log.info(f"{'='*W}")

if __name__ == "__main__":
    main()

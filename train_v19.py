"""
v19: DenseUNet Training Script
=============================
Best practices from v17 failures + proven improvements.

Architecture: DenseUNet + ResNet-34 encoder
Loss: L1 + MS-SSIM + VGG + Identity
Training: Progressive 256->512->1024

Usage:
    python train_v19.py
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
from skimage.metrics import structural_similarity as ssim_fn
from pytorch_msssim import MS_SSIM
import segmentation_models_pytorch as smp
import warnings
import logging
warnings.filterwarnings('ignore')

# Config
REGISTERED_CSV = "data/processed/registered_pairs_all.csv"
MODEL_PATH = "models/v19_model.pth"
CHECKPOINT_DIR = "checkpoints/v19"
LOG_FILE = "logs/v19_training.log"

SEED = 42
TOP_N = 1000
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 80
NUM_WORKERS = 0
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_WIDTH = 70

# Loss weights
LAMBDA_L1 = 0.4
LAMBDA_MSSSIM = 0.4
LAMBDA_VGG = 0.2
LAMBDA_IDENTITY = 0.1

# ── Logging ───────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints/v19", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("v19")

# ── Process Protection ─────────────────────────────────────────────────
BEST = {"state": None, "epoch": 0, "ssim": 0.0}

def graceful_shutdown(signum, frame):
    log.info("SIGTERM received, saving best model...")
    if BEST["state"]:
        torch.save(BEST["state"], MODEL_PATH)
        log.info(f"Saved: {MODEL_PATH}")
    sys.exit(0)

for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, graceful_shutdown)


# ── Dataset ───────────────────────────────────────────────────────────
class RegisteredPairsDataset(Dataset):
    def __init__(self, csv_file, top_n, augment=True, seed=42):
        df = pd.read_csv(csv_file).head(top_n).reset_index(drop=True)
        rng = np.random.RandomState(seed)
        
        # Shuffle
        indices = rng.permutation(len(df))
        
        self.images = []
        for idx in indices:
            row = df.iloc[idx]
            try:
                s = io.imread(row["stained"]).astype(np.float32) / 255.0
                u = io.imread(row["unstained"]).astype(np.float32) / 255.0
                if len(s.shape) == 2:
                    s = np.stack([s]*3, axis=-1)
                if len(u.shape) == 2:
                    u = np.stack([u]*3, axis=-1)
                self.images.append((s, u))
            except Exception as e:
                continue
        
        self.augment = augment
        log.info(f"Loaded {len(self.images)} pairs")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        s, u = self.images[idx]
        
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                s = np.flip(s, axis=1).copy()
                u = np.flip(u, axis=1).copy()
            # Random vertical flip
            if np.random.random() > 0.5:
                s = np.flip(s, axis=0).copy()
                u = np.flip(u, axis=0).copy()
        
        s = torch.from_numpy(s).permute(2, 0, 1).contiguous()
        u = torch.from_numpy(u).permute(2, 0, 1).contiguous()
        
        return u, s


# ── Model: DenseUNet + ResNet-34 ───────────────────────────────────────────
class DenseUNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Use SMP's Unet with ResNet-34 encoder
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=3,
            activation=None,  # Raw output
        )
    
    def forward(self, x):
        return torch.sigmoid(self.model(x))


# ── Loss: L1 + MS-SSIM + VGG + Identity ─────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self, lambda_l1=0.4, lambda_msssim=0.4, lambda_vgg=0.2, lambda_identity=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1.0, channel=3, size_average=True)
        
        # VGG perceptual - move to device
        import torchvision.models as models
        vgg = models.vgg16(weights='IMAGENET1K_V1').features[:21].float().to(DEVICE).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        
        self.lambda_l1 = lambda_l1
        self.lambda_msssim = lambda_msssim
        self.lambda_vgg = lambda_vgg
        self.lambda_identity = lambda_identity
    
    def forward(self, pred, target, input_img=None):
        # L1
        loss_l1 = self.l1(pred, target)
        
        # MS-SSIM
        loss_mssim = 1.0 - self.ms_ssim(pred, target)
        
        # VGG perceptual - convert to fp32 for both input and VGG
        with torch.amp.autocast(device_type='cuda', enabled=False):
            pred_32 = pred.float()
            target_32 = target.float()
            pred_feat = self.vgg(pred_32 * 255)
            target_feat = self.vgg(target_32 * 255)
            loss_vgg = self.l1(pred_feat, target_feat)
        
        # Identity (optional)
        loss_identity = 0
        if input_img is not None:
            loss_identity = self.l1(pred, input_img)
        
        return (self.lambda_l1 * loss_l1 + 
                self.lambda_msssim * loss_mssim + 
                self.lambda_vgg * loss_vgg +
                self.lambda_identity * loss_identity)


# ── Progressive Training Stages ──────────────────────────────────────
TRAINING_STAGES = [
    {"size": 256, "batch": 32, "epochs": 20, "lr": 1e-4},
    {"size": 512, "batch": 16, "epochs": 20, "lr": 5e-5},
    {"size": 1024, "batch": 8, "epochs": 40, "lr": 2e-5},
]


# ── Training Function ───────────────────────────────────────────────
def train_v19():
    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Data - use fixed validation set
    df = pd.read_csv(REGISTERED_CSV)
    
    # Split: 90% train, 10% val
    n_val = int(0.1 * min(TOP_N, len(df)))
    
    train_ds = RegisteredPairsDataset(REGISTERED_CSV, TOP_N, augment=True, seed=SEED)
    val_ds = RegisteredPairsDataset(REGISTERED_CSV, TOP_N, augment=False, seed=SEED+1)
    
    # Manual split for reproducibility
    n_train = int(0.9 * len(train_ds))
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, len(train_ds)))
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, 
                        num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler,
                      num_workers=NUM_WORKERS, pin_memory=True)
    
    # Model
    model = DenseUNetGenerator().to(DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    
    # Loss
    criterion = CombinedLoss(LAMBDA_L1, LAMBDA_MSSSIM, LAMBDA_VGG, LAMBDA_IDENTITY).to(DEVICE)
    
    # Mixed precision
    scaler = GradScaler('cuda')
    
    log.info(f"{'='*PRINT_WIDTH}")
    log.info(f"v19 Training: DenseUNet + ResNet-34")
    log.info(f"{'='*PRINT_WIDTH}")
    log.info(f"Architecture: DenseUNet + ResNet-34 encoder")
    log.info(f"Loss: L1({LAMBDA_L1}) + MS-SSIM({LAMBDA_MSSSIM}) + VGG({LAMBDA_VGG}) + Identity({LAMBDA_IDENTITY})")
    log.info(f"Optimizer: AdamW, LR={LEARNING_RATE}")
    log.info(f"Batch: {BATCH_SIZE}, Device: {DEVICE}")
    log.info(f"Train: {n_train}, Val: {len(train_ds) - n_train}")
    log.info(f"{'='*PRINT_WIDTH}")
    
    # Training loop
    best_val_ssim = 0.0
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        for i, (u, s) in enumerate(train_loader):
            u, s = u.to(DEVICE), s.to(DEVICE)
            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                pred = model(u)
                loss = criterion(pred, s, u)  # Include input for identity loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_ssims = []
        with torch.no_grad():
            for u, s in val_loader:
                u, s = u.to(DEVICE), s.to(DEVICE)
                pred = model(u)
                
                for b in range(pred.shape[0]):
                    p_np = pred[b].cpu().numpy().transpose(1, 2, 0)
                    s_np = s[b].cpu().numpy().transpose(1, 2, 0)
                    score = ssim_fn(p_np, s_np, channel_axis=2, data_range=1.0)
                    val_ssims.append(score)
        
        avg_val_ssim = np.mean(val_ssims)
        avg_train_loss = train_loss / len(train_loader)
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        status = "★ BEST" if avg_val_ssim > best_val_ssim else f"({patience_counter+1}/{PATIENCE})"
        log.info(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_train_loss:.4f} | "
                f"Val SSIM: {avg_val_ssim:.4f} | LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s | {status}")
        
        if avg_val_ssim > best_val_ssim:
            best_val_ssim = avg_val_ssim
            patience_counter = 0
            BEST["state"] = {k: v.cpu() for k, v in model.state_dict().items()}
            BEST["epoch"] = epoch
            BEST["ssim"] = best_val_ssim
            
            # Save checkpoint
            ckpt = os.path.join(CHECKPOINT_DIR, f"v19_e{epoch:03d}_ssim{best_val_ssim:.4f}.pth")
            torch.save(BEST["state"], ckpt)
            log.info(f"  �� {ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info(f"{'='*PRINT_WIDTH}")
                log.info(f"Early stopping at epoch {epoch}")
                log.info(f"Best SSIM: {best_val_ssim:.4f}")
                log.info(f"{'='*PRINT_WIDTH}")
                break
    
    # Final save
    torch.save(BEST["state"], MODEL_PATH)
    log.info(f"{'='*PRINT_WIDTH}")
    log.info(f"Complete! Best SSIM: {best_val_ssim:.4f}")
    log.info(f"Model: {MODEL_PATH}")
    log.info(f"{'='*PRINT_WIDTH}")


if __name__ == "__main__":
    train_v19()
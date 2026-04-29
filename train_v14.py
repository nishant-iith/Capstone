import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image
import numpy as np
import pandas as pd
import csv
from skimage import io
from skimage.metrics import structural_similarity as ssim_fn
import cv2
import collections
import signal
import sys

# ── Config ────────────────────────────────────────────────────────────────────
STAINED_DIR   = "dataset/stained"
UNSTAINED_DIR = "dataset/unstained"
CSV_ALL       = "data/processed/registered_pairs_all.csv"
TRAIN_CSV     = "data/training_patches.csv"
MODEL_PATH     = "models/v14_model.pth"
VALID_OUT_DIR  = "val_results/v14"
CHECKPOINT_DIR = "checkpoints/v14"
BATCH_SIZE     = 16
LEARNING_RATE  = 1e-4
EPOCHS         = 100
NUM_WORKERS    = 0
PATCH_SIZE     = 512
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W = 90

# Global state for graceful shutdown
BEST_MODEL_STATE = {"model": None, "epoch": 0, "ssim": 0.0}

def graceful_shutdown(signum, frame):
    """Save best model and exit on Ctrl+C."""
    print(f"\n\n  {'='*W}")
    print(f"  GRACEFUL SHUTDOWN")
    if BEST_MODEL_STATE["model"] is not None:
        torch.save(BEST_MODEL_STATE["model"], MODEL_PATH)
        print(f"  Best model saved: Epoch {BEST_MODEL_STATE['epoch']}  SSIM {BEST_MODEL_STATE['ssim']:.4f}")
        print(f"  Path: {MODEL_PATH}")
    else:
        print(f"  No best model to save (training incomplete)")
    print(f"  {'='*W}\n")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_shutdown)
# ──────────────────────────────────────────────────────────────────────────────

# ── Model Architecture (U-Net) ────────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

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
        
        self.bottleneck = conv_block(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        self.final = nn.Conv2d(64, 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        b = self.bottleneck(self.pool4(e4))
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.sigmoid(self.final(d1))

# ── Dataset ────────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        print(f"Caching {len(self.df)} patches into RAM...")
        self.s_images = []
        self.u_images = []
        
        for idx, row in self.df.iterrows():
            s_img = io.imread(row["stained"]) / 255.0
            u_img = io.imread(row["unstained"]) / 255.0
            
            # Convert to tensor and permute (C, H, W)
            s_t = torch.FloatTensor(s_img).permute(2, 0, 1)
            u_t = torch.FloatTensor(u_img).permute(2, 0, 1)
            
            self.s_images.append(s_t)
            self.u_images.append(u_t)
            
            if (idx + 1) % 500 == 0:
                print(f"  Loaded {idx + 1}/{len(self.df)} patches...", end='\r')
        print(f"\n  RAM caching complete.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        s_img = self.s_images[idx]
        u_img = self.u_images[idx]
        
        if self.transform:
            if np.random.random() > 0.5:
                s_img = torch.flip(s_img, [2])
                u_img = torch.flip(u_img, [2])
            if np.random.random() > 0.5:
                s_img = torch.flip(s_img, [1])
                u_img = torch.flip(u_img, [1])
                
        return u_img, s_img

# ── Loss & Metrics ──────────────────────────────────────────────────────────────
def ssim_loss(img1, img2):
    # Simple approximation of SSIM loss using 1 - SSIM
    # Note: Real torch-ssim is better, but for a single script, L1 + structural check is used.
    # We use a standard L1 for training and use ssim_fn for validation.
    return nn.L1Loss()(img1, img2)

# ── Generation Utils ───────────────────────────────────────────────────────────
def generate_full_image(model, unstained_path, output_path):
    model.eval()
    u_full = io.imread(unstained_path) / 255.0
    h, w, c = u_full.shape
    
    # Output image buffer
    out_img = np.zeros_like(u_full, dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    # Sliding window with 50% overlap
    stride = 256
    with torch.no_grad():
        for y in range(0, h - PATCH_SIZE + 1, stride):
            for x in range(0, w - PATCH_SIZE + 1, stride):
                patch = u_full[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch_t = torch.FloatTensor(patch).permute(2,0,1).unsqueeze(0).to(DEVICE)
                
                pred = model(patch_t).squeeze(0).permute(1,2,0).cpu().numpy()
                
                # Gaussian weight for blending
                weight = np.outer(np.hanning(PATCH_SIZE), np.hanning(PATCH_SIZE))
                
                out_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred * weight[:, :, np.newaxis]
                weight_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += weight
                
    # Normalize by weight map to avoid seams
    out_img /= (weight_map[:, :, np.newaxis] + 1e-8)
    out_img = (np.clip(out_img, 0, 1) * 255).astype(np.uint8)
    io.imsave(output_path, out_img, check_contrast=False)

# ── Main Trainer ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(VALID_OUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataset = PatchDataset(TRAIN_CSV)
    # Split 90% train, 10% val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    model = UNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    criterion = nn.L1Loss()
    scaler = torch.amp.GradScaler('cuda')
    
    print()
    print(f"  {'='*W}")
    print(f"  {'V14: U-NET BASELINE  |  512×512 PATCHES':^{W}}")
    print(f"  {'='*W}")
    print(f"  Architecture       U-Net (4-level encoder-decoder)")
    print(f"  Loss               L1")
    print(f"  Optimizer          Adam  lr={LEARNING_RATE}")
    print(f"  Device             {str(DEVICE).upper()}")
    print(f"  Batch Size         {BATCH_SIZE}  |  Workers: {NUM_WORKERS} (cached dataset)")
    print(f"  Train Patches      {train_size}  |  Val Patches: {val_size}")
    print(f"  Max Epochs         {EPOCHS}  |  Target: SSIM > 0.70 (patch-level)")
    print(f"  {'='*W}")
    print()

    best_val_ssim = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
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
                print(f"  Epoch {epoch:03d} [{bar}] {i+1:>4}/{len(train_loader)}  Loss: {loss.item():.4f}", end='\r')
        
        # Validation Tier 1: Patch-level SSIM
        model.eval()
        val_ssims = []
        with torch.no_grad():
            for u, s in val_loader:
                u, s = u.to(DEVICE), s.to(DEVICE)
                pred = model(u)
                
                # Convert to numpy for ssim_fn
                # Ensure shapes are (H, W, C) for ssim_fn
                if pred.shape[0] == 1:
                    p_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    s_np = s.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                else:
                    # If it's a batch, we'll just take the first sample for the validation score
                    p_np = pred[0].cpu().numpy().transpose(1, 2, 0)
                    s_np = s[0].cpu().numpy().transpose(1, 2, 0)
                
                # Force images to have 3 dimensions (H, W, C) for multichannel SSIM
                if len(p_np.shape) == 2:
                    p_np = p_np[:, :, np.newaxis]
                if len(s_np.shape) == 2:
                    s_np = s_np[:, :, np.newaxis]
                
                score = ssim_fn(p_np, s_np, channel_axis=2, data_range=1.0)
                val_ssims.append(score)
        
        avg_val_ssim = np.mean(val_ssims)
        scheduler.step(avg_val_ssim)
        train_time = time.time() - t0

        print(f"\n  Epoch {epoch:03d}/{EPOCHS}  |  Loss: {train_loss/len(train_loader):.4f}  |  Val SSIM: {avg_val_ssim:.4f}  |  Time: {train_time:.1f}s")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  → Full-size validation (random test image)...")
            with open(CSV_ALL, "r") as f:
                all_rows = list(csv.DictReader(f))
            test_row = np.random.choice(all_rows)
            prefix = test_row["prefix"]
            u_path = os.path.join(UNSTAINED_DIR, f"{prefix}_unstained.tif")

            out_path = os.path.join(VALID_OUT_DIR, f"epoch_{epoch}_{prefix}.tif")
            generate_full_image(model, u_path, out_path)

            try:
                gen = io.imread(out_path) / 255.0
                gt = io.imread(os.path.join(STAINED_DIR, f"{prefix}_stained.tif")) / 255.0
                if gen.shape == gt.shape:
                    full_ssim = ssim_fn(gen, gt, channel_axis=2, data_range=1.0)
                    print(f"     Full-size SSIM: {full_ssim:.4f}  ({prefix[:40]})")
            except:
                print(f"     Full-size sample saved to {out_path}")

        # Clear cache to prevent OOM in next epoch
        torch.cuda.empty_cache()

        if avg_val_ssim > best_val_ssim:
            best_val_ssim = avg_val_ssim
            torch.save(model.state_dict(), MODEL_PATH)
            BEST_MODEL_STATE["model"] = model.state_dict().copy()
            BEST_MODEL_STATE["epoch"] = epoch
            BEST_MODEL_STATE["ssim"] = best_val_ssim
            print(f"  ★ Best model saved! Val SSIM: {best_val_ssim:.4f}")

if __name__ == "__main__":
    main()

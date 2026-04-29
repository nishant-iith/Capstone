"""Monitor v14 training. Tracks best model, graceful shutdown."""
import os
import re
import time
import shutil
from pathlib import Path
from skimage import io
from skimage.metrics import structural_similarity as ssim_fn

VAL_DIR = "val_results/v14"
DATASET_DIR = "dataset/stained"
BEST_MODEL_PATH = "models/v14_best.pth"
CHECKPOINTS_DIR = "checkpoints_v14_monitor"

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

best_epoch = 0
best_ssim = 0.0
last_checked = 0

def compute_ssim(epoch, prefix):
    """Compute SSIM for epoch result."""
    gen_path = os.path.join(VAL_DIR, f"epoch_{epoch}_{prefix}.tif")
    gt_path = os.path.join(DATASET_DIR, f"{prefix}_stained.tif")
    
    if not os.path.exists(gen_path) or not os.path.exists(gt_path):
        return None
    
    try:
        gen = io.imread(gen_path) / 255.0
        gt = io.imread(gt_path) / 255.0
        if gen.shape == gt.shape:
            return ssim_fn(gen, gt, channel_axis=2, data_range=1.0)
    except:
        pass
    return None

def check_progress():
    """Check latest val results."""
    global best_epoch, best_ssim, last_checked
    
    if not os.path.exists(VAL_DIR):
        return
    
    files = sorted(os.listdir(VAL_DIR), reverse=True)
    
    for fname in files[:3]:  # Check last 3 results
        m = re.match(r'epoch_(\d+)_(.+)\.tif', fname)
        if not m:
            continue
        
        epoch = int(m.group(1))
        prefix = m.group(2)
        
        if epoch <= last_checked:
            continue
        
        ssim = compute_ssim(epoch, prefix)
        if ssim is None:
            continue
        
        last_checked = epoch
        
        status = ""
        if ssim > best_ssim:
            best_ssim = ssim
            best_epoch = epoch
            shutil.copy("models/v14_model.pth", f"{CHECKPOINTS_DIR}/v14_ep{epoch}_ssim{ssim:.4f}.pth")
            status = " ★ NEW BEST"
        
        print(f"  Epoch {epoch:>3d}  →  SSIM: {ssim:.4f}  (best: ep{best_epoch} {best_ssim:.4f}){status}")

print("\n" + "="*70)
print("V14 TRAINING MONITOR — Best Model Tracker")
print("="*70 + "\n")

try:
    while True:
        check_progress()
        time.sleep(30)
except KeyboardInterrupt:
    print(f"\n\n{'='*70}")
    print(f"GRACEFUL STOP")
    print(f"Best model: Epoch {best_epoch}  →  SSIM {best_ssim:.4f}")
    print(f"Checkpoints saved to: {CHECKPOINTS_DIR}/")
    
    if best_ssim > 0:
        src = f"{CHECKPOINTS_DIR}/v14_ep{best_epoch}_ssim{best_ssim:.4f}.pth"
        if os.path.exists(src):
            shutil.copy(src, BEST_MODEL_PATH)
            print(f"Best model copied to: {BEST_MODEL_PATH}")
    print("="*70 + "\n")

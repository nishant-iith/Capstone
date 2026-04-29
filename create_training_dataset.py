import os
import csv
import time
import numpy as np
import multiprocessing as mp
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage import io, registration, transform
from skimage.metrics import structural_similarity as ssim_fn
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
STAINED_DIR   = "dataset/stained"
UNSTAINED_DIR = "dataset/unstained"
CSV_ALL       = "data/processed/registered_pairs_all.csv"
TRAIN_DIR     = "data/training_patches"
TRAIN_S_DIR   = os.path.join(TRAIN_DIR, "stained")
TRAIN_U_DIR   = os.path.join(TRAIN_DIR, "unstained")
TRAIN_CSV     = "data/training_patches.csv"

NUM_WORKERS   = 60
PATCH_SIZE    = 512
SSIM_THRESHOLD = 0.4
TOP_K_SAMPLES  = 1000
LOAD_THREADS  = 48
W = 85 # Print width
# ──────────────────────────────────────────────────────────────────────────────

# Global - shared via fork CoW
_ITEMS = []

def _worker_process_patch(args):
    """
    Worker function to register a single patch and calculate SSIM.
    args: (img_idx, patch_idx, x, y)
    """
    img_idx, patch_idx, x, y = args
    item = _ITEMS[img_idx]
    
    try:
        s_full = item["s_img"]
        u_full = item["u_img"]
        prefix = item["prefix"]
        
        # Crop patch
        s_patch = s_full[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        u_patch = u_full[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        
        if s_patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
            return None

        # Grayscale for TV-L1
        if len(s_patch.shape) == 3:
            s_gray = cv2.cvtColor(s_patch, cv2.COLOR_RGB2GRAY)
            u_gray = cv2.cvtColor(u_patch, cv2.COLOR_RGB2GRAY)
        else:
            s_gray, u_gray = s_patch, u_patch
            
        s_g = s_gray.astype(np.float32) / 255.0
        u_g = u_gray.astype(np.float32) / 255.0
        
        # TV-L1 Registration
        v_f, u_f = registration.optical_flow_tvl1(s_g, u_g)
        
        # Warp
        gy, gx = np.meshgrid(np.arange(PATCH_SIZE, dtype=np.float32), 
                             np.arange(PATCH_SIZE, dtype=np.float32), indexing="ij")
        
        if len(u_patch.shape) == 3:
            warped = np.zeros_like(u_patch, dtype=np.float64)
            for c in range(3):
                warped[:, :, c] = transform.warp(
                    u_patch[:, :, c] / 255.0,
                    np.array([gy + v_f, gx + u_f]),
                    mode="edge",
                )
            warped = (warped * 255).clip(0, 255).astype(np.uint8)
        else:
            warped = transform.warp(u_patch / 255.0, np.array([gy + v_f, gx + u_f]), mode="edge")
            warped = (warped * 255).clip(0, 255).astype(np.uint8)
        
        # Compute SSIM
        axis = 2 if len(s_patch.shape) == 3 else None
        score = ssim_fn(s_patch, warped, channel_axis=axis, data_range=255)
        
        if score >= SSIM_THRESHOLD:
            patch_name = f"{prefix}_p{patch_idx}.tif"
            out_s = os.path.join(TRAIN_S_DIR, patch_name)
            out_u = os.path.join(TRAIN_U_DIR, patch_name)
            
            io.imsave(out_s, s_patch, check_contrast=False)
            io.imsave(out_u, warped, check_contrast=False)
            
            return {"prefix": prefix, "patch_idx": patch_idx, "stained": out_s, "unstained": out_u, "ssim": round(score, 4)}
            
    except Exception:
        pass
    
    return None

def main():
    global _ITEMS
    print(f"\n{ '='*W }\n{'ELITE PATCH REGISTRATION PIPELINE':^{W}}\n{ '='*W }")
    
    os.makedirs(TRAIN_S_DIR, exist_ok=True)
    os.makedirs(TRAIN_U_DIR, exist_ok=True)
    
    # 1. Get Top-K prefixes
    with open(CSV_ALL, "r") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    all_rows.sort(key=lambda x: float(x["ssim"]), reverse=True)
    top_rows = all_rows[:TOP_K_SAMPLES]
    
    # 2. Step 1: Load images to RAM
    print(f"Step 1/2: Loading {len(top_rows)*2} images to RAM ({LOAD_THREADS} threads)")
    t0 = time.time()
    
    def _load(row):
        prefix = row["prefix"]
        s_path = os.path.join(STAINED_DIR, f"{prefix}_stained.tif")
        u_path = os.path.join(UNSTAINED_DIR, f"{prefix}_unstained.tif")
        return {
            "prefix": prefix,
            "s_img": io.imread(s_path),
            "u_img": io.imread(u_path)
        }

    loaded_items = [None] * len(top_rows)
    with ThreadPoolExecutor(max_workers=LOAD_THREADS) as pool:
        futs = {pool.submit(_load, row): i for i, row in enumerate(top_rows)}
        n = 0
        for fut in as_completed(futs):
            loaded_items[futs[fut]] = fut.result()
            n += 1
            if n % 100 == 0 or n == len(top_rows):
                pct = n / len(top_rows) * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"  [{bar}] {n}/{len(top_rows)} images loaded", end='\r')
    
    _ITEMS.extend(loaded_items)
    print(f"\n  RAM caching complete in {time.time()-t0:.1f}s\n")

    # 3. Step 2: Parallel Patch Registration
    print(f"Step 2/2: Registering 512px patches (Workers: {NUM_WORKERS})")
    t1 = time.time()
    
    tasks = []
    for img_idx in range(len(_ITEMS)):
        patch_idx = 0
        for y in [0, 512]:
            for x in [0, 512]:
                tasks.append((img_idx, patch_idx, x, y))
                patch_idx += 1
                
    results = []
    # Sliding window for batch statistics
    batch_ssims = collections.deque(maxlen=NUM_WORKERS)
    
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=NUM_WORKERS) as pool:
        count = 0
        for res in pool.imap_unordered(_worker_process_patch, tasks, chunksize=4):
            count += 1
            if res:
                results.append(res)
                batch_ssims.append(res["ssim"])
            
            if count % 25 == 0 or count == len(tasks):
                elapsed = time.time() - t1
                rate = count / elapsed if elapsed > 0 else 0
                pct = count / len(tasks) * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                
                # Calculate batch stats
                if batch_ssims:
                    b_mean = np.mean(batch_ssims)
                    b_med  = np.median(batch_ssims)
                    b_min  = np.min(batch_ssims)
                    b_max  = np.max(batch_ssims)
                    stats_str = f" | Mean:{b_mean:.3f} Med:{b_med:.3f} Min:{b_min:.3f} Max:{b_max:.3f}"
                else:
                    stats_str = " | No stats yet"
                
                print(f"  [{bar}] {count}/{len(tasks)} | {rate:.1f} p/s | {len(results)} acc{stats_str}", end='\r')


    # 4. Save results
    with open(TRAIN_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prefix", "patch_idx", "stained", "unstained", "ssim"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n\n{'='*W}\nREGISTRATION COMPLETE\n{'='*W}")
    print(f"Total patches accepted: {len(results)}")
    print(f"Training CSV: {TRAIN_CSV}")
    print(f"Total time: {(time.time()-t0)/60:.2f} min")

if __name__ == "__main__":
    main()


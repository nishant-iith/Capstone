import ants
import numpy as np
from skimage import io, registration, transform
from skimage.metrics import structural_similarity as ssim_fn
import os
import csv
import cv2
import time

# Config
STAINED_DIR = "dataset/stained"
UNSTAINED_DIR = "dataset/unstained"
TEST_OUT_DIR = "window_test_results"
# We'll test on the pair that showed interesting results earlier
TEST_PAIR_PREFIX = "AS-5198-23-Z35_patch_16384_23552"

def run_windowed_test():
    os.makedirs(TEST_OUT_DIR, exist_ok=True)
    
    s_path = os.path.join(STAINED_DIR, f"{TEST_PAIR_PREFIX}_stained.tif")
    u_path = os.path.join(UNSTAINED_DIR, f"{TEST_PAIR_PREFIX}_unstained.tif")
    
    if not os.path.exists(s_path) or not os.path.exists(u_path):
        print("Files missing. Exiting.")
        return

    # Load images
    s_full = io.imread(s_path)
    u_full = io.imread(u_path)
    
    # Convert to grayscale for registration
    if len(s_full.shape) == 3:
        s_gray_full = cv2.cvtColor(s_full, cv2.COLOR_RGB2GRAY)
        u_gray_full = cv2.cvtColor(u_full, cv2.COLOR_RGB2GRAY)
    else:
        s_gray_full = s_full
        u_gray_full = u_full

    win_size = 256
    grid_size = 4 # 4x4 = 16 windows
    
    results = []
    
    print(f"Comparing TV-L1 vs SyN on 4x4 grid (256px windows) for {TEST_PAIR_PREFIX}")
    print(f"{'Window':<10} | {'TV SSIM':<10} | {'SyN SSIM':<10} | {'Winner'}")
    print("-" * 50)

    for row in range(grid_size):
        for col in range(grid_size):
            y = row * win_size
            x = col * win_size
            win_id = f"{row},{col}"
            
            # Crop
            s_win = s_gray_full[y:y+win_size, x:x+win_size].astype(np.float32)
            u_win = u_gray_full[y:y+win_size, x:x+win_size].astype(np.float32)
            
            # --- TV-L1 Registration ---
            s_g = s_win / 255.0
            u_g = u_win / 255.0
            v_f, u_f = registration.optical_flow_tvl1(s_g, u_g)
            
            gy, gx = np.meshgrid(np.arange(win_size, dtype=np.float32), 
                                 np.arange(win_size, dtype=np.float32), indexing="ij")
            warped_tv = np.zeros_like(u_win)
            # Basic warp for grayscale
            warped_tv = transform.warp(u_win / 255.0, np.array([gy + v_f, gx + u_f]), mode="edge")
            warped_tv = (warped_tv * 255).clip(0, 255).astype(np.uint8)
            
            ssim_tv = ssim_fn(s_win.astype(np.uint8), warped_tv, data_range=255)
            
            # --- SyN Registration ---
            s_ants = ants.from_numpy(s_win)
            u_ants = ants.from_numpy(u_win)
            
            try:
                reg = ants.registration(fixed=s_ants, moving=u_ants, type_of_transform='SyN')
                warped_syn = reg['warpedmovout'].numpy()
                warped_syn = (np.clip(warped_syn, 0, 255)).astype(np.uint8)
                ssim_syn = ssim_fn(s_win.astype(np.uint8), warped_syn, data_range=255)
            except Exception:
                ssim_syn = 0.0
            
            winner = "SyN" if ssim_syn > ssim_tv else "TV-L1"
            print(f"{win_id:<10} | {ssim_tv:.4f}     | {ssim_syn:.4f}     | {winner}")
            
            results.append({
                "window": win_id,
                "tv_ssim": ssim_tv,
                "syn_ssim": ssim_syn,
                "winner": winner
            })

    # Save results
    res_path = os.path.join(TEST_OUT_DIR, "windowed_comparison.csv")
    with open(res_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["window", "tv_ssim", "syn_ssim", "winner"])
        writer.writeheader()
        writer.writerows(results)

    avg_tv = np.mean([r["tv_ssim"] for r in results])
    avg_syn = np.mean([r["syn_ssim"] for r in results])
    print("-" * 50)
    print(f"AVERAGE TV-L1 SSIM: {avg_tv:.4f}")
    print(f"AVERAGE SyN SSIM:   {avg_syn:.4f}")
    print(f"Overall Winner: {'SyN' if avg_syn > avg_tv else 'TV-L1'}")

if __name__ == "__main__":
    run_windowed_test()

import ants
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim_fn
import os
import csv
import cv2

# Config
STAINED_DIR = "dataset/stained"
UNSTAINED_DIR = "dataset/unstained"
TEST_OUT_DIR = "syn_test_results"
CSV_ALL = "data/processed/registered_pairs_all.csv"

# Pairs selected for testing (only the 0.5 range pair)
TEST_PAIRS = [
    "AS-5198-23-Z35_patch_16384_23552",
]

def get_tv_ssim(prefix):
    with open(CSV_ALL, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["prefix"] == prefix:
                return float(row["ssim"])
    return None

def run_syn_test():
    os.makedirs(TEST_OUT_DIR, exist_ok=True)
    results = []

    for prefix in TEST_PAIRS:
        print(f"Processing {prefix}...")
        
        s_path = os.path.join(STAINED_DIR, f"{prefix}_stained.tif")
        u_path = os.path.join(UNSTAINED_DIR, f"{prefix}_unstained.tif")
        
        if not os.path.exists(s_path) or not os.path.exists(u_path):
            print(f"  Files missing for {prefix}, skipping.")
            continue
        
        # Load as numpy and convert to grayscale
        s_np_full = io.imread(s_path)
        u_np_full = io.imread(u_path)
        
        if len(s_np_full.shape) == 3:
            s_gray = cv2.cvtColor(s_np_full, cv2.COLOR_RGB2GRAY)
            u_gray = cv2.cvtColor(u_np_full, cv2.COLOR_RGB2GRAY)
        else:
            s_gray = s_np_full
            u_gray = u_np_full
            
        # Use full resolution (1024x1024)
        s_crop = s_gray.astype(np.float32)
        u_crop = u_gray.astype(np.float32)
        
        # Convert to ANTs images
        s_img = ants.from_numpy(s_crop)
        u_img = ants.from_numpy(u_crop)
        
        # Perform SyN registration
        try:
            reg = ants.registration(
                fixed=s_img, 
                moving=u_img, 
                type_of_transform='SyN'
            )
            warped_img = reg['warpedmovout']
        except Exception as e:
            print(f"  Registration failed: {e}")
            continue
        
        # Convert to numpy for SSIM calculation
        s_np = s_img.numpy()
        u_warped_np = warped_img.numpy()
        
        # Original pipeline used the full RGB for SSIM. 
        # For this test, we compare grayscale SSIM since registration was on grayscale.
        try:
            sc_syn = ssim_fn(s_np, u_warped_np, data_range=255)
        except Exception as e:
            print(f"  SSIM error for {prefix}: {e}")
            sc_syn = 0.0

        tv_ssim = get_tv_ssim(prefix)
        
        # Save the result for visual check
        u_save = (np.clip(u_warped_np, 0, 255)).astype(np.uint8)
        io.imsave(os.path.join(TEST_OUT_DIR, f"{prefix}_syn_warped.tif"), u_save, check_contrast=False)
        
        results.append({
            "prefix": prefix,
            "tv_ssim": tv_ssim,
            "syn_ssim": round(sc_syn, 4),
            "improvement": round(sc_syn - (tv_ssim if tv_ssim else 0), 4)
        })
        
        print(f"  TV-L1 SSIM: {tv_ssim} -> SyN (Crop) SSIM: {sc_syn:.4f} (Diff: {sc_syn - (tv_ssim if tv_ssim else 0):.4f})")

    # Save results to CSV
    res_path = os.path.join(TEST_OUT_DIR, "syn_vs_tv_results.csv")
    with open(res_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prefix", "tv_ssim", "syn_ssim", "improvement"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {res_path}")

if __name__ == "__main__":
    run_syn_test()


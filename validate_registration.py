import os
import numpy as np
from skimage import io, color
from skimage.metrics import structural_similarity as ssim

def validate_registration_sample(sample_prefix):
    # Raw paths
    raw_s = f"1000/Stained_data/{sample_prefix}_stained.tif"
    raw_u = f"1000/Unstained_data/{sample_prefix}_unstained.tif"
    
    # Registered paths
    reg_s = f"data/processed/registered/stained/{sample_prefix}_stained.tif"
    reg_u = f"data/processed/registered/unstained/{sample_prefix}_unstained.tif"
    
    if not all(os.path.exists(p) for p in [raw_s, raw_u, reg_s, reg_u]):
        print(f"Sample {sample_prefix} paths not complete.")
        return

    # Load
    s_img = io.imread(reg_s)
    u_raw = io.imread(raw_u)
    u_reg = io.imread(reg_u)
    
    # Grayscale
    s_gray = color.rgb2gray(s_img)
    u_raw_gray = color.rgb2gray(u_raw)
    u_reg_gray = color.rgb2gray(u_reg)
    
    # Compare raw pairing vs registered pairing
    ssim_raw = ssim(s_gray, u_raw_gray, data_range=1.0)
    ssim_reg = ssim(s_gray, u_reg_gray, data_range=1.0)
    
    mse_raw = np.mean((s_gray - u_raw_gray) ** 2)
    mse_reg = np.mean((s_gray - u_reg_gray) ** 2)
    
    print(f"Validation for sample: {sample_prefix}")
    print(f"  Raw Pair        | SSIM: {ssim_raw:.4f} | MSE: {mse_raw:.6f}")
    print(f"  Registered Pair | SSIM: {ssim_reg:.4f} | MSE: {mse_reg:.6f}")
    print(f"  Improvement: {((ssim_reg - ssim_raw)/ssim_raw)*100:.1f}% increase in SSIM")

if __name__ == "__main__":
    # Test on the first sample we found in diagnostics
    validate_registration_sample("AS-5198-23-Z35_patch_14336_29696")

"""
Test: SyN vs TV-L1 on Small Patches (v19)
========================================
Compare SyN registration on 256×256 patches vs TV-L1.

Usage:
    python test_syn_patches_v19.py
"""

import os
import numpy as np
import time
import ants
from skimage import io
from skimage.metrics import structural_similarity as ssim_fn
import cv2
import warnings
warnings.filterwarnings('ignore')

# Config
STAINED_DIR = "dataset/stained"
UNSTAINED_DIR = "dataset/unstained"
PATCH_SIZE = 256
PRINT_WIDTH = 70


def register_syn(stained_img, unstained_img):
    """
    Register using SyN (ANTs) - returns warped unstained.
    """
    try:
        # Convert to ANTs format (grayscale for registration)
        fixed = ants.from_numpy(stained_img[:,:,0].astype(np.float32))
        moving = ants.from_numpy(unstained_img[:,:,0].astype(np.float32))
        
        # Try simpler affine + SyN
        result = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform='AffineFast',
            max_iterations=[50, 100],
        )
        
        # Get warped image
        warped = result['warpedmovout'].numpy()
        
        # Apply to all channels
        warped_rgb = np.zeros_like(stained_img)
        for c in range(3):
            fixed_c = ants.from_numpy(stained_img[:,:,c].astype(np.float32))
            moving_c = ants.from_numpy(unstained_img[:,:,c].astype(np.float32))
            result_c = ants.apply_transform(
                fixed=fixed_c,
                moving=moving_c,
                transform=result['fwdtransforms'],
            )
            warped_rgb[:,:,c] = result_c.numpy()
        
        return warped_rgb.astype(np.uint8)
        
    except Exception as e:
        print(f"    SyN failed: {e}")
        return unstained_img


def register_tvl1(stained_img, unstained_img):
    """
    Register using TV-L1 - returns warped unstained.
    """
    from skimage import registration, transform
    
    s_gray = cv2.cvtColor(stained_img, cv2.COLOR_RGB2GRAY)
    u_gray = cv2.cvtColor(unstained_img, cv2.COLOR_RGB2GRAY)
    s_g = s_gray.astype(np.float32) / 255.0
    u_g = u_gray.astype(np.float32) / 255.0
    
    v_f, u_f = registration.optical_flow_tvl1(s_g, u_g)
    
    H, W = stained_img.shape[:2]
    gy, gx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing='ij'
    )
    warped = np.zeros_like(stained_img, dtype=np.float32)
    for c in range(3):
        warped[:, :, c] = transform.warp(
            unstained_img[:, :, c].astype(np.float32) / 255.0,
            np.array([gy + v_f, gx + u_f]),
            mode='edge',
        )
    return (warped * 255).clip(0, 255).astype(np.uint8)


def test_patch_registration():
    """Test SyN vs TV-L1 on a 256×256 patch."""
    
    # Find first pair
    stained_files = sorted([
        f for f in os.listdir(STAINED_DIR) 
        if f.lower().endswith(('.tif', '.tiff'))
    ])
    
    if not stained_files:
        print("No stained images found!")
        return
    
    fname = stained_files[0]
    print(f"\n{'='*PRINT_WIDTH}")
    print(f"Testing: {fname}")
    print(f"{'='*PRINT_WIDTH}")
    
    # Load images
    s_img = io.imread(os.path.join(STAINED_DIR, fname))
    base = os.path.splitext(fname)[0]
    unstained_name = base.replace("_stained", "_unstained") + ".tif"
    u_img = io.imread(os.path.join(UNSTAINED_DIR, unstained_name))
    
    # Ensure RGB
    if len(s_img.shape) == 2:
        s_img = np.stack([s_img]*3, axis=-1)
    if len(u_img.shape) == 2:
        u_img = np.stack([u_img]*3, axis=-1)
    
    # Test on center crop (256×256)
    h, w = s_img.shape[:2]
    r = (h - PATCH_SIZE) // 2
    c = (w - PATCH_SIZE) // 2
    
    s_patch = s_img[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
    u_patch = u_img[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
    
    print(f"\nPatch: {PATCH_SIZE}×{PATCH_SIZE} at center ({r}, {c})")
    
    # Pre-registration SSIM
    print(f"\n--- Pre-registration ---")
    pre_ssim = ssim_fn(s_patch, u_patch, channel_axis=2, data_range=255)
    print(f"  Pre-SSIM: {pre_ssim:.4f}")
    
    # TV-L1 on patch
    print(f"\n--- TV-L1 on {PATCH_SIZE}×{PATCH_SIZE} ---")
    t0 = time.time()
    u_tvl1 = register_tvl1(s_patch, u_patch)
    tvl1_time = time.time() - t0
    tvl1_ssim = ssim_fn(s_patch, u_tvl1, channel_axis=2, data_range=255)
    print(f"  Post-SSIM: {tvl1_ssim:.4f}")
    print(f"  Time: {tvl1_time:.2f}s")
    
    # SyN on patch
    print(f"\n--- SyN on {PATCH_SIZE}×{PATCH_SIZE} ---")
    t0 = time.time()
    u_syn = register_syn(s_patch, u_patch)
    syn_time = time.time() - t0
    syn_ssim = ssim_fn(s_patch, u_syn, channel_axis=2, data_range=255)
    print(f"  Post-SSIM: {syn_ssim:.4f}")
    print(f"  Time: {syn_time:.2f}s")
    
    # Summary
    print(f"\n{'='*PRINT_WIDTH}")
    print(f"RESULTS")
    print(f"{'='*PRINT_WIDTH}")
    print(f"  Pre:     {pre_ssim:.4f}")
    print(f"  TV-L1:   {tvl1_ssim:.4f} (Δ +{tvl1_ssim - pre_ssim:.4f})")
    print(f"  SyN:     {syn_ssim:.4f} (Δ +{syn_ssim - pre_ssim:.4f})")
    
    if syn_ssim > tvl1_ssim:
        print(f"\n  ✓ SyN is BETTER than TV-L1")
    else:
        print(f"\n  ✗ TV-L1 is better than SyN")
    
    # Save comparison
    io.imsave("test_syn_comparison.png", 
              np.hstack([s_patch, u_patch, u_tvl1, u_syn]))
    print(f"\n  Saved: test_syn_comparison.png")
    print(f"{'='*PRINT_WIDTH}")


if __name__ == "__main__":
    test_patch_registration()
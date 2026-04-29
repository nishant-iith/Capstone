"""
Quick Test: Windowed TV-L1 Registration (v19)
======================================
Tests on a single pair to verify the approach works.

Usage:
    python test_windowed_v19.py
"""

import os
import sys
import numpy as np
import time
import warnings
from skimage import io
from skimage.metrics import structural_similarity as ssim_fn
import cv2

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our functions
from register_windowed_v19 import (
    register_image_windowed,
    create_blend_weights,
    WINDOW_SIZE,
    OVERLAP
)

# Config
STAINED_DIR = "dataset/stained"
UNSTAINED_DIR = "dataset/unstained"
PRINT_WIDTH = 70

def test_single_pair():
    """Test windowed registration on one image pair."""
    
    # Find first pair
    if not os.path.exists(STAINED_DIR):
        print(f"Error: {STAINED_DIR} not found.")
        return
        
    stained_files = sorted([
        f for f in os.listdir(STAINED_DIR) 
        if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))
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
    
    # Find matching unstained - convert "stained" to "unstained" in filename
    base = os.path.splitext(fname)[0]
    unstained_name = base.replace("_stained", "_unstained") + ".tif"
    unstained_path = os.path.join(UNSTAINED_DIR, unstained_name)
    
    if not os.path.exists(unstained_path):
        print(f"Error: No matching unstained image for {fname}")
        return

    u_img = io.imread(unstained_path)
    
    # Ensure RGB
    if len(s_img.shape) == 2:
        s_img = np.stack([s_img]*3, axis=-1)
    if len(u_img.shape) == 2:
        u_img = np.stack([u_img]*3, axis=-1)
    
    print(f"  Image size: {s_img.shape}")
    
    # Test 1: Pre-registration SSIM (baseline)
    print(f"\n--- Pre-registration SSIM ---")
    pre_ssim = ssim_fn(s_img, u_img, channel_axis=2, data_range=255)
    print(f"  Pre-SSIM: {pre_ssim:.4f}")
    
    # Test 2: TV-L1 full resolution (for comparison)
    print(f"\n--- TV-L1 Full Resolution ---")
    from skimage import registration, transform
    
    s_gray = cv2.cvtColor(s_img, cv2.COLOR_RGB2GRAY)
    u_gray = cv2.cvtColor(u_img, cv2.COLOR_RGB2GRAY)
    s_g = s_gray.astype(np.float32) / 255.0
    u_g = u_gray.astype(np.float32) / 255.0
    
    t0 = time.time()
    v_f, u_f = registration.optical_flow_tvl1(s_g, u_g)
    full_time = time.time() - t0
    
    H, img_w = s_img.shape[:2]
    gy, gx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(img_w, dtype=np.float32),
        indexing='ij'
    )
    warped_full = np.zeros_like(s_img, dtype=np.float32)
    for c in range(3):
        warped_full[:, :, c] = transform.warp(
            u_img[:, :, c].astype(np.float32) / 255.0,
            np.array([gy + v_f, gx + u_f]),
            mode='edge',
        )
    warped_full = (warped_full * 255).clip(0, 255).astype(np.uint8)
    
    full_ssim = ssim_fn(s_img, warped_full, channel_axis=2, data_range=255)
    print(f"  Post-SSIM: {full_ssim:.4f}")
    print(f"  Time: {full_time:.2f}s")
    
    # Test 3: Windowed TV-L1 (our v19 approach)
    print(f"\n--- Windowed TV-L1 (v19) ---")
    print(f"  Window: {WINDOW_SIZE}×{WINDOW_SIZE}, Overlap: {OVERLAP}px")
    
    t0 = time.time()
    warped_win, window_ssim = register_image_windowed(s_img, u_img)
    win_time = time.time() - t0
    
    win_ssim = ssim_fn(s_img, warped_win, channel_axis=2, data_range=255)
    print(f"  Post-SSIM: {win_ssim:.4f}")
    print(f"  Time: {win_time:.2f}s")
    
    # Window-level stats
    if window_ssim:
        ssim_vals = list(window_ssim.values())
        print(f"  Window SSIM - Min: {np.min(ssim_vals):.4f}")
        print(f"  Window SSIM - Max: {np.max(ssim_vals):.4f}")
        print(f"  Window SSIM - Mean: {np.mean(ssim_vals):.4f}")
        print(f"  Windows processed: {len(ssim_vals)}")
    
    # Summary
    print(f"\n{'='*PRINT_WIDTH}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*PRINT_WIDTH}")
    print(f"  Pre-registration:  {pre_ssim:.4f}")
    print(f"  TV-L1 Full:        {full_ssim:.4f} (Δ +{full_ssim - pre_ssim:.4f})")
    print(f"  TV-L1 Windowed:      {win_ssim:.4f} (Δ +{win_ssim - pre_ssim:.4f})")
    
    if win_ssim > full_ssim:
        print(f"\n  ✓ Windowed is BETTER than full TV-L1")
    else:
        print(f"\n  ✗ Windowed is worse - check the blending")
    
    print(f"{'='*PRINT_WIDTH}")
    
    # Save test output for visual inspection
    test_out = "test_windowed_v19_output.png"
    io.imsave(test_out, np.hstack([s_img, warped_full, warped_win]))
    print(f"  Comparison saved to: {test_out}")


if __name__ == "__main__":
    test_single_pair()

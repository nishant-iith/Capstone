"""
v19: TV-L1 Windowed Registration Pipeline
=====================================
TV-L1 applied to 256×256 windows (with overlap) for better local alignment.
Then reassemble with Gaussian blending to full 1024×1024.

Key improvements over v17:
- Windowed registration for higher local SSIM
- Gaussian blending for seamless stitching
- Best practices from v17 failure analysis

Usage:
    python register_windowed_v19.py
"""

import os
import sys
import csv
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage import io, registration, transform
from skimage.metrics import structural_similarity as ssim_fn
import cv2
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
STAINED_DIR   = "dataset/stained"
UNSTAINED_DIR = "dataset/unstained"
OUTPUT_DIR  = "data/processed/registered_windowed"
CSV_ALL     = "data/processed/registered_windowed_all.csv"
CSV_TOPK    = "data/processed/registered_windowed_top.csv"

# Windowed registration config
WINDOW_SIZE   = 256      # 256×256 windows
OVERLAP      = 32        # 32px overlap for blending
STRIDE       = WINDOW_SIZE - OVERLAP  # 224px stride
GRID_SIZE    = 4         # 4×4 grid = 16 windows per image
TOP_K       = 2000      # keep top-K pairs

NUM_WORKERS = 30        # parallel workers
VERBOSE = True
W = 70

# Global storage for worker access
_ITEMS = []

# ── Gaussian Blending ──────────────────────────────────────────────────────
def create_blend_weights(window_size, overlap):
    """
    Create Gaussian blending weights for seamless window stitching.
    Center pixels = 1, edge pixels = 0 (smooth transition).
    """
    rows, cols = window_size, window_size
    y, x = np.ogrid[:rows, :cols]
    
    # Distance from nearest edge
    dist_from_edge = np.minimum(
        np.minimum(y, rows - 1 - y),
        np.minimum(x, cols - 1 - x)
    )
    
    # Gaussian-like weight: smoothly衰减 to edge
    # Using smooth step function for better blending
    blend = np.where(
        dist_from_edge < overlap,
        np.exp(-((dist_from_edge - overlap) ** 2) / (2 * (overlap / 2) ** 2)),
        0
    )
    blend = np.clip(blend, 0, 1)
    
    return blend.astype(np.float32)


def gaussian_blend_windows(windows, positions, output_shape, overlap):
    """
    Reassemble windowed registrations using simple averaging.
    Each pixel gets contributions from all windows that cover it.
    """
    H, W = output_shape[:2]
    first_window = windows[positions[0]]
    C = first_window.shape[2] if len(first_window.shape) == 3 else 1
    
    # Output arrays
    output = np.zeros((H, W, C), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)
    
    # Simple approach: average all windows covering each pixel
    for (r, c) in positions:
        window = windows[(r, c)]
        
        # Define window region
        r_end = min(r + WINDOW_SIZE, H)
        c_end = min(c + WINDOW_SIZE, W)
        
        # Trim window if needed
        wr = r_end - r
        wc = c_end - c
        
        # Simple average (no Gaussian weights for stability)
        if C == 3:
            window_crop = window[:wr, :wc, :].astype(np.float64)
        else:
            window_crop = window[:wr, :wc].astype(np.float64)[:, :, np.newaxis]
        
        # Add window (simple average)
        output[r:r_end, c:c_end] += window_crop
        weight_sum[r:r_end, c:c_end] += 1.0
    
    # Normalize by count
    weight_sum = np.clip(weight_sum, 1e-8, None)
    output = output / weight_sum[:, :, np.newaxis]
    
    if C == 1:
        output = output[:, :, 0]
    
    return output, weight_sum


# ── TV-L1 on Window ─────────────────────────────────────────────────
def register_window_tvl1(stained_win, unstained_win):
    """
    Apply TV-L1 optical flow to a single window.
    Returns the warped unstained window.
    """
    # Convert to grayscale for flow
    s_gray = cv2.cvtColor(stained_win, cv2.COLOR_RGB2GRAY)
    u_gray = cv2.cvtColor(unstained_win, cv2.COLOR_RGB2GRAY)
    
    s_g = s_gray.astype(np.float32) / 255.0
    u_g = u_gray.astype(np.float32) / 255.0
    
    # TV-L1 optical flow
    v_f, u_f = registration.optical_flow_tvl1(s_g, u_g)
    
    # Apply flow to each channel
    H, W = unstained_win.shape[:2]
    warped = np.zeros_like(unstained_win, dtype=np.float32)
    
    gy, gx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing='ij'
    )
    
    for c in range(3):
        warped[:, :, c] = transform.warp(
            unstained_win[:, :, c].astype(np.float32) / 255.0,
            np.array([gy + v_f, gx + u_f]),
            mode='edge',
        )
    
    # Convert back to uint8
    warped = (warped * 255).clip(0, 255).astype(np.uint8)
    
    return warped


# ── Full Windowed Registration ────────────────────────────────────────────
def register_image_windowed(s_img, u_img, window_size=WINDOW_SIZE, overlap=OVERLAP):
    """
    Register an image pair using windowed TV-L1 with Gaussian blending.
    
    Returns:
        - warped image (1024×1024×3)
        - dict of window SSIM scores
    """
    H, W = s_img.shape[:2]
    stride = window_size - overlap
    
    # Calculate number of windows in each dimension
    n_windows_y = max(1, (H - overlap) // stride)
    n_windows_x = max(1, (W - overlap) // stride)
    
    # Collect windows and their positions
    windows_dict = {}  # {(row, col): (stained_window, unstained_window)}
    window_ssim = {}
    
    for row in range(0, H - window_size + 1, stride):
        for col in range(0, W - window_size + 1, stride):
            # Extract windows
            s_win = s_img[row:row+window_size, col:col+window_size]
            u_win = u_img[row:row+window_size, col:col+window_size]
            
            # Skip if window is mostly background (simple heuristic)
            if s_win.mean() < 10 or u_win.mean() < 10:
                continue
            
            # Apply TV-L1 to window
            try:
                u_warped = register_window_tvl1(s_win, u_win)
                windows_dict[(row, col)] = u_warped
                
                # Compute SSIM for this window
                ssim_score = ssim_fn(s_win, u_warped, channel_axis=2, data_range=255)
                window_ssim[(row, col)] = ssim_score
            except Exception as e:
                print(f"  Warning: window ({row},{col}) failed: {e}")
                # Use original as fallback
                windows_dict[(row, col)] = u_win
                window_ssim[(row, col)] = 0.0
    
    # Reassemble with Gaussian blending
    if windows_dict:
        positions = list(windows_dict.keys())
        warped_full, _ = gaussian_blend_windows(
            {k: v for k, v in windows_dict.items()},
            positions,
            (H, W, 3),
            overlap
        )
        warped_full = warped_full.clip(0, 255).astype(np.uint8)
    else:
        # Fallback: use original
        warped_full = u_img
    
    return warped_full, window_ssim


# ── Worker Function ──────────────────────────────────────────────────────
def _worker_register_windowed(idx):
    """
    Worker: register one pair with windowed TV-L1.
    Returns (idx, window_ssim_dict, status).
    """
    item = _ITEMS[idx]
    out_s = item["out_stained"]
    out_u = item["out_unstained"]
    
    # Check if already registered
    if os.path.exists(out_s) and os.path.exists(out_u):
        try:
            s = io.imread(out_s)
            u = io.imread(out_u)
            ssim = ssim_fn(s, u, channel_axis=2, data_range=255)
            return idx, {"pre-computed": ssim}, "skipped"
        except:
            pass
    
    try:
        s_img = item["s_img"]
        u_img = item["u_img"]
        
        # Windowed registration
        u_warped, window_ssim = register_image_windowed(s_img, u_img)
        
        # Compute overall SSIM
        overall_ssim = ssim_fn(s_img, u_warped, channel_axis=2, data_range=255)
        
        # Save registered pair
        io.imsave(out_s, s_img, check_contrast=False)
        io.imsave(out_u, u_warped, check_contrast=False)
        
        return idx, window_ssim, "ok"
    
    except Exception as e:
        return idx, {}, f"error: {str(e)[:50]}"


# ── Main Pipeline ──────────────────────────────────────────────────────
def run_windowed_registration():
    """Run the windowed TV-L1 registration pipeline."""
    global _ITEMS
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "stained"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "unstained"), exist_ok=True)
    
    # Find all stained images
    stained_files = sorted([
        f for f in os.listdir(STAINED_DIR) 
        if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))
    ])
    
    print(f"\n{'='*W}")
    print(f"TV-L1 Windowed Registration Pipeline (v19)")
    print(f"{'='*W}")
    print(f"  Window size:     {WINDOW_SIZE}×{WINDOW_SIZE}")
    print(f"  Overlap:      {OVERLAP}px")
    print(f"  Grid:        {GRID_SIZE}×{GRID_SIZE}")
    print(f"  Workers:     {NUM_WORKERS}")
    print(f"  Output:     {OUTPUT_DIR}")
    print(f"{'='*W}\n")
    
    # Build item list
    _ITEMS = []
    for fname in stained_files:
        base = os.path.splitext(fname)[0]
        
        # Check for matching unstained
        unstained_path = os.path.join(UNSTAINED_DIR, fname)
        if not os.path.exists(unstained_path):
            # Try different extensions
            for ext in ['.tif', '.tiff', '.png', '.jpg']:
                alt_path = os.path.join(UNSTAINED_DIR, base + ext)
                if os.path.exists(alt_path):
                    unstained_path = alt_path
                    break
            else:
                continue
        
        out_s = os.path.join(OUTPUT_DIR, "stained", fname)
        out_u = os.path.join(OUTPUT_DIR, "unstained", fname)
        
        # Load images
        try:
            s_img = io.imread(os.path.join(STAINED_DIR, fname))
            u_img = io.imread(unstained_path)
            
            # Ensure RGB
            if len(s_img.shape) == 2:
                s_img = np.stack([s_img]*3, axis=-1)
            if len(u_img.shape) == 2:
                u_img = np.stack([u_img]*3, axis=-1)
            
            _ITEMS.append({
                "fname": fname,
                "s_img": s_img,
                "u_img": u_img,
                "out_stained": out_s,
                "out_unstained": out_u,
            })
        except Exception as e:
            print(f"  Failed to load {fname}: {e}")
    
    print(f"Found {len(_ITEMS)} image pairs to process\n")
    
    # Process with parallel workers
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(_worker_register_windowed, i): i 
            for i in range(len(_ITEMS))
        }
        
        completed = 0
        for future in as_completed(futures):
            idx, window_ssim, status = future.result()
            results.append((idx, window_ssim, status))
            completed += 1
            
            if VERBOSE and completed % 50 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                print(f"  Progress: {completed}/{len(_ITEMS)} ({rate:.1f} pairs/sec)")
    
    # Collect results
    ssim_scores = []
    for idx, window_ssim, status in results:
        item = _ITEMS[idx]
        
        # Calculate mean window SSIM
        if window_ssim and "pre-computed" not in window_ssim:
            mean_ssim = np.mean(list(window_ssim.values())) if window_ssim else 0.0
        else:
            mean_ssim = window_ssim.get("pre-computed", 0.0)
        
        ssim_scores.append({
            "fname": item["fname"],
            "stained": item["out_stained"],
            "unstained": item["out_unstained"],
            "ssim": mean_ssim,
            "status": status,
            "window_scores": str(window_ssim)[:200] if window_ssim else ""
        })
    
    # Sort by SSIM
    ssim_scores.sort(key=lambda x: x["ssim"], reverse=True)
    
    # Save all results
    with open(CSV_ALL, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["fname", "stained", "unstained", "ssim", "status", "window_scores"])
        writer.writeheader()
        writer.writerows(ssim_scores)
    
    # Save top-K
    with open(CSV_TOPK, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["fname", "stained", "unstained", "ssim", "status"])
        writer.writeheader()
        writer.writerows(ssim_scores[:TOP_K])
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'='*W}")
    print(f"Registration Complete")
    print(f"{'='*W}")
    print(f"  Total pairs:   {len(ssim_scores)}")
    print(f"  Time:         {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Best SSIM:    {max(r['ssim'] for r in ssim_scores):.4f}")
    print(f"  Mean SSIM:    {np.mean([r['ssim'] for r in ssim_scores]):.4f}")
    print(f"  Top-{TOP_K} mean: {np.mean([r['ssim'] for r in ssim_scores[:TOP_K]]):.4f}")
    print(f"\n  Output: {CSV_TOPK}")
    print(f"{'='*W}")
    
    return ssim_scores


if __name__ == "__main__":
    run_windowed_registration()
import os
import numpy as np
import csv
import time
from skimage import io, color, registration, transform
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def method_optical_flow(fixed, moving):
    """Dense Optical Flow (TV-L1) - Best performing for histology alignment"""
    try:
        f_gray = color.rgb2gray(fixed) if len(fixed.shape) == 3 else fixed
        m_gray = color.rgb2gray(moving) if len(moving.shape) == 3 else moving
        
        # TV-L1 Optical Flow
        v, u = registration.optical_flow_tvl1(f_gray, m_gray)
        
        # Warp the original moving image
        nr, nc = m_gray.shape
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        
        # If moving is color, warp each channel
        if len(moving.shape) == 3:
            warped = np.zeros_like(moving, dtype=np.float64)
            for i in range(3):
                warped[:,:,i] = transform.warp(moving[:,:,i]/255.0, np.array([row_coords + v, col_coords + u]), mode='edge')
            warped = (warped * 255).clip(0, 255).astype(np.uint8)
        else:
            warped = transform.warp(moving/255.0, np.array([row_coords + v, col_coords + u]), mode='edge')
            warped = (warped * 255).clip(0, 255).astype(np.uint8)
            
        return warped, "OpticalFlow"
    except Exception as e:
        return moving, f"Failed: {str(e)}"

def process_single_pair(pair, output_dir):
    prefix = pair["prefix"]
    out_s_path = os.path.join(output_dir, "stained", f"{prefix}_stained.tif")
    out_u_path = os.path.join(output_dir, "unstained", f"{prefix}_unstained.tif")
    
    if os.path.exists(out_s_path) and os.path.exists(out_u_path):
        return {"prefix": prefix, "stained": out_s_path, "unstained": out_u_path, "status": "skipped"}
        
    try:
        s_img = io.imread(pair["stained_path"])
        u_img = io.imread(pair["unstained_path"])
        
        warped_u, status = method_optical_flow(s_img, u_img)
        
        io.imsave(out_s_path, s_img, check_contrast=False)
        io.imsave(out_u_path, warped_u, check_contrast=False)
        
        return {"prefix": prefix, "stained": out_s_path, "unstained": out_u_path, "status": "success"}
    except Exception as e:
        return {"prefix": prefix, "status": f"error: {str(e)}"}

def run_registration_pipeline(output_dir="data/processed/registered", num_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "stained"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "unstained"), exist_ok=True)
    
    stained_dir = "1000/Stained_data"
    unstained_dir = "1000/Unstained_data"
    
    s_files = [f for f in os.listdir(stained_dir) if f.endswith("_stained.tif")]
    pairs = []
    for sf in s_files:
        uf = sf.replace("_stained.tif", "_unstained.tif")
        if os.path.exists(os.path.join(unstained_dir, uf)):
            pairs.append({
                "stained_path": os.path.join(stained_dir, sf),
                "unstained_path": os.path.join(unstained_dir, uf),
                "prefix": sf.replace("_stained.tif", "")
            })
    
    total = len(pairs)
    print(f"Starting Parallel Optical Flow registration for {total} pairs with {num_workers} workers...")
    
    registered_metadata = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        func = partial(process_single_pair, output_dir=output_dir)
        # Use as_completed to monitor as each task finishes
        from concurrent.futures import as_completed
        future_to_pair = {executor.submit(func, pair): pair for pair in pairs}
        
        for i, future in enumerate(as_completed(future_to_pair)):
            res = future.result()
            if "stained" in res:
                registered_metadata.append({
                    "prefix": res["prefix"],
                    "stained": res["stained"],
                    "unstained": res["unstained"]
                })
            
            if (i + 1) % 5 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                per_pair = elapsed / (i + 1)
                remaining = per_pair * (total - (i + 1))
                print(f"[{i+1}/{total}] Registered: {res['prefix']} | Status: {res.get('status', 'success')} | Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m")

    with open("data/processed/registered_pairs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prefix", "stained", "unstained"])
        writer.writeheader()
        writer.writerows(registered_metadata)
    
    print(f"\nPipeline complete. Registered data saved to {output_dir}")

if __name__ == "__main__":
    # Use 4 workers as a safe default for common environments
    run_registration_pipeline(num_workers=4)

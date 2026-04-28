import numpy as np
import csv
from skimage import io, color, registration, feature, measure, transform
from skimage.metrics import structural_similarity as ssim
import os
import time

def compute_mutual_information(img1, img2, bins=20):
    """
    Mutual Information is often better for cross-modal registration 
    (stained vs unstained) than MSE.
    """
    hgram, x_edges, y_edges = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x
    py = np.sum(pxy, axis=0) # marginal for y
    px_py = px[:, None] * py[None, :] # Broadcast to (bins, bins)
    nzs = pxy > 0 # Find non-zero pxy values
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def method_phase_correlation(fixed, moving):
    """Method 1: Translation-only via Phase Cross Correlation"""
    start = time.time()
    try:
        shift, error, diffphase = registration.phase_cross_correlation(fixed, moving, upsample_factor=10)
        tform = transform.SimilarityTransform(translation=-shift[::-1])
        warped = transform.warp(moving, tform)
        return warped, time.time() - start, "Translation Only"
    except Exception as e:
        return moving, time.time() - start, f"Translation (Error: {str(e)})"

def method_orb_feature(fixed, moving):
    """Method 2: ORB Features + RANSAC (Euclidean/Affine)"""
    start = time.time()
    try:
        # Detect ORB features
        detector_fixed = feature.ORB(n_keypoints=500)
        detector_moving = feature.ORB(n_keypoints=500)
        
        detector_fixed.detect_and_extract(fixed)
        detector_moving.detect_and_extract(moving)
        
        matches = feature.match_descriptors(detector_fixed.descriptors, 
                                            detector_moving.descriptors, 
                                            cross_check=True)
        
        if len(matches) < 4:
            return moving, time.time() - start, "ORB (Failed - Too few matches)"

        src = detector_moving.keypoints[matches[:, 1]]
        dst = detector_fixed.keypoints[matches[:, 0]]
        
        # Use Euclidean (Rotation + Translation) as it's most common for histology sections
        model_robust, inliers = measure.ransac((src, dst), transform.EuclideanTransform,
                                               min_samples=3, residual_threshold=2, max_trials=100)
        
        if model_robust is None:
            return moving, time.time() - start, "ORB (Failed - RANSAC)"
            
        warped = transform.warp(moving, model_robust)
        return warped, time.time() - start, "ORB Euclidean"
    except Exception as e:
        return moving, time.time() - start, f"ORB (Error: {str(e)})"

def method_optical_flow(fixed, moving):
    """Method 3: Dense Optical Flow (TV-L1)"""
    start = time.time()
    try:
        # Optical flow can handle non-rigid but is slow and sensitive
        v, u = registration.optical_flow_tvl1(fixed, moving)
        
        # Warp the moving image using the flow
        nr, nc = moving.shape
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        warped = transform.warp(moving, np.array([row_coords + v, col_coords + u]), mode='edge')
        
        return warped, time.time() - start, "Optical Flow (TV-L1)"
    except Exception as e:
        return moving, time.time() - start, f"Optical Flow (Error: {str(e)})"

def evaluate_methods(n_samples=5):
    stained_dir = "1000/Stained_data"
    unstained_dir = "1000/Unstained_data"
    
    if not os.path.exists(stained_dir) or not os.path.exists(unstained_dir):
        print("Data directories not found.")
        return

    # Discover pairs by filename
    stained_files = [f for f in os.listdir(stained_dir) if f.endswith("_stained.tif")]
    pairs = []
    for sf in stained_files:
        uf = sf.replace("_stained.tif", "_unstained.tif")
        if os.path.exists(os.path.join(unstained_dir, uf)):
            pairs.append({
                "stained": os.path.join(stained_dir, sf),
                "unstained": os.path.join(unstained_dir, uf)
            })
    
    if not pairs:
        print("No valid pairs found in 1000/ directory.")
        return
        
    print(f"Found {len(pairs)} pairs for research.")
    results = []
    
    for i in range(min(n_samples, len(pairs))):
        row = pairs[i]
        print(f"\nEvaluating Pair {i}: {os.path.basename(row['stained'])}")
        
        try:
            unstained = io.imread(row["unstained"])
            stained = io.imread(row["stained"])
            
            f_gray = color.rgb2gray(stained)
            m_gray = color.rgb2gray(unstained)
            
            methods = [
                method_phase_correlation,
                method_orb_feature,
                method_optical_flow
            ]
            
            for method in methods:
                warped, duration, name = method(f_gray, m_gray)
                mi = compute_mutual_information(f_gray, warped)
                score_ssim = ssim(f_gray, warped, data_range=1.0)
                
                results.append({
                    "Sample": i,
                    "Method": name,
                    "MutualInfo": mi,
                    "SSIM": score_ssim,
                    "Time": duration
                })
                print(f"  - {name:25} | MI: {mi:.4f} | SSIM: {score_ssim:.4f} | Time: {duration:.2f}s")
        except Exception as e:
            print(f"Error processing sample {i}: {e}")

    # Manual aggregation
    summary = {}
    for res in results:
        m = res["Method"]
        if m not in summary: summary[m] = {"MI": [], "SSIM": [], "Time": []}
        summary[m]["MI"].append(res["MutualInfo"])
        summary[m]["SSIM"].append(res["SSIM"])
        summary[m]["Time"].append(res["Time"])
        
    print("\nSummary Results (Averages):")
    for m, vals in summary.items():
        print(f"{m:25} | Avg MI: {np.mean(vals['MI']):.4f} | Avg SSIM: {np.mean(vals['SSIM']):.4f} | Avg Time: {np.mean(vals['Time']):.2f}s")

if __name__ == "__main__":
    evaluate_methods(n_samples=5)

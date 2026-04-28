import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def get_stain_matrix(img, Io=240, alpha=1, beta=0.15):
    """
    Helper to calculate Macenko stain matrix.
    """
    I = img.reshape((-1, 3))
    # Convert to Optical Density
    OD = -np.log10((I.astype(np.float64) + 1) / Io)
    
    # Remove transparent pixels
    OD_hat = OD[~np.any(OD < beta, axis=1)]
    if len(OD_hat) == 0:
        return None, None
        
    # SVD
    _, _, V = np.linalg.svd(OD_hat, full_matrices=False)
    
    # Project data onto plane spanned by two largest SVD vectors
    T_hat = OD_hat @ V[0:2].T
    
    # Find angles
    phi = np.arctan2(T_hat[:, 1], T_hat[:, 0])
    
    # Find robust extremes
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)
    
    v_min = V[0:2].T @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v_max = V[0:2].T @ np.array([np.cos(max_phi), np.sin(max_phi)])
    
    # Order: [Hematoxylin, Eosin]
    if v_min[0] > v_max[0]:
        stain_matrix = np.array([v_min, v_max]).T
    else:
        stain_matrix = np.array([v_max, v_min]).T
        
    return stain_matrix, OD

def macenko_normalize(source_img, target_stain_matrix, target_max_c, Io=240):
    """
    Normalize a source image to match a pre-calculated target stain matrix.
    """
    source_stain_matrix, OD_source = get_stain_matrix(source_img, Io=Io)
    if source_stain_matrix is None:
        return source_img
        
    # Calculate concentrations
    source_concentrations = np.linalg.lstsq(source_stain_matrix, OD_source.T, rcond=None)[0]
    
    # Normalize concentrations to match target extremes
    source_max_c = np.percentile(source_concentrations, 99, axis=1)
    # Avoid division by zero
    source_max_c[source_max_c == 0] = 1.0
    
    source_concentrations *= (target_max_c / source_max_c)[:, None]
    
    # Reconstruct
    normalized_OD = (target_stain_matrix @ source_concentrations).T
    normalized_img = Io * np.exp(-normalized_OD * np.log(10))
    
    return np.clip(normalized_img, 0, 255).reshape(source_img.shape).astype(np.uint8)

def run_normalization():
    print(f"--- PATH B: STAIN NORMALIZATION (PURE NUMPY MACENKO) ---")
    print(f"Goal: Eliminate staining variance in 1,000 images")

    input_dir = "data/processed/registered/stained"
    output_dir = "data/processed/registered/stained_normalized"
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".tif")]
    
    # 1. Pre-calculate Target Reference from a "Good" slide
    # We'll use the average of the first 5 images to get a very stable target
    print("Status: Calculating reference stain distribution...")
    ref_matrices = []
    ref_max_cs = []
    
    for i in range(min(5, len(files))):
        img = np.array(Image.open(os.path.join(input_dir, files[i])).convert("RGB"))
        sm, OD = get_stain_matrix(img)
        if sm is not None:
            conc = np.linalg.lstsq(sm, OD.T, rcond=None)[0]
            ref_matrices.append(sm)
            ref_max_cs.append(np.percentile(conc, 99, axis=1))
            
    target_stain_matrix = np.mean(ref_matrices, axis=0)
    target_max_c = np.mean(ref_max_cs, axis=0)

    # 2. Normalize entire dataset
    print(f"Status: Normalizing 1,000 images...")
    for f in tqdm(files, desc="Processing", unit="img"):
        out_path = os.path.join(output_dir, f)
        if os.path.exists(out_path): continue
        
        try:
            source = np.array(Image.open(os.path.join(input_dir, f)).convert("RGB"))
            normalized = macenko_normalize(source, target_stain_matrix, target_max_c)
            Image.fromarray(normalized).save(out_path)
        except Exception as e:
            print(f"\n[Error] {f}: {e}")

    # 3. Save Metadata
    csv_path = "data/processed/registered_pairs.csv"
    df = pd.read_csv(csv_path)
    df["stained_normalized"] = df["stained"].str.replace("\\stained\\", "/stained_normalized/", regex=False)
    # Fix for windows/unix path slash mix
    df["stained_normalized"] = df["stained_normalized"].str.replace("/stained/", "/stained_normalized/", regex=False)
    
    df.to_csv("data/processed/registered_pairs_normalized.csv", index=False)
    print(f"\nResult: 1,000 images normalized. Metadata saved.")

if __name__ == "__main__":
    run_normalization()

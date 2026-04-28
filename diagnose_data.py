import os
import numpy as np
from skimage import io

def analyze_image_type(path):
    try:
        img = io.imread(path)
        if len(img.shape) < 3:
            return "Grayscale", 0
        
        # Calculate saturation or channel variance to detect staining
        # H&E stained images have high variance between channels (Pink/Purple)
        # Unstained images are mostly white/gray (R ~ G ~ B)
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        std_channels = np.std([np.mean(r), np.mean(g), np.mean(b)])
        
        # Also check mean brightness
        mean_val = np.mean(img)
        
        return "Colored" if std_channels > 5 else "Neutral", std_channels, mean_val
    except Exception as e:
        return f"Error: {str(e)}", 0, 0

def diagnose_dataset(root_dir="1000"):
    results = []
    for sub in ["stained", "unstained"]:
        path = os.path.join(root_dir, sub)
        if not os.path.exists(path): continue
        
        files = [f for f in os.listdir(path) if f.endswith(".tif")][:10] # Check first 10
        print(f"\nChecking folder: {sub}")
        for f in files:
            fpath = os.path.join(path, f)
            itype, std, mean = analyze_image_type(fpath)
            print(f"  {f:50} | Type: {itype:8} | Ch-Std: {std:6.2f} | Mean: {mean:6.2f}")

if __name__ == "__main__":
    diagnose_dataset()

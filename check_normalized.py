import numpy as np
from PIL import Image
import os

def check_normalized_image(path):
    img = np.array(Image.open(path))
    print(f"Stats for: {os.path.basename(path)}")
    print(f"  Shape: {img.shape}")
    print(f"  Min: {img.min()} | Max: {img.max()} | Mean: {img.mean():.2f}")
    print(f"  Unique values: {len(np.unique(img))}")

if __name__ == "__main__":
    check_normalized_image("data/processed/registered/stained_normalized/AS-5198-23-Z35_patch_14336_29696_stained.tif")

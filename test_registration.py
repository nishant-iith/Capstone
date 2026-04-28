import numpy as np
from skimage import io, color, registration
import os

def test_registration():
    # Use one of the samples from the CSV
    import pandas as pd
    df = pd.read_csv("data/processed/train_pairs.csv")
    row = df.iloc[0]
    
    unstained_path = row["unstained"]
    stained_path = row["stained"]
    
    print(f"Testing registration for:\nUnstained: {unstained_path}\nStained: {stained_path}")
    
    unstained = io.imread(unstained_path)
    stained = io.imread(stained_path)
    
    # Convert to grayscale for registration
    unstained_gray = color.rgb2gray(unstained)
    stained_gray = color.rgb2gray(stained)
    
    # Simple phase cross correlation for translation
    shift, error, diffphase = registration.phase_cross_correlation(stained_gray, unstained_gray, upsample_factor=10)
    
    print(f"Shift found: {shift}")
    print(f"Error: {error}")

if __name__ == "__main__":
    test_registration()

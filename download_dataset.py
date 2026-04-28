"""
Download H&E Stain Dataset from Google Drive
==========================================

This script downloads the dataset from:
https://drive.google.com/drive/folders/1mQuN7dYWo5Z6Sc5qAh3Zyzs6TTsn-RM5

The dataset contains 4 folders with ~500 images each:
1. stained (H&E stained tissue patches)
2. unstained (unstained tissue patches)
3. tes_stain (test stained)
4. tes_unstain (test unstained)

Total: ~2000 TIF images (~6GB)
"""

import gdown
import os
import sys

# Constants
FOLDER_URL = "https://drive.google.com/drive/folders/1mQuN7dYWo5Z6Sc5qAh3Zyzs6TTsn-RM5"
OUTPUT_DIR = "Dataset"

def main():
    print("=" * 60)
    print("H&E STAIN DATASET DOWNLOADER")
    print("=" * 60)
    print()
    print(f"Source URL: {FOLDER_URL}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("-" * 60)
    print("INSTRUCTIONS:")
    print("-" * 60)
    print("""
1. This will download ALL files from the Google Drive folder
2. The dataset has ~2000 images at ~3MB each (~6GB total)
3. Download may take 30-60 minutes depending on internet speed
4. Do NOT close this terminal until download completes

To run this script:
    python download_dataset.py
""")
    print("-" * 60)
    print()
    
    # Create output directory
    print(f"[1/3] Creating output directory: {OUTPUT_DIR}/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"         -> Directory created successfully!")
    print()
    
    # Start download
    print("[2/3] Starting download...")
    print("-" * 60)
    
    try:
        gdown.download_folder(
            FOLDER_URL,
            output=OUTPUT_DIR,
            quiet=False,  # Show progress
            use_cookies=False
        )
        print("-" * 60)
        print()
        print("[3/3] Download completed!")
        print()
        print("=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Dataset saved to: {OUTPUT_DIR}/")
        print()
        print("Next steps:")
        print("1. Check the downloaded files")
        print("2. Update CSV files in data/ to point to correct paths")
        print("3. Run training with: python train_v7.py")
        
    except Exception as e:
        print()
        print("=" * 60)
        print("ERROR!")
        print("=" * 60)
        print(f"Error message: {e}")
        print()
        print("Possible solutions:")
        print("1. Check your internet connection")
        print("2. Try running the script again")
        print("3. Download manually from the Google Drive link")
        sys.exit(1)

if __name__ == "__main__":
    main()
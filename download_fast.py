"""
FAST H&E Dataset Downloader v2
==============================
Uses gdown directly with folder ID - simplest and fastest approach.

Usage:
    python download_fast.py
"""

import gdown
import os

FOLDER_ID = "1mQuN7dYWo5Z6Sc5qAh3Zyzs6TTsn-RM5"
OUTPUT_DIR = "Dataset"

def main():
    print("=" * 60)
    print("FAST H&E DATASET DOWNLOADER")
    print("=" * 60)
    print()
    print(f"Folder ID: {FOLDER_ID}")
    print(f"Output:   {OUTPUT_DIR}/")
    print()
    print("Using gdown with resume=True")
    print("This will download all 4 subfolders (~2000 files)")
    print()
    print("-" * 60)
    
    # Just use gdown directly - it handles everything
    # resume=True skips already downloaded files
    gdown.download_folder(
        id=FOLDER_ID,
        output=OUTPUT_DIR,
        quiet=False,
        use_cookies=False,
        resume=True  # KEY: continues from where left off
    )
    
    print("-" * 60)
    print("DONE!")

if __name__ == "__main__":
    main()
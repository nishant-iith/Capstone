"""
TRUE PARALLEL H&E Dataset Downloader
====================================
Downloads files in PARALLEL using ThreadPoolExecutor.

Usage:
    python download_parallel.py
"""

import gdown
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

FOLDER_ID = "1mQuN7dYWo5Z6Sc5qAh3Zyzs6TTsn-RM5"
OUTPUT_DIR = "Dataset"
MAX_WORKERS = 16  # Parallel connections

def download_single(file_info):
    """Download a single file."""
    file_id = file_info.id
    local_path = file_info.local_path
    filename = os.path.basename(local_path)
    
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000000:
        return {'status': 'skipped', 'file': filename}
    
    try:
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            output=local_path,
            quiet=True,
            resume=True
        )
        return {'status': 'success', 'file': filename}
    except Exception as e:
        return {'status': 'error', 'file': filename, 'error': str(e)}

def main():
    print("=" * 60)
    print("TRUE PARALLEL H&E DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Max workers: {MAX_WORKERS} parallel connections")
    print()
    
    # Get file list
    print("[1/2] Getting file list...")
    files = gdown.download_folder(
        id=FOLDER_ID,
        output=None,
        skip_download=True
    )
    print(f"      Found {len(files)} files")
    for f in files[:4]:
        print(f"      - {f.path.split('/')[-1]}")
    if len(files) > 4:
        print(f"      ... and {len(files)-4} more")
    print()
    
    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in files:
        d = os.path.dirname(f.local_path)
        os.makedirs(d, exist_ok=True)
    
    # Download in parallel
    print("[2/2] Downloading in parallel...")
    print("-" * 60)
    
    success = skipped = errors = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single, f): f for f in files}
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["status"] == "success":
                success += 1
            elif result["status"] == "skipped":
                skipped += 1
            else:
                errors += 1
            
            if i % 50 == 0 or i == len(files):
                print(f"Progress: {i}/{len(files)} (Success:{success} Error:{errors} Skip:{skipped})")
    
    print("-" * 60)
    print(f"DONE! Success:{success} Skipped:{skipped} Errors:{errors}")

if __name__ == "__main__":
    main()
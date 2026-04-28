"""
PARALLEL DOWNLOAD WITH RETRIES
========================
Downloads with retry logic to handle rate limiting.
"""

import gdown
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

FOLDER_ID = "1mQuN7dYWo5Z6Sc5qAh3Zyzs6TTsn-RM5"
OUTPUT_DIR = "Dataset"
MAX_WORKERS = 4  # Fewer workers = less rate limiting
MAX_RETRIES = 3
RETRY_DELAY = 2

def download_single(file_info, retries=MAX_RETRIES):
    """Download with retry logic."""
    file_id = file_info.id
    local_path = file_info.local_path
    filename = os.path.basename(local_path)
    
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000000:
        return {'status': 'skipped', 'file': filename}
    
    for attempt in range(retries):
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                output=local_path,
                quiet=True,
                resume=True
            )
            return {'status': 'success', 'file': filename}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return {'status': 'error', 'file': filename, 'error': str(e)}

def main():
    print("=" * 60)
    print("DOWNLOAD WITH RETRIES")
    print("=" * 60)
    
    files = gdown.download_folder(id=FOLDER_ID, output=None, skip_download=True)
    print(f"Total files: {len(files)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in files:
        d = os.path.dirname(f.local_path)
        os.makedirs(d, exist_ok=True)
    
    print(f"\nDownloading with {MAX_WORKERS} workers, {MAX_RETRIES} retries...")
    print("-" * 60)
    
    success = skipped = errors = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single, f): f for f in files}
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result['status'] == 'success':
                success += 1
            elif result['status'] == 'skipped':
                skipped += 1
            else:
                errors += 1
            
            if i % 100 == 0:
                print(f"Progress: {i}/{len(files)} (OK:{success} Err:{errors})")
    
    print("-" * 60)
    print(f"DONE! Success:{success} Skipped:{skipped} Errors:{errors}")

if __name__ == "__main__":
    main()
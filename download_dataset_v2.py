"""
Download dataset from Google Drive - folder by folder with parallel downloads.
"""

import os
import sys
import gdown
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

PARENT_FOLDER_ID = "1mQuN7dYWo5Z6Sc5qAh3Zyzs6TTsn-RM5"
DATA_DIR = "data"
OUTPUT_DIRS = {
    "stained": f"{DATA_DIR}/raw/stained",
    "unstained": f"{DATA_DIR}/raw/unstained"
}

# Create output directories
for d in OUTPUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

def get_folder_contents(folder_id):
    """Get contents of a Google Drive folder."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    response = requests.get(url)
    
    # Extract file info from the page
    # Look for file IDs and names
    file_pattern = r'"(\w+)",\s*null,\s*null,\s*2,\s*null,\s*null,\s*null,\s*null,\s*null,\s*null,\s*null,\s*null,\s*"([^"]+)"'
    folder_pattern = r'/drive/folders/(\w+)"'
    
    files = []
    subfolders = []
    
    # Parse file entries
    entries = re.findall(r'\["([\w-]+)",\s*null,\s*null,\s*2,\s*null,\s*null,\s*null,\s*null,\s*null,\s*null,\s*null,\s*null,\s*null,\s*"([^"]+)"', response.text)
    for file_id, file_name in entries:
        if '.tif' in file_name:
            files.append((file_id, file_name))
    
    # Parse subfolder entries  
    folder_entries = re.findall(r'href="/drive/folders/(\w+)"[^>]*>([^<]+)<', response.text)
    for subfolder_id, subfolder_name in folder_entries:
        if subfolder_id != folder_id and 'folder' not in subfolder_name.lower():
            subfolders.append((subfolder_id, subfolder_name))
    
    return files, subfolders

def download_file(file_id, output_path, file_name):
    """Download a single file."""
    url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
    try:
        output = os.path.join(output_path, file_name)
        gdown.download(url, output, quiet=True)
        return True, file_name
    except Exception as e:
        return False, f"{file_name}: {e}"

def download_folder_parallel(folder_id, output_path, max_workers=8):
    """Download all files in a folder using parallel downloads."""
    print(f"Getting contents of folder {folder_id}...")
    files, subfolders = get_folder_contents(folder_id)
    
    if subfolders:
        print(f"Found subfolders: {subfolders}")
    
    print(f"Found {len(files)} files to download")
    
    # Download files in parallel
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_file, file_id, output_path, file_name): file_name
            for file_id, file_name in files
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            success, result = future.result()
            if success:
                success_count += 1
                print(f"[{i}/{len(files)}] Downloaded: {result}")
            else:
                error_count += 1
                print(f"[{i}/{len(files)}] Error: {result}")
    
    print(f"Completed: {success_count} success, {error_count} errors")
    return success_count, error_count

def main():
    print("=" * 50)
    print("Downloading Virtual H&E Stain Dataset")
    print("=" * 50)
    
    # Get main folder contents
    print(f"\nGetting contents of main folder {PARENT_FOLDER_ID}...")
    files, subfolders = get_folder_contents(PARENT_FOLDER_ID)
    
    print(f"Found {len(files)} files in root")
    print(f"Found {len(subfolders)} subfolders")
    
    # List subfolders
    if subfolders:
        print("\nSubfolders:")
        for fid, fname in subfolders:
            print(f"  - {fname}: {fid}")
    
    # Download each subfolder
    for i, (folder_id, folder_name) in enumerate(subfolders, 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{len(subfolders)}] Downloading {folder_name}")
        print(f"{'='*50}")
        
        # Determine output directory based on folder name
        if "stained" in folder_name.lower():
            output_path = OUTPUT_DIRS["stained"]
        elif "unstained" in folder_name.lower():
            output_path = OUTPUT_DIRS["unstained"]
        else:
            output_path = f"{DATA_DIR}/raw/{folder_name}"
        
        os.makedirs(output_path, exist_ok=True)
        
        success, errors = download_folder_parallel(folder_id, output_path, max_workers=8)
        print(f"Result: {success} downloaded, {errors} errors")
        
        # Small delay between folders
        if i < len(subfolders):
            time.sleep(2)
    
    # Also download root level files if any
    if files:
        print(f"\n{'='*50}")
        print("Downloading root level files")
        print(f"{'='*50}")
        success, errors = download_folder_parallel(PARENT_FOLDER_ID, DATA_DIR, max_workers=8)
    
    print("\n" + "=" * 50)
    print("Download complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
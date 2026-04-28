"""
Download dataset from Google Drive - folder by folder with parallel downloads.
Uses Google Drive API for folder listing.
"""

import os
import gdown
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from urllib.parse import urlparse

PARENT_FOLDER_ID = "1mQuN7dYWo5Z6Sc5qAh3Zyzs6TTsn-RM5"
DATA_DIR = "data"
OUTPUT_DIRS = {
    "stained": f"{DATA_DIR}/raw/stained",
    "unstained": f"{DATA_DIR}/raw/unstained"
}

for d in OUTPUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

def get_folder_contents(folder_id):
    """Get contents of a Google Drive folder using API."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers)
    html = response.text
    
    files = []
    subfolders = []
    
    # Pattern for file entries in Drive HTML response
    # Files typically have data-id and data-target-href
    file_pattern = r'data-id="([\w-]+)"[^>]*data-target-href="/file[^"]*"[^>]*>\s*<div[^>]*>[^<]*</div>\s*<div[^>]*>[^<]*</div>\s*<div[^>]*>([^<]+)</div>'
    folder_pattern = r'data-id="([\w-]+)"[^>]*data-target-href="/drive/folder[^"]*"[^>]*>\s*<div[^>]*>[^<]*</div>\s*<div[^>]*>[^<]*</div>\s*<div[^>]*>([^<]+)</div>'
    
    # Alternative pattern - look for anchor tags with file/d folder links
    href_pattern = r'href="(/drive/folders/[\w-]+)"[^>]*>([^<]+)</a>'
    folder_href = re.findall(href_pattern, html)
    for fid, fname in folder_href:
        if fid.strip('/').split('/')[-1] != folder_id:
            subfolders.append((fid.strip('/').split('/')[-1], fname.strip()))
    
    # Look for download links for files
    # Pattern: /file/d/FILEID or /uc?id=FILEID
    file_ids = re.findall(r'/file/d/([\w-]+)', html)
    file_hrefs = re.findall(r'/drive/folders/[\w-]+/[\w-]+\?usp=drive_fs_pdf', html)
    
    # More specific pattern for TIF files
    tif_pattern = r'"(\w+)","[^"]*","[^"]*","[^"]*","[^"]*","[^"]*","[^"]*",2,(\d+\.\d+),"[^"]*"'
    matches = re.findall(tif_pattern, html)
    for fid, size in matches:
        if float(size) > 1000:  # Files larger than 1KB
            files.append((fid, None))
    
    return files, subfolders

def main():
    print("=" * 50)
    print("Downloading Virtual H&E Stain Dataset")
    print("=" * 50)
    
    print(f"\nGetting contents of main folder {PARENT_FOLDER_ID}...")
    files, subfolders = get_folder_contents(PARENT_FOLDER_ID)
    
    print(f"Found {len(files)} files in root")
    print(f"Found {len(subfolders)} subfolders")
    
    if subfolders:
        print("\nSubfolders:")
        for fid, fname in subfolders:
            print(f"  - {fname}: {fid}")
    
    # Use gdown's built-in folder download
    if not subfolders:
        # Try direct folder download with gdown
        print("\nUsing gdown to download folder directly...")
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{PARENT_FOLDER_ID}",
            quiet=False,
            use_cookies=False
        )

if __name__ == "__main__":
    main()
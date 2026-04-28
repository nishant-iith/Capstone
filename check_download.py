import gdown
import os

FOLDER_ID = "1mQuN7dYWo5Z6Sc5qAh3Zyzs6TTsn-RM5"
OUTPUT_DIR = "1000"

print("Getting expected file list...")
files = gdown.download_folder(id=FOLDER_ID, output=None, skip_download=True)
print(f"Total expected: {len(files)}")
print()

# Check each file
missing = []
downloaded = 0

for f in files:
    local_path = os.path.join(OUTPUT_DIR, f.path.replace('stained\\', 'stained/').replace('unstained\\', 'unstained/').replace('tes_stain\\', 'tes_stain/').replace('tes_unstain\\', 'tes_unstain/'))
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000000:
        downloaded += 1
    else:
        missing.append(f.path)

print(f"Downloaded: {downloaded}")
print(f"Missing: {len(missing)}")
print()

if missing:
    print("Missing files:")
    for m in missing[:20]:
        print(f"  - {m}")
    if len(missing) > 20:
        print(f"  ... and {len(missing)-20} more")
import os
import glob
import pandas as pd
from pathlib import Path

def get_paired_patches(stained_dir, unstained_dir):
    """
    Matches stained and unstained patches based on their coordinate identifiers.
    Pattern: [ID]_patch_[X]_[Y]_[state].tif
    """
    stained_files = glob.glob(os.path.join(stained_dir, "*_stained.tif"))
    unstained_files = glob.glob(os.path.join(unstained_dir, "*_unstained.tif"))

    # Extract base IDs (everything before _stained or _unstained)
    stained_map = {os.path.basename(f).replace("_stained.tif", ""): f for f in stained_files}
    unstained_map = {os.path.basename(f).replace("_unstained.tif", ""): f for f in unstained_files}

    common_ids = set(stained_map.keys()).intersection(set(unstained_map.keys()))
    
    pairs = []
    for cid in common_ids:
        pairs.append({
            "id": cid,
            "stained": stained_map[cid],
            "unstained": unstained_map[cid]
        })
    
    return pd.DataFrame(pairs)

if __name__ == "__main__":
    # Test with the provided directory structure
    s_dir = "1000-20260412T223922Z-3-001/1000/stained"
    u_dir = "1000-20260412T223922Z-3-001/1000/unstained"
    
    # Note: In the real data, we might have tes_ folders too. 
    # The pipeline should be configurable.
    df = get_paired_patches(s_dir, u_dir)
    print(f"Found {len(df)} paired patches.")
    print(df.head())

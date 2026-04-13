"""
Coordinate-based pairing of unstained and stained tissue patches.

Dataset folder layout (as committed):
  Dataset/Data/stained/    — AS-5198 stained patches   (*_stained.tif)
  Dataset/Data/tes_unstain/ — AS-5198 unstained patches (*_unstained.tif)
  Dataset/Data/unstained/  — AS-7231 stained patches   (*_stained.tif)
  Dataset/Data/tes_stain/  — AS-7231 unstained patches (*_unstained.tif)

Filename pattern: <slide_id>_patch_<X>_<Y>_<state>.tif
Pairing key:      (slide_id, X, Y)  — same spatial coordinate ↔ same tissue region
"""

import re
from pathlib import Path
import pandas as pd

# Pattern: <slide>_patch_<X>_<Y>_<state>.tif
_PATCH_RE = re.compile(r'^(.+?)_patch_(\d+)_(\d+)_(stained|unstained)\.tif$')


def _index_dir(directory: Path) -> dict:
    """
    Scan a directory for .tif files matching the patch pattern.

    Returns:
        dict mapping (slide_id, x, y) -> Path
    """
    index = {}
    for f in directory.glob("*.tif"):
        m = _PATCH_RE.match(f.name)
        if m:
            slide_id, x, y, _ = m.groups()
            key = (slide_id, int(x), int(y))
            index[key] = f
    return index


def get_paired_patches(stained_dir: str, unstained_dir: str) -> pd.DataFrame:
    """
    Match stained and unstained patches by (slide_id, X, Y) coordinate key.

    Args:
        stained_dir:   Directory containing *_stained.tif files
        unstained_dir: Directory containing *_unstained.tif files

    Returns:
        DataFrame with columns: [id, stained, unstained]
          - id:        coordinate key as string "<slide>_<X>_<Y>"
          - stained:   absolute path to stained .tif
          - unstained: absolute path to unstained .tif
        Unpaired files (orphans) are silently excluded.
    """
    stained_idx   = _index_dir(Path(stained_dir))
    unstained_idx = _index_dir(Path(unstained_dir))

    common_keys = set(stained_idx.keys()) & set(unstained_idx.keys())

    rows = []
    for key in sorted(common_keys):
        slide_id, x, y = key
        rows.append({
            "id":        f"{slide_id}_{x}_{y}",
            "stained":   str(stained_idx[key]),
            "unstained": str(unstained_idx[key]),
        })

    df = pd.DataFrame(rows, columns=["id", "stained", "unstained"])
    return df


def get_all_pairs(data_root: str = "Dataset/Data") -> pd.DataFrame:
    """
    Collect all paired patches from both patient cohorts.

    Cohort 1 (AS-5198): stained/   ↔ tes_unstain/
    Cohort 2 (AS-7231): unstained/ ↔ tes_stain/

    Args:
        data_root: Root path containing the four dataset subdirectories

    Returns:
        Combined DataFrame with columns [id, stained, unstained]
    """
    root = Path(data_root)
    cohort1 = get_paired_patches(
        stained_dir   = root / "stained",
        unstained_dir = root / "tes_unstain",
    )
    cohort2 = get_paired_patches(
        stained_dir   = root / "unstained",
        unstained_dir = root / "tes_stain",
    )
    combined = pd.concat([cohort1, cohort2], ignore_index=True)
    return combined


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "Dataset/Data"
    df = get_all_pairs(root)
    print(f"Total paired patches: {len(df)}")
    print(df.head())

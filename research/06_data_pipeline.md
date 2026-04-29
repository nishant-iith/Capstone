# 06: Data Pipeline — From Raw Slides to Training Tensors

> **Bottom Line:** The pipeline runs raw whole-slide images through patch extraction → TV-L1 registration → SSIM scoring → quality filtering → RAM caching → augmentation → batched tensors. Every stage is essential; the **quality filter** (SSIM-based selection of top pairs) provides the largest single improvement to downstream model performance.

---

## 1. Raw Data Sources

| Source | Format | Resolution | Channels | Count |
|--------|--------|-----------|----------|-------|
| Unstained microscopy | TIFF | 1024×1024 patches | 3 (autofluorescence) | 8,885 |
| Stained H&E | TIFF | 1024×1024 patches | 3 (RGB) | 8,885 |

**Acquisition:** Patches are pre-extracted from whole-slide images (WSIs) at the same physical location, then re-imaged after H&E staining. The pairs are nominally registered at acquisition but deform during chemical processing.

---

## 2. Pipeline Stages

```
┌────────────────────┐
│ Raw paired images  │  8,885 pairs (mean SSIM 0.366)
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ TV-L1 Registration │  Dense flow → warp unstained
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Inline SSIM scoring│  Per-pair quality measurement
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Quality filter     │  Top-1000 / Top-2000 / All
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Patch extraction   │  Optional: 512×512 patches
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ RAM caching        │  Pre-load to memory
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Augmentation       │  Flips, rotations
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Normalization      │  [0,1] or [-1,1]
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Batched tensors    │  GPU training
└────────────────────┘
```

---

## 3. Stage 1 — TV-L1 Registration

See [02_image_registration.md](02_image_registration.md) for full method analysis.

**Pipeline (`registration_pipeline.py`):**
- Parallel `ProcessPoolExecutor` with 60 workers
- 8,885 pairs registered in ~25 minutes
- Inline SSIM scoring per pair (no second pass)
- Outputs: `data/processed/registered/{stained,unstained}/`

**Key Engineering Decision:** Score SSIM during registration, not after. Saves ~3 hours of recomputation.

---

## 4. Stage 2 — Quality Filtering

After registration, every pair has an SSIM score. We use this to filter aggressively.

**Tier Definitions:**

| Tier | Count | SSIM Range | Mean SSIM | Used In |
|------|-------|------------|-----------|---------|
| Top-1000 | 1,000 | [0.5363, 0.7463] | 0.6094 | v11, v16, v17 |
| Top-2000 | 2,000 | [0.5000, 0.7463] | 0.5800 | v10 |
| All | 8,885 | [0.20, 0.7463] | 0.42 | v13 (failed) |

**Critical Lesson (v13 failure):** Training on all 8,885 pairs (mean 0.51) capped achievable SSIM at 0.6326, even with a more complex architecture. Reverting to top-1000 enabled SSIM 0.7080-0.712.

**Filtering Code:**
```python
df = pd.read_csv("data/processed/registered_pairs_all.csv")
df_sorted = df.sort_values("ssim", ascending=False)
top_1000 = df_sorted.head(1000)
top_1000.to_csv("data/processed/registered_pairs.csv", index=False)
```

---

## 5. Stage 3 — Patch Extraction (v14, v15 only)

For v14 and v15, full 1024×1024 images were sliced into 512×512 patches with stride=512 (no overlap).

**Why patches?**
- 4× more training samples per pair
- Smaller GPU memory per sample → larger batches
- Each patch sees consistent local context
- v14 baseline showed +0.005 SSIM gain over full-size training at the same model

**Patch Selection:**
- Only patches from top-registered pairs were extracted
- Total: 3,606 patches from ~1,000 source pairs
- Mean patch SSIM: 0.6254

**Extraction Logic:**
```python
def extract_patches(stained, unstained, patch_size=512):
    H, W = stained.shape[:2]
    patches = []
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            s_patch = stained[y:y+patch_size, x:x+patch_size]
            u_patch = unstained[y:y+patch_size, x:x+patch_size]
            patches.append((s_patch, u_patch))
    return patches
```

**Verdict:** ✅ Useful for early experimentation. v17 reverted to full-size (1024×1024) for clinically realistic context.

---

## 6. Stage 4 — RAM Caching

**Problem:** Disk I/O at 1.5 GB/s vs GPU compute at 100s of GB/s → I/O is the bottleneck.

**Solution:** Pre-load all images to RAM before training begins.

**Implementation (`train_v17.py:FullSizeDataset`):**
```python
class FullSizeDataset(Dataset):
    def __init__(self, csv_path, n_samples=1000):
        self.df = pd.read_csv(csv_path).head(n_samples)
        self.cache = []
        print(f"Caching {n_samples} pairs to RAM...")
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            stained = io.imread(row["stained_path"])
            unstained = io.imread(row["unstained_path"])
            self.cache.append((stained, unstained))
        print(f"Cached. Memory: {get_memory_gb():.1f} GB")
```

**Memory Footprint:**
- 1,000 pairs × 2 images × 1024×1024 × 3 bytes ≈ **6 GB RAM**
- A100 server has 256 GB RAM → trivially fits

**Speedup:** Training throughput improved from 0.8 it/s to 1.3 it/s (+62%).

**Critical Engineering Note (v14 OOM):** With `num_workers=60`, each worker process **duplicates** the cache (Python's `fork` does not share memory for mutable lists in this manner). 60 × 6 GB = 360 GB → crash. Fix: `num_workers=0` since data is already in parent process memory.

---

## 7. Stage 5 — Augmentation

**Used Augmentations:**

| Augmentation | Probability | Purpose |
|--------------|-------------|---------|
| Random horizontal flip | 0.5 | Symmetry — slides have no preferred orientation |
| Random vertical flip | 0.5 | Same |
| Random 90° rotation | 0.25 each (0/90/180/270) | Same |

**Not Used:**
- ❌ Color jitter — would change H&E stain semantics
- ❌ Gaussian blur — destroys diagnostic edge information
- ❌ Random crop (at 1024×1024) — already full-resolution
- ❌ Normalization shifts — fixed reference for SSIM comparison

**Implementation (in dataset `__getitem__`):**
```python
if random.random() < 0.5:
    stained = np.fliplr(stained)
    unstained = np.fliplr(unstained)
if random.random() < 0.5:
    stained = np.flipud(stained)
    unstained = np.flipud(unstained)
k = random.randint(0, 3)
stained = np.rot90(stained, k)
unstained = np.rot90(unstained, k)
```

**Critical:** Both stained and unstained must receive the **identical** augmentation. Otherwise, the pair becomes misaligned again.

---

## 8. Stage 6 — Normalization

Two conventions used across versions:

| Convention | Range | Used In | Notes |
|------------|-------|---------|-------|
| `[0, 1]` (Sigmoid output) | 0-1 | v14, v17 | Direct from `image / 255.0` |
| `[-1, 1]` (Tanh output) | -1 to 1 | v8-v15 | `(image / 127.5) - 1.0` |

**SSIM Computation Caveat:** SSIM requires `data_range` to match. The `pytorch_msssim.SSIM(data_range=1.0)` config assumes [0,1]; use `data_range=2.0` if Tanh-normalized.

---

## 9. Stage 7 — DataLoader Configuration

**Optimized v17 Config:**
```python
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,        # Cached in parent → no fork needed
    pin_memory=True,      # Faster H2D transfer
    drop_last=True,       # Consistent batch shape
)
```

**Why `num_workers=0` despite typical advice for `num_workers>0`?**
- Data is already in RAM cache (no disk I/O to parallelize)
- Workers would only add fork overhead and memory duplication
- Profiling showed 0 workers was fastest for this configuration

**Why `pin_memory=True`?**
- Allocates host memory in pinned (non-pageable) region
- Enables async DMA transfers GPU↔CPU
- ~10-15% speedup on H2D transfers

**Why `drop_last=True`?**
- Last batch may be smaller, causing BatchNorm statistics issues
- Validation loss spikes on partial batches

---

## 10. Validation Set Strategy

**v14 Issue (Lesson Learned):** v14 used a **random** test image each epoch → high variance (val_ssim ranged 0.166 to 0.605). Made it nearly impossible to track real progress.

**v17 Attempted Fix (Buggy):**
v17 used `torch.utils.data.random_split(dataset, [900, 100])` to split train/val. Two bugs:
1. **No seed** — split is non-reproducible across runs.
2. **Augmentation leaks into val** — `FullSizeDataset(augment=True)` is wrapped by both train and val Subsets, so val pairs receive random flips each epoch → noisy val SSIM.

```python
# v17 — buggy
dataset = FullSizeDataset(csv, top_n=1000, augment=True)
train_ds, val_ds = random_split(dataset, [900, 100])  # No seed, augment leaks
```

**Required fix for v18:**
```python
# v18 — correct
generator = torch.Generator().manual_seed(42)
train_idx, val_idx = random_split(range(1000), [900, 100], generator=generator)
train_ds = FullSizeDataset(csv, indices=train_idx, augment=True)
val_ds   = FullSizeDataset(csv, indices=val_idx,   augment=False)  # No augment on val
```

**Verdict:** ❌ v17's val pipeline was unreliable. Fix mandatory for v18.

---

## 11. Reproducibility

**Seed Setting:**
```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = False  # Allow optimization
torch.backends.cudnn.benchmark = True       # Auto-tune
```

**Trade-off:** `cudnn.benchmark=True` is non-deterministic but ~10% faster. Determinism is sacrificed for throughput. Acceptable since experiments are reported with mean ± std across runs.

---

## 12. File Inventory

| File | Purpose |
|------|---------|
| `registration_pipeline.py` | TV-L1 registration with inline SSIM |
| `data/processed/registered_pairs_all.csv` | All 8,885 pairs metadata |
| `data/processed/registered_pairs.csv` | Filtered top pairs |
| `data/processed/registered/stained/` | Stained outputs |
| `data/processed/registered/unstained/` | Warped unstained outputs |
| `train_v14.py:PatchDataset` | Patch dataset class |
| `train_v17.py:FullSizeDataset` | Full-size dataset class |

---

## 13. Lessons on Data Engineering

1. **Quality > quantity** — Top-1000 (mean 0.61) beats All-8885 (mean 0.42) by 0.08 SSIM in trained models.
2. **Disk I/O is the bottleneck** — RAM cache saves >40% of wall-clock time.
3. **Validate on a fixed set** — Random validation introduces unmeasurable variance.
4. **Augmentation must be paired** — Same flip/rotate on stained AND unstained, or registration is undone.
5. **`num_workers` is not always good** — With pre-cached data, workers add overhead.
6. **Inline metrics save hours** — Compute SSIM during registration, not as a separate pass.

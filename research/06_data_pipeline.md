# 06: Data Pipeline — From Raw Slides to Training Tensors

> **Bottom Line:** The current best data path is CLAHE TV-L1 registration plus content-quality top-1000 selection, followed by fixed same-prefix evaluation. This produced the final ensemble score of SSIM 0.7838, PSNR 25.16, PCC 0.8794. Pure old-registered data remains a separate legacy distribution where v20 TTA4 is best.

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
│ Registration       │  Gray TV-L1 or CLAHE TV-L1 → warp unstained
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Inline SSIM scoring│  Per-pair quality measurement
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Quality filter     │  Top-K, best-of, positive-gain, content-quality
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

**2026-05-02 CLAHE Update (`registration_pipeline_clahe.py`):**
- Uses CLAHE-normalized grayscale images only for flow estimation.
- Warps the original RGB unstained image with the CLAHE-derived flow.
- Writes a live per-pair log: `logs/registration_clahe_per_pair.csv`.
- Completed 8,885 pairs with 56 workers in 93.5 minutes.
- Final all-pair mean SSIM: 0.5045 versus 0.4134 for old gray TV-L1.
- Mean gain versus old registration: +0.0911 SSIM.
- Negative-gain rows: 80/8885 (0.90%), which motivates positive-only and best-of CSV variants.

---

## 4. Stage 2 — Quality Filtering

After registration, every pair has an SSIM score. We use this to filter aggressively. After the CLAHE rerun, pair selection is no longer a single CSV: the project now keeps multiple top-K variants because "highest full-image SSIM" and "most informative tissue content" select different examples.

**Tier Definitions:**

| Tier | Count | SSIM Range | Mean SSIM | Used In |
|------|-------|------------|-----------|---------|
| Old gray TV-L1 top-1000 | 1,000 | [0.5363, 0.7463] | 0.6094 | v11, v16, v17, v20_fixed |
| CLAHE TV-L1 top-1000 | 1,000 | [0.5877, 0.7509] | 0.6423 | v22 data ablation |
| CLAHE TV-L1 top-1500 | 1,500 | [0.5564, 0.7509] | 0.6185 | candidate diversity run |
| CLAHE TV-L1 top-2000 | 2,000 | [0.5376, 0.7509] | 0.6004 | candidate diversity run |
| CLAHE all | 8,885 | [0.1719, 0.7509] | 0.5045 | metadata/scoring pool |

**Critical Lesson (v13 failure):** Training on all 8,885 pairs (mean 0.51) capped achievable SSIM at 0.6326, even with a more complex architecture. Reverting to top-1000 enabled SSIM 0.7080-0.712.

**Filtering Code:**
```python
df = pd.read_csv("data/processed/registered_pairs_all.csv")
df_sorted = df.sort_values("ssim", ascending=False)
top_1000 = df_sorted.head(1000)
top_1000.to_csv("data/processed/registered_pairs.csv", index=False)
```

**Current CSV Variant Generators:**
- `make_registration_training_csvs.py` writes pure CLAHE, positive-gain, and best-of-old-vs-CLAHE top-K CSVs.
- `score_registration_tissue.py` computes tissue/content-aware post-registration scores.
- `make_content_quality_csvs.py` writes combined high-content/high-SSIM/positive-gain top-K CSVs.

**Recommended v22A CSVs:**

| CSV | Selection Logic | Why It Exists |
|-----|-----------------|---------------|
| `registered_bestof_old_clahe_top1000.csv` | Highest best-of SSIM, fallback to old if old > CLAHE | safest direct replacement for old top-1000 |
| `registered_clahe_positive_top1000.csv` | CLAHE only, excludes negative-gain rows | tests pure CLAHE without known regressions |
| `content_quality_minrgb0.50_positive_top1000.csv` | combined content-region SSIM, full RGB SSIM, content fraction, positive gain | selects harder, tissue-rich, well-registered patches |

**Content-Aware Scoring Result:** Foreground tissue masking was not discriminative because tested patches were nearly all tissue (`tissue_fraction=1.0000`). Edge/content-aware masking was useful: it selected about 50.65% of pixels on average over all 8,885 pairs and generated top-K lists that differ substantially from full-SSIM ranking.

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

**Required fix for all future runs:**
```python
# correct
generator = torch.Generator().manual_seed(42)
train_idx, val_idx = random_split(range(1000), [900, 100], generator=generator)
train_ds = FullSizeDataset(csv, indices=train_idx, augment=True)
val_ds   = FullSizeDataset(csv, indices=val_idx,   augment=False)  # No augment on val
```

**Verdict:** ❌ v17's val pipeline was unreliable. Fixed, explicit validation splits are mandatory for every run.

**2026-05-01 Audit Finding (v19b/v20):**
v19b and the first v20 run used separate shuffled dataset instances for train and validation. Because each instance shuffled independently and then sliced, the nominal validation set was not held out. A prefix audit found **91/100 validation prefixes overlapped training**.

```python
# historical v19b/v20 pattern — leaky
train_ds = RegisteredPairsDataset(csv, n_samples=900, augment=True)
val_ds   = RegisteredPairsDataset(csv, n_samples=100, augment=False)
# each dataset shuffled independently, so val prefixes can also be in train
```

**Current v20_fixed/v22A Strategy:**
Split the dataframe once, pass explicit indices into each dataset, disable validation augmentation, and assert no prefix overlap before training starts.

```python
split_perm = np.random.RandomState(SEED).permutation(len(df))
train_indices = split_perm[:900]
val_indices = split_perm[900:1000]
train_ds = RegisteredPairsDataset(csv, indices=train_indices, augment=True)
val_ds   = RegisteredPairsDataset(csv, indices=val_indices, augment=False)
assert len(set(train_ds.prefixes) & set(val_ds.prefixes)) == 0
```

**Verdict:** v19b 0.7489 and old v20 0.7549 are historical training signals, not clean validation scores. Use v20_fixed metrics from `logs/v20_fixed_training.log` for current clean reporting. For v22A content-quality experiments, do not compare the internal validation SSIM directly to v20_fixed unless both models are evaluated on the same fixed external validation set.

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
| `registration_pipeline.py` | Original gray TV-L1 registration with inline SSIM |
| `registration_pipeline_clahe.py` | CLAHE TV-L1 registration with per-pair SSIM/delta logging |
| `score_registration_tissue.py` | Tissue/content-aware post-registration scoring |
| `make_registration_training_csvs.py` | Pure CLAHE, positive-gain, and best-of top-K CSV generation |
| `make_content_quality_csvs.py` | Combined high-content/high-SSIM CSV generation |
| `data/processed/registered_pairs_all.csv` | All 8,885 pairs metadata |
| `data/processed/registered_pairs.csv` | Filtered top pairs |
| `data/processed/registered_clahe_pairs_all.csv` | All CLAHE-registered pair metadata |
| `data/processed/content_quality_csvs/` | Content-quality training CSV variants |
| `data/processed/registered/stained/` | Stained outputs |
| `data/processed/registered/unstained/` | Warped unstained outputs |
| `train_v14.py:PatchDataset` | Patch dataset class |
| `train_v17.py:FullSizeDataset` | Full-size dataset class |
| `train_v20.py:RegisteredPairsDataset` | Current explicit-index split implementation |

---

## 13. Lessons on Data Engineering

1. **Quality > quantity** — Top-1000 (mean 0.61 old, 0.6423 CLAHE) beats All-8885 by avoiding noisy label pairs.
2. **Disk I/O is the bottleneck** — RAM cache saves >40% of wall-clock time.
3. **Validate on a fixed set** — Random validation introduces unmeasurable variance.
4. **Augmentation must be paired** — Same flip/rotate on stained AND unstained, or registration is undone.
5. **`num_workers` is not always good** — With pre-cached data, workers add overhead.
6. **Inline metrics save hours** — Compute SSIM during registration, not as a separate pass.
7. **Content and quality should be separated** — High-content patches are harder and may have lower full SSIM, but the 1,000-pair test showed they received larger CLAHE gains. Use combined scoring rather than full SSIM alone.
8. **Keep old registration for fallback** — CLAHE improved 99.1% of rows, but a best-of CSV is safer because 0.90% regressed.

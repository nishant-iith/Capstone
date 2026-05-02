# 02: Image Registration — All Methods Evaluated

> **Bottom Line:** TV-L1 Optical Flow was selected historically, and the current best version is CLAHE-driven TV-L1. CLAHE TV-L1 improved all-pair registration mean SSIM from 0.4134 to 0.5045 and enabled the final CLAHE ensemble score of SSIM 0.7838. The old registered path remains useful for the v20 TTA4 fallback.

---

## 1. The Registration Problem

The **unstained** and **stained** images are paired (same tissue slice) but **physically misaligned** because:

| Source of Misalignment | Effect |
|------------------------|--------|
| Slide repositioning under microscope | Global translation + rotation |
| Heat from chemical staining | Local stretching (~1-5%) |
| Liquid chemicals during processing | Local distortion of tissue regions |
| Slide mounting pressure | Compression artifacts |
| Tissue thickness (~5 microns) | Z-axis variations |

**Result:** A nucleus in the unstained image is typically displaced by 5-50 pixels (at 1024×1024 resolution) in the stained image, with **non-uniform** displacement across the field of view.

This is a **non-rigid deformation problem**. Linear or rigid registration cannot solve it.

---

## 2. Methods Evaluated

We benchmarked 4 registration paradigms: Phase Correlation, ORB Feature Matching, TV-L1 Optical Flow, and SyN (ANTs).

---

### 2.1. Phase Correlation (Frequency Domain)

**Mathematical Basis:**
Computes the cross-power spectrum between two images using the Fourier Shift Theorem:

$$
\hat{R}(u,v) = \frac{F^*(u,v) \cdot G(u,v)}{|F^*(u,v) \cdot G(u,v)|}
$$

Inverse FFT yields a peak at the global translation $(\Delta x, \Delta y)$.

**Strengths:**
- Extremely fast: $O(N \log N)$
- Robust to uniform luminance changes
- Well-established (used in satellite imaging)

**Weaknesses for Histology:**
- ❌ Only handles **global rigid translation**
- ❌ Cannot handle rotation > 1°
- ❌ Cannot handle **non-rigid deformation** (the dominant artifact in histology)

**Test Result on Histology:**
- Reduced misalignment for ~20% of pairs (those with mostly translation)
- Failed catastrophically on 80% of pairs with tissue stretching/rotation
- **Average SSIM gain: +0.04** (from 0.366 to 0.41)

**Verdict:** ❌ **Discarded**. Fundamentally insufficient model class.

---

### 2.2. ORB Feature Matching (Feature-Based)

**Mathematical Basis:**
Oriented FAST keypoints + Rotated BRIEF descriptors, matched via Hamming distance, with RANSAC for transformation estimation.

```python
orb = cv2.ORB_create(nfeatures=5000)
kp1, des1 = orb.detectAndCompute(unstained, None)
kp2, des2 = orb.detectAndCompute(stained, None)
matches = bf_matcher.match(des1, des2)
# Filter via RANSAC
M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC)
```

**Strengths:**
- Handles translation + rotation + scale
- Works well in natural images (buildings, faces)
- Real-time capable

**Weaknesses for Histology:**
- ❌ **Texture repetition catastrophe**: Histology contains thousands of nearly-identical nuclei. ORB descriptors confuse one nucleus for another, producing massive false matches.
- ❌ **Cross-modal failure**: A "corner" in grayscale autofluorescence does not look like a "corner" in pink/purple H&E. Descriptor matching across these modalities is unreliable.
- ❌ **RANSAC convergence failure**: With >50% incorrect matches, RANSAC cannot converge to a valid transformation matrix.

**Test Result on Histology:**
- Convergence failure rate: ~70% of pairs
- Where it converged, transformation was often catastrophically wrong
- **Average SSIM gain: +0.02** (and dramatic failures on a third of cases)

**Verdict:** ❌ **Discarded**. Wrong tool for the domain.

---

### 2.3. TV-L1 Optical Flow ⭐ **(SELECTED)**

**Mathematical Basis:**
Total Variation regularization with L1 data fidelity. Solves the variational optimization problem:

$$
\min_{(u,v)} \int \left( |\nabla u| + |\nabla v| \right) dx \;+\; \lambda \int \left| I_{stained}(x) - I_{unstained}(x + (u,v)) \right| dx
$$

This computes a **dense displacement field** $(u(x), v(x))$ — one motion vector per pixel.

**Implementation:**
```python
from skimage.registration import optical_flow_tvl1
v, u = optical_flow_tvl1(reference_image, target_image)
# v, u are arrays of shape (H, W) giving per-pixel displacement
```

**Strengths:**
- ✅ **Dense, non-rigid**: Each pixel has its own displacement vector
- ✅ **Handles tissue stretching**: Local deformation is the primary need
- ✅ **L1 robust to brightness**: Cross-modal (grayscale → color) registration tolerated
- ✅ **TV regularization**: Smooth flow field, no checkerboard artifacts

**Weaknesses:**
- Slower than rigid methods (~2-3 sec per 1024×1024 pair on CPU)
- Memory intensive (multi-resolution pyramid)
- Sensitive to extreme color differences

**Pipeline (`registration_pipeline.py`):**
1. Convert both images to grayscale (for flow estimation)
2. Compute TV-L1 optical flow on grayscale pair
3. Apply the resulting flow field to **each RGB channel** of the unstained image
4. Use `mode='edge'` to avoid black border artifacts
5. Convert back to uint8 with clipping

**Test Results on Full Dataset (8,885 pairs):**

| Metric | Pre-Registration | Post-Registration | Δ |
|--------|------------------|-------------------|----|
| SSIM (Structure) | 0.3666 | **0.6317** | **+72.3%** |
| MSE (Error) | 0.0649 | **0.0113** | **-82.6%** |
| Mutual Information | 0.1583 | **0.5998** | **+278%** |

**Verdict:** ✅ **SELECTED as original primary registration method.**

---

### 2.3B. CLAHE-Driven TV-L1 — Current Best Registration Dataset

**Motivation:** Gray TV-L1 estimates motion from raw grayscale intensity. In this project, raw grayscale is a weak registration driver because the fixed image is H&E-stained and the moving image is unstained. Brightness and stain appearance differ even where tissue morphology matches, so the optical-flow solver can follow stain-domain contrast rather than nuclei/tissue structure.

**Change:** Before computing TV-L1 flow, both images are converted to grayscale and passed through CLAHE (Contrast Limited Adaptive Histogram Equalization). CLAHE is used only to estimate the flow field. The saved registered pair still contains the original RGB stained image and the original RGB unstained image warped by the CLAHE-derived flow.

```python
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
driver = clahe.apply(gray).astype(np.float32) / 255.0
v, u = optical_flow_tvl1(stained_driver, unstained_driver)
warped_rgb = warp_rgb(unstained_rgb, v, u)
```

**Controlled Ablation Before Full Rerun:**

| Test | Samples | Best Variant | Result |
|------|---------|--------------|--------|
| 512px screening | 8 tiered samples | CLAHE TV-L1 | Only candidate worth pursuing; edge-only and gray-at-512 worsened |
| Full-res selected | 8 tiered samples | CLAHE / CLAHE+prefilter | Mean gain about +0.049 SSIM; won 8/8 |
| Full-res larger check | 40 tiered samples | CLAHE default | Mean gain +0.0562 SSIM; won 38/40 |

**Full Dataset Rerun (`registration_pipeline_clahe.py`):**

| Metric | Old Gray TV-L1 | CLAHE TV-L1 | Δ |
|--------|----------------|-------------|---|
| All-pair mean SSIM | 0.4134 | **0.5045** | **+0.0911** |
| Median SSIM | 0.3893 | **0.4948** | **+0.1055** |
| Top-1000 mean SSIM | 0.6094 | **0.6423** | **+0.0329** |
| Top-1000 cutoff | 0.5363 | **0.5877** | **+0.0514** |
| Max SSIM | 0.7463 | **0.7509** | +0.0046 |
| Negative-gain files | n/a | 80/8885 | 0.90% |

**Interpretation:** CLAHE improves registration most where raw gray TV-L1 struggled. On the full dataset the mean pair gain is +0.0911 SSIM, but the top-1000 gain is smaller because the old top-1000 already consisted of easier high-quality pairs. This is still valuable: the top-1000 cutoff moved from 0.5363 to 0.5877 while preserving all 13 slide sources.

**Outputs:**
- `registration_pipeline_clahe.py` — full CLAHE TV-L1 registration pipeline
- `data/processed/registered_clahe/` — registered RGB image pairs
- `data/processed/registered_clahe_pairs_all.csv` — all scored CLAHE pairs
- `logs/registration_clahe_per_pair.csv` — live per-pair SSIM, old SSIM, and delta
- `data/processed/training_csv_variants/` — top-K, positive-only, and best-of-old-vs-CLAHE training CSVs

**Best-of Safety:** Because 0.90% of files worsened under CLAHE, the safest top-K variant is not always pure CLAHE. The `registered_bestof_old_clahe_topK` CSVs fall back to the old gray TV-L1 pair when old SSIM is higher. For top-1000, the best-of CSV uses 968 CLAHE pairs and 32 old fallbacks.

**Verdict:** ✅ **Current best registration dataset for training ablations.** Use CLAHE top-K and best-of top-K for model training comparisons. Do not delete the old gray TV-L1 dataset because it remains useful for best-of fallback and historical comparison.

---

### 2.4. SyN (Symmetric Normalization, ANTs) — Late-Stage Evaluation

**Mathematical Basis:**
Diffeomorphic image registration using the Greedy Symmetric Normalization algorithm, implemented in the **ANTs (Advanced Normalization Tools)** library. Computes a **diffeomorphism** — a smooth, invertible deformation field.

```python
import ants
fixed  = ants.from_numpy(stained_gray)
moving = ants.from_numpy(unstained_gray)
reg = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')
warped = reg['warpedmovout'].numpy()
```

**Strengths:**
- Diffeomorphic guarantee (preserves topology)
- High-precision medical image registration
- Symmetric (forward = inverse)

**Weaknesses:**
- ✗ **Slower than TV-L1** (~10-30 sec per pair vs 2-3 sec)
- ✗ **Less robust at large displacements** (>50 pixels)

**Initial Comparison Test (16-window grid, 4×4, 256px windows on `AS-5198-23-Z35_patch_16384_23552`):**

| Method | Mean SSIM | Notes |
|--------|-----------|-------|
| TV-L1 (full 1024px) | 0.5756 | Better at large-scale alignment |
| SyN (256px windows) | 0.7993 | Better at local fine alignment |
| SyN (full 1024px) | 0.5313 | Slightly worse than TV-L1 |

**Insight:** SyN is **better at small windows** (where displacements are small and local), but **TV-L1 is better at full resolution** (where displacements can be large).

**2026-05-01 Controlled Evidence Update:**
Later saved comparison files do **not** currently justify replacing TV-L1:

| Saved Result | TV-L1 Mean SSIM | SyN Mean SSIM | Winner |
|--------------|-----------------|---------------|--------|
| `syn_test_results/syn_vs_tv_results.csv` | 0.5756 | 0.5313 | TV-L1 |
| `window_test_results/windowed_comparison.csv` | 0.6667 | 0.5128 | TV-L1 on all 16 windows |

This means SyN/windowed SyN should not replace TV-L1. The current best path is still TV-L1-style dense optical flow, but with CLAHE-preprocessed driver images rather than raw grayscale drivers.

**Implementation Files:**
- `test_syn.py` — Single pair full-resolution comparison
- `window_test_syn_tv.py` — Windowed grid comparison

**Verdict:** ⚠️ **Not adopted as primary**. TV-L1-style optical flow retained because:
1. Full-image registration is the primary need
2. TV-L1 is 5-10× faster
3. Pipeline is already validated end-to-end
4. Latest saved SyN/windowed comparisons do not beat TV-L1

---

## 3. Final Pipeline Architecture

The original production registration pipeline (`registration_pipeline.py`):

```python
def register_pair(stained_path, unstained_path):
    # 1. Load both images at full 1024×1024 resolution
    s = io.imread(stained_path)
    u = io.imread(unstained_path)

    # 2. Grayscale for flow estimation
    s_gray = rgb2gray(s)
    u_gray = rgb2gray(u)

    # 3. TV-L1 optical flow on grayscale
    v, du = optical_flow_tvl1(s_gray, u_gray)

    # 4. Apply flow to each RGB channel of unstained
    coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords[0] += v; coords[1] += du
    u_warped = np.zeros_like(u)
    for c in range(3):
        u_warped[..., c] = warp(u[..., c], coords, mode='edge')

    # 5. Compute SSIM inline for quality scoring
    ssim_score = ssim(s, u_warped, channel_axis=2, data_range=255)

    return u_warped, ssim_score
```

**Parallelization:** The original gray TV-L1 pipeline used up to 60 workers. The full CLAHE TV-L1 rerun used 56 workers and completed 8,885 pairs in 93.5 minutes while writing per-pair SSIM logs.

**Output Files:**
- `data/processed/registered/stained/` — Stained (unchanged)
- `data/processed/registered/unstained/` — Warped to align with stained
- `data/processed/registered_pairs_all.csv` — All pairs with SSIM scores
- `data/processed/registered_pairs.csv` — Top-2000 highest SSIM pairs
- `data/processed/registered_clahe_pairs_all.csv` — All CLAHE TV-L1 pairs with SSIM scores
- `logs/registration_clahe_per_pair.csv` — CLAHE per-pair SSIM, old SSIM, delta, and status

---

## 4. Quality Distribution of Registered Pairs

| Pair Quality Tier | Count | SSIM Range |
|-------------------|-------|------------|
| Old gray TV-L1 top 1,000 | 1000 | 0.5363 - 0.7463 (mean 0.6094) |
| CLAHE TV-L1 top 1,000 | 1000 | 0.5877 - 0.7509 (mean 0.6423) |
| CLAHE TV-L1 top 1,500 | 1500 | 0.5564 - 0.7509 (mean 0.6185) |
| CLAHE TV-L1 top 2,000 | 2000 | 0.5376 - 0.7509 (mean 0.6004) |
| CLAHE TV-L1 all registered | 8885 | 0.1719 - 0.7509 (mean 0.5045) |

**Used for training:** Top 1,000-2,000 pairs (varies by experiment). Current candidate CSVs include pure CLAHE top-K, positive-gain-only top-K, best-of-old-vs-CLAHE top-K, content-ranked top-K, and content-quality top-K.

---

## 4.1. Content-Aware Post-Registration Scoring

After the CLAHE rerun, a post-process scorer (`score_registration_tissue.py`) tested whether pair selection should consider tissue richness, not only full-image SSIM.

**Foreground tissue mask result:** Not useful on the first 1,000 completed files because the patches were essentially all tissue (`tissue_fraction = 1.0000`). Tissue-only SSIM was therefore nearly identical to full-image SSIM.

**Content/edge mask result:** Useful. A high-information mask based on Sobel/edge content selected about half the pixels and changed rankings substantially.

| Metric on 1,000-pair test | Value |
|---------------------------|-------|
| Mean full RGB SSIM | 0.5209 |
| Mean content-gray SSIM | 0.5403 |
| Mean content fraction | 0.5140 |
| Top-100 overlap: full RGB vs content-gray ranking | 38/100 |
| Top-500 overlap: full RGB vs content-gray ranking | 364/500 |
| Correlation: content fraction vs CLAHE gain | 0.698 |

**High-content vs low-content behavior on the 1,000-pair test:**

| Bucket | Mean Old SSIM | Mean CLAHE SSIM | Mean Gain | Mean Content-Gray SSIM |
|--------|---------------|-----------------|-----------|------------------------|
| Low-content quartile | 0.5165 | 0.5892 | +0.0727 | 0.5217 |
| High-content quartile | 0.3596 | 0.4949 | +0.1353 | 0.5531 |

**Interpretation:** High-content patches are harder and have lower full-image SSIM, but they benefited more from CLAHE registration. This supports training data variants that select for both final registration quality and high tissue/texture content rather than only full-image SSIM.

**Full content-quality outputs:**
- `data/processed/registered_clahe_content_scores.csv`
- `data/processed/content_ranked_csvs/`
- `data/processed/content_quality_csvs/`

The recommended content-quality top-1000 CSV is `data/processed/content_quality_csvs/content_quality_minrgb0.50_positive_top1000.csv`: all 13 slides, mean full RGB SSIM 0.5601, mean content-gray SSIM 0.6223, mean content fraction 0.5516, mean gain +0.1152, and no negative-gain rows.

---

## 5. Lessons Learned

1. **Non-rigid registration is non-negotiable** for histology. Rigid methods (Phase, ORB) cannot achieve >0.5 SSIM.

2. **Cross-modal robustness matters.** L1-based methods (TV-L1) outperform L2-based methods because brightness assumptions break across staining domains.

3. **Inline SSIM scoring** is critical for downstream filtering. Originally we registered first and scored separately — re-running registration to add scoring saved hours.

4. **Window-based vs full-image trade-off**: SyN excels at small windows, TV-L1 at full images. A hybrid approach (TV-L1 for global, SyN for local refinement) might further improve registration but was not pursued due to complexity vs marginal gain.

5. **Registration is the structural floor.** No GAN can train past 0.7 SSIM if the registration baseline is 0.4. Investing in registration quality has 10× the ROI of model architecture changes early in a project.

---

## 6. Files Referenced

| File | Purpose |
|------|---------|
| `registration_pipeline.py` | Original gray TV-L1 pipeline (parallel, with inline SSIM) |
| `registration_pipeline_clahe.py` | Current CLAHE TV-L1 registration pipeline |
| `score_registration_tissue.py` | Post-registration tissue/content-aware scoring |
| `make_registration_training_csvs.py` | Pure CLAHE, positive-only, and best-of training CSV generator |
| `make_content_quality_csvs.py` | Combined high-content/high-SSIM CSV generator |
| `test_syn.py` | SyN single-image evaluation script |
| `window_test_syn_tv.py` | SyN vs TV-L1 windowed comparison |
| `data/processed/registered_pairs_all.csv` | Full registration metadata (8,885 rows) |
| `ImageRegistration.md` | Original technical analysis (legacy doc) |

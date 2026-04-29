# 02: Image Registration — All Methods Evaluated

> **Bottom Line:** TV-L1 Optical Flow was selected. It provided +72% SSIM gain (0.366 → 0.632) by computing dense pixel-level deformation fields, handling the non-rigid warping of tissue during chemical staining.

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

**Verdict:** ✅ **SELECTED as primary registration method.**

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

**Comparison Test (16-window grid, 4×4, 256px windows on `AS-5198-23-Z35_patch_16384_23552`):**

| Method | Mean SSIM | Notes |
|--------|-----------|-------|
| TV-L1 (full 1024px) | 0.5756 | Better at large-scale alignment |
| SyN (256px windows) | 0.7993 | Better at local fine alignment |
| SyN (full 1024px) | 0.5313 | Slightly worse than TV-L1 |

**Insight:** SyN is **better at small windows** (where displacements are small and local), but **TV-L1 is better at full resolution** (where displacements can be large).

**Implementation Files:**
- `test_syn.py` — Single pair full-resolution comparison
- `window_test_syn_tv.py` — Windowed grid comparison

**Verdict:** ⚠️ **Not adopted as primary**, but identified as potentially useful for **patch-based fine-tuning** in future work. TV-L1 retained as primary because:
1. Full-image registration is the primary need
2. TV-L1 is 5-10× faster
3. Pipeline is already validated end-to-end

---

## 3. Final Pipeline Architecture

The production registration pipeline (`registration_pipeline.py`):

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

**Parallelization:** `ProcessPoolExecutor` with 60 workers reduces 8,885-pair processing time from ~12 hours to ~25 minutes on the A100 server.

**Output Files:**
- `data/processed/registered/stained/` — Stained (unchanged)
- `data/processed/registered/unstained/` — Warped to align with stained
- `data/processed/registered_pairs_all.csv` — All pairs with SSIM scores
- `data/processed/registered_pairs.csv` — Top-2000 highest SSIM pairs

---

## 4. Quality Distribution of Registered Pairs

| Pair Quality Tier | Count | SSIM Range |
|-------------------|-------|------------|
| Top 1,000 | 1000 | 0.5363 - 0.7463 (mean 0.6094) |
| Top 2,000 | 2000 | 0.5000+ |
| All registered | 8885 | 0.20 - 0.7463 (mean 0.42) |

**Used for training:** Top 1,000-2,000 pairs (varies by experiment).

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
| `registration_pipeline.py` | Production TV-L1 pipeline (parallel, with inline SSIM) |
| `test_syn.py` | SyN single-image evaluation script |
| `window_test_syn_tv.py` | SyN vs TV-L1 windowed comparison |
| `data/processed/registered_pairs_all.csv` | Full registration metadata (8,885 rows) |
| `ImageRegistration.md` | Original technical analysis (legacy doc) |

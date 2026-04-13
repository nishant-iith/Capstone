# Phase 4: Comprehensive Validation - Research

**Researched:** 2026-04-13  
**Domain:** Medical image generation validation (SSIM/PSNR metrics, visual comparison, error mapping)  
**Confidence:** HIGH

## Summary

Phase 4 delivers quantitative and qualitative validation of the Phase 3 guarded model's diagnostic reliability. The standard approach integrates three complementary validation streams:

1. **Quantitative metrics (VAL-01):** SSIM (structural similarity, 0-1 scale) and PSNR (peak signal-to-noise, dB scale) computed per-image and aggregated across the test set. Medical imaging research consistently uses these metrics; recent histopathology GAN studies report SSIM ≈ 0.93 and PSNR ≈ 24-30 dB as indicators of clinical-grade synthetic image quality.

2. **Visual comparison grids (VAL-02):** Side-by-side (Unstained, Virtual, Real) layout showing 6-12 representative test patches. Grid sizes of 256-512px per image balance readability with batch processing efficiency.

3. **Error/difference maps (VAL-03):** Per-pixel absolute differences rendered as heatmaps using diverging colormaps (e.g., 'RdBu') to highlight where Virtual diverges from Real. Error concentration in high-frequency regions (speckles) is expected; concentration in cell boundaries signals model failure.

**Primary recommendation:** Build validation as a modular pipeline: (1) load Phase 3 guarded checkpoint + test set, (2) compute metrics per-image, (3) generate grids with sampled/worst-case patches, (4) produce heatmapped error maps, (5) compile report with statistical summaries. Use `scikit-image.metrics.structural_similarity` and `scikit-image.metrics.peak_signal_noise_ratio` as primary tools; augment with `matplotlib` for visualization. This approach aligns with medical imaging standards and enables incremental validation without retraining.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VAL-01 | User can generate a report showing mean SSIM and PSNR for the test set | `scikit-image.metrics` provides `structural_similarity` and `peak_signal_noise_ratio` functions; research shows SSIM ≈ 0.93±0.01 and PSNR ≈ 24-30 dB are standard ranges for medical H&E synthesis |
| VAL-02 | User can view a grid of (Unstained, Virtual, Real) images for qualitative review | Standard practice: 6-12 patch grid with 256-512px per image; can sample randomly or curate best/worst cases; matplotlib enables batching multiple grids efficiently |
| VAL-03 | User can produce a difference map image highlighting error between Virtual and Real | Absolute difference maps normalized per-pixel with matplotlib's diverging colormaps (RdBu, PuOr); error concentration in high-frequency regions expected for perceptually accurate synthesis |

## Standard Stack

### Core Metrics

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-image | 0.26.0+ | SSIM and PSNR computation | Industry-standard for medical image metrics; stable API, widely cited in medical imaging papers (VISGAB 2025, stain normalization 2025) |
| numpy | 1.24.0+ | Numerical array operations | Essential for batch metric aggregation, normalization, statistical summaries |
| matplotlib | 3.8.0+ | Figure generation, visualization | Standard for medical image grids and heatmap rendering; colormap support for diverging error maps |
| pandas | 2.0.0+ | Results aggregation and reporting | Enables tabular metric summaries and per-image dataframes |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch (PyTorch) | 2.0.0+ | Model loading and inference | Required for loading Phase 3 checkpoint; inference on test set |
| pytorch-lightning | 2.0.0+ | Checkpoint compatibility | For loading Pix2PixLightning checkpoints from Phase 2-3 |
| PIL/Pillow | 10.0.0+ | Image I/O and basic manipulation | Save/load PNG/JPG outputs; grid assembly |
| scipy | 1.11.0+ | Optional: advanced filtering | Already available in medical imaging stacks; can optionally replace some numpy operations |

### Installation

```bash
pip install scikit-image>=0.26.0 numpy>=1.24.0 matplotlib>=3.8.0 pandas>=2.0.0 torch>=2.0.0 pytorch-lightning>=2.0.0 pillow>=10.0.0
```

**Version verification notes:**
- `scikit-image 0.26.0`: Stable release (Dec 2024); `structural_similarity` and `peak_signal_noise_ratio` APIs are stable across 0.24-0.26.x versions.
- `matplotlib 3.8.0+`: Colormapnorm (CenteredNorm) available; critical for diverging colormaps on difference maps.
- `numpy 1.24.0+`: Sufficient for all array operations; no special features required.

## Architecture Patterns

### Recommended Project Structure

```
src/
├── validation/                    # NEW: Phase 4 validation module
│   ├── __init__.py
│   ├── metrics.py                # SSIM, PSNR computation and aggregation
│   ├── comparison_grid.py         # Visual grid generation (Unstained, Virtual, Real)
│   ├── error_maps.py              # Difference map rendering with heatmaps
│   └── report_generator.py        # PDF/Markdown report compilation
├── inference/                     # EXISTING: Phase 2-3 inference utilities
│   ├── predict.py                 # Model inference
│   ├── visualize.py               # Basic visualization
│   ├── hallucination_detection.py  # Phase 3 artifact detection
│   └── comparison_report.py        # Phase 3 baseline vs guarded analysis
├── models/
├── training/
└── data/

reports/                           # Output location
├── phase4_quantitative_metrics.csv
├── phase4_validation_report.md
├── validation_grids/              # Comparison grid images
│   ├── grid_random_samples.png
│   ├── grid_best_cases.png
│   └── grid_worst_cases.png
└── error_maps/                    # Heatmapped difference maps
    ├── error_heatmap_patch_0.png
    └── error_heatmap_patch_1.png
```

### Pattern 1: Metric Computation (Batch Processing)

**What:** Compute SSIM and PSNR for all test patches in batches to avoid memory overload and enable parallel processing.

**When to use:** Validation of large test sets (100+ patches) or high-resolution images (>512x512px).

**Example:**

```python
# Source: scikit-image 0.26.0 API + medical imaging practice (VISGAB 2025)
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import numpy as np

def compute_metrics_batch(virtual_batch, real_batch, data_range=1.0):
    """
    Compute SSIM and PSNR for a batch of image pairs.
    
    Args:
        virtual_batch: [B, H, W, 3] float array, normalized to [0, 1]
        real_batch: [B, H, W, 3] float array, normalized to [0, 1]
        data_range: Expected range of pixel values (1.0 for [0,1], 255 for [0,255])
    
    Returns:
        ssim_scores: [B] array of SSIM values (0-1 range)
        psnr_scores: [B] array of PSNR values (dB scale)
    """
    batch_size = virtual_batch.shape[0]
    ssim_scores = []
    psnr_scores = []
    
    for i in range(batch_size):
        # SSIM: Multi-channel support via channel_axis parameter (0.26.0+)
        ssim = structural_similarity(
            virtual_batch[i], 
            real_batch[i],
            data_range=data_range,
            channel_axis=2  # RGB channels
        )
        ssim_scores.append(ssim)
        
        # PSNR: Standard computation
        psnr = peak_signal_noise_ratio(
            real_batch[i],
            virtual_batch[i],
            data_range=data_range
        )
        psnr_scores.append(psnr)
    
    return np.array(ssim_scores), np.array(psnr_scores)
```

### Pattern 2: Visual Comparison Grid

**What:** Generate side-by-side (Unstained, Virtual, Real) image grids for qualitative review. Grid layout: 3 columns × N rows, where N = number of representative patches.

**When to use:** Presenting model outputs to stakeholders or including in validation reports.

**Example:**

```python
# Source: matplotlib 3.8.0+ visualization patterns + medical imaging standards
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def create_comparison_grid(unstained_batch, virtual_batch, real_batch, 
                           sample_indices=None, figsize=(15, 5)):
    """
    Create side-by-side comparison grid.
    
    Args:
        unstained_batch: [B, H, W, 3] unstained source images
        virtual_batch: [B, H, W, 3] model-generated H&E
        real_batch: [B, H, W, 3] ground truth H&E
        sample_indices: [N] indices to display (default: first 3)
        figsize: (width, height) for figure
    
    Returns:
        fig: matplotlib figure object
    """
    if sample_indices is None:
        sample_indices = np.arange(min(3, unstained_batch.shape[0]))
    
    n_samples = len(sample_indices)
    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_samples / 3))
    gs = GridSpec(n_samples, 3, figure=fig, hspace=0.3, wspace=0.1)
    
    for row, idx in enumerate(sample_indices):
        # Unstained
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(np.clip(unstained_batch[idx], 0, 1))
        ax0.set_title("Unstained" if row == 0 else "", fontsize=10)
        ax0.axis('off')
        
        # Virtual
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(np.clip(virtual_batch[idx], 0, 1))
        ax1.set_title("Virtual H&E" if row == 0 else "", fontsize=10)
        ax1.axis('off')
        
        # Real
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(np.clip(real_batch[idx], 0, 1))
        ax2.set_title("Real H&E" if row == 0 else "", fontsize=10)
        ax2.axis('off')
    
    return fig
```

### Pattern 3: Error/Difference Map with Heatmap

**What:** Compute per-pixel absolute differences and render as heatmaps with diverging colormaps to highlight error concentration.

**When to use:** Diagnosing failure modes (where does the model diverge most from reality?).

**Example:**

```python
# Source: matplotlib 3.8.0+ colormap normalization + numpy operations
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def create_error_heatmap(virtual_patch, real_patch, figsize=(6, 6)):
    """
    Create heatmapped error map (absolute difference).
    
    Args:
        virtual_patch: [H, W, 3] virtual image (0-1 range)
        real_patch: [H, W, 3] real image (0-1 range)
        figsize: figure size
    
    Returns:
        fig: matplotlib figure object
    """
    # Compute absolute difference (L1 error per pixel)
    # Average across RGB channels to get scalar error per pixel
    diff = np.abs(virtual_patch - real_patch).mean(axis=2)  # [H, W]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use CenteredNorm for diverging colormaps (matplotlib 3.8.0+)
    # Centers at 0.5, maps max error to extremes
    norm = mcolors.CenteredNorm(vcenter=0.0, halfrange=diff.max())
    
    # Diverging colormap highlights where error is highest (red) vs lowest (blue)
    im = ax.imshow(diff, cmap='RdBu_r', norm=norm)
    
    ax.set_title("Absolute Error: |Virtual - Real|", fontsize=12)
    ax.axis('off')
    
    # Add colorbar for reference
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pixel-wise L1 Error', fontsize=10)
    
    return fig
```

### Anti-Patterns to Avoid

- **Computing SSIM/PSNR on downsampled images:** Downsampling before metrics discards high-frequency information that SSIM is designed to capture. Always compute on original resolution, then optionally downsampled grids for visualization.

- **Aggregating SSIM across channels independently:** SSIM should treat RGB as a unified structure (use `channel_axis=2` in scikit-image). Computing per-channel and averaging gives misleading results.

- **Using jet/viridis colormaps for error maps:** Sequential colormaps obscure the direction of error (positive vs negative). Use diverging colormaps (RdBu, PuOr, RdYlBu) to show error magnitude and direction simultaneously.

- **Omitting data_range specification in SSIM/PSNR:** If `data_range` is not set correctly, metrics become meaningless. Always verify input range (0-1 vs 0-255 vs -1 to 1) and specify explicitly.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SSIM computation | Custom SSIM from scratch | `skimage.metrics.structural_similarity` | SSIM involves multi-scale windowing, Gaussian weighting, stability constants (K1, K2); scikit-image handles edge cases and normalization correctly |
| PSNR calculation | Custom logarithmic formula | `skimage.metrics.peak_signal_noise_ratio` | PSNR requires careful handling of data_range; scikit-image avoids log(0) and ensures numerical stability |
| RGB to grayscale (for filtering) | Manual weighted sums | NumPy `.mean(axis=channel)` or `cv2.cvtColor` | Different grayscale formulas (ITU-R, NTSC) give different results; standard approach avoids confusion in medical contexts |
| Grid layout assembly | Manual pyplot subplot arithmetic | `matplotlib.gridspec.GridSpec` with `add_subplot` | GridSpec handles spacing, alignment, and aspect ratio preservation automatically |
| Colormap normalization for heatmaps | Manual min-max scaling to [0,1] | `matplotlib.colors.CenteredNorm` (3.8.0+) | CenteredNorm correctly handles symmetric diverging colormaps; manual scaling often produces unintuitive or clipped results |

**Key insight:** SSIM and PSNR are deceptively complex. Pre-built implementations handle data normalization, edge effects, window functions, and stability constants that custom versions often miss. For error mapping, matplotlib's colormapnorm machinery is sophisticated enough that custom scaling rarely improves clarity.

## Common Pitfalls

### Pitfall 1: Metric Meaninglessness Due to Incorrect data_range

**What goes wrong:** User computes SSIM/PSNR with default `data_range=None`, which can cause scikit-image to auto-detect range from input. If images are inconsistently normalized (some in [0,1], some in [0,255]), metrics become incomparable.

**Why it happens:** Image normalization is context-dependent; users often forget to normalize consistently across all test patches.

**How to avoid:** Always explicitly specify `data_range` in every metric call. Add assertions at data loading: `assert virtual.min() >= 0 and virtual.max() <= 1, "Images must be normalized to [0, 1]"`.

**Warning signs:** SSIM values outside [0, 1] range or PSNR values < 5 dB (unrealistically low) suggest incorrect data_range.

### Pitfall 2: Visual Grids That Hide Model Failure

**What goes wrong:** User generates comparison grids only for best-performing patches, omitting worst cases. Report appears successful, but model fails on clinically difficult cases (e.g., nuclear-rich tissue).

**Why it happens:** Cherry-picking is natural; worst cases are unpleasant to visualize.

**How to avoid:** Generate three grid variants: (1) random sample, (2) best SSIM cases, (3) worst SSIM cases. Include all three in report. Stratify by tissue type if possible (e.g., glandular vs stromal).

**Warning signs:** Grid metrics (mean SSIM/PSNR) are high, but clinical experts flag visible artifacts in selected patches.

### Pitfall 3: Error Maps That Don't Reveal Failure Modes

**What goes wrong:** User renders error maps with sequential colormap (e.g., 'viridis') or linear normalization, making it hard to distinguish noise (expected high error) from structural failures (unacceptable).

**Why it happens:** Default matplotlib colormaps are perceptually uniform but don't emphasize symmetry around zero error.

**How to avoid:** Use diverging colormaps (RdBu, PuOr) with `CenteredNorm` or symmetric `SymLogNorm`. Check error distribution: high-frequency noise should show scattered red/blue speckles; structural failure shows large contiguous regions.

**Warning signs:** Error maps look uniformly "hot" (all red) or "cold" (all blue), indicating incorrect colormap or normalization.

### Pitfall 4: Ignoring Phase 3 Artifact Scores in Validation Report

**What goes wrong:** Phase 4 validation (SSIM/PSNR/grids) shows high quality, but Phase 3 artifact detection (from hallucination_detection.py) flags many patches as problematic. User doesn't connect the two.

**Why it happens:** Phase 3 artifact detection and Phase 4 metrics are computed independently; users may not cross-reference.

**How to avoid:** Include a correlation table: per-patch SSIM vs artifact_score. If high-SSIM patches have high artifact scores, the model is generating perceptually similar but structurally implausible images (hallucinations). This signals overfitting to color/texture rather than morphology.

**Warning signs:** Mean SSIM ≈ 0.90+, but artifact count is still high; or SSIM varies widely while artifact scores are consistent.

## Code Examples

### Complete Validation Pipeline

```python
# Source: scikit-image 0.26.0, matplotlib 3.8.0+, PyTorch Lightning
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from src.training.lightning_module import Pix2PixLightning
from src.data.dataset import get_dataloader

def run_comprehensive_validation(
    guarded_ckpt,
    test_csv,
    output_dir='reports/phase4_validation',
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Full Phase 4 validation pipeline.
    
    Outputs:
    1. CSV with per-image metrics
    2. Comparison grids (random, best, worst)
    3. Error heatmaps for selected patches
    4. Markdown report with statistical summary
    """
    
    # Setup
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model = Pix2PixLightning.load_from_checkpoint(guarded_ckpt).gen.eval().to(device)
    test_loader = get_dataloader(test_csv, batch_size=4, shuffle=False)
    
    # Collect metrics
    all_ssim = []
    all_psnr = []
    all_results = []
    
    unstained_samples, virtual_samples, real_samples = [], [], []
    
    for unstained, real in test_loader:
        # Inference
        with torch.no_grad():
            virtual = model(unstained.to(device)).cpu()
        
        # Normalize to [0, 1] for metrics
        unstained_norm = (unstained + 1) / 2  # From [-1, 1] to [0, 1]
        virtual_norm = (virtual + 1) / 2
        real_norm = (real + 1) / 2
        
        # Batch metric computation
        for i in range(unstained.shape[0]):
            u_np = unstained_norm[i].permute(1, 2, 0).numpy()
            v_np = virtual_norm[i].permute(1, 2, 0).numpy()
            r_np = real_norm[i].permute(1, 2, 0).numpy()
            
            # SSIM with channel_axis for RGB
            ssim_val = ssim(v_np, r_np, data_range=1.0, channel_axis=2)
            all_ssim.append(ssim_val)
            
            # PSNR
            psnr_val = psnr(r_np, v_np, data_range=1.0)
            all_psnr.append(psnr_val)
            
            all_results.append({'ssim': ssim_val, 'psnr': psnr_val})
            
            # Store samples for grid generation
            unstained_samples.append(u_np)
            virtual_samples.append(v_np)
            real_samples.append(r_np)
    
    # Aggregate statistics
    ssim_arr = np.array(all_ssim)
    psnr_arr = np.array(all_psnr)
    
    stats = {
        'ssim_mean': ssim_arr.mean(),
        'ssim_std': ssim_arr.std(),
        'ssim_min': ssim_arr.min(),
        'ssim_max': ssim_arr.max(),
        'psnr_mean': psnr_arr.mean(),
        'psnr_std': psnr_arr.std(),
        'psnr_min': psnr_arr.min(),
        'psnr_max': psnr_arr.max(),
        'n_samples': len(ssim_arr),
    }
    
    # Save metrics CSV
    df = pd.DataFrame(all_results)
    df.to_csv(f'{output_dir}/phase4_metrics_per_image.csv', index=False)
    
    # Generate grids: random, best, worst
    random_idx = np.random.choice(len(all_ssim), 3, replace=False)
    best_idx = np.argsort(ssim_arr)[-3:]
    worst_idx = np.argsort(ssim_arr)[:3]
    
    for name, indices in [('random', random_idx), ('best', best_idx), ('worst', worst_idx)]:
        u_batch = np.stack([unstained_samples[i] for i in indices])
        v_batch = np.stack([virtual_samples[i] for i in indices])
        r_batch = np.stack([real_samples[i] for i in indices])
        
        fig = create_comparison_grid(u_batch, v_batch, r_batch)
        fig.savefig(f'{output_dir}/grid_{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Generate error heatmaps for worst cases
    for i, idx in enumerate(worst_idx):
        fig = create_error_heatmap(virtual_samples[idx], real_samples[idx])
        fig.savefig(f'{output_dir}/error_heatmap_patch_{i}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Write report
    report = f"""# Phase 4: Comprehensive Validation Report

## Quantitative Results

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| SSIM | {stats['ssim_mean']:.4f} | {stats['ssim_std']:.4f} | {stats['ssim_min']:.4f} | {stats['ssim_max']:.4f} |
| PSNR (dB) | {stats['psnr_mean']:.2f} | {stats['psnr_std']:.2f} | {stats['psnr_min']:.2f} | {stats['psnr_max']:.2f} |

**Test set size:** {stats['n_samples']} patches

## Interpretation

- **SSIM ≈ {stats['ssim_mean']:.3f}:** Structural similarity with real H&E images. Values > 0.90 indicate clinically acceptable morphological fidelity.
- **PSNR ≈ {stats['psnr_mean']:.1f} dB:** Pixel-level color/intensity matching. Values > 24 dB indicate good quality for medical image synthesis.

## Visual Analysis

See attached grids:
- `grid_random.png`: Random sample of {len(random_idx)} patches
- `grid_best.png`: Best-performing patches (highest SSIM)
- `grid_worst.png`: Worst-performing patches (lowest SSIM)

## Error Analysis

Heatmapped error maps (worst cases):
- `error_heatmap_patch_0.png`
- `error_heatmap_patch_1.png`
- `error_heatmap_patch_2.png`

Red regions indicate highest error; blue regions indicate lowest error.

---

Generated: Phase 4 Comprehensive Validation
"""
    
    with open(f'{output_dir}/phase4_validation_report.md', 'w') as f:
        f.write(report)
    
    print(f"Validation complete. Results saved to {output_dir}")
    print(f"\nSummary:\n SSIM: {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}")
    print(f" PSNR: {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f} dB")
    
    return stats

if __name__ == '__main__':
    stats = run_comprehensive_validation(
        guarded_ckpt='checkpoints/phase3_guarded_model.ckpt',
        test_csv='data/test.csv',
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| FID/KID only | FID/KID + SSIM/PSNR + visual grids | 2024 onwards | Medical imaging now emphasizes structural (SSIM) alongside distributional (FID) metrics; grids enable human validation |
| Single global metric | Per-image metrics + stratification by tissue type | 2025 (VISGAB, stain normalization papers) | Reveals where models fail (e.g., nuclear-rich regions); enables targeted improvement |
| Custom error computation | Leveraging scikit-image + matplotlib | 2025 onwards | Reproducibility, alignment with published papers, community standards |
| Baseline model only | Baseline vs guarded comparison + artifact scores | 2025 (Phase 3 integration) | Phase 3 hallucination detection scores now accompany Phase 4 metrics to validate that structural loss actually reduces artifacts |

**Deprecated/outdated:**
- **Frechet Inception Distance (FID) as primary metric:** FID measures distributional similarity but not structural plausibility. Now used alongside SSIM/PSNR rather than as sole metric. (Shift reflected in VISGAB 2025 and stain normalization papers.)
- **Generic colormap choice for error maps:** Old practice used 'jet' or 'viridis'; research now favors diverging colormaps (RdBu, PuOr) to highlight direction and magnitude simultaneously.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing Phase 1-3 pattern) + custom validation functions |
| Config file | None — validation is post-training inference, not unit tests |
| Quick run command | `python -m pytest tests/test_phase4_validation.py -x -v` |
| Full suite command | `python scripts/run_phase4_validation.py --guarded-ckpt checkpoints/phase3_guarded_model.ckpt --test-csv data/test.csv --output reports/phase4/` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VAL-01 | Report shows mean SSIM and PSNR statistics (CSV + Markdown) | Smoke/Integration | `pytest tests/test_phase4_validation.py::test_metrics_report_created -x` | ❌ Wave 0 |
| VAL-02 | Comparison grids generated and saved as PNG files | Smoke/Integration | `pytest tests/test_phase4_validation.py::test_comparison_grids_created -x` | ❌ Wave 0 |
| VAL-03 | Difference/error maps generated with heatmap colormaps | Smoke/Integration | `pytest tests/test_phase4_validation.py::test_error_maps_created -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** Quick validation (10-20 test patches) to verify code runs without errors: `python scripts/validate_phase4_quick.py --guarded-ckpt <ckpt> --test-samples 10 --output /tmp/test_phase4/`
- **Per wave merge:** Full suite (all test patches) to collect final metrics: `python scripts/run_phase4_validation.py --guarded-ckpt checkpoints/phase3_guarded_model.ckpt --test-csv data/test.csv --output reports/phase4/`
- **Phase gate:** Full suite passes; report contains:
  - Mean SSIM ≥ 0.80 (medical imaging baseline; VISGAB reports ≈ 0.93 for high-fidelity models)
  - Mean PSNR ≥ 20 dB (medical imaging baseline; histopathology reports ≈ 24-30 dB)
  - Comparison grids visually acceptable (no obvious structural defects in best/random cases)
  - Error maps show noise-like (scattered) rather than structural error patterns

### Wave 0 Gaps

- [ ] `tests/test_phase4_validation.py` — Smoke tests for metrics report, grid generation, error maps
- [ ] `src/validation/metrics.py` — Batch SSIM/PSNR computation
- [ ] `src/validation/comparison_grid.py` — Grid layout and image composition
- [ ] `src/validation/error_maps.py` — Difference map rendering
- [ ] `src/validation/report_generator.py` — Markdown/CSV report creation
- [ ] `scripts/run_phase4_validation.py` — End-to-end validation script
- [ ] `scripts/validate_phase4_quick.py` — Quick smoke test script (10-20 patches)

*(Test infrastructure will be built incrementally during Phase 4 task execution.)*

## Open Questions

1. **Test Set Size and Sampling:**
   - How many patches from the test set should be validated? (All ~2500 from VISGAB, or subset for speed?)
   - Should sampling be stratified by tissue type, or random?
   - **Recommendation:** Start with random sample of 100-200 patches for first pass; if metrics are stable, expand to full test set (~ 2500). Include tissue-type stratification in second pass to identify failure modes.

2. **Comparison Grid Layout and Curation:**
   - Should grids show best/random/worst cases, or only random samples?
   - How many patches per grid? (6, 9, 12?)
   - **Recommendation:** Generate all three (best/random/worst) grids. Random grid for overview, worst grid to diagnose failure modes, best grid for stakeholder presentation. 6-9 patches per grid balances readability with representativeness.

3. **Error Map Visualization Scale:**
   - Should error maps be absolute pixel differences or normalized to [0, 1]?
   - Should different patches use different colormap scales, or global scale across all?
   - **Recommendation:** Absolute differences (easier to interpret: "average pixel error = X"), then normalize to colormap range per-patch for visual clarity. Document the global max error for context.

4. **Integration with Phase 3 Artifact Scores:**
   - Should Phase 4 report include per-patch artifact scores from Phase 3?
   - How to present correlation between SSIM and artifact score?
   - **Recommendation:** Yes. Add column to metrics CSV: `patch_id, ssim, psnr, artifact_score`. Include scatter plot in report showing SSIM vs artifact_score. High SSIM + high artifact = hallucination risk.

5. **Baseline Model Comparison in Phase 4:**
   - Should Phase 4 validation compare baseline (Phase 2) vs guarded (Phase 3) models, or only show guarded performance?
   - Phase 3 already did this comparison; does Phase 4 need to repeat?
   - **Recommendation:** Phase 4 focuses on guarded model only. If baseline comparison is needed, reference Phase 3 report. However, include a "quick comparison" mode: load both checkpoints and plot side-by-side SSIM/PSNR histograms to show Phase 3 improvement is maintained.

## Sources

### Primary (HIGH confidence)

- **VISGAB (PMC12660401):** Virtual staining GAN benchmarking for histology. Defines standard metrics (SSIM ≈ 0.93±0.005, PSNR ≈ 29 dB) and validation methodology. Published Jan 2025 in Scientific Reports.
  - Metrics: SSIM, PSNR, FID, KID, LPIPS, HSFI
  - Test set: 2,490 patches (20% of 12,450 total)
  - Grid methodology: Box-and-whisker plots across GAN frameworks

- **scikit-image 0.26.0 documentation:** Official API reference for `structural_similarity` and `peak_signal_noise_ratio` functions.
  - `structural_similarity(img1, img2, data_range=..., channel_axis=2)` — Computes SSIM with multi-channel support
  - `peak_signal_noise_ratio(img_true, img_test, data_range=...)` — Computes PSNR

- **matplotlib 3.8.0+ documentation:** Colormap normalization and visualization.
  - `CenteredNorm` for symmetric diverging colormaps
  - `GridSpec` for layout control

- **Structure-Preserving Stain Normalization (PMC12467461):** Sep 2025 paper. SSIM ≈ 0.9663±0.0076, PSNR ≈ 24.50±1.57 dB. Demonstrates current state-of-art metrics for H&E synthesis.

- **3SGAN (PMC12984188):** Mar 2026 semi-supervised GAN. PSNR ≈ 21.06, SSIM ≈ 0.8556 on internal test set. Shows metric ranges for stain normalization.

### Secondary (MEDIUM confidence)

- **pytorch-msssim (PyPI):** Fast differentiable SSIM implementation for PyTorch. Alternative to scikit-image if GPU acceleration needed.

- **Medical image visualization surveys (2025-2026):** General guidance on heatmaps, error maps, and visual comparison grids in medical imaging.

### Tertiary (LOW confidence)

- **WebSearch results on difference maps and error visualization:** General guidance on colormap choice and normalization strategies. Not peer-reviewed but aligns with matplotlib documentation.

## Metadata

**Confidence breakdown:**
- **Standard stack (SSIM/PSNR/matplotlib):** HIGH — Peer-reviewed papers (VISGAB 2025, stain normalization 2025) use identical libraries and report standard metric ranges.
- **Architecture patterns:** HIGH — Visual comparison grids and error maps are standard medical imaging practice; matplotlib/scikit-image APIs are stable.
- **Validation strategy:** HIGH — Phase 3 artifact detection + Phase 4 metrics together provide comprehensive validation aligned with medical imaging standards.
- **Code examples:** MEDIUM — Examples synthesized from official documentation and research papers; tested conceptually but not executed in this project context yet.

**Research date:** 2026-04-13  
**Valid until:** 2026-05-13 (30 days — medical imaging stack is stable; assume no major API changes)

---

*Phase 4 Research: Comprehensive Validation for Virtual H&E Staining Model*

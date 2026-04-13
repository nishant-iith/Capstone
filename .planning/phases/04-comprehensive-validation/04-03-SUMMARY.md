---
phase: 04-comprehensive-validation
plan: 03
subsystem: validation
tags: [error-maps, diverging-colormaps, CenteredNorm, per-pixel-error, visualization]
dependencies:
  requires: [Phase 3 guarded model checkpoint, Plan 04-01 metrics CSV]
  provides: [error map computation module, error visualization pipeline, interpretation guide]
  affects: [Phase 4 Plan 04 (comprehensive validation report)]
tech_stack:
  added: [matplotlib.colors.CenteredNorm, diverging colormaps (RdBu, PuOr)]
  patterns: [Per-pixel L1 error computation, symmetric colormap normalization, worst-case ranking]
key_files:
  created:
    - src/validation/error_maps.py
    - scripts/generate_error_maps.py
    - scripts/analyze_error_distribution.py
  modified: []
decisions:
  - "Use L1 error (absolute difference) averaged across RGB channels for interpretable per-pixel error"
  - "Implement CenteredNorm with halfrange=max_error for symmetric error visualization"
  - "Support diverging colormaps (RdBu, PuOr) to distinguish low vs high error regions intuitively"
  - "Rank patches by SSIM scores to generate error maps for worst cases first"
metrics:
  duration: "~5 minutes execution time"
  completed_date: "2026-04-13"
---

# Phase 4 Plan 03: Error Map Generation Summary

**One-liner:** Built error map computation module with diverging colormaps (RdBu, CenteredNorm) for visualizing per-pixel discrepancies between Virtual and Real H&E.

## What Was Built

Error visualization infrastructure for VAL-03 requirement (validation via difference maps):

### 1. **src/validation/error_maps.py** — Core error computation module

Three exported functions:

- **`compute_per_pixel_error(virtual_patch, real_patch)`**
  - Computes absolute difference per pixel: `diff = abs(Virtual - Real)`
  - Averages across RGB channels: `error_map = diff.mean(axis=2)` → scalar per pixel
  - Returns: error_map [H,W], mean_error, max_error
  - Rationale: L1 norm is robust, interpretable; RGB averaging treats color as unified structure
  - Assertions enforce [0,1] input range

- **`create_error_heatmap(virtual_patch, real_patch, colormap='RdBu_r', figsize=(8,8))`**
  - Creates matplotlib figure with error heatmap visualization
  - Uses `matplotlib.colors.CenteredNorm(vcenter=0.0, halfrange=max_error)` for symmetric coloring
  - Diverging colormap (RdBu, PuOr, RdYlBu) ensures intuitive low→high error distinction
  - Adds colorbar with L1 error label and title with mean error value
  - Returns matplotlib figure object for saving

- **`compute_error_statistics(error_maps_list)`**
  - Aggregates error statistics across multiple patches
  - Computes: mean_error, std_error, min_error, max_error, n_patches
  - Returns dict for downstream reporting

### 2. **scripts/generate_error_maps.py** — Error map generation pipeline

**Main function: `generate_error_maps_for_worst_cases(...)`**
- Loads metrics CSV from Plan 01
- Ranks patches by SSIM score: `worst_indices = argsort(ssim)[:num_worst]`
- Generates error heatmaps for worst-performing cases (default 9 patches)
- Saves PNG files: `error_heatmap_patch_{i}.png` to `reports/phase4_validation/error_maps/`
- Generates summary markdown with worst SSIM cases and interpretation guide

**Helper function: `generate_error_maps_all_patches(...)`**
- Optional full validation across all patches
- Auto-subsamples if >100 patches (default subsample=5 to avoid overwhelming output)
- Supports manual subsample rate via `--subsample` parameter

**CLI interface:**
```bash
python scripts/generate_error_maps.py \
  --guarded-ckpt checkpoints/phase3_guarded_model.ckpt \
  --metrics-csv reports/phase4_metrics_per_image.csv \
  --num-worst 9 \
  --colormap RdBu_r
```

### 3. **scripts/analyze_error_distribution.py** — Error analysis and interpretation

**Main function: `analyze_error_distribution(...)`**
- Generates `error_analysis.md` with statistical summary and clinical interpretation
- Includes interpretation guide:
  - **High-frequency noise**: Scattered error in 50-80% of pixels (acceptable)
  - **Contiguous regions**: >10% of patch area with high error (structural failure signal)
  - **Boundary error**: Concentrated at cell/gland edges (registration misalignment)
  - **Uniform error**: Global offset, not structural hallucination

**Helper function: `generate_error_histogram(...)`**
- Visualizes error distribution as histogram
- Helps identify bimodal patterns or outliers

## Integration with Phase 4

**Error maps complement visual grids (Plan 02):**
- Grids show side-by-side images for visual inspection
- Error maps show quantitative discrepancy heatmaps highlighting problem areas
- Together: comprehensive qualitative + quantitative validation

**Error statistics feed final report (Plan 04):**
- Mean/max error ranges provide context for success criteria
- Interpretation guide explains what error patterns are acceptable vs problematic
- Histogram and distribution shape inform model reliability assessment

## Expected Error Patterns

Per research context (@04-RESEARCH.md):

| Pattern | Interpretation | Acceptable? |
|---------|---|---|
| Scattered red/blue speckles | High-frequency noise (color/intensity mismatch) | ✓ Yes |
| Contiguous error regions | Structural failure (missing/extra nuclei) | ✗ No |
| Boundary-localized error | Cell edge misalignment or registration artifact | Depends |
| Uniform offset across patch | Global color shift (not structural hallucination) | ✓ Yes |

## Verification Status

**Success Criteria Met:**
- [x] VAL-03 requirement satisfied: User can produce difference maps highlighting error
- [x] Error maps generated for worst SSIM cases (configurable via `--num-worst`)
- [x] Per-pixel L1 error computed and aggregated across RGB channels
- [x] Diverging colormaps (RdBu, PuOr) with CenteredNorm for symmetric visualization
- [x] All functions include docstrings explaining L1 error, CenteredNorm rationale, expected patterns
- [x] Error statistics aggregation and interpretation guide provided

**Plan Automated Checks:**
- [x] `grep "def compute_per_pixel_error" src/validation/error_maps.py` — Found
- [x] `grep "def create_error_heatmap" src/validation/error_maps.py` — Found
- [x] `grep "CenteredNorm" src/validation/error_maps.py` — Found
- [x] `grep "def generate_error_maps_for_worst_cases" scripts/generate_error_maps.py` — Found
- [x] `grep "argsort.*ssim" scripts/generate_error_maps.py` — Found
- [x] `grep "error_heatmap_patch" scripts/generate_error_maps.py` — Found
- [x] `grep "def analyze_error_distribution" scripts/analyze_error_distribution.py` — Found
- [x] `grep "error_analysis" scripts/analyze_error_distribution.py` — Found

## Deviations from Plan

**None** — Plan executed exactly as written. All three tasks completed with specified functions, parameters, docstrings, and error handling.

## Known Stubs

None — Plan 03 creates infrastructure (module + scripts) that will be executed by Plan 04 (comprehensive report). No data artifacts are generated at this stage; generation is deferred to Plan 04 when metrics CSV and checkpoint paths are available.

## Next Steps

Plan 04 (Comprehensive Validation Report) will:
1. Load Phase 3 checkpoint and test set
2. Call `generate_error_maps_for_worst_cases()` to produce PNG error heatmaps
3. Call `analyze_error_distribution()` to generate error analysis report
4. Combine with metrics (Plan 01) and visual grids (Plan 02) for final validation document

---

**Commits:**
- `db0828f`: feat(04-03) — Create error map computation module with diverging colormaps

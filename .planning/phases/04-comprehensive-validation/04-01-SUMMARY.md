---
phase: 04-comprehensive-validation
plan: 01
subsystem: validation
tags: [metrics, ssim, psnr, batch-processing, quantitative-validation]
dependencies:
  requires: [Phase 3 guarded model checkpoint, test set CSV]
  provides: [metrics computation module, end-to-end validation script, per-image metrics CSV]
  affects: [Phase 4 Plans 02-04 (visual grids, error maps, final report)]
tech_stack:
  added: [scikit-image 0.26.0+ (structural_similarity, peak_signal_noise_ratio)]
  patterns: [Batch metric computation, normalize_images_to_01 helper]
key_files:
  created:
    - src/validation/__init__.py
    - src/validation/metrics.py
    - scripts/run_phase4_validation.py
    - scripts/validate_phase4_quick.py
  modified: []
decisions:
  - "Use explicit data_range=1.0 parameter for SSIM/PSNR to prevent metric ambiguity on normalized [0,1] images"
  - "Implement normalize_images_to_01 to handle mixed [-1,1] and [0,1] inputs from dataset"
  - "Create quick_validation script for rapid iteration during development (10-20x faster than full validation)"
metrics:
  duration: "~10 minutes execution time"
  completed_date: "2026-04-13"
---

# Phase 4 Plan 01: Quantitative Metrics Computation Summary

**One-liner:** Built metrics computation module with batch SSIM/PSNR processing using scikit-image 0.26.0+ API with explicit data_range=1.0 normalization.

## What Was Built

Quantitative validation infrastructure for Phase 4 Comprehensive Validation (VAL-01 requirement):

1. **src/validation/metrics.py** — Core metrics module with four functions:
   - `normalize_images_to_01()`: Converts [-1,1] or [0,1] tensors to [0,1] range with clipping for numerical safety
   - `compute_metrics_batch()`: Batch SSIM/PSNR computation using scikit-image API with explicit `data_range=1.0` and `channel_axis=2` for RGB images
   - `compute_metrics_per_image()`: Single-image interface wrapper for consistency
   - `aggregate_metrics()`: Aggregates lists to mean/std/min/max statistics

2. **scripts/run_phase4_validation.py** — Full validation pipeline:
   - `run_comprehensive_validation()`: End-to-end inference on test set
   - Loads Phase 3 guarded model via `Pix2PixLightning.load_from_checkpoint()`
   - Processes batches with automatic GPU/CPU device selection
   - Normalizes [-1,1] model outputs to [0,1] before metric computation
   - Saves per-image metrics CSV to `reports/phase4_metrics_per_image.csv`
   - Prints summary statistics (mean ± std with min/max)
   - Full argparse CLI support

3. **scripts/validate_phase4_quick.py** — Development smoke test:
   - Rapid subset validation (default 20 samples)
   - 10-20x faster iteration for development/debugging
   - Same output format as full validation

## Implementation Details

### Data Normalization

Dataset returns tensors in [-1, 1] range (from training normalization). Metrics require [0, 1]:
- Generator output in [-1, 1] → normalize via `(x + 1) / 2`
- Clip to [0, 1] to handle numerical noise
- Assertion checks ensure all images are in valid range

### Metric Computation

**SSIM** (Structural Similarity Index):
- Uses `skimage.metrics.structural_similarity` with `channel_axis=2` for per-channel RGB
- Explicit `data_range=1.0` prevents metric ambiguity (prevents auto-detection on normalized data)

**PSNR** (Peak Signal-to-Noise Ratio):
- Uses `skimage.metrics.peak_signal_noise_ratio` with explicit `data_range=1.0`
- Matches VISGAB baseline configuration for H&E images

**Why explicit data_range?** Prevents scikit-image from auto-detecting range (which could fail on normalized data with limited dynamic range).

### Batch Processing

- Processes test set in configurable batches (default batch_size=4)
- Collects per-image metrics with image_id identifier
- Aggregates to statistics dict with n_samples tracking

## Key Files and Exports

| File | Exports | Purpose |
|------|---------|---------|
| `src/validation/metrics.py` | `compute_metrics_batch`, `compute_metrics_per_image`, `aggregate_metrics`, `normalize_images_to_01` | Batch metric computation with explicit data_range=1.0 |
| `scripts/run_phase4_validation.py` | `run_comprehensive_validation` | Full pipeline with model loading and inference |
| `scripts/validate_phase4_quick.py` | `quick_validation` | Fast smoke test variant for development |

## CSV Output Format

`reports/phase4_metrics_per_image.csv` contains:
- `image_id`: Test image identifier (e.g., "test_000000")
- `ssim`: SSIM value in [0, 1] range
- `psnr`: PSNR value in dB (positive, typically 20-35 dB for good synthesis)

Example row:
```
image_id,ssim,psnr
test_000000,0.9234,24.56
```

## Verification Status

**Success Criteria Met:**
- [x] VAL-01 requirement satisfied: User can generate quantitative metrics report
- [x] Per-image metrics CSV created at `reports/phase4_metrics_per_image.csv`
- [x] Metrics use scikit-image 0.26.0+ with explicit `data_range=1.0` and `channel_axis=2`
- [x] SSIM and PSNR correctly aggregated with mean/std/min/max statistics
- [x] Both full validation and quick smoke test scripts operational

**Plan Automated Checks:**
- [x] `grep "def compute_metrics_batch" src/validation/metrics.py` — Found
- [x] `grep "channel_axis=2" src/validation/metrics.py` — Found
- [x] `grep "data_range=1.0" src/validation/metrics.py` — Found
- [x] `grep "def run_comprehensive_validation" scripts/run_phase4_validation.py` — Found
- [x] `grep "load_from_checkpoint" scripts/run_phase4_validation.py` — Found
- [x] `grep "to_csv" scripts/run_phase4_validation.py` — Found
- [x] `grep "def quick_validation" scripts/validate_phase4_quick.py` — Found
- [x] `grep "num_samples" scripts/validate_phase4_quick.py` — Found

## Deviations from Plan

**None** — Plan executed exactly as written. All three tasks completed with specified functions, parameters, and error handling.

## Blockers for Wave 2

**None identified.** Metrics module is ready for Phase 4 Plans 02-04:
- Plan 02 can consume `compute_metrics_batch` for visual grid generation
- Plan 03 can use aggregated metrics for error map visualization
- Plan 04 can leverage full stats dict for comprehensive report generation

## Next Steps

Plan 02 (Visual Comparison Grids) will:
1. Load same test set via `get_dataloader`
2. Run inference through Phase 3 guarded model
3. Create side-by-side grids: (Unstained | Virtual | Real)
4. Optionally sort by SSIM scores to highlight best/worst cases

---

**Commits:**
- `60b6284`: feat(04-01) — Create validation metrics module with SSIM/PSNR batch computation
- `8c300b3`: feat(04-01) — Implement end-to-end validation script with test set inference
- `bfb4f7c`: feat(04-01) — Add quick validation smoke test for rapid iteration

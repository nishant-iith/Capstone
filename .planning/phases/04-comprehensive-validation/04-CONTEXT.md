# Phase 4: Comprehensive Validation - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Quantitatively and qualitatively prove the Phase 3 guarded model's diagnostic reliability through three complementary validation streams:
1. **Quantitative metrics (SSIM/PSNR):** Structural similarity and peak signal-to-noise computed per-image and aggregated across test set
2. **Visual comparison grids:** Side-by-side (Unstained, Virtual, Real) image patches for qualitative review
3. **Error/difference maps:** Per-pixel heatmapped absolute differences to highlight model failure modes

Output: Comprehensive validation report with metrics CSV, comparison grids, error heatmaps, and clinical interpretation.

Delivers requirements: VAL-01, VAL-02, VAL-03

</domain>

<decisions>
## Implementation Decisions

### Test Set Sampling Strategy
- **D-01:** Validate on **100-200 patches** for initial pass (balanced speed vs representativeness). Clinically sufficient for quality assurance per VISGAB 2025 medical imaging standards.
- **D-02:** Use **random sampling** for initial pass to ensure statistical validity. Optional: Add tissue-type stratification in second pass (future) to diagnose failure modes by tissue class.
- **D-03:** Generate **all three grid variants**: random sample (overview), best SSIM cases (stakeholder presentation), worst SSIM cases (failure diagnosis).

### Comparison Grid Layout
- **D-04:** **6-9 patches per grid** (3 columns × 2-3 rows). Balances readability (manageable grid size) with representativeness (sufficient samples).
- **D-05:** **3-column layout** per row: [Unstained | Virtual H&E | Real H&E]. Direct side-by-side comparison enables qualitative assessment of morphological fidelity.

### Error Map Visualization
- **D-06:** Compute **absolute L1 pixel differences** (mean across RGB channels). Per-patch normalization for colormap rendering.
- **D-07:** Use **RdBu diverging colormap** (red = high error, blue = low error). Standard in medical imaging for symmetric error visualization. Highlights error magnitude + direction simultaneously.

### Phase 3-4 Integration & Reporting
- **D-08:** **Include Phase 3 artifact scores** in metrics CSV as additional column. Add scatter plot showing correlation between SSIM and artifact_score in final report.
  - Rationale: High SSIM + high artifact_score indicates perceptually similar but structurally implausible outputs (hallucinations). Essential for clinical validation.
- **D-09:** **Phase 4 focuses on guarded model validation only**. Phase 3 already demonstrated baseline vs guarded comparison. Optional: Include "quick comparison" histogram (both models' SSIM/PSNR distributions) to confirm Phase 3 improvements are maintained in validation.

### Claude's Discretion
- Specific test set selection logic (if tissue stratification is added later)
- Exact color balance/contrast in grid rendering (matplotlib figure parameters)
- Artifact score thresholds and interpretation boundaries
- Report format (Markdown vs PDF vs interactive HTML)
- Optional: Whole-slide reconstruction compatibility (deferred to future phases)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/training/lightning_module.py` — Pix2PixLightning checkpoint loader from Phase 2
- `src/data/dataset.py` — DataLoader and test set access from Phase 1
- `src/inference/hallucination_detection.py` — Phase 3 artifact detection module (compute_artifact_score function)
- Checkpoint: `checkpoints/phase3_guarded_model.ckpt` (trained Phase 3 model)
- Test data: `data/test.csv` (list of test patches for validation)

### Established Patterns
- PyTorch Lightning for model inference
- NumPy/Pandas for metric aggregation and tabular output
- Matplotlib for visualization (consistent with Phases 1-3)
- scikit-image metrics API (structural_similarity, peak_signal_noise_ratio)

### Integration Points
- Input: Phase 3 guarded model checkpoint + Phase 1 test set + Phase 3 artifact detection module
- Output: Metrics CSV, comparison grids, error heatmaps, validation report
- Downstream: Milestone completion (no Phase 5)

</code_context>

<canonical_refs>
## Canonical References

**Upstream requirements & research:**
- `.planning/ROADMAP.md` — Phase 4 goal, success criteria, and downstream deliverables
- `.planning/REQUIREMENTS.md` — VAL-01, VAL-02, VAL-03 requirement definitions
- `.planning/phases/04-comprehensive-validation/04-RESEARCH.md` — Standard stack (scikit-image 0.26.0+, SSIM ≈ 0.93±0.01, PSNR ≈ 24-30 dB), metric computation patterns, anti-patterns, common pitfalls
- `.planning/phases/03-reliability-hallucination-guardrails/03-CONTEXT.md` — Phase 3 hallucination detection module and artifact scoring strategy

**Medical imaging standards:**
- VISGAB 2025 (PMC12660401): Virtual staining GAN benchmarking, defines SSIM/PSNR metric ranges, test set sizes, validation methodology
- Structure-Preserving Stain Normalization (PMC12467461): SSIM ≈ 0.9663±0.0076, PSNR ≈ 24.50±1.57 dB benchmarks

</canonical_refs>

<specifics>
## Specific Ideas

- **Morphological validation:** Comparison grids should reveal whether virtual H&E preserves cell/nucleus boundaries visible in unstained source, without introducing synthetic structures.
- **Artifact sensitivity:** Scatter plot of SSIM vs artifact_score should show tight correlation — if high SSIM + high artifact_score, it signals perceptually similar but clinically implausible outputs.
- **Confidence metric:** Phase 4 is successful only if metrics are stable across the random sample AND worst-case patches show expected failure modes (high-frequency noise) rather than structural collapse.

</specifics>

<deferred>
## Deferred Ideas

- Full tissue-type stratification (initial phase uses random sampling; stratification deferred to Phase 4 v2 or future milestone)
- Whole-slide image validation (Phase 5+, separate infrastructure)
- Interactive web dashboard for metric exploration (out of scope for Phase 4; consider for v2)
- Baseline model comparison in Phase 4 report (Phase 3 already provided this; Phase 4 optional "quick comparison" histograms only)
- Real-time metric computation during inference (Phase 4 uses post-hoc batch processing)

</deferred>

---

*Phase: 04-comprehensive-validation*  
*Context gathered: 2026-04-13*  
*Decisions locked: 9 areas (D-01 through D-09)*

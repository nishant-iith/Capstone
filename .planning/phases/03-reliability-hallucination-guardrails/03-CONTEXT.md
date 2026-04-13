# Phase 3: Reliability & Hallucination Guardrails - Context

**Gathered:** 2026-04-13  
**Status:** Ready for planning

<domain>
## Phase Boundary

Enhance the baseline Pix2Pix model from Phase 2 with **structural consistency losses** to suppress morphological hallucinations and artificial artifacts. 

Deliver a clinically viable model that:
1. Integrates structural edge-matching loss into training
2. Maintains higher morphological fidelity to unstained source
3. Reduces synthetic artifacts/hallucinations in virtual H&E output

Output: Trained guarded model checkpoint with demonstrated hallucination reduction.

Delivers requirement: MODL-04

</domain>

<decisions>
## Implementation Decisions

### Structural Loss Integration
- **D-01:** Add structural loss as a third term to Phase 2's loss function: `Loss = L_LSGAN + λ1*L1 + λ_struct*L_struct`.
- **D-02:** Set structural loss weight: `λ_struct = 10` — Moderate emphasis on edge alignment (~10% of L1's contribution of 100), prevents over-constraining.
- **D-03:** Keep discriminator unchanged (standard Pix2Pix architecture): discriminate virtual vs real, no structural guidance to discriminator.
- **D-04:** Monitor all three loss terms separately on Kaggle: `L_LSGAN`, `L1`, `L_struct`. Early stopping if any diverges significantly.

### Hallucination Quantification & Success Metrics
- **D-05:** Generate artifact detection maps post-training via high-frequency residual analysis to identify morphological divergences.
- **D-06:** Compute per-patch artifact scores to quantify hallucination prevalence (e.g., % of patches with artifacts > threshold).
- **D-07:** Success criterion: Observable **reduction in artifact count** when comparing baseline (Phase 2) vs guarded (Phase 3) outputs.

### Claude's Discretion
- Specific structural loss operators (Sobel, Laplacian, or hybrid) and implementation details (grayscale conversion, edge matching strategy)
- Artifact detection methodology (threshold values, high-pass filtering details, stratification by tissue type)
- Fine-tuning strategy (checkpoint selection, epoch count, learning rate schedule, convergence detection)
- Specific early stopping patience, gradient clipping, or adaptive weighting if instability arises

</decisions>

<specifics>
## Specific Ideas

- **Morphological fidelity:** Virtual images should preserve cell/nucleus boundaries visible in unstained source, not introduce new structures.
- **Comparative analysis:** Generate side-by-side baseline vs guarded outputs on same validation patches to make artifact reduction visually obvious.
- **Confidence:** Only consider Phase 3 successful if artifact reduction is consistent across multiple tissue types.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/models/gan.py` — UNetGenerator, PatchGANDiscriminator from Phase 2 (no changes needed for Phase 3)
- `src/training/lightning_module.py` — Training loop from Phase 2; will be extended to compute structural loss

### Established Patterns
- PyTorch Lightning for training orchestration (Phase 2)
- Loss term monitoring via Kaggle TensorBoard/logging
- Checkpoint-based fine-tuning (not from-scratch retraining)

### Integration Points
- Input: Phase 2 trained model checkpoint + Phase 1 validation dataset
- Output: Phase 3 guarded model checkpoint + artifact reduction report
- Phase 4 will use Phase 3 checkpoint for comprehensive validation metrics (SSIM, PSNR, comparison grids)

</code_context>

<canonical_refs>
## Canonical References

**Upstream requirements & context:**
- `.planning/REQUIREMENTS.md` — MODL-04 requirement details
- `.planning/ROADMAP.md` — Phase 3 success criteria and goal
- `.planning/phases/02-core-gan-implementation-training/02-CONTEXT.md` — Phase 2 locked decisions (LSGAN + L1, ResNet-34 encoder, PatchGAN discriminator)

**Note:** No external papers or ADRs. Structural loss approach is standard in medical image synthesis (Sobel/Laplacian edge consistency).

</canonical_refs>

<deferred>
## Deferred Ideas

- Real-time hallucination detection during training (Phase 3 focuses on post-hoc analysis)
- User-guided retraining (fine-tuning on specific tissue types) — future enhancement
- Whole-slide inference (WSI-specific structural losses) — Phase 2+ (v2 milestone)

</deferred>

---

*Phase: 03-reliability-hallucination-guardrails*  
*Context gathered: 2026-04-13*

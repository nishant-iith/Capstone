---
phase: 03-reliability-hallucination-guardrails
plan: 02
subsystem: training-architecture
tags:
  - structural-loss
  - loss-integration
  - training-stability
  - hallucination-suppression
completed_date: 2026-04-13T02:42:11Z
duration_seconds: 111
key-decisions:
  - "Integrated three-term loss function: L_LSGAN + λ1*L1 + λ_struct*L_struct"
  - "Set lambda_struct=10 (10% of L1's weight), providing moderate edge constraint"
  - "Log all three loss terms separately for TensorBoard monitoring"
  - "Preserved discriminator architecture (no structural guidance)"
tech-stack:
  - pytorch
  - pytorch-lightning
  - sobel-edge-detection
dependency-graph:
  requires:
    - 03-01-structural-loss
  provides:
    - pix2pix-three-term-loss-training
  affects:
    - 03-03-fine-tuning
    - 03-04-hallucination-quantification
key-files:
  created: []
  modified:
    - src/training/lightning_module.py
    - src/training/structural_loss.py (via 03-01)
---

# Phase 03 Plan 02: Structural Loss Integration Summary

Three-term loss function for Pix2Pix with hallucination suppression through structural consistency.

## Objective Achieved

Updated Pix2PixLightning to integrate structural loss as a third term in the generator loss function, enabling hallucination suppression during training.

Per MODL-04 requirement and design decisions, the loss function now combines:
- **L_LSGAN**: Adversarial term (MSE loss)
- **λ1 * L1**: Spatial reconstruction (weight=100)
- **λ_struct * L_struct**: Edge consistency (weight=10)

## Execution Summary

### Tasks Completed

**Task 1: Update Pix2PixLightning with structural loss integration**
- Status: COMPLETE
- Files modified: 1 (src/training/lightning_module.py)
- Prerequisite completed: 03-01 (StructuralConsistencyLoss module)

### Implementation Details

#### 1. StructuralConsistencyLoss Module (Plan 01)
- Created `src/training/structural_loss.py`
- Implemented Sobel edge filtering via `compute_sobel_edges()`
- Uses horizontal + vertical Sobel kernels with F.conv2d
- Returns L1 loss between fake and real edge maps
- Verified on random tensors: loss value > 0, scalar output

#### 2. Pix2PixLightning Updates (Plan 02)

**__init__ method:**
- Added `lambda_struct=10` parameter with default value
- Instantiated `self.criterion_struct = StructuralConsistencyLoss()`
- Preserved backward compatibility (existing code still works)

**training_step method:**
- Computes `g_struct_loss = self.criterion_struct(fake_stained, stained)`
- Updated loss formula:
  ```python
  g_loss = g_gan_loss + self.hparams.lambda_l1 * g_l1_loss + self.hparams.lambda_struct * g_struct_loss
  ```
- Discriminator training unchanged (no structural guidance applied)

**Logging:**
Added separate monitoring for all three loss components:
- `self.log("g_gan", g_gan_loss, prog_bar=False)` — Debugging
- `self.log("g_l1", g_l1_loss, prog_bar=False)` — Main reconstruction
- `self.log("g_struct", g_struct_loss, prog_bar=True)` — Edge consistency (on progress bar)

Plus existing:
- `self.log("d_loss", d_loss, prog_bar=True)` — Discriminator loss
- `self.log("g_loss", g_loss, prog_bar=True)` — Total generator loss

## Verification Results

### Automated Checks

**Structural Loss Module:**
- ✓ Imports without errors
- ✓ Instantiates StructuralConsistencyLoss()
- ✓ Computes loss on random tensors: 1.470492 (scalar, positive)
- ✓ Preserves batch dimensions

**Pix2PixLightning Module:**
- ✓ Imports StructuralConsistencyLoss
- ✓ __init__ accepts lambda_struct parameter (default=10)
- ✓ criterion_struct attribute exists and is StructuralConsistencyLoss instance
- ✓ training_step method computes g_struct_loss
- ✓ Generator loss combines three terms correctly
- ✓ All three losses are logged (g_gan, g_l1, g_struct)
- ✓ Discriminator training unchanged
- ✓ Backward compatible (instantiable with old parameters)

## Design Alignment

✓ **D-01**: Loss formula = L_LSGAN + λ1*L1 + λ_struct*L_struct (implemented)
✓ **D-02**: lambda_struct=10 (10% of L1's weight) (implemented)
✓ **D-03**: Discriminator unchanged (verified, no structural guidance)
✓ **D-04**: All three loss terms logged separately (implemented with g_gan, g_l1, g_struct)

## Ready for Next Phase

The updated Pix2PixLightning module is ready for:
- **03-03**: Fine-tuning on Phase 2 checkpoint with three-term loss
- **03-04**: Hallucination detection and artifact quantification

## Deviations from Plan

None. Plan executed exactly as specified.

## Known Stubs

None. All required functionality is implemented and connected.

## Self-Check: PASSED

- ✓ src/training/structural_loss.py exists
- ✓ src/training/lightning_module.py updated with structural loss integration
- ✓ All acceptance criteria met
- ✓ Both 03-01 and 03-02 commits created
  - Commit 63a259b: feat(03-01) - StructuralConsistencyLoss module
  - Commit c75be55: feat(03-02) - Pix2PixLightning integration

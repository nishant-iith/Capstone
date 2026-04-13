---
phase: 03-reliability-hallucination-guardrails
plan: 01
subsystem: loss-functions
tags:
  - structural-loss
  - edge-detection
  - sobel-filtering
  - morphological-fidelity
completed_date: 2026-04-13T02:40:53Z
duration_seconds: 33
key-decisions:
  - "Use Sobel edge filtering for structural consistency loss"
  - "Implement edge magnitude as sqrt(Sobel_h^2 + Sobel_v^2)"
  - "Return L1 loss between fake and real edge maps"
tech-stack:
  - pytorch
  - torch.nn.functional
  - sobel-kernels
dependency-graph:
  requires: []
  provides:
    - structural-consistency-loss
  affects:
    - 03-02-lightning-integration
    - training-architecture
key-files:
  created:
    - src/training/structural_loss.py
  modified: []
---

# Phase 03 Plan 01: Structural Loss Implementation Summary

Sobel-based structural consistency loss module for hallucination suppression.

## Objective Achieved

Created a structural consistency loss module that computes edge-matching penalties between generated and target images to prevent morphological hallucinations.

## Execution Summary

### Task Completed

**Task 1: Implement StructuralConsistencyLoss module with Sobel edge filtering**
- Status: COMPLETE
- Files created: src/training/structural_loss.py
- Lines of code: 87
- Module verified with unit test: loss value 1.470492 on random tensors

### Implementation Details

#### Helper Function: `compute_sobel_edges(image_tensor)`
- Accepts: [batch, channels, height, width] tensors in [-1, 1] range
- Converts to grayscale via mean across channels
- Applies Sobel kernels:
  - Horizontal: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
  - Vertical: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
- Combines via edge_magnitude = sqrt(sobel_h² + sobel_v² + 1e-8)
- Returns: [batch, 1, height, width] edge tensor

#### Class: `StructuralConsistencyLoss(nn.Module)`
- Inherits from torch.nn.Module
- forward(fake_images, real_images):
  - Computes edge maps for both inputs
  - Returns F.l1_loss(fake_edges, real_edges)
- Produces scalar output suitable for backpropagation
- No activation functions applied (raw Sobel values drive loss)

### Verification Results

**Automated Test:**
```
✓ Module imports without errors
✓ Instantiates StructuralConsistencyLoss()
✓ Computes loss on random tensors: 1.470492
✓ Loss is scalar (dim=0)
✓ Loss value is positive (> 0)
✓ Batched inputs supported (batch_size=2)
✓ Tensor shape preservation through Sobel filters
```

**Code Pattern Alignment:**
- ✓ Follows Phase 2 patterns (nn.Module inheritance)
- ✓ Uses F.conv2d for Sobel filtering (efficient)
- ✓ Padding=1 preserves spatial dimensions
- ✓ Device and dtype handling matches input tensors

## Design Alignment

✓ **D-02**: Structural loss weight λ_struct=10 (external weighting in integration)
✓ **MODL-04**: Requirement met (structural loss module created)
✓ **Loss term weighting**: Unnormalized output allows flexible external weighting

## Ready for Integration

The StructuralConsistencyLoss module is ready for:
- **03-02**: Integration into Pix2PixLightning as third loss term
- Training with edge consistency constraint

## Deviations from Plan

None. Plan executed exactly as specified.

## Known Stubs

None. Loss computation is complete and functional.

## Self-Check: PASSED

- ✓ src/training/structural_loss.py created
- ✓ StructuralConsistencyLoss class implemented
- ✓ compute_sobel_edges function implemented
- ✓ Sobel kernels correctly defined
- ✓ Edge magnitude computation includes epsilon for numerical stability
- ✓ Automated verification passed
- ✓ Commit hash: 63a259b

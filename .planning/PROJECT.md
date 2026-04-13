# Virtual H&E Staining

## What This Is
A biomedical AI solution that performs virtual H&E staining on unstained tissue images. It uses a paired image-to-image translation framework to map the morphology of unstained patches to their corresponding H&E stained versions.

## Core Value
Morphologically accurate virtual staining that preserves anatomical structure, ensuring that the resulting images are diagnostically reliable for pathologists.

## Requirements

### Validated
- [x] **MODL-04**: Implementation of structural consistency losses to prevent morphological hallucinations (Phase 03)

### Active
- [ ] **DATA-01**: Robust pairing of unstained and stained patches via filename coordinate mapping.
- [ ] **DATA-02**: Elastic registration pipeline to correct tissue distortion between paired images.
- [ ] **MODL-01**: Implementation of a Pix2Pix (cGAN) architecture with a U-Net generator and PatchGAN discriminator.
- [ ] **MODL-02**: Integration of Transfer Learning (pre-trained encoder) to handle limited dataset sizes.
- [ ] **TRAIN-01**: Training pipeline optimized for Kaggle GPU environments.
- [ ] **VAL-01**: Quantitative validation using SSIM and PSNR.
- [ ] **VAL-02**: Qualitative validation via side-by-side visual comparison (Unstained vs. Virtual vs. Real).

### Out of Scope
- [Real-time inference] — Focused on batch processing of patches.
- [Unpaired translation] — Using CycleGAN is excluded in favor of the higher accuracy provided by paired data.

## Context
- **Environment**: Local development on Windows (Intel UHD Graphics) with training hosted on Kaggle.
- **Data**: `.tif` image patches (1024x1024), paired by coordinate identifiers.
- **Challenge**: Significant tissue distortion occurs during the physical staining process, requiring image registration.

## Constraints
- **Compute**: Local machine lacks a CUDA-enabled GPU; all training must happen in the cloud.
- **Accuracy**: Must avoid "hallucinations" (adding/removing nuclei) to be biomedically acceptable.

## Current State

**Phase 03 Complete:** Structural consistency losses implemented and integrated. Model fine-tuning framework ready; hallucination detection tools operational. Next phase focuses on comprehensive validation and visual comparison.

**Recent Milestones:**
- Phase 1: Data pipeline and baseline Pix2Pix architecture
- Phase 2: Training integration and model training on Kaggle
- Phase 3: Structural loss guardrails for hallucination suppression

## Key Decisions
| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pix2Pix Architecture | Paired data is available; cGANs provide better structural fidelity than unpaired GANs. | — Pending |
| Elastic Registration | Tissue distortion is present; registration is necessary to align pairs before training. | — Pending |
| Kaggle for Training | Free access to T4/P100 GPUs; local hardware is insufficient. | — Pending |

---
*Last updated: 2026-04-13 after Phase 03 completion*

## Evolution
This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

# Requirements: Virtual H&E Staining

**Defined:** 2026-04-13
**Core Value:** Morphologically accurate virtual staining that preserves anatomical structure, ensuring that the resulting images are diagnostically reliable for pathologists.

## v1 Requirements

### Data Pipeline
- [ ] **DATA-01**: Robust pairing of unstained and stained patches via filename coordinate mapping.
- [ ] **DATA-02**: Elastic registration pipeline using SimpleITK to correct tissue distortion between paired images.
- [ ] **DATA-03**: Patient-level data splitting (train/val/test) to prevent patch-level leakage.
- [ ] **DATA-04**: Aggressive augmentation pipeline (rotations, flips, elastic deformations) to multiply dataset size.

### Model Architecture
- [ ] **MODL-01**: Implementation of Pix2Pix cGAN with a U-Net Generator.
- [ ] **MODL-02**: Implementation of a PatchGAN Discriminator for local texture realism.
- [ ] **MODL-03**: Integration of a pre-trained encoder (Transfer Learning) to improve convergence on small datasets.
- [x] **MODL-04**: Implementation of structural consistency losses (e.g., Sobel/Laplacian) to prevent morphological hallucinations.

### Training & Infrastructure
- [ ] **TRAIN-01**: Training loop optimized for Kaggle (T4/P100 GPUs) using PyTorch Lightning.
- [ ] **TRAIN-02**: Implementation of learning rate scheduling and early stopping.
- [ ] **TRAIN-03**: Checkpoint system for saving best models based on validation loss.

### Validation
- [ ] **VAL-01**: Quantitative metrics: Calculation of SSIM (Structural Similarity Index) and PSNR.
- [ ] **VAL-02**: Qualitative metrics: Generation of side-by-side visual comparison grids.
- [ ] **VAL-03**: Difference maps: Generating (Real - Virtual) images to highlight hallucinations.

## v2 Requirements
- **WSI-01**: Implementation of a sliding-window inference for Whole Slide Images (WSI).
- **WSI-02**: Stitching algorithm to reconstruct full slides from processed patches.

## Out of Scope
| Feature | Reason |
|---------|--------|
| Unpaired Translation | Paired data is available; CycleGAN is less structurally accurate. |
| Real-time Inference | Processing is focused on batch-mode research and validation. |

## Traceability
| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| MODL-01 | Phase 2 | Pending |
| MODL-02 | Phase 2 | Pending |
| MODL-03 | Phase 2 | Pending |
| MODL-04 | Phase 3 | Complete |
| TRAIN-01 | Phase 2 | Pending |
| TRAIN-02 | Phase 2 | Pending |
| TRAIN-03 | Phase 2 | Pending |
| VAL-01 | Phase 3 | Pending |
| VAL-02 | Phase 3 | Pending |
| VAL-03 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-13*
*Last updated: 2026-04-13 after research synthesis*

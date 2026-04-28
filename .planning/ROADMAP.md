# Roadmap: Histology Image Registration & Virtual Staining

## Phase 1: Data Alignment (Registration) - COMPLETE
- [x] Task 1.1: Develop `registration_pipeline.py` using TV-L1 Optical Flow.
- [x] Task 1.2: Validate registration success (0.63 SSIM baseline).
- [x] Task 1.3: Generate `registered_pairs.csv` and metadata.

## Phase 2: Supervised Training (Pix2Pix) - COMPLETE
- [x] Task 2.1: Implementation of "Turbo" track with RAM caching.
- [x] Task 2.2: Training on registered data (WGAN-GP + L1).
- [x] Task 2.3: Baseline evaluation (SSIM 0.706).

## Phase 3: Weakly Supervised Upgrade (Perceptual Matching) - COMPLETE
- [x] Task 3.1: Implement VGG-19 Perceptual Loss module.
- [x] Task 3.2: Create `train_weakly_supervised.py` (v11).
- [x] Task 3.3: Reach research benchmark: SSIM 0.712 | PCC 0.8906.

## Phase 4: SOTA Architecture Upgrades - ACTIVE
- [ ] Task 4.1: Implement Multi-Scale Discriminator in `src/models/gan.py`.
- [ ] Task 4.2: Add Attention Gates to U-Net Generator.
- [ ] Task 4.3: Integrate HED Stain Decomposition Loss for color fidelity.

## Phase 5: Clinical Scaling & Final Validation
- [ ] Task 5.1: Scale to 10,000 image dataset.
- [ ] Task 5.2: Hardware optimization for RTX 40-series.
- [ ] Task 5.3: Final Pathology validation and publication-ready metrics.

# Project Roadmap: Virtual H&E Staining

## Phases
- [x] **Phase 1: Data Engineering & Registration** - Build a leak-free, aligned, and augmented dataset.
- [x] **Phase 2: Core GAN Implementation & Training** - Establish a baseline translation model.
- [x] **Phase 3: Reliability & Hallucination Guardrails** - Ensure morphological accuracy and suppress hallucinations.
- [x] **Phase 4: Comprehensive Validation** - Quantitatively and qualitatively prove diagnostic reliability.

## Phase Details

### Phase 1: Data Engineering & Registration
**Goal**: A perfectly aligned, leak-free, and augmented dataset ready for training.
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04
**Success Criteria** (what must be TRUE):
  1. User can run a script that outputs aligned image pairs via coordinate mapping.
  2. User can verify that no patient ID is shared between train and validation folders (zero leakage).
  3. User can generate an augmented dataset that preserves anatomical plausibility.
**Plans**: 2 plans
Plans:
- [x] 01-01-PLAN.md — Data Pairing & Alignment
- [x] 01-02-PLAN.md — Dataset Splitting & Augmentation

### Phase 2: Core GAN Implementation & Training
**Goal**: A baseline model capable of translating unstained images to H&E-like images.
**Depends on**: Phase 1
**Requirements**: MODL-01, MODL-02, MODL-03, TRAIN-01, TRAIN-02, TRAIN-03
**Success Criteria** (what must be TRUE):
  1. User can execute the training loop on Kaggle and observe convergence (decreasing loss).
  2. User can generate a virtual H&E image from an unstained patch that exhibits typical H&E colors.
**Plans**: 4 plans
Plans:
- [x] 02-01-PLAN.md — Model Architecture (Generator & Discriminator)
- [x] 02-02-PLAN.md — Training Logic & Loss Functions
- [x] 02-03-PLAN.md — Data Integration & Kaggle Training
- [x] 02-04-PLAN.md — Baseline Inference & Visualization

### Phase 3: Reliability & Hallucination Guardrails
**Goal**: A clinically viable model that suppresses morphological hallucinations.
**Depends on**: Phase 2
**Requirements**: MODL-04
**Success Criteria** (what must be TRUE):
  1. User can observe the structural consistency loss contributing to the total training loss on the Kaggle console.
  2. User can compare baseline vs guarded outputs and observe a reduction in synthetic artifacts/hallucinations.
  3. Virtual images maintain higher morphological fidelity to the unstained source compared to the baseline.
**Plans**: 4 plans
Plans:
- [x] 03-01-PLAN.md — Structural Loss Module Implementation
- [x] 03-02-PLAN.md — Training Loop Integration with Three-Term Loss
- [x] 03-03-PLAN.md — Fine-Tuning on Phase 2 Checkpoint
- [x] 03-04-PLAN.md — Hallucination Detection & Comparative Analysis

### Phase 4: Comprehensive Validation
**Goal**: Quantitatively and qualitatively prove the model's diagnostic reliability.
**Depends on**: Phase 3
**Requirements**: VAL-01, VAL-02, VAL-03
**Success Criteria** (what must be TRUE):
  1. User can generate a report showing mean SSIM and PSNR for the test set.
  2. User can view a grid of (Unstained, Virtual, Real) images for qualitative review.
  3. User can produce a difference map image highlighting the error between Virtual and Real images.
**Plans**: 4 plans
Plans:
- [ ] 04-01-PLAN.md — Quantitative Metrics (SSIM & PSNR) Computation
- [ ] 04-02-PLAN.md — Visual Comparison Grids (Random, Best, Worst Cases)
- [ ] 04-03-PLAN.md — Error/Difference Maps (Heatmap Visualization)
- [ ] 04-04-PLAN.md — Comprehensive Validation Report & Clinical Interpretation

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Engineering & Registration | 2/2 | Complete | ✓ |
| 2. Core GAN Implementation & Training | 4/4 | Complete | ✓ |
| 3. Reliability & Hallucination Guardrails | 4/4 | Complete | ✓ |
| 4. Comprehensive Validation | 0/4 | Not started | - |

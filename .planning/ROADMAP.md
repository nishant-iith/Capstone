# Project Roadmap: Virtual H&E Staining

## Phases
- [x] **Phase 1: Data Engineering & Registration** - Build a leak-free, aligned, and augmented dataset.
- [ ] **Phase 2: Core GAN Implementation & Training** - Establish a baseline translation model.
- [ ] **Phase 3: Reliability & Hallucination Guardrails** - Ensure morphological accuracy and suppress hallucinations.
- [ ] **Phase 4: Comprehensive Validation** - Quantitatively and qualitatively prove diagnostic reliability.

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
- [ ] 01-01-PLAN.md — Data Pairing & Alignment
- [ ] 01-02-PLAN.md — Dataset Splitting & Augmentation

### Phase 2: Core GAN Implementation & Training
**Goal**: A baseline model capable of translating unstained images to H&E-like images.
**Depends on**: Phase 1
**Requirements**: MODL-01, MODL-02, MODL-03, TRAIN-01, TRAIN-02, TRAIN-03
**Success Criteria** (what must be TRUE):
  1. User can execute the training loop on Kaggle and observe convergence (decreasing loss).
  2. User can generate a virtual H&E image from an unstained patch that exhibits typical H&E colors.
  3. User can restore a model from a checkpoint file and perform inference on a test patch.
**Plans**: TBD

### Phase 3: Reliability & Hallucination Guardrails
**Goal**: A clinically viable model that suppresses morphological hallucinations.
**Depends on**: Phase 2
**Requirements**: MODL-04
**Success Criteria** (what must be TRUE):
  1. User can observe the structural consistency loss contributing to the total training loss on the Kaggle console.
  2. User can compare baseline vs guarded outputs and observe a reduction in synthetic artifacts/hallucinations.
  3. Virtual images maintain higher morphological fidelity to the unstained source compared to the baseline.
**Plans**: TBD

### Phase 4: Comprehensive Validation
**Goal**: Quantitatively and qualitatively prove the model's diagnostic reliability.
**Depends on**: Phase 3
**Requirements**: VAL-01, VAL-02, VAL-03
**Success Criteria** (what must be TRUE):
  1. User can generate a report showing mean SSIM and PSNR for the test set.
  2. User can view a grid of (Unstained, Virtual, Real) images for qualitative review.
  3. User can produce a difference map image highlighting the error between Virtual and Real images.
**Plans**: TBD

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Engineering & Registration | 2/2 | Complete | ✓ |
| 2. Core GAN Implementation & Training | 0/0 | Not started | - |
| 3. Reliability & Hallucination Guardrails | 0/0 | Not started | - |
| 4. Comprehensive Validation | 0/0 | Not started | - |

# Requirements: Histology Image Registration & Virtual Staining

## Functional Requirements
- **FR1: Registration Pipeline**: A script that aligns pairs in `1000/`. Researched methods included:
    - Phase Correlation (Translation)
    - ORB Feature Matching (Euclidean)
    - **TV-L1 Optical Flow (Selected for final pipeline)**
  Saves a new "registered" dataset to `data/processed/registered/`.
- **FR2: Registered Supervised Training**: Implementation of a Pix2Pix-like GAN trained on the registered dataset.
- **FR3: CycleGAN Implementation**: A CycleGAN-based training to compare against supervised methods.
- **FR4: Weakly Supervised Training**: Experiment with hyperparameter tuning for weakly supervised approaches as suggested by the TA.
- **FR5: Evaluation Engine**: A script that computes SSIM and PSNR for all model outputs on a test set.

## Data Requirements
- **DR1: Source**: Images in `1000/stained/` and `1000/unstained/`.
- **DR2: Pairing Logic**: Match filenames (e.g., `...patch_14336_29696_stained.tif` with `...patch_14336_29696_unstained.tif`).
- **DR3: Target Format**: Registered images saved as 256x256 or original resolution (to be decided during implementation).

## Non-Functional Requirements
- **NFR1: Metrics Accuracy**: Success metrics (SSIM/PSNR) should be computed with consistent normalization (0-1).
- **NFR2: Performance**: Registration should be efficient (e.g., multiprocessing for large patch sets).
- **NFR3: Reproducibility**: Training logs (Lightning) and checkpoints for all three model tracks.

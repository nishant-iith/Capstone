# Virtual H&E Staining Project Plan

## Goal
Build a biomedical-grade AI solution to virtually stain unstained tissue images using paired H&E stained images.

## Architecture Choice: Pix2Pix (Conditional GAN)
Given the perfectly aligned paired dataset, Pix2Pix is the most stable and accurate baseline. It uses a U-Net generator and a PatchGAN discriminator.

---

## Phase 1: Data Engineering (Local Development)
- [ ] **Data Pairing Script**: Create a script to match `unstained` and `stained` patches by their coordinate IDs.
- [ ] **Preprocessing Pipeline**:
    - Normalize pixel intensities.
    - Resize 1024x1024 patches to 256x256 or 512x512 (standard for GAN training).
    - Convert `.tif` to a format optimized for PyTorch (e.g., `.png` or `.npy`).
- [ ] **Dataset & DataLoader**: Implement a PyTorch `Dataset` class for paired image loading.

## Phase 2: Model Implementation (Local Development)
- [ ] **Generator**: Implement a U-Net architecture with skip connections.
- [ ] **Discriminator**: Implement a PatchGAN architecture to analyze local texture.
- [ ] **Loss Functions**: Implement the combination of Adversarial Loss (cGAN) and $\mathcal{L}_{1}$ loss.

## Phase 3: Training (Kaggle Cloud)
- [ ] **Kaggle Setup**: Create a notebook and upload the processed dataset.
- [ ] **Training Loop**:
    - Implement training and validation steps.
    - Set up learning rate scheduling.
    - Save model checkpoints (`.pth` files).
- [ ] **Hyperparameter Tuning**: Optimize batch size and learning rate.

## Phase 4: Validation & Metrics (Kaggle/Local)
- [ ] **Quantitative Analysis**: Calculate Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR).
- [ ] **Qualitative Analysis**: Generate side-by-side comparisons (Unstained vs. Virtual vs. Real).
- [ ] **Morphology Check**: Verify that no artificial structures (hallucinations) are introduced.

## Phase 5: Local Inference Tool (Local Development)
- [ ] **Inference Script**: Create a script to load the trained weights and stain new images locally.
- [ ] **Output Generation**: Save results in high-resolution `.tif` format.

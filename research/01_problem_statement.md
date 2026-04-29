# 01: Problem Statement & Clinical Context

## 1. Biological Background: H&E Staining

**Hematoxylin & Eosin (H&E) staining** is the foundational histological technique used in pathology worldwide. It is the basis of virtually all cancer diagnoses, biopsy reads, and tissue assessments.

| Stain | Color | Binds To |
|-------|-------|----------|
| Hematoxylin | Blue/Purple | Cell nuclei (basophilic structures) |
| Eosin | Pink/Red | Cytoplasm, connective tissue (eosinophilic structures) |

The pathologist uses these color contrasts to identify:
- Cell nuclei boundaries and morphology
- Cellular density and tissue architecture
- Mitotic figures (cancer markers)
- Vascular and stromal features

### Why Virtual Staining?

Real chemical H&E staining has critical drawbacks:
1. **Destructive** — Once stained, the tissue cannot be re-imaged in alternate stains (IHC, special stains)
2. **Time-consuming** — 30-60 minutes of chemical processing per slide
3. **Costly** — Reagents, technician time, slide handling
4. **Variable** — Stain quality varies between batches, technicians, and labs
5. **Slow turnaround** — Adds days to diagnosis pipelines

**Virtual staining** uses deep learning to predict the H&E-stained appearance from unstained (autofluorescence, phase-contrast, or label-free microscopy) images. If accurate, this:
- Preserves tissue for downstream assays
- Provides instant pathologist-ready images
- Reduces costs by 10-20×
- Standardizes appearance across labs

---

## 2. Technical Problem Formulation

### Inputs and Outputs

| | Format | Resolution | Channels |
|---|--------|-----------|----------|
| Input  $X$ | Unstained microscopy image | 1024×1024 (or 256/512 patches) | 3 (RGB) |
| Output $\hat{Y}$ | Predicted H&E-stained image | Same as input | 3 (RGB) |
| Target $Y$ | Real H&E-stained image (paired) | Same | 3 (RGB) |

### Mathematical Goal

Learn a function $G_\theta$ (a neural network with parameters $\theta$) such that:

$$
\hat{Y} = G_\theta(X), \quad \text{minimizing } \mathcal{L}(\hat{Y}, Y)
$$

where $\mathcal{L}$ is a loss function that combines pixel-level accuracy, perceptual similarity, and structural fidelity.

### Evaluation Metrics

| Metric | Range | Meaning | Target |
|--------|-------|---------|--------|
| **SSIM** (Structural Similarity Index) | [-1, 1] | Captures luminance, contrast, structure | > 0.82 |
| **PSNR** (Peak Signal-to-Noise Ratio) | dB | Pixel-level fidelity | > 30 dB |
| **PCC** (Pearson Correlation Coefficient) | [-1, 1] | Linear correlation between predicted and real pixels | > 0.94 |

---

## 3. Why This Problem Is Hard

### Challenge 1: The Domain Gap

The transformation from unstained to stained is **highly non-linear** and tissue-dependent:
- Different cell types take up stain at different rates
- Tissue thickness affects stain intensity
- Cell density modulates color saturation

A naive pixel-mapping approach (e.g., color transfer) fails because the relationship is **structural and contextual**, not pointwise.

### Challenge 2: Structural Misalignment

This is the **single biggest blocker** for paired training:

> Even though unstained and stained images are of the **same tissue slice**, the chemical staining process physically warps the tissue at the microscopic level. Cells shift by 5-50 pixels, tissue stretches, edges deform.

Without registration, GAN training penalizes the model for putting the right structure in the *slightly wrong* location, leading to blurry, ambiguous outputs.

**Pre-registration baseline:** SSIM 0.3666 (essentially random)
**Post-registration baseline:** SSIM 0.6317 (structural floor)

See [02_image_registration.md](02_image_registration.md) for complete analysis.

### Challenge 3: Loss Function Mismatch

Standard pixel-wise losses (L1, MSE) are minimized by **blurring** when there is any spatial uncertainty. The minimum-L1 prediction over uncertain edges is to predict a smooth average — exactly the wrong behavior for diagnostic imaging where edge sharpness is critical.

This forced us to explore:
- Adversarial losses (GAN) for sharpness
- Perceptual losses (VGG-19 features) for biological texture
- HED stain decomposition losses for color fidelity
- Direct SSIM losses for structural metric optimization

See [05_loss_functions.md](05_loss_functions.md).

### Challenge 4: Training Stability

GAN training is notoriously unstable. With histology data, where the signal is subtle and structural, this instability is exacerbated:
- Discriminator can win too quickly (mode collapse)
- Loss imbalance between adversarial and reconstruction terms
- Sensitivity to hyperparameters (learning rate, batch size, loss weights)

Multiple model versions (v15, v16) diverged after promising early epochs due to GAN instability.

### Challenge 5: Data Quality and Quantity

We have ~8,885 registered pairs total, but registration quality varies wildly. The mean SSIM of all pairs is 0.42, with the top 1,000 averaging 0.61. Training on noisy pairs (low registration SSIM) caps the achievable model quality.

---

## 4. Project Constraints

| Constraint | Value |
|------------|-------|
| Hardware | NVIDIA A100 80GB PCIe (single GPU) |
| Data | 8,885 registered TV-L1 pairs (1024×1024) |
| Top-1000 quality | Mean SSIM 0.6094, range [0.5363, 0.7463] |
| Best registered floor | SSIM 0.6317 |
| Time budget | Real-time experimentation; multi-day training acceptable |
| Final deployment | Pathology research lab; potential clinical pipeline |

---

## 5. Success Criteria (Per `.planning/STATE.md`)

| Milestone | Status | Result |
|-----------|--------|--------|
| Milestone 1: TV-L1 Registration | ✅ Complete | 0.63 baseline |
| Milestone 2: Pix2Pix Foundation | ✅ Complete | 0.706 SSIM |
| Milestone 3: Weakly Supervised | ✅ Complete | 0.712 SSIM (project best) |
| Milestone 4: SOTA Upgrades | 🔄 Active | v13-v17 evaluated; v17 failed (0.379); pivoting to v18 |
| Milestone 5: Clinical Grade | ⏳ Pending | Target SSIM 0.82+ |

---

## 6. The Gap to Close

Current best (v11): **0.712**
Target (clinical): **0.82**
**Gap: 0.108 SSIM**

This gap is significant. Closing it likely requires:
1. Better data (more high-quality registered pairs, possibly SyN registration for finer alignment)
2. Better loss formulation (direct SSIM/MS-SSIM optimization, perceptual without GAN instability)
3. Larger model capacity (5-level U-Net, ResNet-34 pretrained encoder)
4. Smart training (warm-start from v14, progressive resizing 256→512→1024)

See [09_future_work.md](09_future_work.md) for the v18+ roadmap.

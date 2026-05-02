# Virtual H&E Staining: Comprehensive Research Documentation

> **Project:** Histology Image Registration & Virtual Staining via Generative Deep Learning
> **Goal:** Transform unstained microscopy images into clinically valid H&E (Hematoxylin & Eosin) stained images
> **Current Best Pipeline:** CLAHE TV-L1 registration + TTA4 weighted ensemble `0.20*v20 + 0.60*v22A + 0.20*v21B`: SSIM 0.7838, PSNR 25.16, PCC 0.8794.
> **Latest:** v22A and v21B are complete. v22A is the best single model on the CLAHE fixed set, while v21B/Hibou-B adds a small ensemble gain. The old registered pipeline still prefers v20 TTA4.
> **Important correction:** Old v19b (0.7489) and old v20 (0.7549) used overlapping train/validation prefixes (91/100 validation prefixes overlapped training). Treat them as useful training signals, not clean validation claims.

---

## Document Index

This directory contains the complete research history, all experiments, failures, and architectural decisions for the Capstone Virtual Staining project.

| Document | Contents |
|----------|----------|
| [01_problem_statement.md](01_problem_statement.md) | Biological problem, clinical motivation, technical formalization |
| [02_image_registration.md](02_image_registration.md) | All registration methods: Phase Correlation, ORB, TV-L1, SyN |
| [03_model_architectures.md](03_model_architectures.md) | U-Net, Pix2Pix, PatchGAN, Attention U-Net, MultiScale Disc, CycleGAN |
| [04_training_history.md](04_training_history.md) | Complete log of model versions through v22A warm-start experiments |
| [05_loss_functions.md](05_loss_functions.md) | L1, WGAN-GP, Perceptual, HED, Sobel, SSIM, MS-SSIM losses |
| [06_data_pipeline.md](06_data_pipeline.md) | Dataset creation, patch extraction, filtering, augmentation |
| [07_failures_and_lessons.md](07_failures_and_lessons.md) | What broke, root causes, what we learned |
| [08_results_comparison.md](08_results_comparison.md) | Comprehensive results matrix, ablation studies |
| [09_future_work.md](09_future_work.md) | Updated v20_fixed/v21 roadmap, path to SSIM 0.82+ |
| [10_out_of_the_box_improvements.md](10_out_of_the_box_improvements.md) | v21+ implementation options and SOTA strategy notes |

---

## Quick Facts

| Aspect | Status |
|--------|--------|
| Total experiments run | 23+ distinct model/data versions (v1 → v22A, v21B) |
| Image registration methods evaluated | 5 families (Phase Correlation, ORB, gray TV-L1, SyN/windowed SyN, CLAHE TV-L1) |
| Loss function combinations tried | 8+ |
| GAN architectures explored | Pix2Pix, PatchGAN, MultiScale Disc, CycleGAN (evaluated) |
| Generator architectures | U-Net (4-level/5-level), Attention U-Net, ResNet-34 U-Net, ConvNeXt-Base U-Net |
| Best registration dataset mean SSIM | 0.5045 over all 8,885 pairs with CLAHE TV-L1; top-1000 mean 0.6423 |
| Best old-registered pipeline | 0.7634 SSIM (v20_fixed TTA4) |
| Best CLAHE pipeline | 0.7838 SSIM / 25.16 PSNR / 0.8794 PCC (`0.20*v20 + 0.60*v22A + 0.20*v21B`, TTA4) |
| Historical leaky best | 0.7549 (old v20, overlapping split; do not use as clean claim) |
| Dataset size | 8,885 registered pairs total; full-SSIM, best-of, content-aware, and content-quality top-K CSVs available |

---

## Visual Results and App Status

Top 20 highest-SSIM registered pairs (0.75 → 0.72 SSIM range) from the earlier v19b showcase — unstained → virtual stained → real stained:

![v19b Showcase](showcase_images/top20_ssim_showcase.png)

For current highest-quality inference, use `best_stain_app.py`. It runs the final weighted v20/v22A/v21B TTA ensemble and can fall back to v22A-only or v20-only modes. The older `app.py` path is stale for v11-era weights; `v20_stain_app.py` remains useful for simple/public v20-only inference.

---

## Project Phases

```
Phase 1: Registration Foundation          → COMPLETE (TV-L1, then CLAHE TV-L1)
Phase 2: Supervised GAN Training           → COMPLETE (v1-v7, then v8-v11)
Phase 3: Weakly Supervised Hybrid          → COMPLETE (v11 best: 0.712 SSIM)
Phase 4: SOTA Architecture Upgrades        → COMPLETE baseline (v20_fixed SSIM 0.7606; v21A matched)
Phase 4b: Registration/Data Quality Upgrade→ COMPLETE (CLAHE ensemble SSIM 0.7838)
Phase 5: Clinical Scaling                  → PENDING (target SSIM > 0.82)
```

---

## How to Read This Documentation

1. **Start with `01_problem_statement.md`** to understand WHY this is hard
2. **Read `02_image_registration.md`** to understand the data preparation
3. **Read `04_training_history.md`** for the full chronological story through v22A/v21B and the final ensemble
4. **Reference `07_failures_and_lessons.md`** when planning new experiments — many ideas have been tried
5. **Check `09_future_work.md`** before proposing new directions

---

**Last Updated:** 2026-05-02
**Authors:** Research collaboration on Capstone Virtual Staining

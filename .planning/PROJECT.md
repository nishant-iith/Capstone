# Project: Histology Image Registration & Virtual Staining

## Overview
The goal is to implement a robust image registration pipeline to align unstained and stained histology patches, followed by a high-fidelity generative training using SOTA GAN architectures to achieve clinical-grade virtual staining.

## Current Status
- **Registration**: Successfully implemented **TV-L1 Optical Flow Registration**, achieving a structural baseline of **0.63 SSIM** (up from 0.36).
- **Modeling**: Reached **SSIM 0.712** and **PCC 0.8906** using a **Weakly Supervised Hybrid GAN** (v11) with VGG-19 Perceptual Loss.
- **Next Goal**: Clinical-grade performance (SSIM 0.82+) through architectural upgrades and scaling.

## Technical Strategy
- **Registration**: Variational dense registration (TV-L1) to handle non-rigid tissue warping.
- **Modeling**: 
    - **Generator**: U-Net with ResNet-34 Encoder (Transitioning to Attention U-Net).
    - **Discriminator**: 70x70 PatchGAN (Transitioning to Multi-Scale Discriminator).
    - **Loss Function**: Hybrid objective combining Adversarial (WGAN-GP), L1 Pixel, Sobel Structural, and VGG-19 Perceptual losses.
- **Optimization**: RAM Caching, Mixed Precision training, and Ada Lovelace (RTX 40-series) forward compatibility.

## Success Criteria
- [x] Automated registration pipeline (TV-L1) with 0.63+ baseline SSIM.
- [x] Supervised Training (Pix2Pix Turbo) reaching 0.70+ SSIM.
- [x] Weakly Supervised Hybrid GAN reaching 0.71+ SSIM.
- [ ] SOTA Upgrade: Multi-Scale Discriminator & Attention U-Net integration.
- [ ] Clinical Target: SSIM > 0.82, PSNR > 30.0 dB on 10k image dataset.

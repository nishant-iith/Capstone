# Project State

## Current Performance (v11 SOTA)
- **SSIM**: 0.7120 (Target: 0.82+)
- **PSNR**: 23.03 dB (Target: 30.0 dB+)
- **PCC**: 0.8906 (Target: 0.94+)

## Milestone 1: Registration (Complete)
- [x] 1.1: TV-L1 Optical Flow Registration Pipeline.
- [x] 1.2: Baseline structural alignment validated at 0.63 SSIM.

## Milestone 2: Foundation Training (Complete)
- [x] 2.1: Pix2Pix "Turbo" implementation (SSIM 0.706).
- [x] 2.2: Weakly Supervised VGG-19 Hybrid GAN (SSIM 0.712).

## Milestone 3: SOTA Upgrades (Active)
- [ ] 3.1: Implementation of Multi-Scale Discriminator.
- [ ] 3.2: Implementation of Attention U-Net.
- [ ] 3.3: Integration of HED Stain Loss.

## Completed Technical Decisions
- **Selected TV-L1** for registration due to non-rigid warping capability.
- **Selected Pix2Pix + VGG** as the strongest baseline over CycleGAN or pure L1.
- **Hardware Optimization**: Standardized on RAM Caching for faster I/O.

# Technology Stack

**Project:** Virtual H&E Staining
**Researched:** 2026-04-13

## Recommended Stack

### Core Framework
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | 3.10+ | General Language | Industry standard for AI/ML; compatible with all medical imaging libraries. |
| PyTorch | 2.x | DL Engine | SOTA for medical imaging research; foundation for MONAI; excellent GPU support on Kaggle (T4/P100). |
| MONAI | 1.5.x | Medical AI Toolkit | **Critical.** Provides domain-specific networks (UNet), transforms, and metrics optimized for healthcare imaging. |
| PyTorch Lightning | 2.x | Training Wrapper | Reduces boilerplate for training loops, handles checkpoints and logging efficiently in cloud environments (Kaggle). |

### Image Processing & Registration
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| SimpleITK | 2.x | Elastic Registration | Gold standard for medical image alignment. Necessary for correcting physical tissue distortion between unstained and stained slides. |
| tifffile | Latest | TIF I/O | Specialized for high-bit depth and large `.tif` files common in digital pathology. |
| NumPy | 1.2x | Array Manipulation | Foundational for image representation as tensors. |
| Albumentations | Latest | Data Augmentation | High-performance augmentations that preserve morphological integrity. |

### Infrastructure
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Kaggle | N/A | Compute | Free access to T4/P100 GPUs; sufficient for patch-based training of cGANs. |
| CUDA | 12.x | GPU Acceleration | Required for PyTorch training; standard on Kaggle environments. |

### Supporting Libraries
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-image | Latest | Quantitative Metrics | Use for computing SSIM and PSNR for `VAL-01`. |
| Matplotlib | Latest | Visualization | Side-by-side qualitative comparison for `VAL-02`. |
| TensorBoard | Latest | Monitoring | Tracking GAN loss (Generator vs Discriminator) during training. |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| DL Framework | PyTorch | TensorFlow | MONAI is PyTorch-native; PyTorch has superior flexibility for custom GAN architectures. |
| Registration | SimpleITK | OpenCV | OpenCV's registration tools are too general; SimpleITK provides B-spline and elastic transforms required for biological tissue. |
| Architecture | Pix2Pix (cGAN) | CycleGAN | CycleGAN is for unpaired data; since paired data is available, cGANs provide significantly higher structural fidelity and fewer hallucinations. |
| Base Model | U-Net | Swin-UNETR | Transformer-based models are powerful but require significantly more data and compute than available for this capstone. |

## Installation

\`\`\`bash
# Core Frameworks
pip install torch torchvision torchaudio
pip install monai[all]
pip install pytorch-lightning

# Image Processing
pip install SimpleITK tifffile numpy albumentations scikit-image

# Visualization & Monitoring
pip install matplotlib tensorboard
\`\`\`

## Sources

- [Project-MONAI/MONAI GitHub](https://github.com/Project-MONAI/MONAI) (Confirmed v1.5.2 as current stable)
- [PyTorch Ecosystem](https://pytorch.org/ecosystem/)
- [SimpleITK Documentation](https://simpleitk.org/)
- Domain standards for Digital Pathology image-to-image translation (2024-2025)

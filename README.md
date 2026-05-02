# Virtual H&E Staining - Current Final Baseline

## Final Result (2026-05-02)

The current best pipeline is:

```text
CLAHE TV-L1 registration -> TTA4 -> weighted v20/v22A/v21B ensemble
```

Best fixed CLAHE validation result:

| Pipeline | SSIM | PSNR | PCC |
|---|---:|---:|---:|
| `0.30*v20 + 0.50*v22A + 0.20*v21B`, all TTA4 | **0.7838** | 25.08 | 0.8789 |
| `0.20*v20 + 0.60*v22A + 0.20*v21B`, all TTA4 | **0.7838** | **25.16** | **0.8794** |
| v22A TTA4 only | 0.7807 | 25.16 | 0.8782 |
| v21B TTA4 only | 0.7790 | 24.27 | 0.8598 |
| v20 TTA4 on CLAHE set | 0.7759 | 23.70 | 0.8600 |

On the old registered validation set, the best remains `v20_tta4`: **SSIM 0.7634 / PSNR 25.03 / PCC 0.8684**. Do not use v22A/v21B as a drop-in replacement for the old registered input distribution.

Current app:

```bash
python best_stain_app.py
```

The app defaults to the balanced ensemble (`0.20*v20 + 0.60*v22A + 0.20*v21B`) because it ties best SSIM and gives the best PSNR/PCC.

CLI example:

```bash
python best_stain_app.py --input path/to/unstained.tif --output virtual_he.png
```

Final artifacts:

```text
SCRIPT_INDEX.md
APP_DISTRIBUTION.md
best_model_manifest.json
logs/final_v21b_v22a_eval_summary.txt
logs/final_v21b_v22a_eval_per_pair.csv
reports/final_project_progress_report.pdf
docs/index.html
showcase_images/best100_clahe_balanced/
generate_best100_clahe_showcase.py
```

The GitHub Pages source lives in `docs/`. It includes a minimal research-style page, the detailed PDF report, the full HTML report, and 100 generated comparison panels from the newer CLAHE registration set.

The older v11 Pix2Pix/GAN documentation below is retained as project history. It is no longer the current best.

---

# Historical Virtual H&E Staining Pipeline

This repository contains a professional-grade generative pipeline for transforming unstained histology patches into high-fidelity H&E stained images. It utilizes **TV-L1 Optical Flow Registration** and a **Weakly Supervised Perceptual GAN** to achieve clinical-grade structural accuracy.

---

## 🧬 Technical Rationale: Why these methods?

Standard GANs often fail in medical imaging because they treat images like random photos. Our pipeline is specifically engineered for **Histopathology**:

### Registration Deep-Dive: TV-L1 Optical Flow
Before any training can occur, the dataset must be registered. Histology images are "paired" (same tissue slice) but physically "unregistered" due to the staining process.
*   **The Problem:** Staining involves liquid chemicals and heat, which physically warp and stretch the tissue at a microscopic level.
*   **Method Evaluation:**
    *   *Phase Correlation:* Failed. Too rigid; could not handle tissue rotation.
    *   *ORB Matching:* Failed. repetitive cell textures caused "False Positive" matching and RANSAC errors.
    *   *TV-L1 (Winner):* We implemented a **Variational Dense Registration** algorithm. It calculates a motion vector for **every single pixel**, allowing the model to "digitally un-warp" the tissue.
*   **The Result:** Registration alone improved the structural baseline from **0.36 to 0.63 SSIM**, establishing the "Structural Floor" needed for GAN learning.

### Training Methodology: Comparative Research
We explored multiple training tracks to identify the most effective architecture for virtual staining:

#### 1. Baseline Pix2Pix (v1 - v7)
*   **Approach:** Standard Supervised GAN on unregistered data.
*   **Result:** SSIM 0.26.
*   **Verdict:** Discarded. The misalignment created "Gradient Conflict" where the model couldn't decide whether to prioritize color or position.

#### 2. Registered Pix2Pix (Turbo Track)
*   **Approach:** WGAN-GP + L1 Loss on registered data.
*   **Optimization:** Mixed Precision + RAM Caching for 1.3 it/s throughput.
*   **Result:** SSIM 0.706 | PSNR 22.7 dB.
*   **Verdict:** Significant Success. This established that registration is the key to medical GAN stability.

#### 3. Weakly Supervised Hybrid (Research Track) — **Historical v11 Baseline**
*   **Approach:** Pix2Pix + VGG-19 Perceptual Loss + Sobel Edge Loss.
*   **Logic:** Uses a pre-trained VGG-19 network to match **Feature Maps** (shapes/textures) rather than raw pixels.
*   **Result:** **SSIM 0.712 | PSNR 23.03 dB | PCC 0.8906**.
*   **Verdict:** Historical milestone only. The current final CLAHE ensemble reaches **SSIM 0.7838 / PSNR 25.16 / PCC 0.8794**.

#### 4. Macenko Normalization Track (Path B)
*   **Approach:** Standardizing all stained colors to a reference shade before training.
*   **Result:** SSIM 0.28.
*   **Verdict:** Failed. The normalization process was "lossy," stripping away sub-cellular details that the GAN needed to learn textures.

#### 5. CycleGAN (Evaluation)
*   **Approach:** Unpaired Image-to-Image translation.
*   **Verdict:** Skipped. Since we successfully registered our pairs, Supervised Pix2Pix is mathematically more accurate for diagnostic-grade staining.

---

## ⚡ Hardware Transition: Upgrading to RTX 40-series
The pipeline is designed to be **Forward-Compatible**. When you move from your current hardware to an **RTX 4070/4080/4090**, the following happens:

1.  **Automatic Detection:** PyTorch Lightning will automatically detect the new **Ada Lovelace** architecture. No code changes are required for basic operation.
2.  **Tensor Core Acceleration:** The 40-series has 4th-Gen Tensor Cores. The `precision='16-mixed'` flag in the script will automatically utilize these to double your training speed.
3.  **Large Batch Scaling:** With the increased VRAM (12GB - 24GB), you should increase `batch_size` from 8 to **32 or 64** in `train_weakly_supervised.py`. This provides a smoother gradient and is the key to hitting **SSIM > 0.82**.
4.  **Instant Registration:** On a 40-series, the Optical Flow registration can be moved entirely to the GPU, making the pre-processing of 10,000 images nearly instantaneous.

---

## 🚀 Quick Start: The Research Workflow

Follow these historical steps to replicate the earlier v11 result (SSIM 0.712). For the current final app and metrics, use `best_stain_app.py` and the final ensemble artifacts listed at the top of this README.

### 1. Environment Setup (GPU Acceleration)
This pipeline is optimized for **NVIDIA RTX GPUs** and **Python 3.13**. To enable CUDA on Windows for this version of Python, you must use the nightly builds:

```powershell
# Install Torch + Torchvision with CUDA 12.4 Support
python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 --no-deps --force-reinstall
```

### 2. Phase 1: Image Registration (Mandatory)
Before training, all image pairs must be aligned at the pixel level. Our research proved that **TV-L1 Optical Flow** is the only method capable of handling non-rigid tissue deformation.

1.  Place your raw images in `1000/Stained_data` and `1000/Unstained_data`.
2.  Run the parallel registration pipeline:
    ```powershell
    python registration_pipeline.py
    ```
    *   **Output:** Aligned pairs saved to `data/processed/registered/`.
    *   **Metadata:** Automatically generates `data/processed/registered_pairs.csv`.

### 3. Phase 2: High-Fidelity Training (SOTA Track)
The "Weakly Supervised" track is our best performer. It uses **VGG-19 Perceptual Loss** to match biological structures rather than just pixel colors.

```powershell
python train_weakly_supervised.py
```
*   **Speed:** Uses **RAM Caching** to load 1,000+ images into memory for instant GPU access.
*   **Monitoring:** Includes a professional **Research Dashboard** that tracks SSIM, PSNR, and PCC.
*   **Early Stopping:** Automatically saves the best model and stops when convergence is reached.

### 4. Phase 3: Desktop Inference App
For the current best app:
```powershell
python best_stain_app.py
```

For the legacy v11-era app:
```powershell
python app.py
```
*   The legacy app automatically loads the best model from `checkpoints_weakly_supervised/`.
*   Current deployment should prefer `best_stain_app.py` or `v20_stain_app.py`.

---

## 📊 Evaluation Matrix Definitions

To write a high-impact research paper, we utilize these three biological metrics:

| Metric | Biological Meaning | Current Final Best |
| :--- | :--- | :--- |
| **SSIM** | **Structural Fidelity:** Did we put the nucleus in the right spot? | **0.7838** |
| **PSNR** | **Digital Fidelity:** Is the "painting" clean and noise-free? | **25.16 dB** |
| **PCC** | **Correlation:** Does the AI accurately map gray textures to purple stains? | **0.8794** |

---

## 📈 Scalability & Future Work (10x Dataset)
Our research indicates that the current architecture is data-limited. By expanding to **10,000 images** on **RTX 40-series hardware**, the following targets are achievable:
*   **SSIM Target:** 0.82+ (Clinical Grade)
*   **PSNR Target:** 30.0 dB+ (Gold Standard)
*   **Protocol:** Simply add new data, run `registration_pipeline.py`, and relaunch `train_weakly_supervised.py`.

---

## 📂 Project Structure
*   `registration_pipeline.py`: Parallel TV-L1 registration engine.
*   `best_stain_app.py`: The current final v20/v22A/v21B ensemble app.
*   `train_weakly_supervised.py`: Historical v11 training script.
*   `src/models/gan.py`: Corrected U-Net & PatchGAN architectures.
*   `src/models/losses.py`: VGG-19 Perceptual Loss implementation.
*   `PROGRESS_REPORT.md`: Detailed technical history of all experiments.

---
**Author:** Nishant-IITH  
**Project:** Capstone Virtual Staining  
**Status:** Final CLAHE ensemble baseline complete (SSIM 0.7838). Ready for larger-data validation toward SSIM 0.82+.

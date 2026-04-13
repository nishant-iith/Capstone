# Architecture Patterns

**Domain:** Virtual H&E Staining
**Researched:** 2026-04-13
**Confidence:** MEDIUM (Based on standard cGAN/Histology patterns; search tool unavailable)

## Recommended Architecture

The system follows a paired image-to-image translation pipeline based on the Pix2Pix (Conditional GAN) framework. The critical addition is a pre-training registration layer to mitigate the physical tissue distortion that occurs during the H&E staining process.

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| **Data Manager** | Loading `.tif` patches, filename-based coordinate pairing, and normalization. | Registration Engine, Training Orchestrator |
| **Registration Engine** | Performing elastic registration to align unstained images to their stained counterparts. | Data Manager, Training Orchestrator |
| **Model Module** | Implementation of U-Net Generator (image translation) and PatchGAN Discriminator (real/fake detection). | Training Orchestrator, Inference Pipeline |
| **Training Orchestrator** | Managing the GAN training loop, optimization (Adam), and checkpointing on Kaggle. | Model Module, Registration Engine |
| **Inference Pipeline** | Applying the trained Generator to new unstained patches. | Model Module, Validator |
| **Validator** | Calculating quantitative metrics (SSIM, PSNR) and generating qualitative side-by-side comparisons. | Inference Pipeline, Data Manager |

### Data Flow

1. **Preprocessing Flow**:
   `Raw Unstained/Stained Patches` $\rightarrow$ `Data Manager (Pairing)` $\rightarrow$ `Registration Engine (Elastic Alignment)` $\rightarrow$ `Aligned Pairs`.
2. **Training Flow**:
   `Aligned Unstained` $\rightarrow$ `Generator` $\rightarrow$ `Virtual H&E` $\rightarrow$ `Discriminator (vs Real H&E)` $\rightarrow$ `Loss Update`.
3. **Evaluation Flow**:
   `Test Unstained` $\rightarrow$ `Generator` $\rightarrow$ `Virtual H&E` $\rightarrow$ `Validator (vs Real H&E)` $\rightarrow$ `SSIM/PSNR Metrics`.

## Patterns to Follow

### Pattern 1: Conditional GAN (Pix2Pix)
**What:** Using a generator to create an image conditioned on an input image, with a discriminator ensuring the result is indistinguishable from a real image.
**When:** When paired training data (Input $\rightarrow$ Target) is available.
**Example:**
```python
# Simplified Generator flow
input_image = load_unstained()
virtual_he = generator(input_image)
# Discriminator evaluates pair
validity = discriminator(concat([input_image, virtual_he]))
```

### Pattern 2: Elastic Registration
**What:** Using a deformation field to warp one image to match another, correcting for non-rigid tissue shrinkage or stretching.
**When:** When physical processes (like staining) distort the tissue morphology.
**Example:**
```python
# Conceptual registration flow
fixed = stained_image
moving = unstained_image
transformation = compute_elastic_field(fixed, moving)
aligned_unstained = apply_warp(moving, transformation)
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Unpaired Translation (CycleGAN) for Diagnostic Use
**What:** Using CycleGAN when paired data is available.
**Why bad:** Unpaired translation is more prone to "hallucinations" (creating features that don't exist) or omitting critical nuclei, which is unacceptable in biomedical diagnostics.
**Instead:** Use Pix2Pix/cGAN with strict elastic registration of paired images.

### Anti-Pattern 2: Global Scaling for Alignment
**What:** Using only rigid (translation/rotation/scale) registration.
**Why bad:** Histology tissue distorts locally and non-uniformly; rigid registration will leave significant misalignment.
**Instead:** Implement Elastic/Deformable registration.

## Scalability Considerations

| Concern | Patch-based (Current) | WSI (Future Scale) | Cloud Inference |
|---------|-----------------------|---------------------|------------------|
| **Memory** | Fits in GPU VRAM | Requires Tiling/Stitching | Requires optimized ONNX/TensorRT |
| **Compute** | Fast per patch | Extremely slow for whole slide | Needs GPU clusters |
| **Storage** | Standard Disk | Needs Zarr/Ome-Tiff | Object Storage (S3/GCS) |

## Sources

- Pix2Pix: "Image-to-Image Translation with Conditional Adversarial Networks" (Isola et al.)
- General Histology Virtual Staining patterns (Standard biomedical AI approach)
- Project requirements: `.planning/PROJECT.md`

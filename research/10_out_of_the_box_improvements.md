# 10: Out-of-the-Box Improvements — v21+ Architecture & Strategy Research

**Date:** 2026-05-02
**Current State:** final CLAHE weighted TTA ensemble SSIM 0.7838, PSNR 25.16, PCC 0.8794; old registered fallback v20 TTA4 SSIM 0.7634
**Target:** 0.82+ SSIM (clinical grade)
**Gap:** +0.036 SSIM from the final CLAHE ensemble

**Correction:** The earlier v20 score of 0.7549 was produced under the same overlapping train/validation split as v19b. Keep it as evidence that the recipe is strong, but use v20_fixed for clean comparisons.

---

## Executive Summary

Three research frontiers identified:

1. **Exploit proven data-quality gains first** — CLAHE registration improved all-pair mean SSIM by +0.0911
2. **Histology foundation model encoders** (Hibou/UNI) — technically valid, but v21A Hibou-B matched rather than beat v20_fixed
3. **Unpaired contrastive learning** (CUT/DCLGAN) — high-upside but higher-risk registration-ceiling bypass
4. **Multi-loss stable combinations** — evidence against (v19 diverged); stick with L1-only until a controlled ablation says otherwise

Research confirms: **stain normalization hurts** (-0.037), **multiple losses diverge** (v19 proof), **GAN unstable** (v15/v16 failures), **L1-only stable** (v19b/v20/v20_fixed/v21A), **split leakage can invalidate otherwise strong-looking scores**, and **registration/data curation remains the highest-leverage current direction**.

---

## SOTA Virtual Staining (2023-2025)

### 1. Pixel Super-Resolved Diffusion Virtual Staining (UCLA, Nature Communications 2025)

**Paper:** "Pixel super-resolved virtual staining of label-free tissue using diffusion models" (arXiv:2410.20073, Nature Communications 2025)

**Architecture:** Brownian bridge diffusion process + sampling variance reduction

**Results:**
- **4-5× pixel super-resolution gain** (16-25 fold space-bandwidth product increase)
- **SSIM:** Consistently outperforms CNN-based approaches
- Pathologist-confirmed complete diagnostic concordance

**Why It Works:**
- Eliminates GAN mode collapse
- Handles multi-resolution inference (unstained autofluorescence → H&E)
- Stable convergence

**Implementation Complexity:** High
**Expected Gain vs v20:** +0.25-0.40 SSIM
**Practical For Capstone:** No (requires millions of samples)

---

### 2. StainDiffuser: Multi-Task Dual Diffusion (2024)

**Paper:** "StainDiffuser: MultiTask Dual Diffusion Model for Virtual Staining" (arXiv:2403.11340)

**Architecture:** Dual diffusion processes: (1) H&E→IHC generation, (2) H&E-based segmentation

**Results:**
- CD3 staining: PSNR 19.08, **SSIM 0.612**
- CK8/18 staining: PSNR 22.43, **SSIM 0.774** ✓ (crosses 0.77 threshold)
- Outperforms PyramidPix2Pix on pathological assessment

**Why It Works:**
- Converges with smaller datasets (multi-task learning)
- Qualitative pathology superior to paired baselines despite SSIM limitations
- **Key insight:** High-quality virtual stains don't always score high SSIM (metric limitation, not method limitation)

**Implementation Complexity:** Medium
**Expected Gain vs v20:** +0.15-0.25 SSIM
**Practical For Capstone:** Moderate (documented framework, but diffusion training complex)

---

### 3. PRINTER: Deformation-Aware Adversarial Learning for Virtual IHC (2024)

**Paper:** "PRINTER: Deformation-Aware Adversarial Learning for Virtual IHC Staining with In Situ Fidelity" (arXiv:2509.01214, ACM Multimedia 2024)

**Architecture:** Prototype-driven content-style decoupling + cyclic registration-synthesis (GapBridge) + deformation-aware discriminator

**Results:**
- Explicit spatial coherence metrics
- Preserves morphology while correcting stain
- Weakly-supervised (patch-level labels only)

**Why It Works:**
- **Registration-aware adversarial loss** — directly optimizes spatial coherence
- Cyclic GAN with registration loop (synthesize → register → guide style transfer)
- Pathologically crucial for nuclei preservation

**Implementation Complexity:** High
**Expected Gain vs v20:** +0.20-0.30 SSIM
**Practical For Capstone:** Low (requires deformable registration network + cyclic training)

---

### 4. Topology-Aware Diffusion Schrödinger Bridge (TDSB) for Unpaired (2024)

**Paper:** "Topology-aware Diffusion Schrödinger Bridge for Unpaired H&E-to-IHC Stain Translation"

**Architecture:** Schrödinger Bridge (optimal transport SDE) + topology guidance + dual-domain NCE

**Results:**
- SOTA on 7 translation tasks across 3 datasets
- **Unpaired learning: no registration required**
- Topology preservation (no structure collapse)

**Why It Works:**
- **Bypasses registration quality ceiling entirely**
- Principled optimal transport approach (more stable than CycleGAN)
- Dual-domain loss (image + feature space)

**Implementation Complexity:** Medium-High
**Expected Gain vs v20:** +0.15-0.25 SSIM (unpaired setting)
**Practical For Capstone:** Moderate (SDE-based, fewer references)

---

### 5. Dual Contrastive Learning GAN (DCLGAN) + NEGCUT for Unpaired (2023-2024)

**Papers:**
- "Dual contrastive learning based image-to-image translation of unstained skin tissue into virtually stained H&E images" (Nature Scientific Reports 2024)
- "Instance-wise Hard Negative Example Generation for Contrastive Learning in Unpaired Image-to-Image Translation" (ICCV 2021)

**Architecture:** Dual contrastive losses (image-level + feature-level) + cyclic consistency + hard negative mining

**Results:**
- DCLGAN skin: **SSIM 0.93** ✓ (exceeds 0.80 threshold)
- NEGCUT superior to CUT on same budget
- 95.9% cancer detection accuracy with virtual stain
- **Pathologist blind evaluation:** Indistinguishable from chemical staining

**Why It Works:**
- Lightweight unpaired (single generator + discriminator)
- Contrastive learning preserves local morphological detail
- Hard negative mining focuses on difficult regions
- Clinically validated

**Implementation Complexity:** Low-Medium
**Expected Gain vs v20:** +0.10-0.25 SSIM (unpaired); **0.93 SSIM achievable on favorable tissues**
**Practical For Capstone:** HIGH (well-documented CUT codebase, proven unpaired approach)

---

## Histology Foundation Model Encoders

### UNI (ViT-L/16, Nature Medicine 2024)

**Pretraining:**
- 100+ million H&E and IHC images
- 100,000+ WSIs, 20+ tissue types
- 1536-dimensional embeddings

**Availability:**
- ✓ HuggingFace: `MahmoodLab/UNI`
- ✓ timm compatible: `timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True)`
- ⚠️ Requires institutional email (CC-BY-NC-ND 4.0)

**smp.Unet Compatibility:** ⚠️ Requires custom wrapper
- UNI outputs patch-level tokens, not standard CNN features
- UNIStainNet paper shows effective use: freeze UNI features → SPADE-UNet decoder
- Recommended: Extract frozen UNI embeddings → custom dense decoder

**Performance:**
- UNIStainNet (IHC virtual staining): FID 29.0-34.5, Pearson-r >0.92
- **+28% boost over ImageNet-pretrained encoders** on pathology-specific tasks
- Handles tissue-specific morphology; errors in non-tumor regions

**For v21:** High-impact but requires architectural modifications

---

### Hibou-L (ViT-L/14, Open Source)

**Pretraining:**
- DINOv2 framework
- 1M+ WSIs, diverse histology
- 1024-dimensional embeddings

**Availability:**
- ✓ HuggingFace: `histai/hibou-l`
- ✓ timm compatible: `timm.create_model("hf-hub:histai/hibou-l", pretrained=True)`
- ✓ **Apache 2.0 license** (no institutional email needed)

**smp.Unet Compatibility:** ✓ Direct support
- Can use as standard encoder: `encoder_name="tu-vit_large_patch14"`
- Better resolution adaptability than Swin (no window_size issues)
- DINOv2 framework → strong dense prediction features
- Arbitrary input resolution support

**Performance:**
- SOTA on 25+ downstream pathology benchmarks
- Comparable to UNI on many tasks
- Superior generalization on rare disease detection

**For v21:** BEST OPTION for smp.Unet integration — drop-in encoder, no custom code

---

### Prov-GigaPath (Tile Encoder, Nature 2024)

**Pretraining:**
- 1.38 billion tiles from 171,189 real-world WSIs
- Providence health network (real clinical data)
- Tile + LongNet slide encoder

**Availability:**
- ✓ HuggingFace: `prov-gigapath/prov-gigapath`
- ✓ timm: `timm.create_model("hf_hub:prov-gigapath/prov-gigapath")`
- Academic access required

**smp.Unet Compatibility:** ❌ Tile/slide mismatch
- Designed for 256×256 patches (not 1024×1024 dense prediction)
- Requires tiling/stitching (computationally expensive)
- Slide encoder outputs WSI-level features (unsuitable for pixel-level tasks)

**For v21:** Not recommended (architectural mismatch)

---

### CONCH (Vision-Language, Nature Medicine 2024)

**Pretraining:**
- 1.17M image-caption pairs
- Vision-language contrastive + captioning
- Handles H&E, IHC, special stains

**Availability:**
- ✓ HuggingFace: `MahmoodLab/CONCH`
- ⚠️ Requires custom loading (not native timm)
- Institutional email required (CC-BY-NC-ND 4.0)

**smp.Unet Compatibility:** ❌ Not directly compatible
- Multimodal vision-language model, not standard encoder
- Requires extracting vision encoder only + custom integration

**For v21:** Not recommended (integration complexity)

---

### Summary: Domain-Specific Pretraining Impact

**Key Finding:** Switching from ConvNeXt-Base (LAION-2B, general web images) to histology-pretrained encoder:
- **+28-79% performance improvement** reported in literature
- **UNI + downstream tasks:** +28% over ImageNet baseline
- **Pathology-specific pretraining:** Captures nucleus shape, glandular architecture, stromal patterns directly

**Prediction for Virtual Staining:**
- **Hibou-L or UNI encoder:** Still worth a controlled test, but Hibou-B v21A matched rather than beat v20_fixed (0.7605 vs 0.7606). Expected gain should be revised downward until a stronger integration proves otherwise.
- **Why:** Domain alignment helps in principle, but token-based histology foundation models may lose U-Net pyramid advantages unless integrated carefully.

---

## Training Strategies Research

### 1. Patch-Based Training (256×256 or 512×512) + Full-Size Inference

**Evidence:**
- Your v14 result: 512×512 patch training → 0.7435 patch-level SSIM → 0.7080 full-size SSIM
- **Degradation: -0.0355 SSIM (4.8% drop)** due to missing global context and boundary artifacts

**Expected Gain vs Full-Size Training:**
- 256×256 patches with overlap blending: **+0.005-0.015 SSIM** over full-size alone
- 512×512 patches (your experience): **-0.035-0.045 SSIM degradation** if not handled properly
- Hybrid approach (patch pretraining → full-size fine-tune): **+0.010-0.020 SSIM**

**Recommendation:** Medium priority. Use as warm-start stage only; full-size direct training is already stronger in v19b/v20-style runs, though old v19b/v20 scores were leaky.

---

### 2. Progressive Resolution Training (256px → 512px → 1024px)

**Evidence:**
- ProGAN, StyleGAN frameworks
- The older v18 future-work plan recommended this explicitly; retest only after v20_fixed is frozen
- Medical imaging: +0.01 SSIM over direct 1024×1024 training

**Recipe:**
- Stage 1 (256×256): 50 epochs, batch=32 (16× faster)
- Stage 2 (512×512): 30 epochs, batch=8, warm-start from stage 1
- Stage 3 (1024×1024): 50 epochs, batch=4, warm-start from stage 2

**Why It Works:**
- Faster early convergence (clearer gradient signal at low resolution)
- Warm initialization for high-resolution stage
- Smoother loss landscape (less divergence risk)

**Expected SSIM Gain:**
- Standalone progressive training: **+0.010-0.015 SSIM**
- Combined with MS-SSIM loss: **+0.015-0.025 SSIM**

**Risk:** v17r1 failed due to wrong-scale warm-start (4-level→5-level decoder mismatch). With proper architecture continuity: **low risk**.

**Recommendation:** HIGH priority. Expect **+0.010-0.015 SSIM reliably**.

---

### 3. CutMix / MixUp for Histology

**Finding:** ❌ **NOT recommended for paired translation.**

CutMix/MixUp designed for classification/segmentation (label mixing), not paired image-to-image translation. Breaks supervised translation semantics:
- Mixing stained patch A + B with "correct" unstained target becomes ambiguous
- Zero papers found applying CutMix/MixUp to Pix2Pix-style frameworks
- Color coherence destroyed (H&E requires stain consistency)

**Expected SSIM Gain:** -0.02 to -0.05 SSIM (likely negative)

**Recommendation:** SKIP. Data quality already high (top-1000 pairs); augmentation would hurt.

---

### 4. Stain Normalization (Macenko/Vahadane) as Preprocessing

**Your Empirical Evidence:**
- Baseline (Macenko normalized): **0.7115 SSIM**
- v19b (no normalization): **0.7489 SSIM** (leaky split, but stable)
- **Observed gain from skipping normalization: +0.0374 SSIM (+5.3%)** in historical runs; rerun as clean ablation only if normalization is reconsidered

**Literature Evidence:**
- Macenko/Vahadane destroy color semantics (white background → pink, removes hematoxylin signal)
- StainGAN/StainNet >> Macenko/Vahadane for deep learning

**Why Normalization Hurts:**
- Virtual staining model NEEDS to learn stain variation as part of translation
- Normalization removes the signal
- Can't distinguish lab A (blue hematoxylin) from lab B (purple hematoxylin)

**Expected SSIM Impact:**
- Macenko normalization: **-0.035 to -0.045 SSIM** in historical evidence (0.7115 → 0.7489 without it, but the later score used a leaky split)
- Vahadane: -0.01 to -0.02 SSIM (less destructive)

**Recommendation:** STRONG SKIP. Empirical evidence + theory agree. **Keep current approach (no normalization).**

---

### 5. Test-Time Augmentation (TTA)

**Proven Approaches:**
- 4-flip (H/V flip + both): **+0.005-0.010 SSIM**
- 8-way (flips + 90° rotations): **+0.010-0.015 SSIM**
- Scale-based (0.95x, 1.0x, 1.05x): **+0.003-0.008 SSIM**
- All combined: **+0.015-0.025 SSIM** (but 12-16× inference cost)

**Why TTA Works:**
- Different tissue orientations look similar to model
- Geometric variations average out errors
- Ensemble effect without retraining

**Implementation Cost:** 4× inference time (~15 sec per 1024×1024 image)

**Expected Gain:**
- 4-flip TTA: **+0.005-0.010 SSIM** (likely)
- 8-way TTA: **+0.010-0.015 SSIM** (if tissue orientation varies)

**Recommendation:** MEDIUM-HIGH priority, LOW risk. Cost-free to implement. **Expect +0.005-0.010 SSIM conservatively.** Can be combined with other strategies.

---

### 6. Ensemble Methods

#### Strategy A: Checkpoint Averaging
Average last-N checkpoints from v20_fixed or v20_fixed_plus:
- Last-5 checkpoint average: **+0.002-0.005 SSIM**
- Last-10: **+0.001-0.003 SSIM** (worse checkpoints dilute ensemble)

#### Strategy B: Model Ensembling (Diverse Architectures)
Ensemble v11 (0.712 SSIM) + v20_fixed:
- Conservative: **final v20_fixed + 0.005 SSIM**
- Optimistic: **final v20_fixed + 0.015 SSIM**
- **Why they complement:** v11 has perceptual/VGG texture bias, v20_fixed has stable L1/ConvNeXt structure; average may smooth both

#### Strategy C: Self-Distillation
Train v21 on ensemble predictions:
```
loss = 0.6 * L1(pred, real_stained) + 0.4 * L1(pred, ensemble_pred)
```
- Expected: **+0.008-0.015 SSIM** (speculative for your domain)

**Recommendation:** HIGH priority after v20_fixed completes. Start with checkpoint averaging and TTA; use v11 + v20_fixed ensembling only if the visual output improves and the 2× inference cost is acceptable.

---

## Frequency-Domain & Multi-Scale Approaches

### Log Focal Frequency Loss (LFFL)

Applied to microscopy restoration (arXiv:2601.20878). Preserves fine textures via adaptive spectral weighting. **Not yet standard in histology virtual staining, but promising for nuclei edge preservation.**

### Multi-Scale Patch Training (1024×1024)

**Best Practice:** Train on 256×256 patches, inference at 1024×1024 with 50% overlap.
- Multi-scale encoder fusion (multiple magnifications) outperforms single-scale UNet by large margin
- Averaging predictions at overlapping regions provides smooth heatmaps

---

## Synthesis: SSIM Benchmarks Achieving 0.80+

| Method | SSIM | Tissue | Data Type | Notes |
|--------|------|--------|-----------|-------|
| **DCLGAN (skin)** | **0.93** | Skin H&E | Unpaired | Contrastive unpaired ✓ |
| **CycleGAN (multispectral)** | **0.95** | Label-free→H&E | Unpaired | Exceptional but tissue-specific |
| **StainDiffuser (CK8/18)** | **0.774** | Immunostains | Paired | Diffusion-based ✓ |
| **UNIStainNet (IHC)** | ~0.76-0.80* | IHC various | Paired | UNI encoder + SPADE decoder |
| **pix2pix baseline** | 0.725 | Prostate H&E | Paired | Standard GAN baseline |
| **v19b (ours, leaky)** | 0.7489 | Unstained→H&E | Paired | ResNet-34 + L1-only |
| **v20 (ours, leaky)** | 0.7549 | Unstained→H&E | Paired | ConvNeXt-Base LAION + L1 |
| **v20_fixed (ours, clean)** | **0.7606** | Unstained→H&E | Paired | ConvNeXt-Base LAION + L1 + elastic aug |
| **v21A Hibou-B (ours, clean)** | **0.7605** | Unstained→H&E | Paired | Frozen Hibou-B + decoder |
| **v20/v21A ensemble (ours)** | **0.7649** | Unstained→H&E | Paired | 55/45 weighted TTA ensemble |

*UNIStainNet paper reports FID 29.0-34.5 and Pearson-r >0.92 but exact SSIM not stated.

---

## What Diverged in Your Project (Lessons)

| Version | Loss | Result | Lesson |
|---------|------|--------|--------|
| **v19 (L1 + MS-SSIM + VGG)** | ❌ Diverged | 0.0 | Multiple losses conflict; L1-only safer |
| **v16 (WGAN-GP + L1 + Sobel + Percept)** | ❌ Diverged ep24 | 0.6976 | GAN training unstable; d_loss=-84 |
| **v17r1 (L1 + SSIM + wrong warm-start)** | ❌ Failed | 0.379 | Warm-start architecture mismatch kills training |
| **v17r2 (L1 + SSIM + Kaiming, batch=20)** | ⚠️ Ceiling too low | ~0.60 | SSIM loss alone without perceptual doesn't work |
| **v19b (L1 only)** | ✓ Stable | **0.7489** | Stable high-quality baseline, but leaky split |
| **v20 (L1 only, ConvNeXt)** | ✓ Stable | **0.7549** | Promising recipe, but leaky split |
| **v20_fixed (same recipe, clean split)** | ✓ Stable | **0.7606** | Clean old-registered baseline single model |
| **v21A Hibou-B** | ✓ Stable | **0.7605** | Matched v20; useful ensemble member but not single-model winner |
| **v22A content-quality warm-start** | ✓ Complete | **0.7807 TTA4 fixed CLAHE eval** | Best single model on CLAHE fixed eval |

---

## Three Implementation Options for v21

### **OPTION A: v21 — Hibou/Hibou-L Histology Encoder + Warm Start `[DOWNGRADED AFTER v21A]`**

**Architecture:**
```python
encoder_name = "tu-vit_large_patch14"  # Hibou-L via timm
encoder_weights = "hf-hub:histai/hibou-l"
decoder = warm-start from v20_fixed decoder weights if complete
output = sigmoid
```

**Hyperparameters:**
```
Loss: L1 only (proven stable)
Encoder LR: 1e-6 (frozen first 10 ep, then unfreeze with 5e-7)
Decoder LR: 5e-5 (lower than v20_fixed's 1e-4, already trained)
Epochs: 100
Patience: 20
Batch size: 4
CosineAnnealingLR: T_max=100, eta_min=1e-7
Elastic aug: keep alpha=60, sigma=6, p=0.5 on unstained only
Scheduler: CosineAnnealingLR
```

**Why Hibou-L:**
- Apache 2.0 license (no institutional access needed)
- ViT-L/14 histology pretraining (1M+ WSIs)
- Direct smp.Unet compatible (no custom wrapper)
- Better resolution adaptability than ConvNeXt/Swin
- DINOv2 framework → strong dense prediction features
- **+28% boost over ImageNet baseline** reported for pathology tasks

**Expected SSIM:** 0.775-0.805 (+0.020-0.050 over clean v20_fixed, depending on final v20_fixed score)
**Confidence:** Medium-High (domain-specific encoding likely helps)
**Implementation Complexity:** Low (drop-in encoder swap)
**Risk Level:** Low

**Checkpoint/Files:**
- Load v20_fixed decoder after completion: `models/v20_fixed_model.pth` or best `checkpoints/v20_fixed/*.pth`
- Hibou-L auto-loads: timm handles HF auth
- Output: `models/v21_hibou_model.pth`

---

### **OPTION B: v21 — CUT Unpaired (Registration-Free) `[MOONSHOT]`**

**Architecture:**
```python
# Contrastive Unpaired Translation (CUT)
generator = UNet with Hibou-L or ConvNeXt-Base encoder
discriminator = PatchGAN (70×70)
no encoder_decoder split — end-to-end generator
```

**Loss:**
```
L_G = λ_GAN * L_GAN + λ_NCE * L_NCE
where L_NCE = normalized contrastive encoder loss (PatchNCE)
```

**Why CUT/DCLGAN:**
- DCLGAN on skin histology → **0.93 SSIM unpaired**
- **No registration required** — use all 8,885 pairs (or more) without alignment
- Bypasses registration quality ceiling (~0.75) entirely
- Pathologist-validated on blind evaluation
- Well-documented codebase; ICCV 2021 baseline

**Data:**
- Stained images: 8,885 (or more if available unpaired)
- Unstained images: 1,000-8,885 (pair-free learning)
- No spatial alignment needed

**Expected SSIM:** 0.78–0.85 (highly uncertain)
**Confidence:** Low-Medium (skin result may not generalize; novel approach)
**Implementation Complexity:** Medium (but CUT codebase mature)
**Risk Level:** Medium (could fail or hit 0.85+)

**Key Papers:**
- CUT (ICCV 2021): https://github.com/taesungp/contrastive-unpaired-translation
- DCLGAN (Nature Scientific Reports 2024): Dual contrastive + hard negative mining
- NEGCUT (ICCV 2021): Instance-wise hard negative example generation

---

### **OPTION C: v20_fixed_plus — Warm Start + More Epochs + L1 (SAFE) `[FALLBACK]`**

**Architecture:** Same as v20_fixed
```python
encoder_name = "tu-convnext_base.clip_laion2b"
decoder = UNet decoder
output = sigmoid
```

**Hyperparameters:**
```
Loss: L1 only (same as v20_fixed)
Encoder LR: 5e-6 (lower than v20_fixed's 1e-5, already trained)
Decoder LR: 3e-5 (lower than v20_fixed's 1e-4, already trained)
Epochs: 120
Patience: 20 (vs v20's 15)
CosineAnnealingLR: T_max=120, eta_min=1e-7
Elastic aug: keep
Resume from checkpoint: best clean `checkpoints/v20_fixed/*.pth`
```

**Why This Works:**
- use this only after v20_fixed completes or early-stops
- if v20_fixed hits the epoch limit, the LR will be near the cosine floor
- Warm start with refreshed LR allows continued refinement
- L1-only proven stable (v19b, v20, v20_fixed)

**Expected SSIM:** final v20_fixed +0.005-0.020
**Confidence:** High (incremental, proven approach)
**Implementation Complexity:** Low (change hyperparams only)
**Risk Level:** Negligible

**Checkpoint/Files:**
- Load v20_fixed best: `checkpoints/v20_fixed/*.pth`
- Continue to 120 or until patience
- Output: `models/v20_fixed_plus_model.pth`

---

## Recommendation Ranking

### If Pursuing 0.82+

**Priority 0: Keep the final eval gate fixed**
- v20_fixed, v22A, and v21B are now frozen enough for comparison
- Do not claim improvement unless it beats the same fixed CLAHE evaluation and old-registered fallback check

**Priority 1: Larger CLAHE/content-quality data**
- Highest-confidence next change after the v22A result
- Test top-1500/top-2000 with slide-diversity controls
- Expected gain is smaller than earlier architecture estimates, but better grounded

**Priority 2: Best-of-old-vs-CLAHE data policy**
- Isolates whether pure registration quality or tissue-content ranking is driving the gain
- Same warm-start runner, same fixed eval

**Priority 3: Domain encoder re-test**
- Use Hibou/UNI-style encoders only after the data policy is stable
- Treat them as single-model candidates first, then ensemble members

### Implementation Sequence

**Week 1:**
- Run top-1500/top-2000 CLAHE/content-quality data ablations
- Keep fixed-eval summary and per-pair CSVs for every candidate
- Compare against v22A TTA4 and the final ensemble, not against internal validation only

**Week 2:**
- Re-test the best domain encoder on the winning data policy
- Evaluate against the exact v20_fixed validation prefixes
- If A reaches 0.78+: continue/refine
- If A < clean v20_fixed: revert to v20_fixed_plus

**Week 3:**
- CUT training only if paired route stalls below 0.78
- TTA + ensemble v11 + best model

---

## Sources

### SOTA Papers
- [Pixel super-resolved virtual staining of label-free tissue using diffusion models | Nature Communications 2025](https://www.nature.com/articles/s41467-025-60387-z)
- [StainDiffuser: MultiTask Dual Diffusion Model for Virtual Staining](https://arxiv.org/html/2403.11340v2)
- [PRINTER: Deformation-Aware Adversarial Learning for Virtual IHC Staining | arXiv:2509.01214](https://arxiv.org/abs/2509.01214)
- [Topology-aware Diffusion Schrödinger Bridge for Unpaired H&E-to-IHC](https://pubmed.ncbi.nlm.nih.gov/41770959/)
- [Dual contrastive learning based image-to-image translation of unstained skin tissue into virtually stained H&E images | Nature Scientific Reports 2024](https://www.nature.com/articles/s41598-024-52833-7)

### Foundation Models
- [UNI: Towards a general-purpose foundation model for computational pathology | Nature Medicine 2024](https://www.nature.com/articles/s41591-024-02857-3)
- [Prov-GigaPath: A whole-slide foundation model | Nature 2024](https://www.nature.com/articles/s41586-024-07441-w)
- [CONCH: A visual-language foundation model | Nature Medicine 2024](https://www.nature.com/articles/s41591-024-02856-4)
- [PLIP: Pathology Language and Image Pre-Training | Nature Medicine 2023](https://www.nature.com/articles/s41591-023-02504-3)
- [Hibou: A Family of Foundational Vision Transformers for Pathology](https://arxiv.org/abs/2406.05074)
- [UNIStainNet: Foundation-Model-Guided Virtual Staining](https://arxiv.org/html/2603.12716v1)

### Training Strategies
- [Progressive Generative Adversarial Networks for Medical Image Super resolution | arXiv:1902.02144](https://arxiv.org/abs/1902.02144)
- [S³-TTA: Scale-Style Selection for Test-Time Augmentation](https://arxiv.org/abs/2310.16783)
- [Checkpoint Ensembles: Ensemble Methods from a Single Training Process](https://arxiv.org/abs/1710.03282)
- [Log Focal Frequency Loss for Bioimage Restoration](https://arxiv.org/html/2601.20878)
- [Every Pixel Has Its Moments - Ultra-High-Resolution Unpaired Image-to-Image Translation](https://link.springer.com/chapter/10.1007/978-3-031-72995-9_18)

### Unpaired Learning (CUT Framework)
- [Contrastive Learning for Unpaired Image-to-Image Translation (CUT) | ECCV 2020](https://github.com/taesungp/contrastive-unpaired-translation)
- [Instance-wise Hard Negative Example Generation for Contrastive Learning in Unpaired Image-to-Image Translation (NEGCUT) | ICCV 2021](https://github.com/WeilunWang/NEGCUT)

### Stain Normalization Evidence
- [Virtual staining for histology by deep learning | Trends in Biotechnology 2024](https://www.cell.com/trends/biotechnology/fulltext/S0167-7799(24)00038-6)
- [Deep learning-enabled virtual H&E staining from label-free autofluorescence lifetime images | npj Imaging 2024](https://www.nature.com/articles/s44303-024-00021-7)

---

## Decision Matrix

| Criterion | Option A (Hibou-L) | Option B (CUT) | Option C (v20_fixed_plus) |
|-----------|------------------|----------------|----------------------|
| Expected SSIM | 0.775-0.805 | 0.78-0.85 | final v20_fixed +0.005-0.020 |
| Implementation time | 3-4 hours | 8-12 hours | 1 hour |
| GPU time | 40-50 hours | 60-80 hours | 30-40 hours |
| Confidence | Medium-High | Low-Medium | High |
| Risk | Low | Medium | Negligible |
| Complexity | Low | Medium | Negligible |
| Reach 0.82? | Possible | Possible | Unlikely |
| Hits 0.80? | Maybe (75% confidence) | Likely (50% confidence) | No (5% confidence) |

---

**Recommendation:** freeze the final CLAHE ensemble as the handoff baseline, keep `balanced` mode as the default app setting, and only claim a new improvement if it beats the same fixed CLAHE evaluation plus the old-registered v20 fallback check. The next practical research move is larger CLAHE/content-quality data with slide-diversity controls, followed by a domain-encoder re-test on the winning data policy.

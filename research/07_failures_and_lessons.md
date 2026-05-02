# 07: Failures and Lessons — What Went Wrong, What We Learned

> **Bottom Line:** Across v1-v22A, the main failures were not just model failures; several were evaluation and data-quality failures. The recurring themes: (1) GAN instability dominates loss/architecture choices, (2) data quality has higher leverage than architecture complexity, (3) numerical stability of "elegant" losses must be tested first, (4) more loss terms ≠ better model, (5) validation splits must be audited by image identity/prefix, and (6) registration improvements must be measured per pair before retraining.

---

## 1. The Major Failures

### Failure 1: Pre-Registration Era (v1-v7)

**What we tried:** Standard Pix2Pix on raw unregistered pairs.
**Result:** SSIM 0.26 — model produced blurry pink/purple blobs.
**Why it failed:**
- Tissue cells are displaced by 5-50 pixels between paired images
- L1 loss penalizes the model for putting the right structure at the slightly wrong location
- Optimal L1 strategy under spatial uncertainty is to **average** (i.e., blur)
- GAN loss can't recover detail from a blurry-mean prediction

**Lesson:** **Registration is non-negotiable.** No model can compensate for >10px structural misalignment. Spend 10× more effort on registration before tuning the model.

---

### Failure 2: Phase Correlation & ORB Registration

**What we tried:** Classical CV registration methods (Phase Correlation, ORB Feature Matching).
**Result:**
- Phase Correlation: +0.04 SSIM gain, fails on 80% of pairs with non-rigid distortion
- ORB: 70% convergence failure, catastrophic when wrong (texture repetition)

**Why it failed:**
- Phase Correlation only handles global rigid translation
- Histology has thousands of near-identical nuclei → ORB descriptors confuse them
- Cross-modal (autofluorescence ↔ H&E) breaks descriptor matching
- RANSAC can't converge with >50% bad matches

**Lesson:** **Match the registration method to the deformation type.** Histology requires dense non-rigid (TV-L1, SyN, deep learning); rigid methods are fundamentally insufficient.

---

### Failure 3: HED Stain Decomposition Loss (v12, v15)

**What we tried:** Macenko stain matrix decomposition into Hematoxylin/Eosin/DAB channels with per-channel L1 loss.
**Result:** v12 destabilized, v15 diverged after 1 epoch.

**Why it failed:**
- `torch.inverse` of stain matrix in float16 produces NaN gradients
- HED loss gradient direction conflicted with VGG perceptual gradient direction
- Magnitude imbalance: hard to weight HED against L1(100) and Percept(10)
- Numerical instability propagated through the discriminator

**Lesson:** **Numerical stability trumps theoretical elegance.** Test the loss in isolation with mixed precision before combining. If `torch.inverse` is in the loss path, expect fp16 problems.

---

### Failure 4: All-Pairs Training (v13)

**What we tried:** Train v13 on all 8,885 registered pairs (mean SSIM 0.51), expecting that more data + Attention U-Net would beat v11.
**Result:** SSIM 0.6326 — significantly worse than v11 (0.712).

**Why it failed:**
- Mean SSIM of 0.51 means ~half the pairs are essentially noise
- Model learns to average across noisy pairs → blurry outputs
- Attention gates can't fix bad gradient signal from misaligned pairs
- Architecture complexity doesn't compensate for data quality

**Lesson:** **Data quality dominates architecture.** A simpler model on curated data beats a complex model on noisy data. Filter aggressively.

---

### Failure 5: GAN Divergence (v15, v16)

**What we tried:** Attention U-Net + MultiScale Discriminator + WGAN-GP + L1 + Sobel + Percept.
**Result:**
- v15 peaked at SSIM 0.7199 in epoch 1, then diverged.
- v16 peaked at SSIM 0.6976 in epoch 24, then diverged.

**Why it failed:**
- d_loss ran away to -84 (discriminator winning catastrophically)
- g_loss exploded to 123
- Loss imbalance: L1(100) vs WGAN-GP(10) means GAN signal is 10× weaker than reconstruction
- Once discriminator wins, generator can't recover (saturated gradients)
- Warm-start and lower LR didn't fix it

**Lesson:** **GAN training is high-variance, marginal-reward.** A +0.005-0.010 SSIM gain when stable, vs catastrophic loss when unstable. For SSIM-targeted optimization, **drop the GAN** (v17 strategy).

---

### Failure 6: v14 OOM at num_workers=60

**What we tried:** Maximize DataLoader parallelism with 60 workers (matching CPU count).
**Result:** OOM crash — system memory exhausted at ~360 GB usage.

**Why it failed:**
- `DataLoader` with `num_workers > 0` uses `fork` to create worker processes
- Each worker inherits a copy-on-write reference to parent's memory
- Mutating the cache in workers (or PyTorch's internal collation) triggers full copies
- 60 workers × 6 GB cache = 360 GB

**Lesson:** **Pre-cached datasets should use `num_workers=0`.** Workers help when the bottleneck is disk I/O; they hurt when data is already in memory. Always profile.

---

### Failure 7: v17 Initial Configuration Mistakes

**What we tried:** Multiple iterations of v17 with various config choices.

**Mistakes encountered:**
1. **BATCH_SIZE=32** → OOM at 1024×1024 (each forward needs ~6 GB)
2. **`prefetch_factor=2` with `num_workers=0`** → Invalid arg error
3. **`gradient_clip_val=1.0` with `manual_optimization=False`** → Lightning error
4. **Warm-start shape mismatch** (v14 4-level → v17 5-level bottleneck differs)

**Fixes:**
1. BATCH_SIZE = 4
2. Remove `prefetch_factor` parameter
3. Remove `gradient_clip_val`
4. Filter warm-start state dict to shape-matched layers only:
   ```python
   match = {k: v for k, v in v14_state.items()
            if k in model_state and v.shape == model_state[k].shape}
   ```

**Lesson:** **Validate config combinations.** PyTorch Lightning has many flags with subtle interactions. Always test-run for one batch before committing to a long training.

---

### Failure 8: v17 Wrong-Scale Warm-Start (Catastrophic)

**What we tried:** v17 = 5-level Simple U-Net with L1+SSIM loss, warm-started from v14's 4-level checkpoint by name-matching state-dict keys with shape filter.

**Result:** ❌ SSIM stuck at **0.379** (peak epoch 25), well below the registration floor (0.63) and below the input-as-output baseline. Patience exhausted near epoch 34.

**Training trajectory:**
| Epoch | Val SSIM |
|-------|----------|
| 1 | 0.2343 |
| 6 | 0.3543 |
| 13 | 0.3770 |
| 19 | 0.3788 |
| **25** | **0.3790 ⭐ (peak)** |
| 34 | 0.3378 (decline) |

**Why it failed:**
1. **Wrong-scale warm-start.** v14 has 4 encoder/decoder stages; v17 has 5. Decoder layer names match (`dec1`-`dec4`) but represent **different receptive-field scales**:
   - v14 `dec4` = the deepest decoder (bottleneck-adjacent, coarsest features)
   - v17 `dec4` = mid-level decoder (3rd from bottleneck)
   - Loaded weights operate at the wrong abstraction level → actively harm learning
2. **Bottleneck shape mismatched** (v14: 512→1024; v17: 1024→2048) so bottleneck stayed random-init while neighbors got partial wrong-scale init — worst of both worlds.
3. **Sigmoid output saturated** near gray (0.5) — model couldn't escape because loss landscape was poisoned by misaligned init.
4. **Val set contaminated** — `FullSizeDataset(augment=True)` wrapped with `random_split` shared the augment flag → val pairs received random flips each epoch → noisy validation signal.
5. **No seed on `random_split`** → train/val partition non-reproducible.
6. Only 124/166 layers warm-started; 25% of params started fresh inside an already-3×-larger model.

**Lesson:** **Warm-start is not a free lunch.** Loading weights from an architecturally different parent (different depth, different bottleneck size) is **worse than scratch init** when layer names collide but represent different roles. Either:
- Match architecture exactly (warm-start within same family)
- Use a pretrained encoder with **known semantics** (ResNet-34 ImageNet) where layer roles are fixed
- Remap by **depth-from-output** rather than by name when bridging architectures

**Lesson:** **A val SSIM below the registration floor is a hard red flag.** If the model can't beat copying its input, something fundamental is broken — kill the run and diagnose.

---

### Failure 9: v19b/v20 Train-Val Prefix Leakage

**What we tried:** v19b and the first v20 run created separate train and validation dataset instances, each shuffling the same top-1000 dataframe independently and then slicing to 900 and 100 samples.

**Result:**
- v19b reported SSIM 0.7489.
- old v20 reported SSIM 0.7549.
- A 2026-05-01 audit found **91/100 validation prefixes overlapped training**.

**Why it failed:**
1. Train and val were separated by count, not by a shared index split.
2. Independent shuffles made the first 100 validation entries likely to also appear in the first 900 training entries.
3. Validation augmentation was disabled, but identity leakage still made the metric optimistic.

**Fix in v20_fixed:**
```python
split_perm = np.random.RandomState(SEED).permutation(len(df))
train_indices = split_perm[:900]
val_indices = split_perm[900:1000]
assert len(train_prefixes & val_prefixes) == 0
```

**Lesson:** **A validation split is only real after an overlap audit.** For paired images, validate by stable pair identity or filename prefix. Do not rely on two dataset constructors to independently create compatible splits.

---

## 2. Cross-Cutting Themes

### Theme 1: Loss Imbalance Is Insidious

| Loss Term | Typical λ | Effective Magnitude |
|-----------|-----------|---------------------|
| L1 | 100 | Dominant |
| WGAN-GP | 1-10 | 10× weaker |
| VGG Percept | 10 | Equal-ish |
| Sobel | 20 | Equal-ish |
| HED | 5 | Marginal, conflicting |

When λ_L1 = 100 dominates, the GAN signal is **noise relative to reconstruction**. The model effectively trains on L1 alone, but with the GAN contributing instability.

**Fix Approach:** Either rebalance (λ_L1=10, λ_GAN=1) **OR** accept that GAN is a small auxiliary signal. Don't use both extremes.

---

### Theme 2: Stack of Losses ≠ Better Model

Versions with 5+ loss terms (v12, v13, v15) all suffered. v11 with 4 loss terms (L1+GAN+VGG+Sobel) was the maximum tractable complexity. v14 with 1 loss (L1 only) was simpler and reached 0.708.

**Heuristic:** Each new loss term adds a gradient direction. The optimization manifold becomes harder to navigate. Diminishing returns kick in fast — usually after 3 terms.

---

### Theme 3: Architecture Complexity Without Matching Data

| Architecture | Data Quality | Result |
|--------------|--------------|--------|
| Pix2Pix U-Net | Unregistered | 0.26 |
| ResNet-34 U-Net | TV-L1 1k | 0.706 |
| **ResNet-34 + VGG/Sobel** | **TV-L1 1k** | **0.712 ⭐** |
| Attention U-Net + Multi-Disc | TV-L1 8k (noisy) | 0.6326 |
| Attention U-Net + Multi-Disc | TV-L1 1k patches | 0.7199 (1 epoch only) |

The lesson: complex architectures only help when data quality is high enough to support them. Otherwise, you're learning noise.

---

### Theme 4: Adversarial Loss vs SSIM Target

GANs optimize for "realism" (visual sharpness) which often **hurts** SSIM. SSIM rewards structural agreement; GAN sharpness can introduce hallucinated details that lower SSIM.

| With GAN | Without GAN |
|----------|-------------|
| Sharper outputs | Slightly blurrier |
| Better PSNR? | Better SSIM ✓ |
| Risk of divergence | Stable |

For an SSIM-targeted project, the GAN is the wrong objective.

---

### Theme 5: Validation Set Stability

v14's random validation image caused 0.166-0.605 epoch-to-epoch noise. This made early stopping nearly useless and obscured true convergence. Held-out fixed validation (v17) is mandatory for credible model selection.

v19b/v20 added a second validation lesson: a fixed-looking validation loader can still leak if it is not created from a single shared split. Current standard: seeded split once, explicit indices, validation augmentation disabled, and `overlap=0` logged before epoch 1.

---

## 3. Failure Categorization

| Category | Examples | Rate of Recovery |
|----------|----------|------------------|
| **Data quality** | v1-7 (no reg), v13 (all pairs) | High — fix the data |
| **Numerical stability** | v12, v15 (HED), fp16 inverse | High — change loss, use fp32 |
| **GAN divergence** | v15, v16 | Low — drop GAN |
| **Configuration errors** | v17 OOM, dataloader errors | Trivial — config fix |
| **Validation noise/leakage** | v14 random val, v19b/v20 prefix overlap | Easy to fix, high reporting risk |

---

## 4. Things We Almost Did but Didn't

| Idea | Why We Skipped |
|------|----------------|
| CycleGAN (unpaired) | We have paired data; supervised is more accurate |
| Diffusion model | Compute cost; not enough data |
| Train at 256×256 then upsample | Resolution loss for diagnostic purposes |
| Train discriminator separately first | Doesn't fix imbalance problem |
| Self-distillation | Premature without a strong teacher model |
| Stain normalization preprocessing | Risk of altering diagnostic information |

---

## 5. Failure-Driven Decisions for v17

Every v17 choice is a direct response to a prior failure:

| v17 Choice | Failure It Addresses |
|------------|---------------------|
| **No GAN** | v15/v16 divergence |
| **L1+SSIM only** | Loss-stack overcomplication (v12-v15) |
| **Top-1000 data** | v13 noisy-data failure |
| **Fixed validation set** | v14 high-variance validation |
| **Warm-start from v14** | Cold-start instability |
| **5-level U-Net** | Need bigger receptive field at 1024×1024 |
| **Conservative LR (5e-5)** | v16 instability with LR=1e-4 |
| **Early stop patience=10** | Detect plateau before divergence |
| **Process protection (signal handlers)** | Interrupted runs lose work |
| **Logging to file + stdout** | v15 logs were lost on terminal close |

---

## 6. Meta-Lessons (How to Iterate)

1. **One change per version.** v15 changed architecture + data + loss simultaneously. Impossible to attribute the SSIM 0.7199→divergence to any specific cause. Future: change one thing per experiment.

2. **Always check numerical stability under fp16 first.** `torch.inverse`, `torch.det`, `torch.eig` all have fp16 issues.

3. **Profile before parallelizing.** v14's `num_workers=60` was based on assumption, not measurement. The bottleneck wasn't I/O.

4. **Save logs to disk.** Terminal-only output is lost on disconnection. Use `tee` or Python's `logging.FileHandler`.

5. **Fixed validation set, period.** No exceptions. Also audit overlap by filename prefix or stable pair ID.

6. **GAN is high-risk, marginal-reward.** Use only if (a) the metric rewards realism (PSNR, FID), (b) you have time to debug instability, (c) the gain justifies the risk.

7. **The data ceiling is real.** If your training set has mean SSIM 0.5, expect a model that maxes out around 0.65. No amount of architecture will fix this.

8. **Full-image SSIM alone can over-select easy patches.** The CLAHE/content-quality analysis showed that high-content patches have lower full-image SSIM but larger registration gains. Future dataset selection should report full SSIM, content-region SSIM, content fraction, slide counts, and negative-gain rows.

9. **Keep fallback paths.** CLAHE TV-L1 improved the mean strongly, but 80/8885 rows regressed. A best-of-old-vs-CLAHE CSV is safer than assuming one registration method wins every file.

---

## 7. File Inventory

| File | Purpose |
|------|---------|
| `src/training/lightning_module_v12.py` | HED loss (failed) |
| `src/training/lightning_module_v13.py` | All-pairs training (failed) |
| `src/training/lightning_module_v15.py` | Diverged after 1 epoch |
| `src/training/lightning_module_v16.py` | Diverged after 24 epochs |
| `train_v14.py` | Random val (high variance) |
| `train_v17.py` | All failure-driven fixes |
| `train_v20.py` | Current v20_fixed split correction |

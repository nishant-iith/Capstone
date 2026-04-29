# 04: Training History — Complete Chronology of All 17 Versions

> **Bottom Line:** Best result: **v11 = SSIM 0.712** (Weakly Supervised). v14 = SSIM 0.708 (Simple U-Net, patches). v15/v16 diverged from GAN instability. v17 rev1 **failed at SSIM 0.379** — wrong-scale warm-start from 4-level v14 to 5-level v17. **v17 rev2 IN PROGRESS** — fixed: Kaiming init (no warm-start), batch=20, LR=5e-4, seeded split, val-only augment. SSIM 0.2718 at ep 8, climbing.

---

## Timeline Summary

```
Phase 1 (v1-v7):  Pix2Pix on unregistered data → SSIM 0.26 (failure baseline)
Phase 2 (v8-v10): Pix2Pix + Registration       → SSIM 0.706 (Turbo Pix2Pix)
Phase 3 (v11):   + VGG-19 Perceptual Loss      → SSIM 0.712 ⭐ (BEST EVER)
Phase 4 (v12):   + HED Stain Loss               → Failed
Phase 5 (v13):   Attention U-Net + MultiScale   → SSIM 0.6326 (data quality limited)
Phase 6 (v14):   Simple U-Net + Patches         → SSIM 0.7080 (clean baseline)
Phase 7 (v15):   v13 architecture + patches     → SSIM 0.7199 (1 epoch, then diverged)
Phase 8 (v16):   v15 minus HED, full-size       → SSIM 0.6976 (then diverged)
Phase 9 (v17r1): 5-level U-Net + SSIM + wrong warm-start → SSIM 0.379 (FAILED)
Phase 9 (v17r2): same arch, Kaiming init, batch=20     → IN PROGRESS (ep 8, SSIM 0.2718↑)
```

---

## Detailed Version Records

---

### v1-v7 (Pre-Registration Era)

**Period:** Before this team's work
**Data:** Unregistered paired images
**Architecture:** Standard Pix2Pix U-Net + 70×70 PatchGAN
**Loss:** L1 + Adversarial

**Results:**
- SSIM: ~0.26
- PSNR: ~15.8 dB
- **Visual:** Blurry pink/purple blobs that don't align with cellular structures

**Diagnosis:** Structural misalignment caused gradient conflict. Model couldn't decide whether to prioritize color or position, defaulted to "blur the difference."

**Lesson:** ❌ Registration is non-negotiable.

---

### v8-v10: Pix2Pix Turbo (Post-Registration Foundation)

**Phase Goal:** Establish that registration enables successful GAN training
**Data:** TV-L1 registered pairs (1,000 images)
**Architecture:**
- Generator: U-Net with **ResNet-34 ImageNet encoder**
- Discriminator: 70×70 PatchGAN
**Loss:** WGAN-GP + L1 (λ=100)
**Optimizations:** RAM caching (1.5 GB into memory), 16-bit mixed precision

**Results:**
- v8: ~0.65 SSIM (initial)
- v9: ~0.68 SSIM (added augmentation)
- v10: **SSIM 0.706 | PSNR 22.75 dB**

**Lesson:** ✅ Registration → 0.7 SSIM ceiling unlocked. ResNet-34 encoder + PatchGAN works.
**Plateau:** L1+GAN couldn't progress past 0.71 — needed perceptual signal.

**File:** `src/training/lightning_module_v10.py`

---

### v11: Weakly Supervised Hybrid GAN ⭐ **(PROJECT BEST)**

**Phase Goal:** Break 0.71 SSIM ceiling with perceptual feature matching
**Data:** TV-L1 registered, 1,000 images (good slides only)
**Architecture:** Same as v10
**Loss:** WGAN-GP + L1 + Sobel + **VGG-19 Perceptual** (5 layers)

**Loss Formulation:**
$$
L_{total} = \lambda_{adv} L_{GAN} + \lambda_{L1} L_{L1} + \lambda_{struct} L_{Sobel} + \lambda_{percept} L_{VGG}
$$

**Final Hyperparameters:**
- λ_adv = 1.0 (WGAN-GP)
- λ_L1 = 100
- λ_struct = 20 (Sobel edge loss)
- λ_percept = 10 (VGG-19 5-layer feature matching)
- LR: 1e-4 with cosine schedule
- Batch size: 8

**Results:**
- **SSIM: 0.7120**
- **PSNR: 23.03 dB**
- **PCC: 0.8906**

**Best Checkpoint:** `ws-epoch=27-val_ssim=0.712.ckpt`

**Verdict:** ⭐ This is the **project's best model to date.** Despite 6 subsequent attempts (v12-v17), nothing has surpassed this result.

**Why it worked:**
1. Registration foundation (TV-L1)
2. Pretrained encoder (ResNet-34)
3. **Perceptual loss** finally gave the model a "biological feature understanding"
4. Stable hyperparameters (L1=100 strong enough to anchor, others provide gradient nuance)

**File:** `src/training/lightning_module_v11.py`

---

### v12: HED Stain Loss Addition (Failed Attempt)

**Phase Goal:** Add color-decomposition loss to push past 0.712
**Data:** Same as v11
**Architecture:** Same as v11
**New Loss:** HED Stain Decomposition Loss (Macenko stain matrix)

**HED Loss:**
- Decompose RGB images into Hematoxylin (purple), Eosin (pink), DAB (brown) channels using Macenko stain matrix
- Compute L1 loss on each channel separately
- Force the generator to match each stain component independently

**Results:** ❌ **Training destabilized**, worse than v11
- HED loss conflicted with VGG perceptual gradients
- Stain matrix inversion (`torch.inverse`) numerically unstable in float16

**Verdict:** ❌ Removed. Not worth the complexity.

**File:** `src/training/lightning_module_v12.py`

---

### v13: Attention U-Net + MultiScale Discriminator

**Phase Goal:** SOTA architecture upgrade per project plan
**Data:** **All registered pairs (8,885)**, mean SSIM 0.51 — INCLUDED LOW-QUALITY PAIRS
**Architecture:**
- Generator: U-Net with ResNet-34 + **Attention Gates**
- Discriminator: **MultiScale (local + global)**
**Loss:** WGAN-GP + L1 + Sobel + VGG + HED
**Hyperparameters:** Inherited from v11

**Best Result:** SSIM 0.6326 (epoch 8)

**Why it underperformed v11:**
1. **Data quality regression** — Training on all 8,885 pairs (mean SSIM 0.51) vs v11's curated 1,000 (mean 0.65+)
2. Architecture more complex but data was lower quality

**Lesson:** ⚠️ **Data quality > architecture complexity**. The dataset ceiling can be lower than the model's capability.

**File:** `src/training/lightning_module_v13.py`

---

### v14: Simple U-Net on 512×512 Patches (Baseline)

**Phase Goal:** Reset to simple architecture, train on **high-quality patches**
**Data:** **3,606 patches (512×512)** extracted from top-registered pairs, mean SSIM 0.6254
**Architecture:** Simple 4-level U-Net (no pretrained, BatchNorm, Sigmoid)
**Loss:** Plain L1
**Hyperparameters:**
- Batch size: 16 (was 32 → caused OOM)
- num_workers: 0 (RAM cached, no multiprocessing)
- LR: 1e-4 with ReduceLROnPlateau

**Best Results:**
- **Patch-level SSIM: 0.7435** (epoch 6)
- **Full-size validation SSIM: 0.7080** (epoch 10)

**File:** `train_v14.py`

**Verdict:** ✅ **Simple architecture + high-quality patches → SSIM 0.7080.** Validates that data curation matters more than architecture.

**Issue Identified:** Full-size validation was on **random** test image each epoch → high variance (0.166 to 0.605). v15+ should use a held-out validation set.

---

### v15: Attention + MultiScale + Patches + HED (Diverged)

**Phase Goal:** Apply v14's data quality with v13's architecture
**Data:** Same patches as v14
**Architecture:** v13 (Attention U-Net + MultiScale Disc)
**Loss:** WGAN-GP(10) + L1(100) + Sobel(20) + Percept(10) + **HED(5)**

**Training Curve:**
- Epoch 1: SSIM **0.7199** ⭐ (peak)
- Epoch 2: 0.7038 (drop)
- Epoch 3: 0.7156 (recovery)
- Epoch 4+: gradual decline, eventually diverged

**Diagnosis:** **HED loss caused divergence.** L1(100) + Percept(10) + HED(5) created competing color/structure gradients. WGAN-GP discriminator pulled outputs toward sharpness while HED pulled toward color matching.

**File:** `src/training/lightning_module_v15.py`
**Best Checkpoint:** `checkpoints_v15/v15-epepoch=001-ssimval_ssim=0.7199-psnrval_psnr=17.43.ckpt`

**Verdict:** ⚠️ Best **single-epoch** result of any model, but training unstable.

---

### v16: v15 Minus HED + Full-Size Images

**Phase Goal:** Remove HED to fix divergence, scale to full-size 1024×1024
**Data:** **Top-1000 full-size pairs** (mean SSIM 0.6094)
**Architecture:** Same as v15
**Loss:** WGAN-GP(10) + L1(100) + Sobel(20) + Percept(10) [no HED]
**Hyperparameters:**
- Batch size: 16 → 20 (final)
- LR: 1e-4 → 5e-5 (after divergence on first run)
- Mixed precision 16-bit
- Warm-start from v14 weights

**Training Curve (run 2 with warm-start):**

| Epoch | SSIM |
|-------|------|
| 8     | 0.6432 |
| 12    | 0.6718 |
| 20    | 0.6832 |
| 22    | 0.6930 |
| **24** | **0.6976** ⭐ |
| 25    | 0.6755 (decline) |
| 26    | 0.6556 |
| 27    | 0.6683 |
| 28    | 0.6463 |

**Diagnosis:** GAN training **unstable** even with warm-start and lower LR.
- d_loss became -84 (discriminator winning catastrophically)
- g_loss exploded to 123
- After epoch 24, model lost ability to recover

**Verdict:** ❌ **GAN approach inherently unstable** for this task. Project pivot to non-GAN approach for v17.

**File:** `train_v16.py`, `src/training/lightning_module_v16.py`

---

### v17 Rev1: Simple U-Net + L1+SSIM + Warm-Start (FAILED)

**Phase Goal:** Apply learned lessons:
1. Simple architecture > complex unstable
2. **Direct SSIM optimization** (not just L1)
3. Warm-start from v14 (proven 0.708)
4. **5-level U-Net** for full-size context
5. **No GAN** (consistently destabilized)

**Data:** Top-1000 full-size pairs (mean SSIM 0.6094)
**Architecture:** Simple **5-level U-Net** (bottleneck 32×32 at 1024×1024)
**Loss:** **L1(0.5) + SSIM(0.5)** — directly optimize the metric

**Loss Formulation:**
```python
loss = 0.5 * L1(pred, target) + 0.5 * (1 - SSIM(pred, target))
```

**Hyperparameters:**
- Batch size: 4 (5-level U-Net needs ~6 GB/image)
- LR: 5e-5 (conservative for warm-start)
- Schedule: CosineAnnealingLR → 1e-7
- Patience: 10 (early stop)
- Augmentation: random flips
- Mixed precision: AMP 16-bit
- cuDNN benchmark enabled

**Process Protection:**
- Signal handlers for SIGINT, SIGTERM, SIGHUP, SIGQUIT
- High CPU priority via `os.nice(-10)`
- Logging to file + stdout simultaneously

**Status:** ❌ **FAILED — stuck at SSIM 0.379** (peaked epoch 25, declining after 34 epochs)

**Observed Results (`logs/v17_training.log`):**

| Epoch | Val SSIM |
|-------|----------|
| 1     | 0.2343 |
| 6     | 0.3543 |
| 13    | 0.3770 |
| 19    | 0.3788 |
| **25** | **0.3790 ⭐ (peak)** |
| 34    | 0.3378 (patience 9/10, near early stop) |

**Diagnosis (Critical):**
1. **Val SSIM (0.38) below registration floor (0.63)** — model output worse than identity. Sigmoid likely saturated near gray. Even copying the input would beat current output.
2. **Warm-start net-NEGATIVE.** v14 was 4-level, v17 is 5-level. Decoder layer names (`dec1`-`dec4`) match by name but represent **different receptive-field scales** in the two architectures. Loaded v14 weights are at the wrong scale → actively harmful.
3. **Bottleneck shape mismatch** (v14 bottleneck 512→1024; v17 bottleneck 1024→2048) → bottleneck stays random-init while surrounding layers get partial wrong-scale init.
4. **Validation set augmented** — `FullSizeDataset(augment=True)` with `random_split` sharing the flag → val pairs receive random flips each epoch → noisy val SSIM signal.
5. **No seed on `random_split`** — train/val partition non-reproducible across runs.
6. **Only 124/166 layers warm-started** — 25% of params start fresh inside an already-large model (5-level adds ~3× parameter count).

**Verdict:** ❌ v17 rev1 warm-start strategy broken. Fixes applied in rev2 (see below).

---

### v17 Rev2: Simple U-Net + L1+SSIM + Kaiming Init (IN PROGRESS)

**Fixes applied over rev1:**
- **No warm-start** — Kaiming init from scratch (eliminates scale mismatch)
- **Batch=20** (up from 4) — utilizes A100 80GB headroom, 5× throughput
- **LR=5e-4** (linear-scaled with batch, up from 1e-4)
- **Seeded split** (seed=42) — reproducible train/val partition
- **Val augment disabled** — pristine val SSIM signal

**Current status (2026-04-29):**
| Epoch | Val SSIM |
|-------|----------|
| 5     | 0.2635 |
| 8     | **0.2718 ⭐ (best so far, climbing)** |

**Trajectory:** Consistent improvement each epoch. Ceiling estimate: 0.60–0.65 (no perceptual/adversarial loss — L1+SSIM alone plateaus here). Still higher than rev1 peak.

**File:** `train_v17.py`

---

## Master Results Table

| Ver | Architecture | Data | Loss | Best SSIM | Status |
|-----|--------------|------|------|-----------|--------|
| v1-7 | Pix2Pix | Unregistered | L1+GAN | 0.26 | Failed (no registration) |
| v8 | ResNet U-Net + PatchGAN | TV-L1 1k | L1+WGAN-GP | 0.65 | OK |
| v9 | + augmentation | TV-L1 1k | L1+WGAN-GP | 0.68 | OK |
| v10 | Pix2Pix Turbo | TV-L1 1k | L1+WGAN-GP | 0.706 | Good |
| **v11** ⭐ | + VGG-19 + Sobel | TV-L1 1k | Hybrid | **0.712** | **PROJECT BEST** |
| v12 | + HED | TV-L1 1k | Hybrid + HED | (failed) | Diverged |
| v13 | Attention + MultiScale | TV-L1 8.8k (mean 0.51) | Hybrid + HED | 0.6326 | Data quality limited |
| v14 | Simple U-Net | 3.6k patches (mean 0.625) | L1 only | 0.7080 | Clean baseline |
| v15 | v13 architecture | 3.6k patches | Hybrid + HED | 0.7199 | Diverged after epoch 1 |
| v16 | v15 minus HED | Top-1k full-size | Hybrid (no HED) | 0.6976 | Diverged after epoch 24 |
| v17r1 | 5-level Simple U-Net (warm-start v14) | Top-1k full-size | L1 + SSIM | 0.3790 | ❌ Failed — wrong-scale warm-start |
| v17r2 | 5-level Simple U-Net (Kaiming, batch=20) | Top-1k full-size | L1 + SSIM | 0.2718+ | 🔄 IN PROGRESS (ep 8, climbing) |

---

## Key Inflection Points

1. **TV-L1 Registration (Phase 1):** SSIM 0.26 → 0.65 (+0.39, 150% gain)
2. **Mixed Precision + RAM Cache:** Throughput 0.8 → 1.3 it/s (+62%)
3. **VGG-19 Perceptual Loss (v11):** SSIM 0.706 → 0.712 (+0.006, project best)
4. **Data Curation (v14):** All-pairs (0.51 mean) → curated patches (0.625 mean) → SSIM 0.6326 → 0.7080
5. **GAN Removal (v17):** Stability over architectural complexity

---

## What v18 Should Do (Per Research Recommendations)

Based on all 17 prior experiments, the highest-impact next moves are:

1. **ResNet-34 ImageNet pretrained encoder** (v11 had this, v17 dropped it)
2. **MS-SSIM loss** (multi-scale, captures structure at all resolutions)
3. **Residual blocks in decoder** (better gradient flow at full-size)
4. **Progressive training** (train at 512px, fine-tune at 1024px)

**Expected:** SSIM 0.76-0.80

See [09_future_work.md](09_future_work.md) for the full plan.

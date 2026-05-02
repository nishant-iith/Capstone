# 04: Training History — Complete Chronology Through Final Ensemble

> **Bottom Line:** The best current pipeline is **CLAHE TV-L1 + TTA4 weighted v20/v22A/v21B ensemble = SSIM 0.7838, PSNR 25.16, PCC 0.8794**. v20_fixed remains the best old-registered fallback at SSIM 0.7634 with TTA4. Historical v19b (0.7489) and old v20 (0.7549) were stable and useful, but their validation split leaked: 91/100 validation prefixes overlapped training.

---

## Timeline Summary

```
Phase 1 (v1-v7):  Pix2Pix on unregistered data → SSIM 0.26 (failure baseline)
Phase 2 (v8-v10): Pix2Pix + Registration       → SSIM 0.706 (Turbo Pix2Pix)
Phase 3 (v11):   + VGG-19 Perceptual Loss      → SSIM 0.712
Phase 4 (v12):   + HED Stain Loss               → Failed
Phase 5 (v13):   Attention U-Net + MultiScale   → SSIM 0.6326 (data quality limited)
Phase 6 (v14):   Simple U-Net + Patches         → SSIM 0.7080 (clean baseline)
Phase 7 (v15):   v13 architecture + patches     → SSIM 0.7199 (1 epoch, then diverged)
Phase 8 (v16):   v15 minus HED, full-size       → SSIM 0.6976 (then diverged)
Phase 9  (v17r1): 5-level U-Net + SSIM + wrong warm-start → SSIM 0.379 (FAILED)
Phase 9  (v17r2): same arch, Kaiming init, batch=20     → abandoned (ceiling ~0.62)
Phase 10 (v19):   DenseUNet + ResNet-34 + L1+MS-SSIM+VGG → diverged (FAILED)
Phase 10 (v19b):  DenseUNet + ResNet-34 + L1 only        → SSIM 0.7489 (leaky split)
Phase 11 (v20):   ConvNeXt-Base LAION + L1 + elastic aug → SSIM 0.7549 (leaky split)
Phase 11 (v20_fixed): same v20 recipe + clean split      → SSIM 0.7606 clean
Phase 12 (v21A): Hibou-B frozen feature decoder          → SSIM 0.7605 clean
Phase 13 (v22A): v20 warm-start + content-quality CLAHE  → internal SSIM 0.7655; fixed CLAHE TTA 0.7807
Phase 14 (v21B): Hibou-B warm-start + content-quality    → internal SSIM 0.7544; fixed CLAHE TTA 0.7790
Final ensemble: 0.20*v20 + 0.60*v22A + 0.20*v21B TTA4 → fixed CLAHE SSIM 0.7838
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

### v11: Weakly Supervised Hybrid GAN

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

**Verdict:** This was the first stable high-quality model and the long-running baseline. Later clean-split v20_fixed has surpassed its SSIM, but v11 remains important because it proved the value of registration, pretrained encoding, and perceptual loss.

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
| v11 | + VGG-19 + Sobel | TV-L1 1k | Hybrid | 0.712 | Historic clean baseline |
| v12 | + HED | TV-L1 1k | Hybrid + HED | (failed) | Diverged |
| v13 | Attention + MultiScale | TV-L1 8.8k (mean 0.51) | Hybrid + HED | 0.6326 | Data quality limited |
| v14 | Simple U-Net | 3.6k patches (mean 0.625) | L1 only | 0.7080 | Clean baseline |
| v15 | v13 architecture | 3.6k patches | Hybrid + HED | 0.7199 | Diverged after epoch 1 |
| v16 | v15 minus HED | Top-1k full-size | Hybrid (no HED) | 0.6976 | Diverged after epoch 24 |
| v17r1 | 5-level Simple U-Net (warm-start v14) | Top-1k full-size | L1 + SSIM | 0.3790 | ❌ Failed — wrong-scale warm-start |
| v17r2 | 5-level Simple U-Net (Kaiming, batch=20) | Top-1k full-size | L1 + SSIM | ~0.60 ceiling | ❌ Abandoned — ceiling too low |
| v19 | DenseUNet + ResNet-34 | Top-1k full-size | L1+MS-SSIM+VGG | diverged | ❌ Failed |
| v19b | DenseUNet + ResNet-34 | Top-1k full-size | L1 only | 0.7489 | Stable but leaky split |
| v20 | ConvNeXt-Base LAION-2B U-Net | Top-1k full-size | L1 + elastic aug | 0.7549 | Stable but leaky split |
| **v20_fixed** ⭐ | **ConvNeXt-Base LAION-2B U-Net** | **Top-1k full-size, clean 900/100 split** | **L1 + elastic aug** | **0.7606** | **Clean old-registered baseline single model** |
| v21A | Hibou-B frozen feature decoder | Top-1k full-size, clean split | L1 + elastic aug | 0.7605 | Matched v20, ensemble member |
| v22A | v20 warm-start ConvNeXt U-Net | content-quality CLAHE top-1000 | L1 + elastic aug | 0.7655 internal; 0.7807 TTA4 fixed CLAHE eval | Best single model on CLAHE fixed eval |
| v21B | Hibou-B decoder warm-start | content-quality CLAHE top-1000 | L1 + elastic aug | 0.7544 internal; 0.7790 TTA4 fixed CLAHE eval | Useful ensemble complement, not best single |
| Final ensemble | v20 + v22A + v21B TTA4 | CLAHE fixed eval | Inference blend | 0.7838 SSIM | Best overall fixed-eval result |

---

### v19b: DenseUNet + ResNet-34 + L1-Only (Historical Leaky Baseline)

**Phase Goal:** Apply v19 arch with stable L1-only loss (v19 full-loss diverged)
**Data:** Top-1000 full-size pairs (mean SSIM 0.6094)
**Architecture:** `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet")` — ImageNet pretrained encoder, sigmoid output
**Loss:** **L1 ONLY** — simplicity → stability
**Hyperparameters:**
- Batch: 8, LR: 1e-4, AdamW (weight_decay=1e-4)
- Schedule: CosineAnnealingLR (T_max=80, eta_min=1e-7)
- Epochs: 80, patience: 15, grad clip: 1.0
- Mixed precision AMP, cuDNN benchmark

**Training Curve:**

| Epoch | Val SSIM |
|-------|----------|
| 1 | 0.6467 |
| 5 | 0.7226 |
| 17 | 0.7394 |
| 40 | 0.7462 |
| 61 | 0.7484 |
| 76 | 0.7486 |
| **80** | **0.7489 ⭐** |

**Behavior:** Monotonic improvement, no divergence. Loss declined 0.1938 → 0.0301. Cosine LR kept squeezing gains to final epoch.

**Verdict:** Stable and valuable as an architecture/loss signal, but no longer a clean validation claim. A later audit found the train and validation datasets were independently shuffled and sliced, which produced 91/100 overlapping validation prefixes. Keep the lesson: pretrained encoder + L1 + cosine LR is stable. Do not report 0.7489 as clean generalization.

**File:** `train_v19b.py`, model: `models/v19b_model.pth`, best ckpt: `checkpoints/v19b/v19b_e080_ssim0.7489.pth`

---

### v20: ConvNeXt-Base LAION-2B + L1 + Elastic Aug (Historical Leaky Baseline)

**Phase Goal:** Test whether a stronger general-purpose pretrained encoder beats ResNet-34 while keeping the stable L1-only recipe.
**Data:** Top-1000 full-size TV-L1 pairs
**Architecture:** `smp.Unet(encoder_name="tu-convnext_base.clip_laion2b", encoder_weights="laion2b")`
**Loss:** L1 only
**Augmentation:** Elastic deformation on the unstained input only (`alpha=60`, `sigma=6`, `p=0.5`), plus paired geometric augmentation

**Historical Result:** SSIM 0.7549 at epoch 74, best checkpoint `checkpoints/v20/v20_e074_ssim0.7549.pth`.

**Caveat:** Same overlapping-split bug as v19b. Treat 0.7549 as evidence that ConvNeXt + L1 + elastic aug is promising, not as a clean validation score.

**File:** `train_v20.py` before the 2026-05-01 split fix, model: `models/v20_model.pth`

---

### v20_fixed: ConvNeXt-Base LAION-2B + L1 + Elastic Aug (Clean Baseline)

**Phase Goal:** Rerun v20 correctly with a single deterministic split and no train/val prefix overlap.
**Data:** Top-1000 full-size TV-L1 pairs, split once with seed 42 into 900 train / 100 val.
**Architecture:** Same as v20: ConvNeXt-Base LAION-2B encoder + U-Net decoder.
**Loss:** L1 only.
**Augmentation:** Elastic deformation on unstained training inputs only; validation augmentation disabled.

**Split Fix:**
```python
split_perm = np.random.RandomState(SEED).permutation(len(df))
train_indices = split_perm[:900]
val_indices = split_perm[900:1000]
assert len(train_prefixes & val_prefixes) == 0
```

**Final Clean Result (completed 2026-05-01):**

| Epoch | Val SSIM | PSNR | PCC | Notes |
|-------|----------|------|-----|-------|
| 23 | 0.7538 | 24.58 | 0.8590 | earlier snapshot |
| 54 | 0.7597 | 24.88 | 0.8640 | late-stage gain |
| **77** | **0.7606** | **24.92** | **0.8652** | **best checkpoint** |
| 80 | 0.7606 | 24.93 | 0.8653 | final epoch |

**Verdict:** Best clean single-model baseline. It beats the old leaky v20 score under an audited `overlap=0` split and is the strongest practical model for the app. It also remains slightly ahead of the v21A Hibou-B frozen-feature experiment.

**Files:** `train_v20.py`, `logs/v20_fixed_training.log`, `checkpoints/v20_fixed/`, final model path `models/v20_fixed_model.pth`

---

### v21A: Hibou-B Frozen Feature Decoder

**Phase Goal:** Test whether histology-pretrained Hibou-B features beat the clean ConvNeXt v20 baseline.

**Setup:**
- Backbone: `histai/hibou-b`, loaded through Hugging Face with `trust_remote_code=True`.
- Feature shape: DINOv2-style tokens with registers, 768-dimensional hidden state.
- Training design: frozen Hibou-B encoder plus high-resolution input-skip decoder.
- Data: same clean 900/100 split discipline, no train/val overlap.

**Result:** Best epoch 97: SSIM 0.7605, PSNR 24.77, PCC 0.8634. Early stopped at epoch 117.

**Inference/ensemble ablation:**

| Setup | SSIM | PSNR | PCC |
|-------|------|------|-----|
| v20 base | 0.7606 | 24.92 | 0.8652 |
| v20 TTA4 | 0.7634 | 25.03 | 0.8684 |
| v21A base | 0.7605 | 24.77 | 0.8634 |
| v21A TTA4 | 0.7626 | 24.86 | 0.8659 |
| 55/45 v20/v21A TTA ensemble | **0.7649** | **25.11** | **0.8710** |

**Verdict:** Hibou-B is technically valid and complementary enough for a small ensemble gain, but it did not beat v20_fixed as a single model. Because it also introduces gated-model and redistribution complexity, v20 remains the primary deployable model.

**Files:** `train_v21a_hibou_b.py`, `logs/v21a_hibou_b_training.log`, `eval_tta_ensemble.py`, `eval_weighted_ensemble_fast.py`

---

### Registration/Data Upgrade: CLAHE TV-L1 and Content-Quality CSVs

**Phase Goal:** Improve paired-label quality before changing the model. The central hypothesis was that v20 may be capped by registration noise more than architecture capacity.

**Registration result:** Full CLAHE TV-L1 registration on 8,885 pairs improved all-pair mean SSIM from 0.4134 to 0.5045, a mean gain of +0.0911. The top-1000 mean increased from 0.6094 to 0.6423 while preserving all 13 slides.

**Content-aware selection result:** Foreground tissue masking was not useful because these patches are essentially all tissue, but edge/content-aware scoring changed rankings substantially. The selected content-quality top-1000 has mean full RGB SSIM 0.5601, mean content-gray SSIM 0.6223, mean content fraction 0.5516, mean gain +0.1152, and zero negative-gain rows.

**Training CSVs created:**
- `data/processed/training_csv_variants/registered_bestof_old_clahe_top1000.csv`
- `data/processed/training_csv_variants/registered_clahe_positive_top1000.csv`
- `data/processed/content_quality_csvs/content_quality_minrgb0.50_positive_top1000.csv`

**Interpretation:** Full-SSIM top-1000 is cleaner/easier, but content-quality top-1000 is richer and has much larger registration improvement. These validation scores will not be directly comparable unless evaluated on a fixed external validation set, because the content-quality validation split is harder and compositionally different.

---

### v22A: v20 Warm-Start on Content-Quality Top-1000 (Active)

**Phase Goal:** Test whether the new high-content, positive-gain CLAHE dataset can improve a v20-style model quickly.

**Setup:**
- Architecture: same ConvNeXt-Base LAION-2B U-Net as v20_fixed.
- Initialization: warm-start from `models/v20_fixed_model.pth`.
- Data: `data/processed/content_quality_csvs/content_quality_minrgb0.50_positive_top1000.csv`.
- Split: seed 42, 900 train / 100 val, `overlap=0`.
- LR: encoder 5e-6, decoder 5e-5.
- Max epochs: 40, patience 10.

**Early log snapshot (2026-05-02):**

| Epoch | Val SSIM | PSNR | PCC | Notes |
|-------|----------|------|-----|-------|
| 1 | 0.6974 | 21.62 | 0.9158 | warm-start adaptation begins |
| 2 | 0.7360 | 23.82 | 0.9318 | large immediate improvement |
| 3 | 0.7283 | 23.14 | 0.9314 | temporary dip |
| 4 | 0.7374 | 23.72 | 0.9331 | early best |
| 6 | 0.7514 | 24.04 | 0.9368 | large jump |
| 8 | 0.7528 | 24.33 | 0.9395 | new best |
| 9 | 0.7570 | 24.56 | 0.9409 | earlier best |
| 12 | 0.7531 | 24.55 | 0.9402 | patience 3/10 |
| 13 | 0.7563 | 24.61 | 0.9407 | patience 4/10; close to best |
| 14 | 0.7526 | 24.42 | 0.9409 | patience 5/10 |
| **15** | **0.7580** | **24.68** | **0.9421** | **new best; checkpoint saved** |

**Important comparison caveat:** This validation split is not the same as the v20_fixed validation split. It is content-rich and harder, so the internal SSIM should not be compared numerically against v20_fixed's 0.7606 as if it were the same test set. Use it to judge convergence and then run a fixed external evaluation for model claims.

**Files:** `train_v20_csv_variant.py`, `logs/v22a_content_quality_top1000_ft_training.log`, `checkpoints/v22a_content_quality_top1000_ft/`

## Key Inflection Points

1. **TV-L1 Registration (Phase 1):** SSIM 0.26 → 0.65 (+0.39, 150% gain)
2. **Mixed Precision + RAM Cache:** Throughput 0.8 → 1.3 it/s (+62%)
3. **VGG-19 Perceptual Loss (v11):** SSIM 0.706 → 0.712 (+0.006)
4. **Data Curation (v14):** All-pairs (0.51 mean) → curated patches (0.625 mean) → SSIM 0.6326 → 0.7080
5. **GAN Removal (v17):** Stability over architectural complexity
6. **ResNet-34 encoder + cosine LR (v19b):** SSIM 0.712 → 0.7489, but later found leaky
7. **ConvNeXt-Base LAION + elastic aug + clean split (v20_fixed):** SSIM 0.7606 with overlap=0
8. **Histology encoder ablation (v21A Hibou-B):** matched v20 at SSIM 0.7605, ensemble TTA reached 0.7649
9. **CLAHE registration/data curation (v22A/v21B input):** all-pair registration mean SSIM +0.0911; final CLAHE fixed-eval ensemble reached SSIM 0.7838

---

## What the Next Version Should Do

The next version should continue changing one thing at a time:

1. **Scale the proven CLAHE/content-quality recipe** from top-1000 to top-1500/top-2000 while preserving slide diversity.
2. **Test best-of-old-vs-CLAHE selection** with the same warm-start runner to isolate registration quality from content-quality ranking.
3. **Keep v22A as the deployable base** unless a larger clean-data run beats the final TTA ensemble on the same fixed eval.
4. **Use Hibou-style models as ensemble complements** until a domain encoder wins as a single deployable model.

See [09_future_work.md](09_future_work.md) and [10_out_of_the_box_improvements.md](10_out_of_the_box_improvements.md) for the current plan.

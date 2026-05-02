# 09: Future Work — The Path from Final Ensemble to 0.82+ SSIM

> **Bottom Line:** v22A and v21B are complete. The current best pipeline is CLAHE TV-L1 + weighted TTA ensemble `0.20*v20 + 0.60*v22A + 0.20*v21B`, scoring SSIM 0.7838, PSNR 25.16, PCC 0.8794 on the fixed CLAHE same-prefix evaluation. The next path to 0.82+ is broader validated CLAHE data, not another immediate architecture swap.

---

## 0. Current Immediate Roadmap (2026-05-02)

Completed:

1. **v22A content-quality warm-start:** best internal SSIM 0.7655; fixed CLAHE TTA4 SSIM 0.7807.
2. **v21B Hibou-B content-quality warm-start:** best internal SSIM 0.7544; fixed CLAHE TTA4 SSIM 0.7790.
3. **Final fixed eval:** best CLAHE ensemble SSIM 0.7838, PSNR 25.16, PCC 0.8794.

Next:

1. **Promote the final ensemble app** with `best_stain_app.py` and keep v20 fallback for old registered inputs.
2. **Scale CLAHE/content-quality training beyond top-1000** only after preserving a fixed holdout set.
3. **Evaluate on a truly external slide/slide-region split** before making clinical-grade claims.
4. **Improve registration failure handling** for the remaining difficult patches; registration remains the highest-leverage bottleneck.

---

## 1. Historical v18 Plan (Superseded by v19b/v20_fixed)

The section below is retained for research history. Its main ideas (pretrained encoder, no GAN, progressive training) were partly validated by later v19b/v20 work, but the version labels are stale. The active baseline is now v20_fixed, not v18.

**Goal:** Combine the best lessons from all prior versions into a single coherent model.

### v18 Architecture

```
Input (3, 1024, 1024)
    ↓
ResNet-34 ImageNet pretrained encoder (5 stages)
    ├─ stage1 → (64, 512, 512)
    ├─ stage2 → (64, 256, 256)
    ├─ stage3 → (128, 128, 128)
    ├─ stage4 → (256, 64, 64)
    └─ stage5 → (512, 32, 32)
    ↓
Bottleneck (1024, 32, 32) with residual blocks ×3
    ↓
U-Net decoder with attention gates + residual blocks
    ↓
Conv1×1 + Sigmoid → (3, 1024, 1024)
```

### v18 Loss

```python
loss = 0.4 * L1(pred, target) +
       0.4 * (1 - MS_SSIM(pred, target)) +
       0.2 * VGG_perceptual(pred, target)
# No GAN — proven unnecessary and unstable
```

### v18 Training Strategy

1. **Stage 1 — Pretrain at 256×256** (1 hour): warm initialization, learns coarse mapping
2. **Stage 2 — Fine-tune at 512×512** (2 hours): learns mid-scale structure
3. **Stage 3 — Final at 1024×1024** (8 hours): full-resolution refinement

### v18 Hyperparameters

| Param | Value |
|-------|-------|
| Batch size | 4 (1024px), 8 (512px), 32 (256px) |
| LR | 1e-4 → cosine to 1e-7 |
| Optimizer | AdamW (β1=0.9, β2=0.999) |
| Weight decay | 1e-4 |
| Mixed precision | bf16 (more stable than fp16) |
| Augmentation | Random flip + 90° rotation |
| Early stop patience | 15 |

**Predicted SSIM:** 0.76-0.80 (v17 result + 0.02-0.04 from architecture+loss improvements)

---

## 2. High-Impact Improvements (Ranked by Expected Δ SSIM)

### 2.1. ResNet-34 ImageNet Pretrained Encoder — **+0.01-0.02**

**Rationale:**
- v11 had this and reached 0.712
- v17 (currently training) dropped this for simplicity
- ImageNet features encode universal visual priors (edges, color, texture)
- Histology benefits from these despite domain gap

**Implementation:**
```python
import segmentation_models_pytorch as smp
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,
    activation="sigmoid",
)
```

**Risk:** Low. Well-validated.

---

### 2.2. MS-SSIM Loss — **+0.01-0.02**

**Rationale:**
- Single-scale SSIM only matches structure at one resolution
- MS-SSIM matches at 5 scales → captures fine, mid, coarse simultaneously
- Better aligns with human perceptual judgment
- Direct generalization of v17's approach

**Implementation:**
```python
from pytorch_msssim import MS_SSIM
ms_ssim = MS_SSIM(data_range=1.0, channel=3, win_size=11)
loss = 0.4 * L1(p, t) + 0.4 * (1 - ms_ssim(p, t)) + 0.2 * vgg(p, t)
```

**Risk:** Low. Drop-in replacement for SSIM.

---

### 2.3. Progressive Training — **+0.01**

**Rationale:**
- Models trained from scratch at 1024×1024 see fewer effective gradient updates per epoch
- Pretraining at 256×256 (16× faster) → fine-tune at 512×512 → final at 1024×1024
- Each stage warm-starts the next → smoother convergence
- Used in StyleGAN, ProGAN with strong results

**Implementation:**
```python
# Stage 1: 256×256, 50 epochs, batch 32
train_at_resolution(256, batch_size=32, epochs=50)
# Stage 2: 512×512, 30 epochs, batch 8, warm-start from stage 1
train_at_resolution(512, batch_size=8, epochs=30, warm_start="stage1.ckpt")
# Stage 3: 1024×1024, 50 epochs, batch 4, warm-start from stage 2
train_at_resolution(1024, batch_size=4, epochs=50, warm_start="stage2.ckpt")
```

**Risk:** Low. Adds engineering complexity but well-understood.

---

### 2.4. SyN Local Refinement on Top of TV-L1 — **+0.01-0.02 (data ceiling raise)**

**Rationale:**
- TV-L1 is good at global registration; SyN is better at small windows
- A hybrid approach: TV-L1 first, then SyN on each 256×256 sub-window
- Raises the data quality ceiling → enables higher model SSIM

**Pipeline:**
```python
def hybrid_register(stained, unstained):
    # Step 1: TV-L1 global
    flow = optical_flow_tvl1(rgb2gray(stained), rgb2gray(unstained))
    coarse = warp_with_flow(unstained, flow)
    # Step 2: SyN local refinement on 256×256 windows
    refined = np.zeros_like(coarse)
    for i in range(4):
        for j in range(4):
            window_s = stained[i*256:(i+1)*256, j*256:(j+1)*256]
            window_c = coarse[i*256:(i+1)*256, j*256:(j+1)*256]
            refined[i*256:(i+1)*256, j*256:(j+1)*256] = syn_register(window_s, window_c)
    return refined
```

**Cost:** +30 sec/pair × 8,885 pairs ≈ 75 hours (run once). Worth it.

**Risk:** Medium. SyN edge cases at window boundaries — need overlap+blend.

---

### 2.5. Self-Distillation — **+0.01-0.02**

**Rationale:**
- Train v18 → use v18 predictions as additional ground truth → train v19
- Smooths the loss landscape; reduces overfitting to imperfect registrations
- Used to good effect in noisy-label settings

**Implementation:**
```python
teacher = load_v18()
teacher.eval()
for batch in loader:
    pseudo = teacher(batch.unstained).detach()
    pred = student(batch.unstained)
    loss = 0.7 * L1(pred, batch.stained) + 0.3 * L1(pred, pseudo)
```

**Risk:** Medium. Distillation can amplify teacher errors.

---

### 2.6. Ensemble Inference — **+0.005-0.01**

**Rationale:**
- Average predictions from v11, v17, v18 at inference time
- Different models make different errors → averaging cancels noise
- Common wins in Kaggle, image generation challenges

**Implementation:**
```python
def ensemble_predict(x):
    p1 = v11_model(x)
    p2 = v17_model(x)
    p3 = v18_model(x)
    return (p1 + p2 + p3) / 3
```

**Cost:** 3× inference time. Fine for non-real-time pathology use.

**Risk:** Low.

---

### 2.7. Larger Curated Dataset — **+0.01-0.015**

**Rationale:**
- Top-1000 (mean SSIM 0.609) is small
- Top-2000 has lower mean (~0.55) — noisier
- Solution: go back to source slides, register more pairs with SyN, expand top-3000 with mean ≥ 0.55

**Practical Steps:**
1. Re-register all 8,885 pairs with hybrid TV-L1 + SyN
2. New SSIM scores → expand top tier
3. Train v19 on top-3000 with hybrid registration

**Risk:** Low; requires compute time.

---

## 3. Speculative / Long-Term Ideas

### 3.1. Diffusion Model

**Pros:** SOTA on natural image generation; smoother distributions than GAN.
**Cons:** Massive compute cost; needs more data; SSIM not the natural metric.
**Verdict:** Worth exploring after v18 if compute budget expands.

---

### 3.2. Foundation Model (Histology)

**Pros:** Pretrained on ~10M+ histology slides → strong domain priors.
**Cons:** Models like CTransPath, Phikon, UNI exist but are encoder-only; need decoder design.
**Verdict:** Replace ResNet-34 encoder with histology foundation model in v20.

---

### 3.3. Conditional Stain Style Variation

**Pros:** Model could output multiple plausible stain "styles" (different labs).
**Cons:** Adds modeling complexity; not the primary goal.
**Verdict:** Skip until base model crosses 0.82.

---

### 3.4. 3D Context (Z-Stack Inputs)

**Pros:** Tissue is 3D; multiple z-planes give more information.
**Cons:** Most data is 2D; redoing acquisition is expensive.
**Verdict:** Defer until clinical pipeline integration.

---

### 3.5. Active Learning / Hard Example Mining

**Pros:** Focus training on the worst-predicted pairs.
**Cons:** Can amplify registration errors; needs careful curation.
**Verdict:** Worth a quick experiment in v19.

---

## 4. Anti-Patterns to Avoid

| Anti-Pattern | Why It's Bad |
|--------------|--------------|
| **Adding more loss terms** | Diminishing returns; gradient conflicts (v12-v15) |
| **Adding GAN back in** | v15/v16 instability; SSIM-incompatible objective |
| **Training on all 8,885 pairs** | v13 data-quality failure |
| **Random validation set** | v14 high variance; can't track convergence |
| **Unaudited train/val split** | v19b/v20 prefix leakage; clean metrics require `overlap=0` |
| **Aggressive augmentation** | Color jitter destroys H&E semantics |
| **Increasing batch size for "speed"** | OOM at 1024×1024; gradient noise increases |
| **Dropping ResNet encoder** | v17 may miss this; v11→v18 should keep it |

---

## 5. Recommended Roadmap

```
v20_fixed (frozen clean baseline)
  ├─ Best epoch 77: SSIM 0.7606 / PSNR 24.92 / PCC 0.8652
  ├─ TTA4: SSIM 0.7634
  └─ App/deployment baseline

v21A (histology encoder ablation)
  ├─ Hibou-B frozen encoder + decoder
  ├─ Best: SSIM 0.7605 / PSNR 24.77 / PCC 0.8634
  ├─ 55/45 v20/v21A TTA ensemble: SSIM 0.7649
  └─ Keep for ensemble/secondary ablation, not primary app path

v22A (data/registration scaling, completed)
  ├─ CLAHE TV-L1 registration: all-pair mean SSIM 0.5045, +0.0911 vs old
  ├─ Content-quality top-1000: mean full SSIM 0.5601, content-gray 0.6223, gain +0.1152
  ├─ Warm-start v20 from models/v20_fixed_model.pth
  ├─ v22A TTA4 on CLAHE fixed eval: SSIM 0.7807 / PSNR 25.16 / PCC 0.8782
  └─ Final v20+v22A+v21B TTA4 ensemble: SSIM 0.7838 / PSNR 25.08 / PCC 0.8789

v22B/v22C (next data ablations)
  ├─ v22B: best-of-old-vs-CLAHE top-1000 warm-start
  ├─ v22C: content-quality or best-of top-1500 warm-start
  └─ Pick by fixed external validation, not internal split alone
```

---

## 6. Resource Budget for Next Runs

| Resource | Estimate |
|----------|----------|
| v22A content-quality warm-start | Complete; use `models/v22a_content_quality_top1000_ft_model.pth` |
| v22B best-of warm-start | ~1-1.5 GPU hours |
| TTA/checkpoint averaging | Minutes to 1 hour for validation sweep |
| v21B Hibou-B rerun on new data | Complete for top-1000; future reruns should use larger data only if the fixed eval gate is clear |
| RAM | 32 GB is enough; cached top-1000 pairs use about 6 GB |
| Disk | Keep only latest best checkpoint per run where possible |
| Risk | Low for continuation/TTA; medium for new encoder compatibility |

---

## 7. Open Research Questions

1. **How far does content-quality selection scale?** Top-1000 improved the CLAHE fixed eval; top-1500/top-2000 should test diversity.
2. **Which data policy is best: full-SSIM, best-of, positive-only, or content-quality?**
3. **Is top-1000 too narrow?** Top-1500 may trade slightly lower registration quality for better slide/tissue diversity.
4. **Can Hibou-style models win as single models?** v21B helped the final ensemble but still did not beat v22A alone.
5. **Can we design an external validation set that is stable across data-selection experiments?**

---

## 8. Closing Synthesis

The journey from 0.26 (v1-7) to 0.712 (v11) was mostly the story of three things:
1. **Registration** — TV-L1 unlocked GAN training
2. **Encoder** — ImageNet ResNet-34 added universal priors
3. **Perceptual loss** — VGG-19 features added biological awareness

The journey from 0.712 to clean 0.7606 added two more:
1. **Stable non-adversarial training** — L1-only was more reliable than stacked losses
2. **Stronger pretrained encoder** — ConvNeXt-Base LAION improved the clean baseline when paired with a correct split

The remaining path to 0.82 is likely:
1. **Clean evaluation discipline** — no unaudited validation splits
2. **Domain-specific pretraining** — histology encoder ablation against v20_fixed
3. **Better data when proven** — CLAHE registration has now passed controlled, full-dataset, and fixed-evaluation checks; the next question is whether scaling beyond top-1000 improves without losing registration quality

GANs are not in either list. **The path forward is non-adversarial.**

---

## 9. Files Referenced

| File | Purpose |
|------|---------|
| `best_stain_app.py` | Final app for balanced v20/v22A/v21B TTA ensemble |
| `best_model_manifest.json` | Final model weights, modes, scores, and runtime notes |
| `APP_DISTRIBUTION.md` | App run/build/distribution handoff |
| `train_v20.py` | v20_fixed clean baseline |
| `train_v20_csv_variant.py` | v22A/v22B warm-start runner for CSV variants |
| `logs/v20_fixed_training.log` | v20_fixed clean metrics |
| `logs/v22a_content_quality_top1000_ft_training.log` | Completed v22A warm-start metrics |
| `v20_stain_app.py` | Legacy/simple inference app for v20-style weights |
| `registration_pipeline_clahe.py` | CLAHE TV-L1 registration pipeline |
| `data/processed/content_quality_csvs/` | Content-quality training CSVs |
| `research/05_loss_functions.md` | MS-SSIM details |
| `research/02_image_registration.md` | SyN evaluation |
| `research/04_training_history.md` | All prior version results |

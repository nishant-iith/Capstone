# 09: Future Work — The Path from 0.712 to 0.82+ SSIM

> **Bottom Line:** The clinical target (SSIM 0.82+) is achievable through a combination of better encoder (ResNet-34 ImageNet), better loss (MS-SSIM + perceptual without GAN), better data (SyN registration refinement, larger curated set), and better training (progressive resizing, warm-start chain). v18 is the recommended next major iteration; v19+ should pursue self-distillation and ensemble methods.

---

## 1. Recommended Next Model: v18

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
| **Aggressive augmentation** | Color jitter destroys H&E semantics |
| **Increasing batch size for "speed"** | OOM at 1024×1024; gradient noise increases |
| **Dropping ResNet encoder** | v17 may miss this; v11→v18 should keep it |

---

## 5. Recommended Roadmap

```
v17 (current)
  ├─ If SSIM ≥ 0.73: Build v18 with all improvements
  └─ If SSIM < 0.73: Investigate why; rerun with fixes

v18 (next major)
  ├─ ResNet-34 + MS-SSIM + Progressive training
  ├─ Target: 0.76-0.80
  └─ If achieved: build v19

v19 (refinement)
  ├─ + SyN-refined registration
  ├─ + Larger top-3000 dataset
  ├─ + Self-distillation from v18
  └─ Target: 0.80-0.82

v20 (clinical-grade)
  ├─ + Histology foundation model encoder
  ├─ + Ensemble (v11 + v17 + v18 + v19)
  └─ Target: 0.82-0.85
```

---

## 6. Resource Budget for v18

| Resource | Estimate |
|----------|----------|
| GPU hours | 12-15 hours (A100 80GB) |
| RAM | 32 GB |
| Disk | 20 GB (dataset cached + checkpoints) |
| Engineering effort | 2-3 days |
| Risk | Low (all components proven individually) |

---

## 7. Open Research Questions

1. **What is the registration ceiling?** — Can hybrid TV-L1+SyN reach mean SSIM 0.75 on the top tier?
2. **Does GAN ever help when stable?** — Could a carefully balanced GAN (λ=0.5) help v18 cross 0.80?
3. **Is the ImageNet→histology domain gap exploitable?** — Would a histology-pretrained encoder (CTransPath) outperform ResNet-34?
4. **Can we get more registered pairs?** — How does scaling top-tier from 1k to 3k affect convergence?
5. **What's the irreducible noise floor?** — How much SSIM is lost purely to registration imperfection?

---

## 8. Closing Synthesis

The journey from 0.26 (v1-7) to 0.712 (v11) is mostly the story of three things:
1. **Registration** — TV-L1 unlocked GAN training
2. **Encoder** — ImageNet ResNet-34 added universal priors
3. **Perceptual loss** — VGG-19 features added biological awareness

The journey from 0.712 to 0.82 is likely the story of three more:
1. **Direct metric optimization** — MS-SSIM instead of indirect GAN
2. **Better data** — SyN refinement + larger curated set
3. **Compositional training** — Progressive resizing + warm-start chain

GANs are not in either list. **The path forward is non-adversarial.**

---

## 9. Files Referenced

| File | Purpose |
|------|---------|
| `train_v17.py` | Current SOTA attempt |
| `train_v18.py` (TBD) | Next iteration — to be written |
| `research/05_loss_functions.md` | MS-SSIM details |
| `research/02_image_registration.md` | SyN evaluation |
| `research/04_training_history.md` | All prior version results |

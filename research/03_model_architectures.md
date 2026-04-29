# 03: Model Architectures — All Generators & Discriminators

> **Bottom Line:** We explored five major architectural families. The most successful was the **U-Net Generator with ResNet-34 Encoder + 70×70 PatchGAN Discriminator**, used in v11 (best result: 0.712 SSIM). We later experimented with **Attention U-Net + MultiScale Discriminator** (v15, v16) and pure **U-Net (4-level and 5-level)** (v14, v17).

---

## 1. Generator Architectures

### 1.1. Pix2Pix U-Net (Original — v1-v7)

The original Pix2Pix paper (Isola et al., 2017) U-Net with `inplace_abn` instance normalization.

```
Input (3, 256, 256)
  ↓ Conv → InstanceNorm → LeakyReLU
  ↓ Encoder (8 down-sampling blocks)
  ↓ Bottleneck (1×1 spatial)
  ↓ Decoder (8 up-sampling blocks with skip connections)
  ↓ Tanh → Output (3, 256, 256)
```

**Used in:** v1-v7 (all failed without registration)
**Best Result:** SSIM 0.26 (random data → random output)

---

### 1.2. U-Net with ResNet-34 Encoder ⭐ **(BEST — v11)**

Modern U-Net with ImageNet-pretrained ResNet-34 encoder. Inspired by `segmentation_models_pytorch` design.

```
Input (3, 256, 256)
  ↓
ResNet-34 Encoder (ImageNet pretrained)
  ├─ conv1 + maxpool   →  (64, 64, 64)
  ├─ layer1            → (64, 64, 64)
  ├─ layer2            → (128, 32, 32)
  ├─ layer3            → (256, 16, 16)
  └─ layer4            → (512, 8, 8)
  ↓
Bottleneck (1024 channels)
  ↓
U-Net Decoder (with skip connections)
  ├─ up0: 1024 + 512 → 512   (8×8 → 16×16)
  ├─ up1: 512 + 256 → 256    (16×16 → 32×32)
  ├─ up2: 256 + 128 → 128    (32×32 → 64×64)
  ├─ up3: 128 + 64 → 64      (64×64 → 128×128)
  ├─ up4: 64 + 0 → 32        (128×128 → 256×256)
  └─ Conv 1×1 + Tanh         →  (3, 256, 256)
```

**File:** `src/models/gan.py:UNetGenerator`

**Key Properties:**
- ImageNet pretrained features → strong color/edge/texture priors
- Skip connections at every encoder stage
- InstanceNorm in decoder (not BatchNorm — better for image translation)
- Output range: $[-1, 1]$ (Tanh activation)

**Used in:** v8-v15 (multiple variants)
**Best Result:** SSIM **0.712** in v11 (Weakly Supervised)

---

### 1.3. Attention U-Net (v15, v16)

U-Net with **attention gates** on every skip connection. The attention gate computes a soft mask over the encoder features based on a gating signal from the decoder, suppressing irrelevant background.

```python
class AttentionGate(nn.Module):
    """g = decoder gating signal, x = encoder skip features"""
    def __init__(self, f_g, f_x, f_int):
        self.W_g = Conv1×1(f_g, f_int)
        self.W_x = Conv1×1(f_x, f_int, stride=2)
        self.psi = Conv1×1(f_int, 1) → InstanceNorm → Sigmoid

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = ReLU(g1 + x1)
        psi = self.psi(psi)  # mask in [0, 1]
        return x * psi       # attention-weighted skip
```

**Reference:** Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas" (2018)

**File:** `src/models/gan.py:AttentionGate, UNetGenerator (with attention)`

**Theoretical Benefits:**
- Suppresses background gradients (focuses on cells)
- Reduces false positive features in skip connections
- Empirically helps on small/sparse object segmentation

**Used in:** v15, v16
**Result:** v15 peaked at SSIM 0.7199 (1 epoch only, then diverged); v16 peaked at 0.6976 then diverged.

**Verdict on Attention:** ⚠️ Theoretically sound, but training was unstable due to the GAN losses (not the architecture). Architecture itself is fine.

---

### 1.4. Simple 4-Level U-Net (v13, v14)

Plain U-Net without pretrained encoder, BatchNorm, ReLU, Sigmoid output range $[0, 1]$.

```
Encoder (5 conv_block stages):
  enc1: 3 → 64    (1024×1024)
  enc2: 64 → 128  (512×512)
  enc3: 128 → 256 (256×256)
  enc4: 256 → 512 (128×128)
Bottleneck:
  512 → 1024      (64×64)
Decoder (with skip cat):
  dec4: 1024+512 → 512
  dec3: 512+256 → 256
  dec2: 256+128 → 128
  dec1: 128+64 → 64
Output:
  Conv1×1 → Sigmoid → (3, 1024, 1024)
```

**File:** `train_v14.py` (model defined inline)

**Used in:** v14
**Result:** SSIM **0.7080** (full-size validation, epoch 10) on 512×512 patches

---

### 1.5. Simple 5-Level U-Net (v17 — FAILED)

Same as 4-level, but adds one more encoder/decoder stage to capture **larger receptive field at full 1024×1024 resolution**.

```
enc1: 3 → 64     (1024×1024)
enc2: 64 → 128   (512×512)
enc3: 128 → 256  (256×256)
enc4: 256 → 512  (128×128)
enc5: 512 → 1024 (64×64)
Bottleneck:
  1024 → 2048    (32×32)  ← deepest representation
Decoder mirrors encoder
Output (3, 1024, 1024)
```

**File:** `train_v17.py`

**Rationale:**
- At 1024×1024, a 4-level U-Net has bottleneck at 64×64 — small but not capturing **tissue-level context**
- 5-level U-Net has bottleneck at 32×32 — captures larger structures (cell clusters, tissue boundaries)
- Increases parameters by ~3× but A100 80GB has memory headroom

**Used in:** v17 (FAILED, SSIM 0.379)
**Result:** ❌ Warm-start from v14 (4-level) loaded decoder layers at the wrong receptive-field scale. v17's `dec4` is mid-level while v14's `dec4` is bottleneck-adjacent — same name, different role. Bottleneck shape mismatched (skipped during load). Net effect: decoder partially initialized with **wrong-scale features**, sigmoid output saturated near gray, val SSIM (0.38) ended up below the registration floor (0.63).
**Architectural lesson:** Warm-start only across architectures that share **depth**, or remap layers by **depth from output**, not by name.

---

## 2. Discriminator Architectures

### 2.1. 70×70 PatchGAN ⭐ **(STANDARD)**

The classic Pix2Pix discriminator. Receives concatenated `(input, target_or_fake)` and outputs a grid of patch-level real/fake scores. Each output pixel corresponds to a 70×70 receptive field in the input.

```
Input: concat(unstained, stained_or_fake)  →  (6, H, W)

Conv4×4 stride=2 → 64
LReLU(0.2)
Conv4×4 stride=2 → 128
InstanceNorm + LReLU
Conv4×4 stride=2 → 256
InstanceNorm + LReLU
Conv4×4 stride=1 → 512
InstanceNorm + LReLU
Conv4×4 stride=1 → 1
(no sigmoid — used with WGAN-GP)
```

**Why "70×70"?** With four 4×4 convolutions, two with stride=2, the effective receptive field of each output pixel is approximately 70×70 in the input.

**Why patch-based?** Forces the discriminator to focus on local texture realism rather than global structure (which the L1 loss handles).

**Used in:** v8-v13, v14 (no disc), v15-v16 (Multi-scale variant)
**Verdict:** ✅ Solid foundation; harder to outperform than expected

---

### 2.2. MultiScale Discriminator (v15, v16)

Two PatchGAN discriminators operating at different scales:

```
                      Input (concat unstained + stained)
                              ↓
                ┌─────────────┴─────────────┐
                ↓                           ↓
        Local PatchGAN              Global PatchGAN
        (15×15 receptive)           (downsampled 2×, 7×7)
                ↓                           ↓
         Patch logits                 Patch logits
        (texture detail)            (tissue structure)
                ↓                           ↓
                └─────────────┬─────────────┘
                              ↓
                     Average → adversarial loss
```

**File:** `src/models/gan.py:MultiScaleDiscriminator`

**Theoretical Benefits:**
- **Local (15×15):** Catches fine cellular textures, edges
- **Global (7×7 after downsample):** Catches tissue-level architectural patterns
- More gradient signal to generator

**Used in:** v15, v16
**Result:** Both diverged. Adversarial complexity exacerbated GAN instability.
**Verdict:** ⚠️ Theoretically stronger than single-scale, but only useful with **stable training**, which we struggled to achieve with WGAN-GP + L1(100) imbalance.

---

## 3. CycleGAN (Considered, NOT Used)

CycleGAN learns unpaired image-to-image translation using **cycle consistency**:
$F(G(X)) \approx X$ and $G(F(Y)) \approx Y$.

**Why considered:** Allows training without registration.
**Why rejected:**
1. We **have** paired data (after TV-L1 registration). Supervised Pix2Pix is mathematically more accurate.
2. Cycle consistency does not enforce structural fidelity at the pixel level — bad for diagnostic-grade outputs.
3. Unpaired training tends to "hallucinate" features that look stained but don't correspond to real tissue.

**File:** `src/training/lightning_module_cyclegan.py` (implemented but never used in production)

**Verdict:** ❌ **Not used** for production. Reserved for ablation/comparison only.

---

## 4. Architectural Comparison Matrix

| Model | Generator | Discriminator | Skip Conn. | Pretrained | Loss | Best SSIM |
|-------|-----------|---------------|------------|------------|------|-----------|
| v1-v7 | Pix2Pix U-Net | 70×70 PatchGAN | Standard | No | L1+GAN | 0.26 |
| v8-v10 | ResNet-34 U-Net | 70×70 PatchGAN | Standard | ImageNet | L1+WGAN-GP | 0.706 |
| **v11** ⭐ | ResNet-34 U-Net | 70×70 PatchGAN | Standard | ImageNet | L1+WGAN-GP+VGG+Sobel | **0.712** |
| v12 | ResNet-34 U-Net | 70×70 PatchGAN | Standard | ImageNet | + HED | (failed) |
| v13 | Attention U-Net | MultiScale | Attention | ImageNet | Hybrid | 0.6326 |
| v14 | Simple 4-level U-Net | None (no GAN) | Standard | No | L1 only | 0.7080 |
| v15 | Attention U-Net | MultiScale | Attention | ImageNet | Hybrid + HED | 0.7199 (then diverged) |
| v16 | Attention U-Net | MultiScale | Attention | ImageNet | Hybrid (no HED) | 0.6976 (then diverged) |
| v17 | Simple 5-level U-Net | None (no GAN) | Standard | No (broken warm-start from v14) | L1+SSIM | 0.379 ❌ |

---

## 5. Key Architectural Insights

### Insight 1: Pretrained Encoder is Highly Beneficial
The transition from random-init U-Net (v1-v7, achieved 0.26) to ResNet-34 ImageNet encoder (v8+, achieved 0.706+) was a major leap. ImageNet features encode useful priors (edges, color patterns) that transfer to histology.

### Insight 2: Simpler ≠ Worse
The simple U-Net (v14) **outperformed** the complex Attention U-Net + MultiScale Disc (v16) on full-size images:
- v14 (simple, patches): 0.7080
- v16 (complex, full-size): 0.6976

**Reason:** GAN training instability with complex architectures negated their theoretical benefits.

### Insight 3: Discriminator is a Double-Edged Sword
| With Disc | Without Disc |
|-----------|--------------|
| Sharper outputs | Slightly blurrier |
| Risk of divergence | Stable training |
| Better PSNR/perceptual | Better SSIM (paradoxically) |

**Conclusion:** For SSIM-targeted optimization, removing the discriminator (v14, v17) actually helps because adversarial loss can pull the model away from the SSIM-minimum solution.

### Insight 4: 5-Level vs 4-Level U-Net
At 1024×1024 input, a 4-level U-Net has bottleneck at 64×64. This may not capture full tissue-level context. The 5-level variant (bottleneck 32×32) was tested in v17 but training failed due to warm-start scale mismatch — not the architecture's fault. **5-level remains a valid choice for v18, but with proper init** (ResNet-34 ImageNet) instead of warm-start from a 4-level parent.

### Insight 5: Activation Functions
- **Tanh output** (Pix2Pix style, $[-1, 1]$) requires `(image - 0.5) * 2` normalization
- **Sigmoid output** ($[0, 1]$) maps directly to image space
- Both work; Sigmoid is simpler and used in v14, v17.

---

## 6. Files Referenced

| File | Purpose |
|------|---------|
| `src/models/gan.py` | UNetGenerator (Attention), MultiScaleDiscriminator |
| `src/models/losses.py` | PerceptualLoss, HEDStainLoss |
| `train_v14.py` | Simple 4-level U-Net (defined inline) |
| `train_v17.py` | Simple 5-level U-Net (defined inline) |
| `src/training/lightning_module_v11.py` | v11 weakly supervised (best model) |
| `src/training/lightning_module_v15.py` | v15 (multi-scale + attention + HED) |
| `src/training/lightning_module_v16.py` | v16 (multi-scale + attention, no HED) |

# 03: Model Architectures — All Generators & Discriminators

> **Bottom Line:** The strongest single CLAHE model is the ConvNeXt-Base v22A warm-start, but the best overall result is an ensemble: `0.20*v20 + 0.60*v22A + 0.20*v21B`, all TTA4, with SSIM 0.7838, PSNR 25.16, PCC 0.8794. Hibou-B did not win alone but improved the final ensemble slightly.

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

### 1.6. ConvNeXt-Base LAION-2B U-Net (v20, v20_fixed, v22A warm-start)

U-Net decoder from `segmentation_models_pytorch` with a timm ConvNeXt-Base encoder pretrained on LAION-2B/CLIP-style web images.

```python
model = smp.Unet(
    encoder_name="tu-convnext_base.clip_laion2b",
    encoder_weights="laion2b",
    in_channels=3,
    classes=3,
    activation="sigmoid",
)
```

**File:** `train_v20.py`

**Used in:** v20 and v20_fixed

**Result:**
- old v20: SSIM 0.7549, but leaky train/val split
- v20_fixed: SSIM 0.7606, PSNR 24.92, PCC 0.8652, clean 900/100 split with `overlap=0`
- v22A warm-start: same architecture initialized from `models/v20_fixed_model.pth`, trained on content-quality CLAHE top-1000; best single CLAHE fixed-eval result with TTA4 was SSIM 0.7807, PSNR 25.16, PCC 0.8782

**Verdict:** Current best deployable architecture family. The main lesson is that a stronger pretrained encoder plus a simple L1 objective beats adding more losses or adversarial machinery. Better registered labels did help on the CLAHE fixed evaluation, and the best final result came from using v22A as the largest-weight member of a small TTA ensemble.

### 1.7. Hibou-B Frozen Feature Decoder (v21A)

Hibou-B is a histology-pretrained DINOv2-style model with register tokens. It cannot be dropped into `segmentation_models_pytorch.Unet` as a standard encoder because its token downsampling pattern is not a normal CNN pyramid. The v21A workaround froze Hibou-B and trained a lightweight high-resolution decoder with input skips.

**Result:** SSIM 0.7605, PSNR 24.77, PCC 0.8634. This matched v20_fixed but did not beat it.

**Ensemble Result:** v20/v21A 55/45 weighted 4-flip TTA reached SSIM 0.7649, PSNR 25.11, PCC 0.8710.

**Verdict:** Useful as a complementary model and research ablation, but not the primary deployable architecture due to gated dependency, redistribution complexity, and no single-model win.

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
| v19b | ResNet-34 smp.Unet | None | Standard | ImageNet | L1 only | 0.7489 (leaky) |
| v20 | ConvNeXt-Base smp.Unet | None | Standard | LAION-2B | L1 + elastic aug | 0.7549 (leaky) |
| **v20_fixed** ⭐ | **ConvNeXt-Base smp.Unet** | **None** | **Standard** | **LAION-2B** | **L1 + elastic aug** | **0.7606 clean** |
| v21A | Hibou-B frozen + decoder | None | Token decoder | Histology | L1 + elastic aug | 0.7605 clean |
| v22A | ConvNeXt-Base smp.Unet warm-start | None | Standard | LAION-2B | L1 + elastic aug | 0.7807 TTA4 on CLAHE fixed eval |

---

## 5. Key Architectural Insights

### Insight 1: Pretrained Encoder is Highly Beneficial
The transition from random-init U-Net (v1-v7, achieved 0.26) to ResNet-34 ImageNet encoder (v8+, achieved 0.706+) was a major leap. ImageNet features encode useful priors (edges, color patterns) that transfer to histology.

v20_fixed extends this: ConvNeXt-Base LAION-2B reached 0.7606 on the clean split. The v21A result shows that histology-pretrained features are not automatically better when the decoder and feature hierarchy are constrained. The current priority is to test better labels/data with the stable v20 architecture before adding another encoder variable.

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
At 1024×1024 input, a 4-level U-Net has bottleneck at 64×64. This may not capture full tissue-level context. The 5-level variant (bottleneck 32×32) was tested in v17 but training failed due to warm-start scale mismatch — not the architecture's fault. A 5-level model remains a valid future ablation, but only with proper initialization and the same clean split used by v20_fixed.

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
| `train_v20.py` | ConvNeXt-Base LAION-2B U-Net (`v20_fixed`) |
| `src/training/lightning_module_v11.py` | v11 weakly supervised (best model) |
| `src/training/lightning_module_v15.py` | v15 (multi-scale + attention + HED) |
| `src/training/lightning_module_v16.py` | v16 (multi-scale + attention, no HED) |

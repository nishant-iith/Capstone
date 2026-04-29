# 05: Loss Functions — All Components Explored

> **Bottom Line:** L1 + Perceptual (VGG-19) + Sobel was the winning combination (v11). Adding HED stain decomposition destabilized training. Direct SSIM optimization was tried in v17 but failed (SSIM 0.379) — not the loss's fault, but a wrong-scale warm-start from a differently-shaped parent model. MS-SSIM remains promising for v18. WGAN-GP adversarial loss is powerful but requires careful balancing — without it, models are more stable but slightly blurrier.

---

## 1. L1 Loss (Pixel-Wise Reconstruction)

**Formula:**
$$
L_{L1} = \frac{1}{N} \sum_i |G(x_i) - y_i|
$$

**Properties:**
- Penalizes per-pixel deviation from ground truth
- Robust to outliers (compared to L2/MSE)
- **Failure mode:** When uncertain, model predicts the **mean** → blurry outputs

**Used in:** Every model from v1 onwards. Always with weight λ=100 in hybrid losses.

**Verdict:** ✅ **Foundational.** Cannot be removed without losing color/intensity grounding.

---

## 2. WGAN-GP (Wasserstein GAN with Gradient Penalty)

**Formula:**
$$
L_{D} = \mathbb{E}_{\hat{x}}[D(\hat{x})] - \mathbb{E}_{x}[D(x)] + \lambda_{GP} \, \mathbb{E}_{\tilde{x}}[(\| \nabla_{\tilde{x}} D(\tilde{x}) \|_2 - 1)^2]
$$

$$
L_{G} = -\mathbb{E}_{\hat{x}}[D(\hat{x})]
$$

where $\tilde{x}$ is a random interpolation between real and fake samples.

**Why WGAN-GP over Standard GAN?**
- Standard GAN uses BCE loss → unstable, mode collapse, vanishing gradients
- WGAN minimizes Wasserstein distance → continuous gradient signal
- Gradient Penalty (GP) replaces weight clipping for Lipschitz constraint

**Hyperparameters Used:**
- λ_GP = 10
- Adam β1 = 0.0, β2 = 0.9 (recommended for WGAN-GP)

**Used in:** v8-v13, v15, v16

**Failure Mode Observed:**
- In v15/v16, the discriminator "won" too aggressively (d_loss → -84)
- Generator couldn't keep up → instability → divergence
- Imbalance with L1(100) caused this — GAN signal was 10× weaker than reconstruction

**Verdict:** ⚠️ **Powerful but unstable.** Removed in v14 and v17 for stability.

---

## 3. VGG-19 Perceptual Loss ⭐ **(v11 KEY INNOVATION)**

**Formula:**
$$
L_{percept} = \sum_{l \in \{1, 2, 3, 4, 5\}} \frac{1}{H_l W_l C_l} \| \phi_l(G(x)) - \phi_l(y) \|_1
$$

where $\phi_l$ is the activation at the $l$-th VGG-19 layer (ReLU after each block).

**Implementation (`src/models/losses.py:PerceptualLoss`):**
```python
class PerceptualLoss(nn.Module):
    def __init__(self):
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList([
            vgg[:4],    # relu1_2 — fine textures, edges
            vgg[4:9],   # relu2_2 — local patterns
            vgg[9:18],  # relu3_4 — mid-level shapes
            vgg[18:27], # relu4_4 — object parts
            vgg[27:36]  # relu5_4 — global structures
        ])
        # Freeze VGG (only used as feature extractor)
        for p in self.parameters():
            p.requires_grad = False
```

**Why It Works for Histology:**
- Forces the model to match **biological textures** (cell membrane patterns, nuclear chromatin) rather than just pixel colors
- Multi-layer matching captures both fine cellular detail AND tissue-level architecture
- ImageNet-pretrained VGG has implicit knowledge of edges, textures, color patterns

**Hyperparameters:**
- λ_percept = 10 (in v11 hybrid loss)

**Used in:** v11 (project best), v15, v16

**Verdict:** ✅ **Critical for breaking 0.71 SSIM ceiling.** Adds ~0.005-0.010 SSIM improvement.

---

## 4. Sobel Structural Loss (Edge Loss)

**Formula:**
$$
L_{Sobel} = \frac{1}{N} \sum_i \| \text{Sobel}(G(x_i)) - \text{Sobel}(y_i) \|_1
$$

**Implementation:**
```python
def sobel(img):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=img.dtype, device=img.device)
    kx = kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(img, kx, padding=1, groups=3)
    gy = F.conv2d(img, ky, padding=1, groups=3)
    return torch.sqrt(gx**2 + gy**2 + 1e-6)
```

**Why It Works:**
- Histology = cell edges and membrane boundaries are diagnostic
- Sobel filter = explicit edge detector
- Forces generator to match **edge maps** as well as raw pixels

**Hyperparameters:**
- λ_struct = 20 (in v11)

**Used in:** v11 (best), v13, v15, v16

**Verdict:** ✅ Helps slightly. Stable, no instability concerns.

---

## 5. HED Stain Decomposition Loss (Failed)

**Concept:**
H&E staining can be decomposed via the Macenko stain matrix into:
- **H**ematoxylin (purple, binds nuclei)
- **E**osin (pink, binds cytoplasm)
- **D**AB / residual (rare)

The HED loss decomposes both predicted and ground-truth images into these three channels and computes L1 on each.

**Mathematical Formulation:**
1. Convert RGB → optical density: $OD = -\log_{10}(I / I_{max})$
2. Project onto stain matrix: $C = M^{-1} \cdot OD$
3. Compute L1 on each channel of $C$

**Implementation (`src/models/losses.py:HEDStainLoss`):**
```python
class HEDStainLoss(nn.Module):
    def __init__(self):
        self.stain_matrix = torch.tensor([
            [0.65, 0.70, 0.29],   # Hematoxylin
            [0.07, 0.99, 0.11],   # Eosin
            [0.27, 0.57, 0.78],   # DAB
        ]).T  # transpose for matmul
        self.stain_inv = torch.inverse(self.stain_matrix.to(torch.float64))

    def forward(self, fake, real):
        OD_fake = self._rgb_to_od(fake)
        OD_real = self._rgb_to_od(real)
        C_fake = OD_fake @ self.stain_inv.float()
        C_real = OD_real @ self.stain_inv.float()
        return F.l1_loss(C_fake, C_real)
```

**Failure Modes:**
1. **Numerical instability**: `torch.inverse` of stain matrix in float16 → NaN gradients
2. **Loss conflict**: HED gradient direction conflicted with VGG perceptual direction
3. **Magnitude imbalance**: Hard to weight properly against L1 and adversarial

**Used in:** v12 (failed), v13, v15 (caused divergence)

**Verdict:** ❌ **Removed.** Theoretically elegant but practically unstable.

---

## 6. Direct SSIM Loss (v17)

**Formula:**
$$
L_{SSIM} = 1 - SSIM(G(x), y)
$$

where SSIM is the structural similarity index:
$$
SSIM(p, q) = \frac{(2\mu_p\mu_q + C_1)(2\sigma_{pq} + C_2)}{(\mu_p^2 + \mu_q^2 + C_1)(\sigma_p^2 + \sigma_q^2 + C_2)}
$$

**Implementation (using `pytorch_msssim`):**
```python
from pytorch_msssim import SSIM as SSIM_Module

class L1SSIMLoss(nn.Module):
    def __init__(self, lambda_l1=0.5, lambda_ssim=0.5):
        self.l1 = nn.L1Loss()
        self.ssim = SSIM_Module(data_range=1.0, channel=3)

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_loss = 1.0 - self.ssim(pred, target)
        return self.lambda_l1 * l1_loss + self.lambda_ssim * ssim_loss
```

**Why?**
- We measure performance with SSIM
- Optimizing L1 only indirectly improves SSIM
- Directly optimizing SSIM should give a more direct gradient signal

**Used in:** v17 (failed at SSIM 0.379)

**Verdict:** ⚠️ **Inconclusive — loss config is sound, but v17's wrong-scale warm-start prevented evaluating the loss in isolation.** Re-test in v18 with proper init (ResNet-34 ImageNet) before drawing conclusions. The L1+SSIM combination itself is well-motivated and theoretically expected to give +0.01-0.02 SSIM over L1-only.

---

## 7. MS-SSIM Loss (Recommended for v18)

**Formula:**
Multi-Scale SSIM computes SSIM at 5 progressively downsampled scales:
$$
MS\text{-}SSIM(p, q) = [l_M(p,q)]^{\alpha_M} \cdot \prod_{j=1}^{M} [c_j(p,q)]^{\beta_j} [s_j(p,q)]^{\gamma_j}
$$

**Why Better Than SSIM:**
- Single-scale SSIM only captures structure at one resolution
- MS-SSIM captures fine, mid, and coarse structures simultaneously
- Better aligns with human perceptual judgment

**Implementation:**
```python
from pytorch_msssim import MS_SSIM
ms_ssim_loss = 1.0 - MS_SSIM(data_range=1.0, channel=3)(pred, target)
```

**Used in:** Not yet (recommended for v18)

**Expected Gain:** +0.01-0.02 SSIM over single-scale SSIM

---

## 8. Loss Combination Matrix

| Version | L1 | WGAN-GP | VGG | Sobel | HED | SSIM | MS-SSIM | Result |
|---------|-----|---------|-----|-------|-----|------|---------|--------|
| v1-7    | ✓ (100) | ✓ | – | – | – | – | – | 0.26 (no reg) |
| v10     | ✓ (100) | ✓ (10) | – | – | – | – | – | 0.706 |
| **v11** | ✓ (100) | ✓ (1) | ✓ (10) | ✓ (20) | – | – | – | **0.712** ⭐ |
| v12     | ✓ (100) | ✓ (1) | ✓ (10) | ✓ (20) | ✓ (5) | – | – | failed |
| v13     | ✓ (100) | ✓ (10) | ✓ (10) | ✓ (20) | ✓ (5) | – | – | 0.6326 |
| v14     | ✓ (1) | – | – | – | – | – | – | 0.7080 |
| v15     | ✓ (100) | ✓ (10) | ✓ (10) | ✓ (20) | ✓ (5) | – | – | 0.7199 → diverged |
| v16     | ✓ (100) | ✓ (10) | ✓ (10) | ✓ (20) | – | – | – | 0.6976 → diverged |
| v17     | ✓ (0.5) | – | – | – | – | ✓ (0.5) | – | 0.379 ❌ (warm-start scale mismatch) |
| v18*    | ✓ (0.4) | – | ✓ (0.2) | – | – | – | ✓ (0.4) | predicted 0.76+ |

*v18 is hypothetical (recommended)

---

## 9. Lessons on Loss Engineering

### Lesson 1: **L1 is the Anchor**
Every successful model uses L1 with weight ≥ 100 (or λ=0.5 in normalized formulations). It provides the basic intensity grounding.

### Lesson 2: **Adversarial is High-Variance**
GAN loss can give a +0.005-0.010 SSIM boost when stable, but causes catastrophic divergence when unstable. **High-risk, marginal-reward.**

### Lesson 3: **Don't Stack Too Many Conflicting Losses**
v12, v13, v15 with 5+ loss terms (L1 + WGAN-GP + VGG + Sobel + HED) all suffered from competing gradients. v11's 4-term formulation was the maximum tractable complexity.

### Lesson 4: **Optimize What You Measure (Untested)**
v14 (L1 only) → 0.7080. v17 (L1+SSIM) was supposed to validate this but failed for **architectural** reasons (warm-start scale mismatch), not loss reasons. **Direct SSIM optimization** is still theoretically motivated; re-test in v18 with proper initialization.

### Lesson 5: **Loss Magnitudes Matter**
λ_L1 = 100 vs λ_GP = 10 means the model effectively ignores adversarial. Either rebalance or accept that GAN is a small auxiliary signal.

### Lesson 6: **Numerical Stability Trumps Theory**
HED loss is theoretically excellent (decomposes the actual stain physics). Practically, `torch.inverse` in fp16 produces NaNs. **Always test numerical stability first.**

---

## 10. Files Referenced

| File | Purpose |
|------|---------|
| `src/models/losses.py` | PerceptualLoss, HEDStainLoss |
| `train_v17.py:L1SSIMLoss` | L1 + SSIM combined loss |
| `src/training/lightning_module_v11.py` | Best loss configuration (project best) |
| `src/training/lightning_module_v15.py` | All 5 losses (v15 — caused divergence) |
| `src/training/lightning_module_v16.py` | 4 losses without HED (still diverged) |

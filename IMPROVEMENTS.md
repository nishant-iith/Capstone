# Potential Improvements

Current best: **SSIM 0.2696** (v7 + TTA, single training slide, 500 patches)

---

## 1. More Training Data — Highest Impact

**Expected gain: SSIM 0.35–0.45+**

Current model trained on 1 slide (AS-7231-22-Z17, ~500 patches). Hit dataset ceiling.

- Add 3–5 more paired unstained/stained slides
- Each new slide contributes ~500 additional patches
- This alone will 2–3× the improvement achievable by any other method

> Root cause of current ceiling: model learned one slide's stain distribution and cannot generalize.

---

## 2. Larger Encoder Architecture — Medium Impact

**Expected gain: +3–8%**

Current generator uses ResNet-34 encoder (44.2M params total).

- Swap encoder to ResNet-50 or EfficientNet-B4
- Deepen bottleneck (1024 → 2048 channels)
- Same U-Net decoder structure

---

## 3. Perceptual Loss (carefully) — Medium Impact

**Expected gain: +2–5%**

v8 attempted perceptual loss but failed — too many conflicting loss terms.

Correct approach: replace L1 with perceptual, do not stack both:

```python
# Instead of: wgan + L1 + structural
loss_G = wgan_loss + lambda_perceptual * perceptual_loss(fake, real)
```

Use VGG-16 features at layers relu2_2 and relu3_3.

---

## 4. Larger Patch Size — Medium Impact

**Expected gain: +3–6%**

Current: 256×256 patches. More context per patch improves structure learning.

- Try 512×512
- Requires 4× more GPU memory → reduce batch size to 1–2
- May need gradient accumulation to compensate

---

## 5. Richer Augmentation — Low-Medium Impact

**Expected gain: +1–3%**

Current augmentation: rotation only (90/180/270° + h-flip).

Add:
- Color jitter (±10% brightness, contrast, saturation)
- Gaussian blur (σ=0.5–1.0)
- Random crops + resize back to 256×256
- Elastic deformation (small σ to avoid distortion artifacts — v9 used too-large deformation)

---

## Summary

| Improvement | Expected Gain | Effort |
|-------------|--------------|--------|
| More training slides | +30–70% | High (data collection) |
| Larger encoder | +3–8% | Medium |
| Perceptual loss | +2–5% | Medium |
| Larger patches (512×512) | +3–6% | Low |
| Richer augmentation | +1–3% | Low |

**Bottom line:** Options 2–5 yield marginal gains on existing data. Only adding more training slides breaks through the current ceiling.

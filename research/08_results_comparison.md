# 08: Results Comparison — Comprehensive Metrics & Ablations

> **Bottom Line:** The current best fixed-eval pipeline is **CLAHE TV-L1 + TTA4 weighted v20/v22A/v21B ensemble = SSIM 0.7838, PSNR 25.16, PCC 0.8794**. On the old registered input distribution, **v20 TTA4 remains best at SSIM 0.7634**. Historical v19b (0.7489) and old v20 (0.7549) were stable but used an overlapping train/validation split, so they should not be reported as clean validation.

---

## 1. Master Results Table

| Ver | Architecture | Disc | Data | Loss | SSIM | PSNR | PCC | Status |
|-----|--------------|------|------|------|------|------|-----|--------|
| v1-7 | Pix2Pix U-Net | 70×70 | Unreg | L1+GAN | 0.26 | ~15.8 | – | Failed |
| v8 | ResNet-34 U-Net | 70×70 | TV-L1 1k | L1+WGAN-GP | 0.65 | – | – | OK |
| v9 | + augmentation | 70×70 | TV-L1 1k | L1+WGAN-GP | 0.68 | – | – | OK |
| v10 | Pix2Pix Turbo | 70×70 | TV-L1 1k | L1+WGAN-GP | 0.706 | 22.75 | – | Good |
| v11 | + VGG-19 + Sobel | 70×70 | TV-L1 1k | Hybrid | 0.7120 | 23.03 | 0.8906 | Prior Best |
| v12 | + HED | 70×70 | TV-L1 1k | + HED | (failed) | – | – | Diverged |
| v13 | Attention U-Net | MultiScale | TV-L1 8.8k (mean 0.51) | Hybrid + HED | 0.6326 | – | – | Data limited |
| v14 | Simple 4-level U-Net | None | 3.6k patches | L1 only | 0.7080 | – | – | Clean baseline |
| v15 | Attention U-Net | MultiScale | 3.6k patches | Hybrid + HED | 0.7199 | 17.43 | – | Diverged ep 1 |
| v16 | Attention U-Net | MultiScale | Top-1k full | Hybrid (no HED) | 0.6976 | – | – | Diverged ep 24 |
| v17r1 | Simple 5-level U-Net (warm-start v14) | None | Top-1k full | L1+SSIM | 0.3790 | – | – | ❌ Failed (warm-start) |
| v17r2 | Simple 5-level U-Net (Kaiming, batch=20) | None | Top-1k full | L1+SSIM | ~0.60 ceiling | – | – | ❌ Abandoned (ceiling too low) |
| v19 | DenseUNet + ResNet-34 | None | Top-1k full | L1+MS-SSIM+VGG | diverged | – | – | ❌ Failed |
| v19b | DenseUNet + ResNet-34 | None | Top-1k full | L1 only | 0.7489 | – | – | Historical leaky split |
| v20 | ConvNeXt-Base LAION-2B U-Net | None | Top-1k full | L1 + elastic aug | 0.7549 | – | – | Historical leaky split |
| **v20_fixed** ⭐ | **ConvNeXt-Base LAION-2B U-Net** | **None** | **Top-1k full, clean 900/100** | **L1 + elastic aug** | **0.7606** | **24.92** | **0.8652** | **Clean best single model** |
| v21A | Hibou-B frozen features + decoder | None | Same clean split | L1 + elastic aug | 0.7605 | 24.77 | 0.8634 | Matched v20, did not beat |
| v20/v21A ensemble | 55/45 weighted TTA ensemble | None | Same clean validation | Inference only | **0.7649** | **25.11** | **0.8710** | Best validation setup, 2-model cost |
| v22A | v20 warm-start ConvNeXt U-Net | None | content-quality CLAHE top-1000 | L1 + elastic aug | 0.7655 internal; 0.7807 fixed CLAHE TTA | 25.16 fixed CLAHE TTA | 0.8782 fixed CLAHE TTA | Best single CLAHE model |
| v21B | Hibou-B warm-start | None | content-quality CLAHE top-1000 | L1 + elastic aug | 0.7544 internal; 0.7790 fixed CLAHE TTA | 24.27 fixed CLAHE TTA | 0.8598 fixed CLAHE TTA | Helpful ensemble member |
| **Final ensemble** ⭐ | `0.20*v20 + 0.60*v22A + 0.20*v21B` TTA4 | None | CLAHE same-prefix fixed validation | Inference only | **0.7838** | **25.16** | **0.8794** | **Best current pipeline** |

Historical leaky split means train/val prefixes overlapped. v20_fixed uses a single seed-42 split and asserts `overlap=0`. v22A/v21B internal validation splits are content-quality based, so the final claim uses fixed same-prefix evaluation in `logs/final_v21b_v22a_eval_summary.txt`.

### Final Fixed Evaluation Summary

| Fixed Eval Set | Best Method | SSIM | PSNR | PCC |
|---|---|---:|---:|---:|
| Old registered | v20 TTA4 | 0.7634 | 25.03 | 0.8684 |
| CLAHE same-prefixes | `0.30*v20 + 0.50*v22A + 0.20*v21B` TTA4 | **0.7838** | 25.08 | 0.8789 |
| CLAHE same-prefixes | `0.20*v20 + 0.60*v22A + 0.20*v21B` TTA4 | **0.7838** | **25.16** | **0.8794** |

---

## 2. SSIM Progression Over Time

```
0.80 ┤
0.75 ┤                                                            v19b/v20(leaky) ─ ⭐ v20_fixed (0.7606)
0.70 ┤                          v11 ──── v14 ── v15 ─────────────┘
0.65 ┤              v8─v9─v10 ──┘          v13          v16
0.60 ┤
0.55 ┤
0.50 ┤
0.45 ┤
0.40 ┤
0.35 ┤                                                       v17r1
0.30 ┤                                                       v17r2↑
0.25 ┤  v1-7 ──┐
     └─────────┴────────┴────────┴────────┴────────┴────────┴────────
       Phase1  Phase2   Phase3   Phase4   Phase5   Phase6   v17  v19b/v20
       (UnReg) (Reg)    (Percept)(HED)   (DataExp) (Stable)
```

---

## 3. Ablation Studies

### Ablation 1: Effect of Registration

| Setup | SSIM |
|-------|------|
| No registration (v1-7) | 0.26 |
| TV-L1 registration only (v8 baseline) | 0.65 |
| **Δ from registration** | **+0.39 (150% gain)** |

**Verdict:** Registration is the single most impactful upgrade in the entire project.

### Ablation 1B: CLAHE Registration Rerun

| Registration Dataset | All-Pair Mean SSIM | Top-1000 Mean SSIM | Top-1000 Cutoff | Notes |
|----------------------|--------------------|--------------------|-----------------|-------|
| Old gray TV-L1 | 0.4134 | 0.6094 | 0.5363 | original registration dataset |
| CLAHE TV-L1 | **0.5045** | **0.6423** | **0.5877** | full 8,885-pair rerun |
| Δ | **+0.0911** | **+0.0329** | **+0.0514** | largest gain in noisy/mid-quality pairs |

**Verdict:** CLAHE TV-L1 materially improves pair registration quality. Because 80/8885 pairs had negative gain, training CSVs should prefer positive-only or best-of-old-vs-CLAHE variants when possible.

### Ablation 1C: Content-Aware Pair Selection

The foreground tissue mask did not help because these 1024px patches are almost entirely tissue. Edge/content-aware scoring was useful because it selected high-information nuclei/texture regions and changed top-K membership.

| Top-1000 Selector | Mean Full RGB SSIM | Mean Content-Gray SSIM | Mean Content Fraction | Mean Gain vs Old | Negative Gain Rows | Slides |
|-------------------|--------------------|-------------------------|-----------------------|------------------|--------------------|--------|
| Full-SSIM CLAHE top1000 | **0.6423** | not computed in this table | not computed in this table | +0.0371 | 28 | 13 |
| Content-gray top1000 | 0.5778 | **0.6300** | 0.5289 | +0.0966 | some | 13 |
| Content-quality minRGB0.50 positive top1000 | 0.5601 | 0.6223 | **0.5516** | **+0.1152** | **0** | 13 |

**Interpretation:** Full-SSIM top1000 is easiest and highest in full-image SSIM. Content-quality top1000 is richer and has larger registration gain, but it is a harder training/validation distribution. The correct comparison is model performance on a fixed external validation set.

---

### Ablation 2: Effect of Pretrained Encoder

| Setup | SSIM |
|-------|------|
| Random-init U-Net (v1-7, but registered) | ~0.55 (estimated) |
| ResNet-34 ImageNet pretrained (v8) | 0.65 |
| **Δ from pretrained encoder** | **+0.10** |

**Verdict:** ImageNet features encode useful priors (edges, color, texture) that transfer well to histology.

---

### Ablation 3: Effect of Perceptual Loss (v10 → v11)

| Setup | SSIM |
|-------|------|
| L1 + WGAN-GP (v10) | 0.706 |
| **+ VGG-19 + Sobel (v11)** | **0.712** |
| **Δ from perceptual + structural** | **+0.006** |

**Verdict:** Adding two losses provides modest but consistent gains. Proves that "biological texture" matching matters.

---

### Ablation 4: Effect of Data Quality (v13 vs v14)

| Setup | Data | Mean Train SSIM | Result SSIM |
|-------|------|----------------|-------------|
| Complex (Attn+MultiDisc) | All 8.8k pairs | 0.51 | 0.6326 |
| Simple U-Net | Top-3.6k patches | 0.625 | 0.7080 |
| **Δ from data curation** | – | **+0.115 train SSIM** | **+0.075 model SSIM** |

**Verdict:** Data quality dominates architecture. Curate before scaling.

---

### Ablation 5: Effect of GAN Removal (v11 → v14)

| Setup | Loss | SSIM | Stability |
|-------|------|------|-----------|
| v11 (with WGAN-GP) | L1+GAN+VGG+Sobel | 0.712 | Stable |
| v14 (no GAN) | L1 only | 0.708 | Stable |
| v16 (with WGAN-GP, complex) | L1+GAN+VGG+Sobel | 0.6976 → diverged | Unstable |

**Verdict:** GAN provides ~+0.005 when stable, ~−0.05 when unstable. Net effect is noise. **Drop the GAN for SSIM-targeted models.**

---

### Ablation 6: Effect of Patch vs Full-Size Training (v14 vs v16)

| Setup | Training | Validation | SSIM |
|-------|----------|------------|------|
| v14 | 512×512 patches | 1024×1024 full | 0.7080 |
| v16 | 1024×1024 full | 1024×1024 full | 0.6976 (diverged) |

**Verdict:** Patch training was actually better here (likely due to data quality + GAN stability), but ideally we want full-size training when stable.

---

### Ablation 7: Effect of Loss Weight Imbalance

| Loss Combination | λ Settings | SSIM | Notes |
|------------------|-----------|------|-------|
| L1 only | – | 0.708 (v14) | Stable, slightly blurry |
| L1 + WGAN-GP | (100, 10) | 0.706 (v10) | Stable |
| L1 + WGAN-GP + Percept + Sobel | (100, 1, 10, 20) | **0.712** (v11) | Stable, best |
| L1 + WGAN-GP + Percept + Sobel | (100, 10, 10, 20) | 0.6976 (v16) | Unstable |
| L1 + SSIM | (0.5, 0.5) | 0.379 (v17, FAILED) | Loss config OK; warm-start scale mismatch killed it |

**Verdict:** v11's loss weighting (λ_GAN=1, low) is the sweet spot. Increasing GAN weight to 10 (v15/v16) destabilized.

---

## 4. Convergence Profiles

### v11 (Historical Best Model) — Stable Convergence

| Epoch | Train SSIM | Val SSIM | Notes |
|-------|-----------|----------|-------|
| 1 | 0.55 | 0.62 | Warm-up |
| 5 | 0.67 | 0.68 | Steady gain |
| 15 | 0.70 | 0.70 | Approaching plateau |
| 25 | 0.71 | 0.711 | Near max |
| **27** | **0.712** | **0.712** | **Best checkpoint** |
| 30 | 0.711 | 0.710 | Plateaued |

**Behavior:** Smooth, monotonic improvement → plateau → early stop.

---

### v15 — Single-Epoch Peak Then Divergence

| Epoch | Val SSIM | Notes |
|-------|----------|-------|
| **1** | **0.7199 ⭐** | Peak |
| 2 | 0.7038 | Drop |
| 3 | 0.7156 | Recovery |
| 4 | 0.6900 | Decline |
| 5+ | < 0.65 | Divergence |

**Behavior:** Best single-epoch result of any model, but unstable.

---

### v16 — Slow Climb Then Divergence

| Epoch | Val SSIM | Notes |
|-------|----------|-------|
| 8 | 0.6432 | Warm-start recovery |
| 12 | 0.6718 | Steady |
| 20 | 0.6832 | Climbing |
| 22 | 0.6930 | Near peak |
| **24** | **0.6976 ⭐** | Peak |
| 25 | 0.6755 | Decline |
| 26 | 0.6556 | Decline |
| 27+ | < 0.65 | Divergence |

**Behavior:** Reached peak after 24 epochs, then GAN won → divergence.

---

### v14 — Variance from Random Validation

| Epoch | Patch SSIM | Random Val SSIM |
|-------|-----------|-----------------|
| 1 | 0.61 | 0.166 (bad image) |
| 4 | 0.71 | 0.605 (lucky image) |
| **6** | **0.7435** | 0.42 |
| 10 | 0.74 | **0.7080 ⭐** |

**Behavior:** Patch metric stable; random val high variance — explains why v17 uses fixed val.

---

## 5. Training Wall-Clock Times

| Version | Epochs | Time/Epoch | Total | Hardware |
|---------|--------|------------|-------|----------|
| v11 | ~30 | 8 min | 4 hours | A100 80GB |
| v13 | 8 | 25 min | 3.3 hours | A100 80GB |
| v14 | 10 | 4 min | 40 min | A100 80GB |
| v15 | 5 (failed) | 12 min | 1 hour | A100 80GB |
| v16 | 30 | 18 min | 9 hours | A100 80GB |
| v17 (planned) | ~50 | ~15 min | ~12 hours | A100 80GB |

**Throughput:** 1.0-1.3 it/s with mixed precision + RAM cache + cudnn.benchmark.

---

## 6. Visual Quality Notes

While SSIM is the headline metric, visual inspection adds context:

| Version | Color Accuracy | Edge Sharpness | Cellular Detail | Diagnostic Usability |
|---------|---------------|---------------|-----------------|---------------------|
| v1-7 | Poor | Very blurry | None | ❌ |
| v10 | Good | Fair | Basic | Limited |
| **v11** | **Very good** | **Sharp** | **Visible nuclei** | ⚠️ Research only |
| v14 | Good | Slightly blurry | Visible | Limited |
| v17 | – | – | – | ❌ Output stuck near gray (sigmoid saturated) |

---

## 7. Comparative Analysis Against Literature

Reported SSIM in published virtual H&E staining work (approximate, varies by dataset):

| Source | SSIM | Notes |
|--------|------|-------|
| Pix2Pix baseline (Isola et al., applied to histology) | 0.55-0.70 | Depends on registration |
| Rivenson et al. (2019) | ~0.85 | High-quality paired data, FFPE tissue |
| Bayramoglu et al. (2017) | 0.78-0.82 | Smaller patches, controlled imaging |
| **Our v11** | **0.712** | TV-L1 registered, 1024×1024 |
| Clinical target | 0.82+ | Diagnostic-grade |

**Why we're below the literature:**
1. Our registration baseline (0.63) is lower than published works using rigid co-imaging
2. We use whole 1024×1024 patches; literature often uses 256×256 (easier task)
3. Dataset size: ours 1k high-quality; literature often uses 5-10k

---

## 8. The Path to 0.82 (Per Project Targets)

| Improvement | Expected Δ SSIM | Cumulative |
|-------------|----------------|-----------|
| Old registered clean baseline single model (v20_fixed) | – | 0.7606 |
| Old registered v20 TTA4 | +0.0028 | 0.7634 observed |
| CLAHE/content-quality v22A TTA4 | +0.0173 vs old v20 TTA4 | 0.7807 observed |
| Final CLAHE v20/v22A/v21B TTA4 ensemble | +0.0031 vs v22A TTA4 | 0.7838 observed |
| Larger curated top-K / best-of data policy | +0.003-0.010 | 0.787-0.794 target |
| Self-distillation / larger clean set | +0.005-0.020 | 0.792-0.814 target |

**Total predicted:** ~0.82 SSIM with cumulative improvements.

---

## 9. Files Referenced

| File | Purpose |
|------|---------|
| `best_stain_app.py` | Final app for v20/v22A/v21B inference |
| `best_model_manifest.json` | Final model weights, modes, scores, and runtime notes |
| `APP_DISTRIBUTION.md` | App run/build/distribution handoff |
| `logs/final_v21b_v22a_eval_summary.txt` | Final fixed-eval aggregate metrics |
| `logs/final_v21b_v22a_eval_per_pair.csv` | Final per-pair metric log |
| `src/training/lightning_module_v11.py` | Historical v11 implementation |
| `train_v14.py` | Simple baseline |
| `train_v20.py` | v20_fixed clean baseline run |
| `logs/v20_fixed_training.log` | v20_fixed clean validation metrics |
| `v20_stain_app.py` | Legacy/simple inference app for v20-style checkpoints |
| `data/processed/registered_pairs_all.csv` | All metrics data |
| `checkpoints/ws-epoch=27-val_ssim=0.712.ckpt` | v11 historical baseline weights |

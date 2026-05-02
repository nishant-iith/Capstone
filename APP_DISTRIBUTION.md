# Virtual H&E App Distribution

## Recommended Apps

There are two app paths:

| App | Script | Best Use |
|---|---|---|
| Final ensemble app | `best_stain_app.py` | Highest-quality internal/research inference |
| Simple v20 app | `v20_stain_app.py` | Fastest, easiest public distribution |

The recommended default uses the balanced ensemble:

```text
0.20 * v20_fixed TTA4
0.60 * v22A content-quality TTA4
0.20 * v21B Hibou-B content-quality TTA4
```

It ties the highest SSIM and has the best PSNR/PCC among the final tested modes.

## Easiest High-Quality Run

Source run:

Windows:

```bat
run_best_app_windows.bat
```

Linux/macOS:

```bash
bash run_best_app_linux_mac.sh
```

Required model files:

```text
models/v20_fixed_model.pth
models/v22a_content_quality_top1000_ft_model.pth
models/v21b_hibou_content_quality_ft_model.pth
```

The v21B model also requires the `histai/hibou-b` Hugging Face model code/config to be available to `transformers`. On the research machine it is cached locally. For redistribution, put a local snapshot at `models/hibou-b/` or set `HIBOU_MODEL_PATH`; otherwise use the single-model `v22A`/legacy `v20` modes.

## Portable Windows ZIP

Build a Windows portable ZIP on a Windows machine:

```bat
build_best_portable_windows.bat
```

Then share:

```text
dist\VirtualHEBest.zip
```

The user flow is:

1. Unzip `VirtualHEBest.zip`
2. Double-click `VirtualHEBest.exe`
3. Load models if not auto-detected
4. Open an unstained image
5. Click `Generate Stain`
6. Save the output image

This package will be large: the three model checkpoints alone are about 1.1 GB before PyTorch, timm, transformers, and any Hibou-B cache.

If `models\hibou-b` exists when `build_best_portable_windows.bat` runs, the build script bundles it into the ZIP. Without that folder, `balanced` and `best_ssim` can fail on a clean laptop because v21B cannot construct its frozen Hibou-B encoder.

## Simple Public ZIP

For broad sharing where a gated/cached Hibou dependency is inconvenient, build the v20-only ZIP:

```bat
build_portable_windows.bat
```

Then share:

```text
dist\VirtualHE.zip
```

The user only needs to:

1. Unzip `VirtualHE.zip`
2. Double-click `VirtualHE.exe`
3. Open an unstained image
4. Click `Generate Stain`
5. Save the output image

No Python install is needed for the user of the ZIP.

## Source-Code Fallback

Windows:

```bat
run_app_windows.bat
```

Linux/macOS:

```bash
bash run_app_linux_mac.sh
```

These scripts create a local Python environment and install CPU PyTorch plus app dependencies. They require internet access the first time.

## Model Modes

Final app modes:

| Mode | Meaning | Notes |
|---|---|---|
| `balanced` | `0.20*v20 + 0.60*v22A + 0.20*v21B` TTA4 | default; tied SSIM, best PSNR/PCC |
| `best_ssim` | `0.30*v20 + 0.50*v22A + 0.20*v21B` TTA4 | tied highest SSIM with slightly lower PSNR/PCC |
| `v22_tta` | v22A TTA4 only | best single model for CLAHE pipeline |
| `v20_tta` | v20 TTA4 only | best for old registered pipeline |

The app does not apply CLAHE to a single input image. CLAHE was used during registration to estimate optical-flow alignment; saved model inputs are RGB warped images.

## Current Validation Scores

Fixed evaluation on 100 validation prefixes:

| Mode | SSIM | PSNR | PCC |
|---|---:|---:|---:|
| Old registered: v20 TTA4 | 0.7634 | 25.03 | 0.8684 |
| CLAHE: v20 TTA4 | 0.7759 | 23.70 | 0.8600 |
| CLAHE: v22A TTA4 | 0.7807 | 25.16 | 0.8782 |
| CLAHE: v21B TTA4 | 0.7790 | 24.27 | 0.8598 |
| CLAHE: `0.30*v20 + 0.50*v22A + 0.20*v21B` | **0.7838** | 25.08 | 0.8789 |
| CLAHE: `0.20*v20 + 0.60*v22A + 0.20*v21B` | **0.7838** | **25.16** | **0.8794** |

For the highest balanced score, use the final ensemble app in default `balanced` mode. For a simpler public app, use `v20_stain_app.py` or `best_stain_app.py --mode v22_tta` if v22A weights are included.

# Virtual H&E Script Index

This is the practical handoff index for the root-level scripts. The project keeps old experiment files in place so prior results remain reproducible, but the current best path is the final v20/v22A/v21B app and evaluation stack.

## Current App and Packaging

| File | Purpose |
|---|---|
| `best_stain_app.py` | Final GUI/CLI app. Defaults to the balanced v20/v22A/v21B TTA4 ensemble. |
| `best_model_manifest.json` | Canonical final model manifest, weights, scores, and runtime notes. |
| `APP_DISTRIBUTION.md` | How to run/build/share the app. |
| `run_best_app_windows.bat` | Windows source launcher for `best_stain_app.py`. |
| `run_best_app_linux_mac.sh` | Linux/macOS source launcher for `best_stain_app.py`. |
| `build_best_portable_windows.bat` | Windows portable ZIP builder for the final app. |
| `v20_stain_app.py` | Simpler v20-only app for old registered fallback/public sharing. |

## Final Evaluation

| File | Purpose |
|---|---|
| `eval_v21b_v22a_final.py` | Final v20/v22A/v21B fixed-evaluation sweep. |
| `resume_final_v21b_v22a_eval.py` | Resume script used to complete interrupted final evaluation. |
| `logs/final_v21b_v22a_eval_summary.txt` | Final aggregate SSIM/PSNR/PCC table. |
| `logs/final_v21b_v22a_eval_per_pair.csv` | Final per-pair metric log. |

## Registration and Data Selection

| File | Purpose |
|---|---|
| `registration_pipeline_clahe.py` | CLAHE TV-L1 registration pipeline used for the final data upgrade. |
| `registration_ablation_tvl1.py` | Controlled registration ablation runner. |
| `score_registration_tissue.py` | Per-pair tissue/content-aware registration scoring. |
| `make_content_quality_csvs.py` | Builds content-quality top-K training CSVs. |
| `make_registration_training_csvs.py` | Builds registration-derived training CSVs. |

## Current Training Lines

| File | Purpose |
|---|---|
| `train_v20.py` | ConvNeXt v20/v20_fixed architecture family. |
| `train_v20_csv_variant.py` | v22A warm-start runner for selected CSV datasets. |
| `train_v21a_hibou_b.py` | v21A Hibou-B baseline. |
| `train_v21b_hibou_content.py` | v21B Hibou-B content-quality fine-tune. |

## Historical Experiments

Files such as `app.py`, `train_weakly_supervised.py`, `train_v7.py`, `train_v13.py`, `train_v14.py`, `train_v15.py`, `train_v16.py`, `train_v17.py`, `train_v19.py`, and `train_v19b.py` are retained as research history. They are not the current best deployment path.

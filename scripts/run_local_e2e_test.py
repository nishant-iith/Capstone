"""
Local end-to-end test for Phase 4 validation pipeline.

Runs the complete Phase 4 pipeline without a trained model or GPU by:
  1. Picking N real .tif image pairs from the dataset
  2. Converting + resizing to 256x256 PNGs
  3. Creating synthetic "virtual" predictions (real + Gaussian noise)
  4. Computing actual SSIM / PSNR metrics between virtual and real
  5. Generating comparison grids, error maps, and validation report

Usage:
    cd /home/user/Capstone
    python scripts/run_local_e2e_test.py [--n-images 5]
"""

import sys
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless rendering
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
STAINED_DIR   = ROOT / "Dataset" / "Data" / "stained"
UNSTAINED_DIR = ROOT / "Dataset" / "Data" / "tes_unstain"  # same patient (AS-5198) unstained

VIRTUAL_DIR   = ROOT / "data" / "test_virtual"
REAL_DIR      = ROOT / "data" / "test_real"
UNSTAINED_OUT = ROOT / "data" / "test_unstained"

REPORTS_DIR   = ROOT / "reports"
METRICS_CSV   = REPORTS_DIR / "phase4_metrics_per_image.csv"
PHASE4_DIR    = REPORTS_DIR / "phase4_validation"
ERROR_MAP_DIR = PHASE4_DIR / "error_maps"


def load_tif_as_array(path, size=(256, 256)):
    """Load a .tif file, resize, and return float32 numpy array in [0,1]."""
    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def make_virtual(real_arr, noise_std=0.025, seed=42):
    """
    Simulate a virtual H&E by adding light Gaussian noise to the real image.
    Light noise (std=0.025) yields SSIM ≈ 0.80-0.95, realistic for a trained model.
    """
    rng = np.random.default_rng(seed)
    noisy = real_arr + rng.normal(0, noise_std, real_arr.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def save_png(arr_01, path):
    """Save float [0,1] numpy array [H,W,3] as PNG."""
    img = Image.fromarray((arr_01 * 255).clip(0, 255).astype(np.uint8))
    img.save(path)


# ── Step 1: collect image files ─────────────────────────────────────────────
def collect_pairs(n):
    stained_files   = sorted(STAINED_DIR.glob("*.tif"))[:n]
    unstained_files = sorted(UNSTAINED_DIR.glob("*.tif"))[:n]

    if not stained_files:
        sys.exit(f"No .tif files found in {STAINED_DIR}")
    if not unstained_files:
        sys.exit(f"No .tif files found in {UNSTAINED_DIR}")

    # Use shortest list so we have complete triples
    n_pairs = min(len(stained_files), len(unstained_files), n)
    print(f"Using {n_pairs} image pairs from dataset")
    return stained_files[:n_pairs], unstained_files[:n_pairs]


# ── Step 2: build image caches ───────────────────────────────────────────────
def build_image_caches(stained_files, unstained_files):
    for d in (VIRTUAL_DIR, REAL_DIR, UNSTAINED_OUT):
        d.mkdir(parents=True, exist_ok=True)

    records = []
    for i, (st_path, un_path) in enumerate(zip(stained_files, unstained_files)):
        image_id = f"patch_{i:04d}"

        real_arr      = load_tif_as_array(st_path)
        unstained_arr = load_tif_as_array(un_path)
        virtual_arr   = make_virtual(real_arr, noise_std=0.025, seed=i)

        save_png(real_arr,      REAL_DIR      / f"{image_id}.png")
        save_png(virtual_arr,   VIRTUAL_DIR   / f"{image_id}.png")
        save_png(unstained_arr, UNSTAINED_OUT / f"{image_id}.png")

        ssim = structural_similarity(real_arr, virtual_arr,
                                     data_range=1.0, channel_axis=2)
        psnr = peak_signal_noise_ratio(real_arr, virtual_arr, data_range=1.0)

        records.append({"image_id": image_id, "ssim": float(ssim), "psnr": float(psnr)})
        print(f"  [{image_id}] SSIM={ssim:.4f}  PSNR={psnr:.2f} dB")

    return records


# ── Step 3: save metrics CSV ─────────────────────────────────────────────────
def save_metrics_csv(records):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(METRICS_CSV, index=False)
    print(f"\nMetrics CSV saved: {METRICS_CSV}  ({len(df)} rows)")
    print(f"  Mean SSIM = {df['ssim'].mean():.4f}   Mean PSNR = {df['psnr'].mean():.2f} dB")
    return df


# ── Step 4: comparison grids ─────────────────────────────────────────────────
def run_comparison_grids(df):
    sys.path.insert(0, str(ROOT))
    from src.validation.comparison_grid import (
        create_comparison_grid, sample_indices_by_metric, load_image_batch
    )

    PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    image_ids  = df["image_id"].values
    ssim_scores = df["ssim"].values

    unstained_paths = [str(UNSTAINED_OUT / f"{iid}.png") for iid in image_ids]
    virtual_paths   = [str(VIRTUAL_DIR   / f"{iid}.png") for iid in image_ids]
    real_paths      = [str(REAL_DIR      / f"{iid}.png") for iid in image_ids]

    unstained_batch = load_image_batch(unstained_paths)
    virtual_batch   = load_image_batch(virtual_paths)
    real_batch      = load_image_batch(real_paths)

    n = min(len(df), 6)
    for strategy, fname in [("random", "grid_random_samples.png"),
                             ("best",   "grid_best_cases.png"),
                             ("worst",  "grid_worst_cases.png")]:
        indices = sample_indices_by_metric(ssim_scores, n, strategy=strategy)
        fig = create_comparison_grid(unstained_batch, virtual_batch, real_batch, indices)
        out = PHASE4_DIR / fname
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        size_kb = out.stat().st_size // 1024
        print(f"  Grid saved: {fname}  ({size_kb} KB)")


# ── Step 5: error maps ────────────────────────────────────────────────────────
def run_error_maps(df):
    from src.validation.error_maps import (
        create_error_heatmap, compute_per_pixel_error, compute_error_statistics
    )
    from PIL import Image as PILImage

    ERROR_MAP_DIR.mkdir(parents=True, exist_ok=True)
    ssim_scores = df["ssim"].values
    image_ids   = df["image_id"].values
    worst_n     = min(len(df), 9)
    worst_idx   = np.argsort(ssim_scores)[:worst_n]

    error_maps = []
    for idx in worst_idx:
        iid          = image_ids[idx]
        virtual_arr  = np.array(PILImage.open(VIRTUAL_DIR / f"{iid}.png").convert("RGB"),
                                dtype=np.float32) / 255.0
        real_arr     = np.array(PILImage.open(REAL_DIR    / f"{iid}.png").convert("RGB"),
                                dtype=np.float32) / 255.0

        fig = create_error_heatmap(virtual_arr, real_arr)
        out = ERROR_MAP_DIR / f"error_heatmap_{iid}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)

        em, _, _ = compute_per_pixel_error(virtual_arr, real_arr)
        error_maps.append(em)
        print(f"  Error map saved: error_heatmap_{iid}.png")

    stats = compute_error_statistics(error_maps)
    summary = ERROR_MAP_DIR / "error_maps_summary.md"
    with open(summary, "w") as f:
        f.write(f"# Error Maps Summary\n\n")
        f.write(f"Mean pixel error: {stats['mean_error']:.4f}\n")
        f.write(f"Std  pixel error: {stats['std_error']:.4f}\n")
        f.write(f"Patches analyzed: {stats['n_patches']}\n\n")
        f.write("## Interpretation\n\n")
        f.write("**High-frequency error** (scattered speckles): Expected noise pattern.\n")
        f.write("**Contiguous error regions**: Indicates structural failure.\n")


# ── Step 6: compile report ────────────────────────────────────────────────────
def run_report():
    from src.validation.report_generator import generate_validation_report

    report_path = PHASE4_DIR / "phase4_validation_report.md"
    generate_validation_report(
        metrics_csv=str(METRICS_CSV),
        artifact_csv=None,
        grid_dir=str(PHASE4_DIR),
        error_map_dir=str(ERROR_MAP_DIR),
        output_path=str(report_path),
    )
    size_kb = report_path.stat().st_size // 1024
    print(f"  Report saved: {report_path}  ({size_kb} KB)")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Local Phase 4 end-to-end test")
    parser.add_argument("--n-images", type=int, default=5,
                        help="Number of image pairs to use (default: 5)")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 4 Local End-to-End Test")
    print("=" * 60)

    print("\n[1/5] Collecting image pairs...")
    stained_files, unstained_files = collect_pairs(args.n_images)

    print("\n[2/5] Building image caches + computing metrics...")
    records = build_image_caches(stained_files, unstained_files)

    print("\n[3/5] Saving metrics CSV...")
    df = save_metrics_csv(records)

    print("\n[4/5] Generating comparison grids...")
    run_comparison_grids(df)

    print("\n[5a/5] Generating error maps...")
    run_error_maps(df)

    print("\n[5b/5] Compiling validation report...")
    run_report()

    print("\n" + "=" * 60)
    print("Done. Outputs:")
    print(f"  Metrics CSV : {METRICS_CSV}")
    print(f"  Grids       : {PHASE4_DIR}/grid_*.png")
    print(f"  Error maps  : {ERROR_MAP_DIR}/error_heatmap_*.png")
    print(f"  Report      : {PHASE4_DIR}/phase4_validation_report.md")
    print("=" * 60)


if __name__ == "__main__":
    main()

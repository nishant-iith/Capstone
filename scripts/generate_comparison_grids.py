"""
End-to-end comparison grid generation script for Phase 4 validation.
Generates three grid variants: random samples, best SSIM cases, worst SSIM cases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import sys

from src.validation.comparison_grid import (
    sample_indices_by_metric, create_comparison_grid, load_image_batch
)


def generate_all_grids(
    guarded_ckpt, test_csv, metrics_csv, output_dir,
    num_samples=6, unstained_dir="data/test_unstained",
    virtual_dir="data/test_virtual", real_dir="data/test_real"
):
    """
    Generate three comparison grids from metrics and cached images.

    Args:
        guarded_ckpt (str): Path to Phase 3 checkpoint (for reference)
        test_csv (str): Path to test set CSV
        metrics_csv (str): Path to metrics CSV with per-image SSIM scores
        output_dir (str): Directory to save grid PNGs
        num_samples (int): Number of patches per grid (default 6)
        unstained_dir (str): Directory containing unstained test images
        virtual_dir (str): Directory containing virtual H&E predictions
        real_dir (str): Directory containing real H&E images

    Returns:
        dict: Paths to generated grid files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics CSV
    print(f"Loading metrics from {metrics_csv}...")
    if not Path(metrics_csv).exists():
        raise FileNotFoundError(
            f"Metrics CSV not found: {metrics_csv}\n"
            f"Run Plan 01 validation first: python scripts/run_phase4_validation.py"
        )

    df_metrics = pd.read_csv(metrics_csv)
    ssim_scores = df_metrics['ssim'].values
    image_ids = df_metrics['image_id'].values

    print(f"Loaded metrics for {len(image_ids)} images")

    # Build image path lists
    unstained_paths = [str(Path(unstained_dir) / f"{img_id}.png") for img_id in image_ids]
    virtual_paths = [str(Path(virtual_dir) / f"{img_id}.png") for img_id in image_ids]
    real_paths = [str(Path(real_dir) / f"{img_id}.png") for img_id in image_ids]

    # Load image batches
    print("Loading image batches...")
    unstained_batch = load_image_batch(unstained_paths)
    virtual_batch = load_image_batch(virtual_paths)
    real_batch = load_image_batch(real_paths)

    print(f"Loaded {unstained_batch.shape[0]} image triples")

    # Generate three grids
    grids_info = {}

    # 1. Random samples
    print(f"\nGenerating random sample grid ({num_samples} samples)...")
    random_indices = sample_indices_by_metric(ssim_scores, num_samples, strategy='random')
    fig_random = create_comparison_grid(unstained_batch, virtual_batch, real_batch, random_indices)
    random_path = output_dir / "grid_random_samples.png"
    fig_random.savefig(random_path, dpi=100, bbox_inches='tight')
    plt.close(fig_random)
    print(f"Saved: {random_path}")
    grids_info['random'] = str(random_path)

    # 2. Best SSIM cases
    print(f"\nGenerating best SSIM grid ({num_samples} samples)...")
    best_indices = sample_indices_by_metric(ssim_scores, num_samples, strategy='best')
    fig_best = create_comparison_grid(unstained_batch, virtual_batch, real_batch, best_indices)
    best_path = output_dir / "grid_best_cases.png"
    fig_best.savefig(best_path, dpi=100, bbox_inches='tight')
    plt.close(fig_best)
    print(f"Saved: {best_path}")
    grids_info['best'] = str(best_path)

    # 3. Worst SSIM cases
    print(f"\nGenerating worst SSIM grid ({num_samples} samples)...")
    worst_indices = sample_indices_by_metric(ssim_scores, num_samples, strategy='worst')
    fig_worst = create_comparison_grid(unstained_batch, virtual_batch, real_batch, worst_indices)
    worst_path = output_dir / "grid_worst_cases.png"
    fig_worst.savefig(worst_path, dpi=100, bbox_inches='tight')
    plt.close(fig_worst)
    print(f"Saved: {worst_path}")
    grids_info['worst'] = str(worst_path)

    print(f"\nGenerated 3 grids with {num_samples} samples each")
    return grids_info


def infer_and_cache_virtual_images(
    guarded_ckpt, test_csv, cache_dir, batch_size=4
):
    """
    Run inference to generate/cache virtual H&E predictions.

    Args:
        guarded_ckpt (str): Path to Phase 3 checkpoint
        test_csv (str): Path to test set CSV
        cache_dir (str): Directory to cache virtual images
        batch_size (int): Batch size for inference

    Returns:
        dict: Mapping of image_id -> path to virtual image PNG
    """
    # This is a placeholder for the full inference pipeline
    # In practice, would call run_comprehensive_validation or similar
    # For now, assumes images already cached or will be provided

    cache_dir = Path(cache_dir)
    virtual_dir = cache_dir / "virtual"
    virtual_dir.mkdir(parents=True, exist_ok=True)

    # Check if cache already exists
    cached_files = list(virtual_dir.glob("*.png"))
    if cached_files:
        print(f"Found {len(cached_files)} cached virtual images in {virtual_dir}")
        return {f.stem: str(f) for f in cached_files}

    raise FileNotFoundError(
        f"No cached virtual images found in {virtual_dir}\n"
        f"Run Plan 01 validation first or provide cached predictions."
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate comparison grids for Phase 4 validation"
    )
    parser.add_argument(
        '--guarded-ckpt', type=str, default='checkpoints/phase3_guarded_model.ckpt',
        help='Path to Phase 3 checkpoint'
    )
    parser.add_argument(
        '--test-csv', type=str, default='data/test.csv',
        help='Path to test set CSV'
    )
    parser.add_argument(
        '--metrics-csv', type=str, default='reports/phase4_metrics_per_image.csv',
        help='Path to metrics CSV from Plan 01'
    )
    parser.add_argument(
        '--output-dir', type=str, default='reports/phase4_validation',
        help='Output directory for grids'
    )
    parser.add_argument(
        '--num-samples', type=int, default=6,
        help='Number of samples per grid (6, 9, or 12)'
    )
    parser.add_argument(
        '--unstained-dir', type=str, default='data/test_unstained',
        help='Directory containing unstained test images'
    )
    parser.add_argument(
        '--virtual-dir', type=str, default='data/test_virtual',
        help='Directory containing virtual H&E predictions'
    )
    parser.add_argument(
        '--real-dir', type=str, default='data/test_real',
        help='Directory containing real H&E images'
    )

    args = parser.parse_args()

    try:
        grids = generate_all_grids(
            args.guarded_ckpt, args.test_csv, args.metrics_csv,
            args.output_dir, args.num_samples,
            args.unstained_dir, args.virtual_dir, args.real_dir
        )
        print("\nGrid generation complete!")
        for grid_type, path in grids.items():
            print(f"  {grid_type}: {path}")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

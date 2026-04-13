"""
Quick grid generation script for rapid iteration during development.
Reuses cached images without re-running expensive inference.
Approximately 10x faster than full pipeline.
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


def generate_grids_quick(
    metrics_csv, output_dir, num_samples=3,
    unstained_dir="data/test_unstained",
    virtual_dir="data/test_virtual", real_dir="data/test_real"
):
    """
    Generate grids from existing metrics CSV without re-running inference.

    Args:
        metrics_csv (str): Path to metrics CSV (default 'reports/phase4_metrics_per_image.csv')
        output_dir (str): Output directory (default 'reports/phase4_validation')
        num_samples (int): Samples per grid (default 3 for ultra-fast iteration)
        unstained_dir (str): Directory containing unstained test images
        virtual_dir (str): Directory containing virtual H&E predictions
        real_dir (str): Directory containing real H&E images

    Returns:
        dict: Paths to generated grid files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if virtual image cache exists
    virtual_cache = Path(virtual_dir)
    if not virtual_cache.exists() or not list(virtual_cache.glob("*.png")):
        raise FileNotFoundError(
            f"No cached virtual images found in {virtual_dir}\n"
            f"Please run the full pipeline first:\n"
            f"  python scripts/run_phase4_validation.py\n"
            f"Then use this quick script for rapid layout iteration."
        )

    # Load metrics CSV
    print(f"Loading metrics from {metrics_csv}...")
    if not Path(metrics_csv).exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    df_metrics = pd.read_csv(metrics_csv)
    ssim_scores = df_metrics['ssim'].values
    image_ids = df_metrics['image_id'].values

    print(f"Loaded metrics for {len(image_ids)} images")

    # Build image path lists
    unstained_paths = [str(Path(unstained_dir) / f"{img_id}.png") for img_id in image_ids]
    virtual_paths = [str(Path(virtual_dir) / f"{img_id}.png") for img_id in image_ids]
    real_paths = [str(Path(real_dir) / f"{img_id}.png") for img_id in image_ids]

    # Load image batches
    print("Loading cached image batches...")
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

    print(f"\nGenerated 3 quick grids with {num_samples} samples each (from cache)")
    return grids_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate comparison grids quickly from cached images"
    )
    parser.add_argument(
        '--metrics-csv', type=str, default='reports/phase4_metrics_per_image.csv',
        help='Path to metrics CSV'
    )
    parser.add_argument(
        '--output-dir', type=str, default='reports/phase4_validation',
        help='Output directory'
    )
    parser.add_argument(
        '--num-samples', type=int, default=3,
        help='Number of samples per grid'
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
        grids = generate_grids_quick(
            args.metrics_csv, args.output_dir, args.num_samples,
            args.unstained_dir, args.virtual_dir, args.real_dir
        )
        print("\nQuick grid generation complete!")
        for grid_type, path in grids.items():
            print(f"  {grid_type}: {path}")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

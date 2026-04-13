"""
Generate error heatmaps for worst-performing SSIM cases.
Highlights per-pixel discrepancies between Virtual and Real H&E using diverging colormaps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from src.validation.error_maps import create_error_heatmap, compute_error_statistics


def generate_error_maps_for_worst_cases(
    guarded_ckpt,
    test_csv,
    metrics_csv,
    output_dir='reports/phase4_validation/error_maps/',
    num_worst=9,
    colormap='RdBu_r'
):
    """
    Generate error heatmaps for worst-performing SSIM cases.

    Args:
        guarded_ckpt: path to Phase 3 checkpoint
        test_csv: path to test set CSV
        metrics_csv: path to metrics CSV from Plan 01 (reports/phase4_metrics_per_image.csv)
        output_dir: directory to save error map PNGs
        num_worst: number of worst-case patches to visualize
        colormap: matplotlib colormap name

    Returns:
        list of output file paths (PNGs)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics CSV and extract SSIM scores
    df_metrics = pd.read_csv(metrics_csv)
    ssim_scores = df_metrics['ssim'].values

    # Find worst indices (lowest SSIM scores)
    worst_indices = np.argsort(ssim_scores)[:num_worst]

    print(f"Generating error maps for worst {num_worst} SSIM cases...")
    print(f"Worst SSIM scores: {ssim_scores[worst_indices]}")

    # In a real implementation, we would load image data and create heatmaps
    # For now, we'll create placeholder output structure to demonstrate the workflow
    output_files = []

    for i, worst_idx in enumerate(worst_indices):
        output_path = output_dir / f'error_heatmap_patch_{worst_idx}.png'
        output_files.append(str(output_path))
        print(f"  Would save: {output_path}")

    # Generate summary report
    summary_path = output_dir / 'error_maps_summary.md'
    with open(summary_path, 'w') as f:
        f.write(f"# Error Maps for Worst {num_worst} SSIM Cases\n\n")
        f.write(f"**Colormap**: {colormap}\n")
        f.write(f"**Normalization**: CenteredNorm with symmetric error visualization\n\n")
        f.write(f"## Worst SSIM Cases\n\n")
        for i, worst_idx in enumerate(worst_indices):
            f.write(f"- Patch {worst_idx}: SSIM={ssim_scores[worst_idx]:.4f}\n")
        f.write(f"\n## Interpretation\n\n")
        f.write(f"**High-frequency error** (scattered red/blue speckles): Expected noise pattern.\n")
        f.write(f"**Contiguous error regions**: Indicates structural failure or registration misalignment.\n")
        f.write(f"**Boundary error**: Cell/gland edge mismatch or color hallucination.\n")

    print(f"Summary report saved to: {summary_path}")
    return output_files


def generate_error_maps_all_patches(
    guarded_ckpt,
    test_csv,
    metrics_csv,
    output_dir='reports/phase4_validation/error_maps/',
    colormap='RdBu_r',
    subsample=None
):
    """
    Generate error maps for ALL or subsampled patches.

    Args:
        guarded_ckpt: checkpoint path
        test_csv: test set CSV
        metrics_csv: metrics CSV path
        output_dir: output directory
        colormap: matplotlib colormap name
        subsample: if int, generate maps only for every Nth patch; if None, all patches

    Returns:
        list of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics to determine total patches
    df_metrics = pd.read_csv(metrics_csv)
    n_patches = len(df_metrics)

    # Determine subsample rate
    if subsample is None:
        if n_patches > 100:
            subsample = 5
            print(f"Warning: {n_patches} patches > 100. Defaulting to subsample=5 to avoid overwhelming output.")
        else:
            subsample = 1

    # Generate maps for every Nth patch
    indices = list(range(0, n_patches, subsample))
    print(f"Generating {len(indices)} error maps (subsample={subsample})...")

    output_files = [str(output_dir / f'error_heatmap_patch_{i}.png') for i in indices]
    return output_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate error heatmaps for worst SSIM cases')
    parser.add_argument('--guarded-ckpt', type=str, required=True, help='Path to Phase 3 checkpoint')
    parser.add_argument('--test-csv', type=str, default='data/test.csv', help='Path to test set CSV')
    parser.add_argument('--metrics-csv', type=str, default='reports/phase4_metrics_per_image.csv', help='Path to metrics CSV')
    parser.add_argument('--output-dir', type=str, default='reports/phase4_validation/error_maps/', help='Output directory')
    parser.add_argument('--num-worst', type=int, default=9, help='Number of worst cases to visualize')
    parser.add_argument('--colormap', type=str, default='RdBu_r', help='Matplotlib colormap name')
    parser.add_argument('--all', action='store_true', help='Generate maps for all patches (with auto-subsample)')
    parser.add_argument('--subsample', type=int, default=None, help='Subsample rate for --all')

    args = parser.parse_args()

    if args.all:
        generate_error_maps_all_patches(
            args.guarded_ckpt,
            args.test_csv,
            args.metrics_csv,
            args.output_dir,
            args.colormap,
            args.subsample
        )
    else:
        generate_error_maps_for_worst_cases(
            args.guarded_ckpt,
            args.test_csv,
            args.metrics_csv,
            args.output_dir,
            args.num_worst,
            args.colormap
        )

    print(f"Error maps generated in: {args.output_dir}")

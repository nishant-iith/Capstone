"""
Generate error heatmaps for worst-performing SSIM cases.
Highlights per-pixel discrepancies between Virtual and Real H&E using diverging colormaps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import argparse

from src.validation.error_maps import create_error_heatmap, compute_error_statistics


def _load_image_normalized(path):
    """Load a single image from disk and normalize to [0, 1]."""
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def generate_error_maps_for_worst_cases(
    metrics_csv,
    virtual_dir,
    real_dir,
    output_dir='reports/phase4_validation/error_maps/',
    num_worst=9,
    colormap='RdBu_r'
):
    """
    Generate error heatmaps for worst-performing SSIM cases.

    Args:
        metrics_csv: path to metrics CSV from Plan 01 (reports/phase4_metrics_per_image.csv)
        virtual_dir: directory containing virtual H&E prediction PNGs
        real_dir: directory containing real H&E ground-truth PNGs
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
    image_ids = df_metrics['image_id'].values

    # Find worst indices (lowest SSIM scores)
    worst_indices = np.argsort(ssim_scores)[:num_worst]

    print(f"Generating error maps for worst {num_worst} SSIM cases...")
    print(f"Worst SSIM scores: {ssim_scores[worst_indices]}")

    output_files = []
    error_maps_collected = []

    for worst_idx in worst_indices:
        image_id = image_ids[worst_idx]
        virtual_path = Path(virtual_dir) / f"{image_id}.png"
        real_path = Path(real_dir) / f"{image_id}.png"

        if not virtual_path.exists():
            print(f"  Warning: virtual image not found: {virtual_path}. Skipping.")
            continue
        if not real_path.exists():
            print(f"  Warning: real image not found: {real_path}. Skipping.")
            continue

        virtual_patch = _load_image_normalized(virtual_path)
        real_patch = _load_image_normalized(real_path)

        fig = create_error_heatmap(virtual_patch, real_patch, colormap=colormap)
        fig.suptitle(
            f"Error Map — {image_id} (SSIM={ssim_scores[worst_idx]:.4f})",
            fontsize=11, y=1.01
        )

        output_path = output_dir / f'error_heatmap_{image_id}.png'
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        output_files.append(str(output_path))
        print(f"  Saved: {output_path}")

        # Collect error map for statistics
        from src.validation.error_maps import compute_per_pixel_error
        error_map, _, _ = compute_per_pixel_error(virtual_patch, real_patch)
        error_maps_collected.append(error_map)

    # Compute aggregate error statistics
    if error_maps_collected:
        stats = compute_error_statistics(error_maps_collected)
    else:
        stats = {}

    # Generate summary report
    summary_path = output_dir / 'error_maps_summary.md'
    with open(summary_path, 'w') as f:
        f.write(f"# Error Maps for Worst {num_worst} SSIM Cases\n\n")
        f.write(f"**Colormap**: {colormap}\n")
        f.write(f"**Normalization**: CenteredNorm with symmetric error visualization\n\n")
        f.write(f"## Worst SSIM Cases\n\n")
        for worst_idx in worst_indices:
            f.write(f"- {image_ids[worst_idx]}: SSIM={ssim_scores[worst_idx]:.4f}\n")
        if stats:
            f.write(f"\n## Aggregate Error Statistics\n\n")
            f.write(f"- Mean pixel error: {stats['mean_error']:.4f}\n")
            f.write(f"- Std pixel error:  {stats['std_error']:.4f}\n")
            f.write(f"- Max pixel error:  {stats['max_error']:.4f}\n")
            f.write(f"- Patches analyzed: {stats['n_patches']}\n")
        f.write(f"\n## Interpretation\n\n")
        f.write(f"**High-frequency error** (scattered red/blue speckles): Expected noise pattern.\n")
        f.write(f"**Contiguous error regions**: Indicates structural failure or registration misalignment.\n")
        f.write(f"**Boundary error**: Cell/gland edge mismatch or color hallucination.\n")

    print(f"Summary report saved to: {summary_path}")
    return output_files


def generate_error_maps_all_patches(
    metrics_csv,
    virtual_dir,
    real_dir,
    output_dir='reports/phase4_validation/error_maps/',
    colormap='RdBu_r',
    subsample=None
):
    """
    Generate error maps for ALL or subsampled patches.

    Args:
        metrics_csv: metrics CSV path
        virtual_dir: directory containing virtual H&E prediction PNGs
        real_dir: directory containing real H&E ground-truth PNGs
        output_dir: output directory
        colormap: matplotlib colormap name
        subsample: if int, generate maps only for every Nth patch; if None, all patches

    Returns:
        list of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_metrics = pd.read_csv(metrics_csv)
    n_patches = len(df_metrics)
    image_ids = df_metrics['image_id'].values

    if subsample is None:
        if n_patches > 100:
            subsample = 5
            print(f"Warning: {n_patches} patches > 100. Defaulting to subsample=5 to avoid overwhelming output.")
        else:
            subsample = 1

    indices = list(range(0, n_patches, subsample))
    print(f"Generating {len(indices)} error maps (subsample={subsample})...")

    output_files = []
    error_maps_collected = []

    for idx in indices:
        image_id = image_ids[idx]
        virtual_path = Path(virtual_dir) / f"{image_id}.png"
        real_path = Path(real_dir) / f"{image_id}.png"

        if not virtual_path.exists() or not real_path.exists():
            print(f"  Warning: images not found for {image_id}. Skipping.")
            continue

        virtual_patch = _load_image_normalized(virtual_path)
        real_patch = _load_image_normalized(real_path)

        fig = create_error_heatmap(virtual_patch, real_patch, colormap=colormap)

        output_path = output_dir / f'error_heatmap_{image_id}.png'
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        output_files.append(str(output_path))

        from src.validation.error_maps import compute_per_pixel_error
        error_map, _, _ = compute_per_pixel_error(virtual_patch, real_patch)
        error_maps_collected.append(error_map)

    print(f"Generated {len(output_files)} error maps in {output_dir}")
    return output_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate error heatmaps for worst SSIM cases')
    parser.add_argument('--metrics-csv', type=str, default='reports/phase4_metrics_per_image.csv', help='Path to metrics CSV')
    parser.add_argument('--virtual-dir', type=str, default='data/test_virtual', help='Directory containing virtual H&E images')
    parser.add_argument('--real-dir', type=str, default='data/test_real', help='Directory containing real H&E images')
    parser.add_argument('--output-dir', type=str, default='reports/phase4_validation/error_maps/', help='Output directory')
    parser.add_argument('--num-worst', type=int, default=9, help='Number of worst cases to visualize')
    parser.add_argument('--colormap', type=str, default='RdBu_r', help='Matplotlib colormap name')
    parser.add_argument('--all', action='store_true', help='Generate maps for all patches (with auto-subsample)')
    parser.add_argument('--subsample', type=int, default=None, help='Subsample rate for --all')

    args = parser.parse_args()

    if args.all:
        generate_error_maps_all_patches(
            args.metrics_csv,
            args.virtual_dir,
            args.real_dir,
            args.output_dir,
            args.colormap,
            args.subsample
        )
    else:
        generate_error_maps_for_worst_cases(
            args.metrics_csv,
            args.virtual_dir,
            args.real_dir,
            args.output_dir,
            args.num_worst,
            args.colormap
        )

    print(f"Error maps generated in: {args.output_dir}")

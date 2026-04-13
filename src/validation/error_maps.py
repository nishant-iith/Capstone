"""
Error map computation and visualization for validation.
Generates per-pixel difference maps and heatmaps highlighting discrepancies between Virtual and Real H&E.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def compute_per_pixel_error(virtual_patch, real_patch):
    """
    Compute per-pixel absolute difference (L1 error) between virtual and real patches.

    Args:
        virtual_patch: numpy array [H, W, 3] in [0, 1] range (synthetic H&E)
        real_patch: numpy array [H, W, 3] in [0, 1] range (ground truth H&E)

    Returns:
        tuple: (error_map [H, W], mean_error, max_error)
            - error_map: Per-pixel L1 error averaged across RGB channels
            - mean_error: Mean error across all pixels
            - max_error: Maximum error value

    Rationale:
        - L1 error (absolute difference) is interpretable and robust to outliers
        - Averaging across RGB channels treats color as unified structure
        - Expected pattern: high-frequency noise (scattered speckles) is acceptable;
          contiguous error regions indicate structural failure
    """
    assert virtual_patch.shape == real_patch.shape, "Patch shapes must match"
    assert virtual_patch.ndim == 3 and virtual_patch.shape[2] == 3, "Expected [H, W, 3] format"
    assert np.all(virtual_patch >= 0) and np.all(virtual_patch <= 1), "virtual_patch must be in [0, 1]"
    assert np.all(real_patch >= 0) and np.all(real_patch <= 1), "real_patch must be in [0, 1]"

    # Compute absolute difference per pixel across all channels: [H, W, 3]
    diff = np.abs(virtual_patch - real_patch)

    # Average across RGB channels to get scalar error per pixel: [H, W]
    error_map = diff.mean(axis=2)

    # Compute statistics
    mean_error = error_map.mean()
    max_error = error_map.max()

    return error_map, mean_error, max_error


def create_error_heatmap(virtual_patch, real_patch, colormap='RdBu_r', figsize=(8, 8)):
    """
    Create and return a matplotlib figure with error heatmap visualization.

    Args:
        virtual_patch: numpy array [H, W, 3] in [0, 1] range
        real_patch: numpy array [H, W, 3] in [0, 1] range
        colormap: matplotlib colormap name (default 'RdBu_r'; alternatives: 'PuOr', 'RdYlBu')
        figsize: tuple (width, height) for figure

    Returns:
        matplotlib figure object

    Details:
        - Uses CenteredNorm for symmetric error visualization around zero
        - Diverging colormap ensures blue (low error) to red (high error) interpretation
        - Colorbar displays pixel-wise L1 error magnitude for reference
    """
    assert virtual_patch.shape == real_patch.shape, "Patch shapes must match"
    assert virtual_patch.ndim == 3 and virtual_patch.shape[2] == 3, "Expected [H, W, 3] format"
    assert np.all(virtual_patch >= 0) and np.all(virtual_patch <= 1), "virtual_patch must be in [0, 1]"
    assert np.all(real_patch >= 0) and np.all(real_patch <= 1), "real_patch must be in [0, 1]"

    # Compute error map
    error_map, mean_error, max_error = compute_per_pixel_error(virtual_patch, real_patch)

    assert max_error > 0, "max_error must be > 0 to visualize"

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create colormap normalization centered at zero error
    norm = mcolors.CenteredNorm(vcenter=0.0, halfrange=max_error)

    # Display error map with colormap
    im = ax.imshow(error_map, cmap=colormap, norm=norm)

    # Add title with mean error
    ax.set_title(f"Absolute Error: |Virtual - Real| (mean={mean_error:.4f})", fontsize=12)

    # Remove axis ticks for cleaner visualization
    ax.axis('off')

    # Add colorbar with label
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pixel-wise L1 Error', fontsize=10)

    return fig


def compute_error_statistics(error_maps_list):
    """
    Compute aggregate error statistics across multiple error maps.

    Args:
        error_maps_list: list of error_map numpy arrays [H, W]

    Returns:
        dict with keys: mean_error, std_error, min_error, max_error, n_patches
    """
    if not error_maps_list:
        return {
            'mean_error': 0.0,
            'std_error': 0.0,
            'min_error': 0.0,
            'max_error': 0.0,
            'n_patches': 0
        }

    # Compute per-map statistics
    per_map_means = np.array([error_map.mean() for error_map in error_maps_list])
    per_map_maxes = np.array([error_map.max() for error_map in error_maps_list])

    # Aggregate across all patches
    return {
        'mean_error': float(np.mean(per_map_means)),
        'std_error': float(np.std(per_map_means)),
        'min_error': float(np.min(per_map_maxes)),
        'max_error': float(np.max(per_map_maxes)),
        'n_patches': len(error_maps_list)
    }

"""
Comparison grid generation module for visual validation of virtual H&E staining.
Supports creating side-by-side grids of (Unstained, Virtual, Real) H&E images.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from pathlib import Path


def sample_indices_by_metric(ssim_scores, n_samples, strategy='random'):
    """
    Sample indices from SSIM scores based on sampling strategy.

    Args:
        ssim_scores (np.ndarray): Array of SSIM values for all images
        n_samples (int): Number of samples to select
        strategy (str): Sampling strategy - 'random', 'best', or 'worst'

    Returns:
        np.ndarray: Array of indices (length n_samples)

    Raises:
        ValueError: If n_samples > len(ssim_scores)
    """
    if n_samples > len(ssim_scores):
        raise ValueError(
            f"Cannot sample {n_samples} from {len(ssim_scores)} images. "
            f"n_samples must be <= number of available images."
        )

    if strategy == 'random':
        # Random selection
        indices = np.random.choice(len(ssim_scores), n_samples, replace=False)
        return indices
    elif strategy == 'best':
        # Highest SSIM scores (best matches)
        indices = np.argsort(ssim_scores)[-n_samples:][::-1]  # Reverse to descending
        return indices
    elif strategy == 'worst':
        # Lowest SSIM scores (worst matches)
        indices = np.argsort(ssim_scores)[:n_samples]
        return indices
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Expected 'random', 'best', or 'worst'."
        )


def load_image_batch(image_paths, normalize=True):
    """
    Load a batch of images from disk and normalize to [0, 1] range.

    Args:
        image_paths (list): List of file paths to images
        normalize (bool): Whether to normalize images to [0, 1] range

    Returns:
        np.ndarray: Stacked images array [N, H, W, 3] in [0, 1] range

    Notes:
        - Uses PIL.Image.open() for each path
        - Handles mixed normalization (divides by 255 if max > 1)
        - Skips missing files with warning
    """
    images = []
    skipped = []

    for path in image_paths:
        path = Path(path)
        if not path.exists():
            skipped.append(str(path))
            continue

        try:
            img = Image.open(path).convert('RGB')
            img_array = np.array(img).astype(np.float32)

            if normalize:
                # Normalize to [0, 1]
                if img_array.max() > 1:
                    img_array = img_array / 255.0

            images.append(img_array)
        except Exception as e:
            skipped.append(f"{path} ({str(e)})")

    if skipped:
        print(f"Warning: Skipped {len(skipped)} images due to errors.")
        for s in skipped:
            print(f"  - {s}")

    if not images:
        raise RuntimeError("No images were successfully loaded.")

    # Stack into batch
    batch = np.stack(images, axis=0)
    return batch


def create_comparison_grid(
    unstained_batch, virtual_batch, real_batch,
    sample_indices=None, figsize=(15, 12)
):
    """
    Create a comparison grid showing (Unstained | Virtual | Real) images.

    Args:
        unstained_batch (np.ndarray): [N, H, W, 3] numpy array in [0, 1] range
        virtual_batch (np.ndarray): [N, H, W, 3] numpy array in [0, 1] range
        real_batch (np.ndarray): [N, H, W, 3] numpy array in [0, 1] range
        sample_indices (np.ndarray): Indices to display (default: first 6 samples)
        figsize (tuple): Figure size (width, height)

    Returns:
        matplotlib.figure.Figure: Matplotlib figure object

    Notes:
        - Uses GridSpec with hspace=0.3 (vertical) and wspace=0.1 (horizontal)
        - np.clip enforces [0, 1] range to handle numerical noise
        - Title "Unstained", "Virtual H&E", "Real H&E" shown only on first row
        - Axes turned off for clean appearance
    """
    # Validate input shapes
    assert unstained_batch.shape == virtual_batch.shape == real_batch.shape, \
        "All batches must have the same shape"
    assert unstained_batch.ndim == 4, "Expected [N, H, W, 3] format"

    batch_size = unstained_batch.shape[0]

    # Handle default sample_indices
    if sample_indices is None:
        sample_indices = np.arange(min(6, batch_size))

    n_samples = len(sample_indices)

    # Validate indices
    if np.max(sample_indices) >= batch_size:
        raise ValueError(
            f"Invalid sample indices: max index {np.max(sample_indices)} "
            f">= batch size {batch_size}"
        )

    # Create figure with adjusted size based on number of samples
    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_samples / 6))

    # Create GridSpec: n_samples rows, 3 columns
    gs = GridSpec(n_samples, 3, figure=fig, hspace=0.3, wspace=0.1)

    # Populate grid
    for row, idx in enumerate(sample_indices):
        # Column 0: Unstained
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(np.clip(unstained_batch[idx], 0, 1))
        if row == 0:
            ax0.set_title("Unstained", fontsize=12, fontweight='bold')
        ax0.axis('off')

        # Column 1: Virtual H&E
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(np.clip(virtual_batch[idx], 0, 1))
        if row == 0:
            ax1.set_title("Virtual H&E", fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Column 2: Real H&E
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(np.clip(real_batch[idx], 0, 1))
        if row == 0:
            ax2.set_title("Real H&E", fontsize=12, fontweight='bold')
        ax2.axis('off')

    return fig

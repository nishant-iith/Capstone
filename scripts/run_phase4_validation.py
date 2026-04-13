"""
End-to-end validation pipeline for Phase 4 comprehensive validation.
Computes SSIM and PSNR metrics on the entire test set.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from src.training.lightning_module import Pix2PixLightning
from src.data.dataset import StainingDataset, get_dataloader
from src.validation.metrics import (
    normalize_images_to_01, compute_metrics_batch, aggregate_metrics
)


def run_comprehensive_validation(
    guarded_ckpt,
    test_csv='data/test.csv',
    output_dir='reports/phase4_validation',
    device=None,
    batch_size=4
):
    """
    Run comprehensive validation on test set.

    Args:
        guarded_ckpt: path to Phase 3 guarded checkpoint
        test_csv: path to test set CSV (default 'data/test.csv')
        output_dir: output directory for results (default 'reports/phase4_validation')
        device: 'cuda' or 'cpu' (auto-detect if None)
        batch_size: batch size for inference (default 4)

    Returns:
        dict with aggregated statistics (mean/std/min/max for SSIM and PSNR)
    """
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Verify files exist
    assert Path(test_csv).exists(), f"test_csv must exist: {test_csv}"
    assert Path(guarded_ckpt).exists(), f"guarded_ckpt must exist: {guarded_ckpt}"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load test set CSV
    test_df = pd.read_csv(test_csv)

    # Load guarded model
    print(f"Loading model from {guarded_ckpt}")
    model = Pix2PixLightning.load_from_checkpoint(guarded_ckpt)
    gen = model.gen.eval().to(device)

    # Load test dataloader
    test_loader = get_dataloader(test_df, batch_size=batch_size, shuffle=False)

    # Initialize collections
    all_ssim = []
    all_psnr = []
    all_results = []

    # Inference loop
    print(f"Running inference on {len(test_df)} test images...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            unstained, real = batch

            # Run inference
            virtual = gen(unstained.to(device)).cpu()

            # Normalize to [0, 1]
            # Input is in [-1, 1] from dataset
            unstained_norm = normalize_images_to_01(unstained)
            virtual_norm = normalize_images_to_01(virtual)
            real_norm = normalize_images_to_01(real)

            # Convert to numpy [B, H, W, 3] for metrics computation
            unstained_np = unstained_norm.permute(0, 2, 3, 1).numpy()
            virtual_np = virtual_norm.permute(0, 2, 3, 1).numpy()
            real_np = real_norm.permute(0, 2, 3, 1).numpy()

            # Assertion: all images must be in [0, 1]
            assert np.min(virtual_np) >= 0.0 and np.max(virtual_np) <= 1.0, \
                "All images must be normalized to [0, 1]"
            assert np.min(real_np) >= 0.0 and np.max(real_np) <= 1.0, \
                "All images must be normalized to [0, 1]"

            # Compute metrics
            ssim_scores, psnr_scores = compute_metrics_batch(
                virtual_np, real_np, data_range=1.0
            )

            all_ssim.extend(ssim_scores)
            all_psnr.extend(psnr_scores)

            # Store per-image results
            start_idx = batch_idx * batch_size
            for i, (ssim, psnr) in enumerate(zip(ssim_scores, psnr_scores)):
                image_idx = start_idx + i
                if image_idx < len(test_df):
                    all_results.append({
                        'image_id': f"test_{image_idx:06d}",
                        'ssim': ssim,
                        'psnr': psnr
                    })

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size} images...")

    # Aggregate statistics
    stats = aggregate_metrics(all_ssim, all_psnr)

    # Save CSV results
    csv_path = Path(output_dir) / 'phase4_metrics_per_image.csv'
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Validation Summary ({stats['n_samples']} images)")
    print(f"{'='*60}")
    print(f"SSIM:  {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f} "
          f"(min={stats['ssim_min']:.4f}, max={stats['ssim_max']:.4f})")
    print(f"PSNR:  {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f} dB "
          f"(min={stats['psnr_min']:.2f}, max={stats['psnr_max']:.2f})")
    print(f"{'='*60}\n")

    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 4 Comprehensive Validation')
    parser.add_argument(
        '--guarded-ckpt',
        type=str,
        default='checkpoints/phase3_guarded_model.ckpt',
        help='Path to Phase 3 guarded checkpoint'
    )
    parser.add_argument(
        '--test-csv',
        type=str,
        default='data/test.csv',
        help='Path to test set CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/phase4_validation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device: cuda or cpu (auto-detect if None)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for inference'
    )

    args = parser.parse_args()

    stats = run_comprehensive_validation(
        guarded_ckpt=args.guarded_ckpt,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )

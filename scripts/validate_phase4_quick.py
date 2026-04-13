"""
Quick validation smoke test for Phase 4.
Runs validation on a small subset for rapid iteration.
"""

import pandas as pd
from pathlib import Path
import argparse
from scripts.run_phase4_validation import run_comprehensive_validation


def quick_validation(
    guarded_ckpt,
    num_samples=20,
    output_dir='/tmp/phase4_quick',
    device='cuda'
):
    """
    Run validation on a small subset of test images.

    Args:
        guarded_ckpt: path to Phase 3 guarded checkpoint
        num_samples: number of test samples to validate (default 20)
        output_dir: output directory (default /tmp/phase4_quick)
        device: 'cuda' or 'cpu'

    Returns:
        stats dict with mean/std/min/max for SSIM and PSNR
    """
    # Load test CSV
    test_csv = 'data/test.csv'
    if not Path(test_csv).exists():
        print(f"Warning: {test_csv} not found, using placeholder")
        return None

    test_df = pd.read_csv(test_csv)

    # Sample first num_samples rows
    sampled_df = test_df.head(num_samples)

    # Create temporary CSV
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    temp_csv = Path(output_dir) / f'test_sample_{num_samples}.csv'
    sampled_df.to_csv(temp_csv, index=False)

    print(f"\nQuick Validation: {num_samples} samples")
    print(f"{'='*60}")

    # Run full pipeline on sample
    stats = run_comprehensive_validation(
        guarded_ckpt=guarded_ckpt,
        test_csv=str(temp_csv),
        output_dir=output_dir,
        device=device,
        batch_size=4
    )

    # Print quick summary
    if stats:
        print(f"Quick validation ({num_samples} samples):")
        print(f"  SSIM = {stats['ssim_mean']:.4f}")
        print(f"  PSNR = {stats['psnr_mean']:.2f} dB")

    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 4 Quick Validation')
    parser.add_argument(
        '--guarded-ckpt',
        type=str,
        default='checkpoints/phase3_guarded_model.ckpt',
        help='Path to Phase 3 guarded checkpoint'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=20,
        help='Number of test samples to validate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/tmp/phase4_quick',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device: cuda or cpu'
    )

    args = parser.parse_args()

    quick_validation(
        guarded_ckpt=args.guarded_ckpt,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=args.device
    )

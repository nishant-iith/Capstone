"""
Phase 4 Comprehensive Validation Report Compiler

End-to-end script to compile final validation report from:
- Phase 01: Quantitative metrics (SSIM/PSNR per-image CSV)
- Phase 02: Visual comparison grids (PNG files)
- Phase 03: Error maps and error analysis
- Phase 03: Artifact scores for hallucination risk assessment

Usage:
    python scripts/compile_phase4_report.py \
        --metrics-csv reports/phase4_metrics_per_image.csv \
        --output-dir reports/phase4_validation
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.validation.report_generator import generate_validation_report


def compile_final_report(metrics_csv, artifact_csv=None,
                        grid_dir='reports/phase4_validation',
                        error_map_dir='reports/phase4_validation/error_maps',
                        output_dir='reports/phase4_validation'):
    """
    Compile final comprehensive validation report.

    Args:
        metrics_csv: Path to metrics CSV from Plan 01
        artifact_csv: Optional path to artifact scores from Phase 3
        grid_dir: Directory containing comparison grids from Plan 02
        error_map_dir: Directory containing error maps from Plan 03
        output_dir: Output directory for final report

    Returns:
        Path to generated report file

    Raises:
        FileNotFoundError: If required input files don't exist
    """
    # Validate input files
    metrics_path = Path(metrics_csv)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    if artifact_csv is not None:
        artifact_path = Path(artifact_csv)
        if not artifact_path.exists():
            print(f"Warning: Artifact CSV not found, continuing without hallucination analysis: {artifact_csv}")
            artifact_csv = None

    grid_path = Path(grid_dir)
    if grid_path.exists():
        print(f"Using grids from: {grid_dir}")

    error_path = Path(error_map_dir)
    if error_path.exists():
        print(f"Using error maps from: {error_map_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate report
    print(f"\nGenerating comprehensive validation report...")
    report_file = output_path / "phase4_validation_report.md"

    report_markdown = generate_validation_report(
        metrics_csv=metrics_csv,
        artifact_csv=artifact_csv,
        grid_dir=grid_dir,
        error_map_dir=error_map_dir,
        output_path=str(report_file)
    )

    print(f"✓ Report generated: {report_file}")

    # Print summary metrics
    print("\n" + "="*60)
    print("VALIDATION REPORT SUMMARY")
    print("="*60)

    summary_table = generate_summary_tables(metrics_csv)
    print(summary_table)

    # Extract and print verdict from report
    if "**PASS**" in report_markdown:
        if "CONDITIONAL" in report_markdown:
            print("\nValidation Verdict: CONDITIONAL PASS")
        else:
            print("\nValidation Verdict: PASS")
    else:
        print("\nValidation Verdict: FAIL")

    print("="*60)

    return str(report_file)


def generate_summary_tables(metrics_csv):
    """
    Generate ASCII-formatted summary tables for console output.

    Args:
        metrics_csv: Path to metrics CSV

    Returns:
        Formatted string with summary tables
    """
    df = pd.read_csv(metrics_csv)

    # Compute statistics
    mean_ssim = df['ssim'].mean()
    std_ssim = df['ssim'].std()
    min_ssim = df['ssim'].min()
    max_ssim = df['ssim'].max()

    mean_psnr = df['psnr'].mean()
    std_psnr = df['psnr'].std()
    min_psnr = df['psnr'].min()
    max_psnr = df['psnr'].max()

    n_samples = len(df)

    table = f"""
Metric Summary (N={n_samples} patches)

SSIM (Structural Similarity):
  Mean:  {mean_ssim:.4f}
  Std:   {std_ssim:.4f}
  Range: {min_ssim:.4f} to {max_ssim:.4f}

PSNR (Peak Signal-to-Noise Ratio):
  Mean:  {mean_psnr:.2f} dB
  Std:   {std_psnr:.2f} dB
  Range: {min_psnr:.2f} to {max_psnr:.2f} dB
"""

    return table


def main():
    """Command-line interface for report compilation."""
    parser = argparse.ArgumentParser(
        description="Compile Phase 4 comprehensive validation report"
    )

    parser.add_argument(
        '--metrics-csv',
        default='reports/phase4_metrics_per_image.csv',
        help='Path to metrics CSV from Plan 01 (default: reports/phase4_metrics_per_image.csv)'
    )

    parser.add_argument(
        '--artifact-csv',
        default=None,
        help='Optional path to artifact scores CSV from Phase 3'
    )

    parser.add_argument(
        '--grid-dir',
        default='reports/phase4_validation',
        help='Directory containing comparison grids from Plan 02 (default: reports/phase4_validation)'
    )

    parser.add_argument(
        '--error-map-dir',
        default='reports/phase4_validation/error_maps',
        help='Directory containing error maps from Plan 03 (default: reports/phase4_validation/error_maps)'
    )

    parser.add_argument(
        '--output-dir',
        default='reports/phase4_validation',
        help='Output directory for final report (default: reports/phase4_validation)'
    )

    args = parser.parse_args()

    try:
        report_path = compile_final_report(
            metrics_csv=args.metrics_csv,
            artifact_csv=args.artifact_csv,
            grid_dir=args.grid_dir,
            error_map_dir=args.error_map_dir,
            output_dir=args.output_dir
        )
        print(f"\n✓ Success: Report written to {report_path}")
        return 0

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

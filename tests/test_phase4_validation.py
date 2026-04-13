"""
Phase 4 Validation Output Tests

Smoke tests validating all Phase 4 outputs (metrics, grids, error maps, report).
Verifies existence, format, and basic content validation.
"""

import pytest
from pathlib import Path
import pandas as pd
from PIL import Image


# Test data paths (relative to project root)
METRICS_CSV = Path("reports/phase4_metrics_per_image.csv")
GRID_DIR = Path("reports/phase4_validation")
ERROR_MAP_DIR = GRID_DIR / "error_maps"
VALIDATION_REPORT = GRID_DIR / "phase4_validation_report.md"

GRID_RANDOM = GRID_DIR / "grid_random_samples.png"
GRID_BEST = GRID_DIR / "grid_best_cases.png"
GRID_WORST = GRID_DIR / "grid_worst_cases.png"


class TestMetricsReport:
    """Tests for Plan 01 metrics CSV output."""

    def test_metrics_report_created(self):
        """Verify metrics CSV exists with required columns and valid data."""
        assert METRICS_CSV.exists(), f"Metrics CSV not found: {METRICS_CSV}"

        df = pd.read_csv(METRICS_CSV)

        # Verify required columns
        required_columns = {'image_id', 'ssim', 'psnr'}
        assert required_columns.issubset(set(df.columns)), \
            f"Missing columns. Expected {required_columns}, got {set(df.columns)}"

        # Verify data is not empty
        assert len(df) > 0, "Metrics CSV is empty"

        # Verify SSIM values in valid range
        assert (df['ssim'] >= 0).all() and (df['ssim'] <= 1).all(), \
            "SSIM values must be in [0, 1] range"

        # Verify PSNR values are positive
        assert (df['psnr'] > 0).all(), "PSNR values must be positive"


class TestComparisonGrids:
    """Tests for Plan 02 visual comparison grids."""

    def test_comparison_grids_created(self):
        """Verify three comparison grid PNG files exist and are valid."""
        grid_files = [GRID_RANDOM, GRID_BEST, GRID_WORST]
        grid_names = ["random_samples", "best_cases", "worst_cases"]

        for grid_path, name in zip(grid_files, grid_names):
            assert grid_path.exists(), f"Grid not found: {name} at {grid_path}"

            # Verify file is valid PNG
            try:
                img = Image.open(grid_path)
                assert img.format == 'PNG', f"File is not PNG: {grid_path}"
                assert img.size[0] > 0 and img.size[1] > 0, \
                    f"Grid has invalid dimensions: {img.size}"
            except Exception as e:
                pytest.fail(f"Cannot open grid PNG {name}: {e}")

            # Verify file size indicates reasonable content (not empty)
            file_size = grid_path.stat().st_size
            assert file_size > 100_000, \
                f"Grid file too small ({file_size} bytes), may be empty: {name}"


class TestErrorMaps:
    """Tests for Plan 03 error map outputs."""

    def test_error_maps_created(self):
        """Verify error map directory exists with PNG files."""
        assert ERROR_MAP_DIR.exists(), \
            f"Error map directory not found: {ERROR_MAP_DIR}"

        # List PNG files in error_maps/ (matches both error_heatmap_patch_* and error_heatmap_<id>_*)
        error_map_files = list(ERROR_MAP_DIR.glob("error_heatmap_*.png"))

        assert len(error_map_files) >= 3, \
            f"Expected >= 3 error maps, found {len(error_map_files)}"

        # Verify each file is valid PNG and reasonable size
        for error_path in error_map_files:
            try:
                img = Image.open(error_path)
                assert img.format == 'PNG', f"File is not PNG: {error_path}"
            except Exception as e:
                pytest.fail(f"Cannot open error map PNG: {error_path}: {e}")

            file_size = error_path.stat().st_size
            assert file_size > 50_000, \
                f"Error map too small ({file_size} bytes): {error_path}"


class TestValidationReport:
    """Tests for Plan 04 comprehensive validation report."""

    def test_validation_report_created(self):
        """Verify comprehensive validation report markdown exists."""
        assert VALIDATION_REPORT.exists(), \
            f"Validation report not found: {VALIDATION_REPORT}"

        with open(VALIDATION_REPORT, 'r') as f:
            content = f.read()

        # Verify report is not empty
        assert len(content) > 1000, \
            "Validation report is too small, may be incomplete"

        # Verify required sections are present
        required_sections = {
            "Quantitative Results": "Phase 4 quantitative metrics section",
            "Visual Analysis": "Visual comparison grid section",
            "Error Analysis": "Error map and per-pixel difference section",
            "Validation Verdict": "Pass/Fail verdict section",
        }

        for section_name, description in required_sections.items():
            assert section_name in content, \
                f"Missing section: {section_name} ({description})"

        # Verify metric numbers are in the report
        assert "SSIM" in content, "Report missing SSIM metrics"
        assert "PSNR" in content, "Report missing PSNR metrics"

    def test_metrics_in_expected_range(self):
        """Verify metrics CSV values are within reasonable ranges."""
        assert METRICS_CSV.exists(), "Metrics CSV must exist for this test"

        df = pd.read_csv(METRICS_CSV)

        mean_ssim = df['ssim'].mean()
        mean_psnr = df['psnr'].mean()

        # Reasonable range for medical imaging (not too good to be fake, not too bad)
        assert 0.65 <= mean_ssim <= 0.99, \
            f"Mean SSIM {mean_ssim:.4f} outside reasonable range [0.65, 0.99]"

        assert 15 <= mean_psnr <= 40, \
            f"Mean PSNR {mean_psnr:.2f} dB outside reasonable range [15, 40]"


class TestReportIntegration:
    """Tests for integration of all Phase 4 components."""

    def test_all_outputs_exist(self):
        """Verify all Phase 4 outputs are present."""
        outputs = {
            "Metrics CSV": METRICS_CSV,
            "Random grid": GRID_RANDOM,
            "Best grid": GRID_BEST,
            "Worst grid": GRID_WORST,
            "Error maps directory": ERROR_MAP_DIR,
            "Validation report": VALIDATION_REPORT,
        }

        missing = [name for name, path in outputs.items() if not path.exists()]

        assert not missing, f"Missing outputs: {', '.join(missing)}"

    def test_report_references_data(self):
        """Verify report markdown references grid and error map files."""
        assert VALIDATION_REPORT.exists(), "Validation report must exist"

        with open(VALIDATION_REPORT, 'r') as f:
            content = f.read()

        # Verify report mentions grids
        grid_references = ['grid_random_samples', 'grid_best_cases', 'grid_worst_cases']
        for ref in grid_references:
            assert ref in content, \
                f"Report should reference grid file: {ref}"

        # Verify report mentions error maps
        assert 'error_heatmap' in content, \
            "Report should reference error map files"


class TestMetricStatistics:
    """Tests for metric statistical properties."""

    def test_ssim_statistics_computed(self):
        """Verify SSIM statistics are computed correctly."""
        df = pd.read_csv(METRICS_CSV)

        mean = df['ssim'].mean()
        std = df['ssim'].std()
        min_val = df['ssim'].min()
        max_val = df['ssim'].max()

        # Basic sanity checks
        assert min_val <= mean <= max_val, \
            "Mean should be between min and max"

        assert std >= 0, "Std should be non-negative"

        # SSIM range check
        assert 0 <= min_val and max_val <= 1, \
            "SSIM values must be in [0, 1]"

    def test_psnr_statistics_computed(self):
        """Verify PSNR statistics are computed correctly."""
        df = pd.read_csv(METRICS_CSV)

        mean = df['psnr'].mean()
        std = df['psnr'].std()
        min_val = df['psnr'].min()
        max_val = df['psnr'].max()

        # Basic sanity checks
        assert min_val <= mean <= max_val, \
            "Mean should be between min and max"

        assert std >= 0, "Std should be non-negative"

        # PSNR should be positive (dB scale)
        assert min_val > 0, "PSNR values must be positive"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

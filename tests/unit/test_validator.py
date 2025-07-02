"""Unit tests for the validator module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from simpulse.validator import OptimizationValidator


class TestOptimizationValidator:
    """Test cases for OptimizationValidator class."""

    def test_validator_initialization(self):
        """Test validator initialization with default settings."""
        validator = OptimizationValidator()

        assert validator.timeout == 300  # 5 minutes default
        assert validator.max_retries == 3

    def test_validator_initialization_custom(self):
        """Test validator initialization with custom settings."""
        validator = OptimizationValidator(timeout=600, max_retries=5)

        assert validator.timeout == 600
        assert validator.max_retries == 5

    @patch("subprocess.run")
    def test_check_lean_syntax_success(self, mock_run):
        """Test successful Lean syntax checking."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        validator = OptimizationValidator()
        result = validator._check_lean_syntax(Path("test.lean"))

        assert result is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_check_lean_syntax_failure(self, mock_run):
        """Test failed Lean syntax checking."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error: invalid syntax")

        validator = OptimizationValidator()
        result = validator._check_lean_syntax(Path("test.lean"))

        assert result is False

    @patch("subprocess.run")
    def test_check_lean_syntax_timeout(self, mock_run):
        """Test Lean syntax checking with timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("lean", 300)

        validator = OptimizationValidator()
        result = validator._check_lean_syntax(Path("test.lean"))

        assert result is False

    @patch("subprocess.run")
    def test_measure_compilation_time_success(self, mock_run):
        """Test successful compilation time measurement."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        validator = OptimizationValidator()

        with patch("time.time", side_effect=[0.0, 5.0]):  # 5 second compilation
            time_taken = validator._measure_compilation_time(Path("test.lean"))

        assert time_taken == 5.0

    @patch("subprocess.run")
    def test_measure_compilation_time_failure(self, mock_run):
        """Test compilation time measurement with failure."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")

        validator = OptimizationValidator()
        time_taken = validator._measure_compilation_time(Path("test.lean"))

        assert time_taken is None

    def test_validate_correctness_success(self, sample_lean_file):
        """Test correctness validation with successful compilation."""
        validator = OptimizationValidator()

        with patch.object(validator, "_check_lean_syntax", return_value=True):
            result = validator.validate_correctness(sample_lean_file)

        assert result is True

    def test_validate_correctness_failure(self, sample_lean_file):
        """Test correctness validation with compilation failure."""
        validator = OptimizationValidator()

        with patch.object(validator, "_check_lean_syntax", return_value=False):
            result = validator.validate_correctness(sample_lean_file)

        assert result is False

    def test_validate_performance_improvement(self, sample_lean_file, temp_dir):
        """Test performance validation showing improvement."""
        validator = OptimizationValidator()

        # Create optimized version
        optimized_file = temp_dir / "optimized.lean"
        optimized_file.write_text(sample_lean_file.read_text())

        # Mock compilation times: original=10s, optimized=8s (20% improvement)
        def mock_compilation_time(file_path):
            if "optimized" in str(file_path):
                return 8.0
            else:
                return 10.0

        with patch.object(
            validator, "_measure_compilation_time", side_effect=mock_compilation_time
        ):
            result = validator.validate_performance(sample_lean_file, optimized_file, runs=3)

        assert result is not None
        assert result["original_mean"] == 10.0
        assert result["optimized_mean"] == 8.0
        assert result["improvement_percent"] == 20.0
        assert result["speedup"] == 1.25

    def test_validate_performance_regression(self, sample_lean_file, temp_dir):
        """Test performance validation showing regression."""
        validator = OptimizationValidator()

        optimized_file = temp_dir / "optimized.lean"
        optimized_file.write_text(sample_lean_file.read_text())

        # Mock compilation times: original=8s, optimized=10s (regression)
        def mock_compilation_time(file_path):
            if "optimized" in str(file_path):
                return 10.0
            else:
                return 8.0

        with patch.object(
            validator, "_measure_compilation_time", side_effect=mock_compilation_time
        ):
            result = validator.validate_performance(sample_lean_file, optimized_file, runs=3)

        assert result is not None
        assert result["improvement_percent"] == -25.0  # Negative = regression
        assert result["speedup"] < 1.0

    def test_validate_performance_no_change(self, sample_lean_file, temp_dir):
        """Test performance validation with no significant change."""
        validator = OptimizationValidator()

        optimized_file = temp_dir / "optimized.lean"
        optimized_file.write_text(sample_lean_file.read_text())

        # Mock equal compilation times
        with patch.object(validator, "_measure_compilation_time", return_value=10.0):
            result = validator.validate_performance(sample_lean_file, optimized_file, runs=3)

        assert result is not None
        assert result["improvement_percent"] == 0.0
        assert result["speedup"] == 1.0

    def test_validate_performance_compilation_failure(self, sample_lean_file, temp_dir):
        """Test performance validation with compilation failure."""
        validator = OptimizationValidator()

        optimized_file = temp_dir / "optimized.lean"
        optimized_file.write_text("invalid lean syntax")

        # Mock compilation failure
        def mock_compilation_time(file_path):
            if "optimized" in str(file_path):
                return None  # Compilation failed
            else:
                return 10.0

        with patch.object(
            validator, "_measure_compilation_time", side_effect=mock_compilation_time
        ):
            result = validator.validate_performance(sample_lean_file, optimized_file, runs=3)

        assert result is None

    def test_validate_optimization_complete(self, sample_lean_file, temp_dir):
        """Test complete optimization validation (correctness + performance)."""
        validator = OptimizationValidator()

        optimized_file = temp_dir / "optimized.lean"
        optimized_file.write_text(sample_lean_file.read_text())

        # Mock successful validation
        with (
            patch.object(validator, "validate_correctness", return_value=True),
            patch.object(validator, "validate_performance") as mock_perf,
        ):

            mock_perf.return_value = {
                "original_mean": 10.0,
                "optimized_mean": 8.0,
                "improvement_percent": 20.0,
                "speedup": 1.25,
            }

            result = validator.validate_optimization(sample_lean_file, optimized_file)

        assert result["correctness"] is True
        assert result["performance"]["improvement_percent"] == 20.0

    def test_validate_optimization_correctness_failure(self, sample_lean_file, temp_dir):
        """Test optimization validation with correctness failure."""
        validator = OptimizationValidator()

        optimized_file = temp_dir / "broken.lean"
        optimized_file.write_text("invalid syntax")

        with patch.object(validator, "validate_correctness", return_value=False):
            result = validator.validate_optimization(sample_lean_file, optimized_file)

        assert result["correctness"] is False
        assert result["performance"] is None

    def test_calculate_statistics(self):
        """Test calculation of performance statistics."""
        validator = OptimizationValidator()

        original_times = [10.0, 11.0, 9.0, 10.5, 9.5]
        optimized_times = [8.0, 8.5, 7.5, 8.2, 7.8]

        stats = validator._calculate_statistics(original_times, optimized_times)

        assert stats["original_mean"] == 10.0
        assert stats["optimized_mean"] == 8.0
        assert stats["improvement_percent"] == 20.0
        assert stats["speedup"] == 1.25
        assert "original_std" in stats
        assert "optimized_std" in stats

    def test_calculate_statistics_with_variation(self):
        """Test statistics calculation with time variation."""
        validator = OptimizationValidator()

        # More realistic times with variation
        original_times = [10.2, 9.8, 10.5, 9.9, 10.1]
        optimized_times = [8.1, 7.9, 8.3, 8.0, 8.2]

        stats = validator._calculate_statistics(original_times, optimized_times)

        assert abs(stats["original_mean"] - 10.1) < 0.1
        assert abs(stats["optimized_mean"] - 8.1) < 0.1
        assert 18 < stats["improvement_percent"] < 22  # Around 20%
        assert stats["original_std"] > 0
        assert stats["optimized_std"] > 0

    def test_is_significant_improvement(self):
        """Test determination of significant improvement."""
        validator = OptimizationValidator()

        # Clear improvement
        assert validator._is_significant_improvement(10.0, 8.0, 0.2, 0.2) is True

        # Marginal improvement (might not be significant)
        assert validator._is_significant_improvement(10.0, 9.9, 0.5, 0.5) is False

        # High variation (might not be significant)
        assert validator._is_significant_improvement(10.0, 8.0, 2.0, 2.0) is False

    @patch("subprocess.run")
    def test_retry_on_failure(self, mock_run):
        """Test retry mechanism on compilation failure."""
        # Fail twice, then succeed
        mock_run.side_effect = [
            MagicMock(returncode=1),  # First failure
            MagicMock(returncode=1),  # Second failure
            MagicMock(returncode=0),  # Success
        ]

        validator = OptimizationValidator(max_retries=3)
        result = validator._check_lean_syntax(Path("test.lean"))

        assert result is True
        assert mock_run.call_count == 3

    @patch("subprocess.run")
    def test_max_retries_exceeded(self, mock_run):
        """Test behavior when max retries are exceeded."""
        mock_run.return_value = MagicMock(returncode=1)  # Always fail

        validator = OptimizationValidator(max_retries=2)
        result = validator._check_lean_syntax(Path("test.lean"))

        assert result is False
        assert mock_run.call_count == 2

    def test_validate_with_custom_timeout(self, sample_lean_file):
        """Test validation with custom timeout setting."""
        validator = OptimizationValidator(timeout=60)  # 1 minute

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            validator._check_lean_syntax(sample_lean_file)

            # Check that timeout is passed to subprocess
            call_args = mock_run.call_args
            assert call_args.kwargs.get("timeout") == 60

    def test_empty_file_handling(self, temp_dir):
        """Test handling of empty Lean files."""
        validator = OptimizationValidator()

        empty_file = temp_dir / "empty.lean"
        empty_file.write_text("")

        with patch.object(validator, "_check_lean_syntax", return_value=True):
            result = validator.validate_correctness(empty_file)

        assert result is True

    def test_large_file_performance(self, temp_dir):
        """Test performance validation with larger files."""
        validator = OptimizationValidator()

        # Create a larger Lean file
        large_content = "\n".join(
            [f"@[simp] theorem rule_{i} : {i} = {i} := rfl" for i in range(100)]
        )

        large_file = temp_dir / "large.lean"
        large_file.write_text(large_content)

        optimized_file = temp_dir / "large_optimized.lean"
        optimized_file.write_text(large_content)

        # Mock longer compilation times for large files
        def mock_compilation_time(file_path):
            return 30.0 if "optimized" in str(file_path) else 35.0

        with patch.object(
            validator, "_measure_compilation_time", side_effect=mock_compilation_time
        ):
            result = validator.validate_performance(large_file, optimized_file, runs=2)

        assert result is not None
        assert result["improvement_percent"] > 0

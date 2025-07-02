"""Validation and performance measurement for Lean 4 optimizations."""

import statistics
import subprocess
import time
from pathlib import Path


class OptimizationValidator:
    """Validates that optimizations preserve correctness and measure performance."""

    def __init__(self, timeout: int = 300, max_retries: int = 3):
        """Initialize the validator.

        Args:
            timeout: Timeout for Lean commands in seconds.
            max_retries: Maximum number of retries for failed operations.
        """
        self.timeout = timeout
        self.max_retries = max_retries

    def validate_correctness(self, file_path: Path) -> bool:
        """Validate that a Lean file compiles correctly.

        Args:
            file_path: Path to the Lean file to validate.

        Returns:
            True if file compiles successfully, False otherwise.
        """
        return self._check_lean_syntax(file_path)

    def validate_performance(
        self, original_file: Path, optimized_file: Path, runs: int = 5
    ) -> dict | None:
        """Measure performance difference between original and optimized files.

        Args:
            original_file: Path to the original Lean file.
            optimized_file: Path to the optimized Lean file.
            runs: Number of benchmark runs to average.

        Returns:
            Dictionary with performance statistics, or None if measurement failed.
        """
        # Measure original performance
        original_times = []
        for _ in range(runs):
            time_taken = self._measure_compilation_time(original_file)
            if time_taken is None:
                return None
            original_times.append(time_taken)

        # Measure optimized performance
        optimized_times = []
        for _ in range(runs):
            time_taken = self._measure_compilation_time(optimized_file)
            if time_taken is None:
                return None
            optimized_times.append(time_taken)

        return self._calculate_statistics(original_times, optimized_times)

    def validate_optimization(self, original_file: Path, optimized_file: Path) -> dict:
        """Perform complete validation: correctness + performance.

        Args:
            original_file: Path to the original Lean file.
            optimized_file: Path to the optimized Lean file.

        Returns:
            Dictionary with validation results.
        """
        # Check correctness first
        correctness_valid = self.validate_correctness(optimized_file)

        result = {"correctness": correctness_valid, "performance": None}

        # Only measure performance if correctness is valid
        if correctness_valid:
            performance = self.validate_performance(original_file, optimized_file)
            result["performance"] = performance

        return result

    def _check_lean_syntax(self, file_path: Path) -> bool:
        """Check if a Lean file has valid syntax.

        Args:
            file_path: Path to the Lean file.

        Returns:
            True if syntax is valid, False otherwise.
        """
        for attempt in range(self.max_retries):
            try:
                result = subprocess.run(
                    ["lean", "--check", str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                return result.returncode == 0

            except subprocess.TimeoutExpired:
                if attempt == self.max_retries - 1:
                    return False
                time.sleep(1)  # Brief pause before retry

            except FileNotFoundError:
                return False

        return False

    def _measure_compilation_time(self, file_path: Path) -> float | None:
        """Measure the time it takes to compile a Lean file.

        Args:
            file_path: Path to the Lean file.

        Returns:
            Compilation time in seconds, or None if compilation failed.
        """
        try:
            start_time = time.time()

            result = subprocess.run(
                ["lean", "--make", str(file_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            end_time = time.time()

            if result.returncode == 0:
                return end_time - start_time
            else:
                return None

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _calculate_statistics(
        self, original_times: list[float], optimized_times: list[float]
    ) -> dict:
        """Calculate performance statistics from timing measurements.

        Args:
            original_times: List of original compilation times.
            optimized_times: List of optimized compilation times.

        Returns:
            Dictionary with calculated statistics.
        """
        original_mean = statistics.mean(original_times)
        optimized_mean = statistics.mean(optimized_times)
        original_std = statistics.stdev(original_times) if len(original_times) > 1 else 0.0
        optimized_std = statistics.stdev(optimized_times) if len(optimized_times) > 1 else 0.0

        # Calculate improvement
        if original_mean > 0:
            improvement_percent = ((original_mean - optimized_mean) / original_mean) * 100
            speedup = original_mean / optimized_mean if optimized_mean > 0 else float("inf")
        else:
            improvement_percent = 0.0
            speedup = 1.0

        # Determine if improvement is significant
        significant = self._is_significant_improvement(
            original_mean, optimized_mean, original_std, optimized_std
        )

        return {
            "original_mean": original_mean,
            "optimized_mean": optimized_mean,
            "original_std": original_std,
            "optimized_std": optimized_std,
            "improvement_percent": improvement_percent,
            "speedup": speedup,
            "significant": significant,
            "runs": len(original_times),
        }

    def _is_significant_improvement(
        self, original_mean: float, optimized_mean: float, original_std: float, optimized_std: float
    ) -> bool:
        """Determine if performance improvement is statistically significant.

        Args:
            original_mean: Mean of original times.
            optimized_mean: Mean of optimized times.
            original_std: Standard deviation of original times.
            optimized_std: Standard deviation of optimized times.

        Returns:
            True if improvement is significant, False otherwise.
        """
        # Simple heuristic: improvement must be larger than combined standard deviations
        # and at least 5% improvement
        if original_mean <= optimized_mean:
            return False

        improvement = original_mean - optimized_mean
        combined_std = original_std + optimized_std
        improvement_percent = improvement / original_mean

        return improvement > combined_std and improvement_percent >= 0.05

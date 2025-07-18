"""
Performance Measurement Framework for Lean 4 Projects

Provides real performance measurement capabilities to validate optimization
effectiveness, replacing theoretical estimates with actual timing data.
"""

import json
import logging
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMeasurement:
    """Single performance measurement result."""

    file_path: str
    compilation_time: float  # seconds
    success: bool
    error_message: str | None = None
    memory_usage: int | None = None  # MB

    def __post_init__(self):
        if self.compilation_time < 0:
            self.compilation_time = 0.0


@dataclass
class PerformanceReport:
    """Complete performance analysis report."""

    measurements: list[PerformanceMeasurement] = field(default_factory=list)
    total_time: float = 0.0
    average_time: float = 0.0
    median_time: float = 0.0
    success_rate: float = 0.0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def __post_init__(self):
        if self.measurements:
            successful = [m for m in self.measurements if m.success]
            times = [m.compilation_time for m in successful]

            self.total_time = sum(times)
            self.average_time = statistics.mean(times) if times else 0.0
            self.median_time = statistics.median(times) if times else 0.0
            self.success_rate = len(successful) / len(self.measurements)

    def compare_with(self, other: "PerformanceReport") -> "PerformanceComparison":
        """Compare this report with another to show improvement/regression."""
        return PerformanceComparison(baseline=other, optimized=self)


@dataclass
class PerformanceComparison:
    """Comparison between baseline and optimized performance."""

    baseline: PerformanceReport
    optimized: PerformanceReport

    @property
    def time_improvement_percent(self) -> float:
        """Calculate percentage improvement in compilation time."""
        if self.baseline.total_time == 0:
            return 0.0
        improvement = (
            self.baseline.total_time - self.optimized.total_time
        ) / self.baseline.total_time
        return improvement * 100

    @property
    def average_time_improvement_percent(self) -> float:
        """Calculate percentage improvement in average compilation time."""
        if self.baseline.average_time == 0:
            return 0.0
        improvement = (
            self.baseline.average_time - self.optimized.average_time
        ) / self.baseline.average_time
        return improvement * 100

    @property
    def is_improvement(self) -> bool:
        """Check if optimization resulted in actual improvement."""
        return (
            self.time_improvement_percent > 0
            and self.optimized.success_rate >= self.baseline.success_rate
        )

    def summary(self) -> str:
        """Generate human-readable summary of comparison."""
        if not self.is_improvement:
            return f"No improvement detected. Time change: {self.time_improvement_percent:+.1f}%"

        return (
            f"Performance improved by {self.time_improvement_percent:.1f}% "
            f"(avg: {self.average_time_improvement_percent:.1f}%)"
        )


class PerformanceMeasurer:
    """Measures Lean compilation performance with high accuracy."""

    def __init__(self, project_path: Path, lean_executable: str = "lean"):
        self.project_path = Path(project_path)
        self.lean_executable = lean_executable
        self.measurements_cache: dict[str, list[PerformanceMeasurement]] = {}

    def measure_file(self, file_path: Path, runs: int = 3) -> PerformanceMeasurement:
        """Measure compilation time for a single Lean file."""
        times = []
        last_error = None

        for _ in range(runs):
            try:
                start_time = time.perf_counter()

                result = subprocess.run(
                    [self.lean_executable, str(file_path)],
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout per file
                )

                end_time = time.perf_counter()
                compilation_time = end_time - start_time

                if result.returncode == 0:
                    times.append(compilation_time)
                else:
                    last_error = result.stderr
                    logger.warning(f"Compilation failed for {file_path}: {result.stderr}")

            except subprocess.TimeoutExpired:
                last_error = "Compilation timed out after 5 minutes"
                logger.error(f"Timeout measuring {file_path}")
            except Exception as e:
                last_error = str(e)
                logger.error(f"Error measuring {file_path}: {e}")

        if times:
            # Use median time to reduce impact of outliers
            best_time = statistics.median(times)
            return PerformanceMeasurement(
                file_path=str(file_path), compilation_time=best_time, success=True
            )
        else:
            return PerformanceMeasurement(
                file_path=str(file_path),
                compilation_time=0.0,
                success=False,
                error_message=last_error,
            )

    def measure_project(
        self, files: list[Path] | None = None, runs_per_file: int = 3, max_files: int | None = None
    ) -> PerformanceReport:
        """Measure compilation performance for entire project or file subset."""
        if files is None:
            files = list(self.project_path.glob("**/*.lean"))

        # Filter out test files and examples that might not be representative
        files = [
            f
            for f in files
            if not any(excluded in str(f).lower() for excluded in ["test", "example", "benchmark"])
        ]

        if max_files and len(files) > max_files:
            # Sample representative files if too many
            import random

            files = random.sample(files, max_files)
            logger.info(f"Sampling {max_files} files from {len(files)} total")

        measurements = []
        total_files = len(files)

        logger.info(f"Measuring performance of {total_files} files...")

        for i, file_path in enumerate(files, 1):
            logger.info(f"Measuring {i}/{total_files}: {file_path.name}")

            measurement = self.measure_file(file_path, runs=runs_per_file)
            measurements.append(measurement)

            # Log progress
            if measurement.success:
                logger.debug(f"  ✓ {measurement.compilation_time:.2f}s")
            else:
                logger.debug(f"  ✗ Failed: {measurement.error_message}")

        report = PerformanceReport(measurements=measurements)
        logger.info(
            f"Performance measurement complete: {report.success_rate:.1%} success rate, "
            f"{report.total_time:.1f}s total, {report.average_time:.2f}s average"
        )

        return report

    def benchmark_optimization(
        self, baseline_files: list[Path], optimized_files: list[Path], runs_per_file: int = 3
    ) -> PerformanceComparison:
        """Benchmark performance before and after optimization."""
        logger.info("Measuring baseline performance...")
        baseline_report = self.measure_project(baseline_files, runs_per_file)

        logger.info("Measuring optimized performance...")
        optimized_report = self.measure_project(optimized_files, runs_per_file)

        comparison = baseline_report.compare_with(optimized_report)
        logger.info(f"Benchmark complete: {comparison.summary()}")

        return comparison

    def save_report(self, report: PerformanceReport, output_path: Path) -> None:
        """Save performance report to JSON file."""
        data = {
            "timestamp": report.timestamp,
            "total_time": report.total_time,
            "average_time": report.average_time,
            "median_time": report.median_time,
            "success_rate": report.success_rate,
            "measurements": [
                {
                    "file_path": m.file_path,
                    "compilation_time": m.compilation_time,
                    "success": m.success,
                    "error_message": m.error_message,
                }
                for m in report.measurements
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Performance report saved to {output_path}")

    def load_report(self, input_path: Path) -> PerformanceReport:
        """Load performance report from JSON file."""
        with open(input_path) as f:
            data = json.load(f)

        measurements = [
            PerformanceMeasurement(
                file_path=m["file_path"],
                compilation_time=m["compilation_time"],
                success=m["success"],
                error_message=m.get("error_message"),
            )
            for m in data["measurements"]
        ]

        return PerformanceReport(measurements=measurements)


class OptimizationValidator:
    """Validates that optimizations actually improve performance."""

    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.measurer = PerformanceMeasurer(project_path)

    def validate_optimization(
        self,
        original_files: list[Path],
        optimized_files: list[Path],
        min_improvement_percent: float = 5.0,
    ) -> tuple[bool, PerformanceComparison]:
        """Validate that optimization provides measurable improvement."""
        comparison = self.measurer.benchmark_optimization(original_files, optimized_files)

        is_valid = (
            comparison.is_improvement
            and comparison.time_improvement_percent >= min_improvement_percent
        )

        if is_valid:
            logger.info(f"✓ Optimization validated: {comparison.summary()}")
        else:
            logger.warning(f"✗ Optimization not validated: {comparison.summary()}")

        return is_valid, comparison

    def create_backup_and_validate(
        self, files_to_optimize: list[Path], optimization_func, min_improvement_percent: float = 5.0
    ) -> tuple[bool, PerformanceComparison]:
        """Create backups, apply optimization, and validate improvement."""
        # Create backups
        backup_dir = self.project_path / ".simpulse_backup"
        backup_dir.mkdir(exist_ok=True)

        backup_files = []
        for file_path in files_to_optimize:
            backup_path = backup_dir / file_path.name
            backup_path.write_text(file_path.read_text())
            backup_files.append(backup_path)

        try:
            # Measure baseline
            logger.info("Measuring baseline performance...")
            baseline_report = self.measurer.measure_project(files_to_optimize)

            # Apply optimization
            logger.info("Applying optimization...")
            optimization_func()

            # Measure optimized performance
            logger.info("Measuring optimized performance...")
            optimized_report = self.measurer.measure_project(files_to_optimize)

            # Compare and validate
            comparison = baseline_report.compare_with(optimized_report)
            is_valid = (
                comparison.is_improvement
                and comparison.time_improvement_percent >= min_improvement_percent
            )

            if not is_valid:
                # Restore backups if optimization didn't help
                logger.warning("Optimization not effective, restoring backups...")
                for original_file, backup_file in zip(
                    files_to_optimize, backup_files, strict=False
                ):
                    original_file.write_text(backup_file.read_text())

            return is_valid, comparison

        finally:
            # Clean up backups
            for backup_file in backup_files:
                backup_file.unlink(missing_ok=True)
            if backup_dir.exists() and not list(backup_dir.iterdir()):
                backup_dir.rmdir()


# Example usage and testing
if __name__ == "__main__":
    import tempfile

    # Test with a sample project
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)

        # Create a simple test file
        test_file = project_path / "test.lean"
        test_file.write_text(
            """
theorem simple_test : 1 + 1 = 2 := by simp
theorem another_test : [1, 2].length = 2 := by simp
"""
        )

        measurer = PerformanceMeasurer(project_path)
        measurement = measurer.measure_file(test_file)

        print(
            f"Test measurement: {measurement.compilation_time:.3f}s, success: {measurement.success}"
        )

        if measurement.success:
            print("✓ Performance measurement framework working correctly")
        else:
            print(f"✗ Measurement failed: {measurement.error_message}")

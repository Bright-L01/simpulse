"""
Real performance validation for Lean optimizations.
Measures actual compilation time differences.
"""

import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class PerformanceMetrics:
    """Actual performance measurements."""

    wall_time: float
    cpu_time: float
    memory_peak_mb: float
    exit_code: int
    simp_calls: int = 0
    optimization_applied: bool = False

    @property
    def successful(self) -> bool:
        return self.exit_code == 0


class PerformanceValidator:
    """Validates that optimizations actually improve performance."""

    def __init__(self, lean_path: str = None, iterations: int = 3):
        from ..config import get_lean_command

        self.lean_path = lean_path or get_lean_command()
        self.iterations = iterations
        self.min_improvement_threshold = 1.05  # Require 5% improvement

    def validate_optimization(
        self, baseline_file: Path, optimized_file: Path, require_improvement: bool = True
    ) -> Dict[str, any]:
        """
        Validate that optimization actually improves performance.

        Args:
            baseline_file: Original Lean file
            optimized_file: Lean file with optimization applied
            require_improvement: If True, raise error if no improvement

        Returns:
            Dict with performance comparison results
        """
        # Measure baseline performance
        baseline_metrics = self._measure_performance(baseline_file, "baseline")
        if not baseline_metrics.successful:
            raise ValueError(f"Baseline file failed compilation: {baseline_file}")

        # Measure optimized performance
        optimized_metrics = self._measure_performance(optimized_file, "optimized")
        if not optimized_metrics.successful:
            raise ValueError(f"Optimized file failed compilation: {optimized_file}")

        # Calculate improvement
        speedup = baseline_metrics.wall_time / optimized_metrics.wall_time
        time_saved = baseline_metrics.wall_time - optimized_metrics.wall_time
        time_saved_percent = (time_saved / baseline_metrics.wall_time) * 100

        # Check if improvement meets threshold
        improvement_achieved = speedup >= self.min_improvement_threshold

        if require_improvement and not improvement_achieved:
            raise ValueError(
                f"Optimization failed to improve performance. "
                f"Speedup: {speedup:.2f}x (required: {self.min_improvement_threshold}x)"
            )

        return {
            "baseline": {
                "wall_time": baseline_metrics.wall_time,
                "cpu_time": baseline_metrics.cpu_time,
                "memory_peak_mb": baseline_metrics.memory_peak_mb,
            },
            "optimized": {
                "wall_time": optimized_metrics.wall_time,
                "cpu_time": optimized_metrics.cpu_time,
                "memory_peak_mb": optimized_metrics.memory_peak_mb,
            },
            "improvement": {
                "speedup": speedup,
                "time_saved_seconds": time_saved,
                "time_saved_percent": time_saved_percent,
                "meets_threshold": improvement_achieved,
            },
            "validation": {
                "iterations": self.iterations,
                "threshold": self.min_improvement_threshold,
                "status": "PASSED" if improvement_achieved else "FAILED",
            },
        }

    def _measure_performance(self, lean_file: Path, label: str) -> PerformanceMetrics:
        """Measure actual Lean compilation performance."""
        times = []

        for i in range(self.iterations):
            start_time = time.time()
            start_cpu = time.process_time()

            # Run Lean compilation
            result = subprocess.run(
                [self.lean_path, str(lean_file)], capture_output=True, text=True
            )

            end_time = time.time()
            end_cpu = time.process_time()

            times.append(
                {
                    "wall_time": end_time - start_time,
                    "cpu_time": end_cpu - start_cpu,
                    "exit_code": result.returncode,
                }
            )

        # Calculate averages (excluding failed runs)
        successful_runs = [t for t in times if t["exit_code"] == 0]
        if not successful_runs:
            return PerformanceMetrics(
                wall_time=float("inf"), cpu_time=float("inf"), memory_peak_mb=0, exit_code=1
            )

        avg_wall_time = statistics.mean([t["wall_time"] for t in successful_runs])
        avg_cpu_time = statistics.mean([t["cpu_time"] for t in successful_runs])

        return PerformanceMetrics(
            wall_time=avg_wall_time,
            cpu_time=avg_cpu_time,
            memory_peak_mb=0,  # TODO: Implement memory tracking
            exit_code=0,
        )

    def validate_optimization_safety(self, original_file: Path, optimized_file: Path) -> bool:
        """
        Verify optimization doesn't break correctness.

        Both files should compile successfully with same results.
        """
        # Compile both files
        original_result = subprocess.run(
            [self.lean_path, "--", str(original_file)], capture_output=True, text=True
        )

        optimized_result = subprocess.run(
            [self.lean_path, "--", str(optimized_file)], capture_output=True, text=True
        )

        # Both should succeed
        if original_result.returncode != 0 or optimized_result.returncode != 0:
            return False

        # TODO: Compare theorem outputs for semantic equivalence
        return True

    def create_performance_report(
        self, validation_results: Dict[str, any], output_file: Optional[Path] = None
    ) -> str:
        """Create human-readable performance validation report."""
        report = f"""
PERFORMANCE VALIDATION REPORT
============================

Baseline Performance:
  Wall Time: {validation_results['baseline']['wall_time']:.3f}s
  CPU Time:  {validation_results['baseline']['cpu_time']:.3f}s

Optimized Performance:
  Wall Time: {validation_results['optimized']['wall_time']:.3f}s
  CPU Time:  {validation_results['optimized']['cpu_time']:.3f}s

Improvement Metrics:
  Speedup:      {validation_results['improvement']['speedup']:.2f}x
  Time Saved:   {validation_results['improvement']['time_saved_seconds']:.3f}s
  Improvement:  {validation_results['improvement']['time_saved_percent']:.1f}%

Validation Status: {validation_results['validation']['status']}
  Required speedup: {validation_results['validation']['threshold']:.2f}x
  Achieved speedup: {validation_results['improvement']['speedup']:.2f}x
"""

        if output_file:
            output_file.write_text(report)

        return report


def main():
    """Example usage of performance validator."""
    import sys

    if len(sys.argv) != 3:
        print("Usage: performance_validator.py <baseline.lean> <optimized.lean>")
        sys.exit(1)

    validator = PerformanceValidator()

    try:
        results = validator.validate_optimization(Path(sys.argv[1]), Path(sys.argv[2]))

        report = validator.create_performance_report(results)
        print(report)

        if results["validation"]["status"] == "PASSED":
            print("✅ Optimization validated successfully!")
            sys.exit(0)
        else:
            print("❌ Optimization failed validation!")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Validation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

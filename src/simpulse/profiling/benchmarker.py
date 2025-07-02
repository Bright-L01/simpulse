"""Performance benchmarker."""

import statistics
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel


class BenchmarkResult(BaseModel):
    """Benchmark results."""

    mean: float
    stdev: float
    runs: int
    times: list[float]


class ComparisonResult(BaseModel):
    """Comparison benchmark results."""

    baseline_mean: float
    optimized_mean: float
    improvement_percentage: float


class Benchmarker:
    """Run performance benchmarks."""

    def benchmark(self, project_path: Path, runs: int = 3) -> BenchmarkResult:
        """Run benchmark on a project."""
        times = []

        for _ in range(runs):
            # Clean build
            subprocess.run(["lake", "clean"], cwd=project_path, capture_output=True)

            # Time build
            start = time.time()
            result = subprocess.run(["lake", "build"], cwd=project_path, capture_output=True)

            if result.returncode == 0:
                times.append(time.time() - start)

        return BenchmarkResult(
            mean=statistics.mean(times) if times else 0,
            stdev=statistics.stdev(times) if len(times) > 1 else 0,
            runs=len(times),
            times=times,
        )

    def compare(
        self, project_path: Path, optimization_plan: Path, runs: int = 3
    ) -> ComparisonResult:
        """Compare baseline vs optimized performance."""
        # Would implement full comparison logic
        baseline = self.benchmark(project_path, runs)
        # Apply optimization and benchmark again
        optimized = baseline  # Placeholder

        improvement = ((baseline.mean - optimized.mean) / baseline.mean) * 100

        return ComparisonResult(
            baseline_mean=baseline.mean,
            optimized_mean=optimized.mean,
            improvement_percentage=improvement,
        )

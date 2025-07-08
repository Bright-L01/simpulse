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
            from ..config import get_lake_command

            subprocess.run([get_lake_command(), "clean"], cwd=project_path, capture_output=True)

            # Time build
            start = time.time()
            result = subprocess.run(
                [get_lake_command(), "build"], cwd=project_path, capture_output=True
            )

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
        import json
        import shutil
        import tempfile

        # Benchmark baseline performance
        baseline = self.benchmark(project_path, runs)

        # Create temporary copy of project for optimization
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_project = Path(temp_dir) / "optimized_project"
            shutil.copytree(
                project_path, temp_project, ignore=shutil.ignore_patterns("*.bak", "build", ".lake")
            )

            # Apply optimizations
            try:
                if optimization_plan.exists():
                    # Load optimization plan
                    with open(optimization_plan) as f:
                        plan_data = json.load(f)

                    # Apply changes to temporary project
                    for change in plan_data.get("changes", []):
                        file_path = temp_project / change["file_path"]
                        if file_path.exists():
                            content = file_path.read_text()
                            # Apply optimization change
                            old_pattern = f"@[simp] theorem {change['rule_name']}"
                            new_pattern = f"@[simp, priority := {change['new_priority']}] theorem {change['rule_name']}"
                            content = content.replace(old_pattern, new_pattern)
                            file_path.write_text(content)

                # Benchmark optimized performance
                optimized = self.benchmark(temp_project, runs)

            except Exception:
                # Fallback if optimization fails
                optimized = baseline

        improvement = (
            ((baseline.mean - optimized.mean) / baseline.mean) * 100 if baseline.mean > 0 else 0
        )

        return ComparisonResult(
            baseline_mean=baseline.mean,
            optimized_mean=optimized.mean,
            improvement_percentage=improvement,
        )

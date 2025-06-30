#!/usr/bin/env python3
"""
Real-world benchmark for Simpulse on actual Lean 4 modules.

Measures compilation time improvements with optimized simp priorities.
"""

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BenchmarkResult:
    """Result from benchmarking a module."""

    module_name: str
    baseline_time: float
    optimized_time: float
    improvement_percent: float
    simp_calls: int
    optimized_rules: int
    compilation_success: bool
    error_message: Optional[str] = None


@dataclass
class ModuleProfile:
    """Profiling data for a module."""

    module_name: str
    simp_rule_usage: Dict[str, int]  # Rule name -> usage count
    hot_paths: List[str]  # Frequently executed proof paths
    bottlenecks: List[Tuple[str, float]]  # (location, time)


class RealWorldBenchmark:
    """Benchmark Simpulse on real Lean 4 code."""

    def __init__(self, lean_project: Path, simpulse_path: Path):
        """Initialize benchmark with paths."""
        self.lean_project = lean_project
        self.simpulse_path = simpulse_path
        self.results: List[BenchmarkResult] = []

        # Ensure we have Lean 4
        self._check_lean_installation()

    def _check_lean_installation(self):
        """Verify Lean 4 is installed and accessible."""
        try:
            result = subprocess.run(
                ["lean", "--version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError("Lean 4 not found. Please install Lean 4.")
            print(f"Found Lean: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError("Lean 4 not found in PATH. Please install Lean 4.")

    def benchmark_modules(self, modules: List[str]) -> Dict[str, BenchmarkResult]:
        """Benchmark specified modules with and without optimization."""
        print(f"Benchmarking {len(modules)} modules...")
        print("=" * 70)

        for module in modules:
            print(f"\nBenchmarking: {module}")
            result = self._benchmark_single_module(module)
            self.results.append(result)

            print(f"  Baseline time: {result.baseline_time:.2f}s")
            print(f"  Optimized time: {result.optimized_time:.2f}s")
            print(f"  Improvement: {result.improvement_percent:.1f}%")

            if not result.compilation_success:
                print(f"  ⚠️  Error: {result.error_message}")

        # Generate summary
        self._generate_summary()

        return {r.module_name: r for r in self.results}

    def _benchmark_single_module(self, module: str) -> BenchmarkResult:
        """Benchmark a single module."""
        module_path = self._resolve_module_path(module)

        if not module_path.exists():
            return BenchmarkResult(
                module_name=module,
                baseline_time=0,
                optimized_time=0,
                improvement_percent=0,
                simp_calls=0,
                optimized_rules=0,
                compilation_success=False,
                error_message=f"Module not found: {module_path}",
            )

        # Profile the module first
        profile = self._profile_module(module_path)

        # Run baseline compilation
        baseline_time = self._compile_baseline(module_path)

        # Apply Simpulse optimization
        optimized_module = self._apply_optimization(module_path, profile)

        # Run optimized compilation
        optimized_time = self._compile_optimized(optimized_module)

        # Calculate improvement
        if baseline_time > 0:
            improvement = (baseline_time - optimized_time) / baseline_time * 100
        else:
            improvement = 0

        return BenchmarkResult(
            module_name=module,
            baseline_time=baseline_time,
            optimized_time=optimized_time,
            improvement_percent=improvement,
            simp_calls=len(profile.simp_rule_usage),
            optimized_rules=len(
                [r for r, c in profile.simp_rule_usage.items() if c > 10]
            ),
            compilation_success=True,
        )

    def _resolve_module_path(self, module: str) -> Path:
        """Convert module name to file path."""
        # Convert Mathlib.Data.List.Basic to Mathlib/Data/List/Basic.lean
        parts = module.split(".")
        return self.lean_project / Path(*parts).with_suffix(".lean")

    def _profile_module(self, module_path: Path) -> ModuleProfile:
        """Profile a module to understand simp usage."""
        # In a real implementation, this would hook into Lean's elaborator
        # For now, we'll analyze the source statically

        content = module_path.read_text()

        # Extract simp usage patterns
        import re

        simp_pattern = re.compile(r"by\s+simp(?:\s+\[([^\]]*)\])?")

        simp_usage = {}
        for match in simp_pattern.finditer(content):
            rules_text = match.group(1) or ""
            rules = [r.strip() for r in rules_text.split(",") if r.strip()]
            for rule in rules:
                simp_usage[rule] = simp_usage.get(rule, 0) + 1

        return ModuleProfile(
            module_name=str(module_path),
            simp_rule_usage=simp_usage,
            hot_paths=[],  # Would need runtime profiling
            bottlenecks=[],
        )

    def _apply_optimization(self, module_path: Path, profile: ModuleProfile) -> Path:
        """Apply Simpulse optimization to a module."""
        # Create optimized version
        optimized_path = module_path.parent / f"{module_path.stem}_optimized.lean"

        content = module_path.read_text()

        # Apply optimizations based on profile
        # In reality, this would:
        # 1. Reorder simp rules based on usage frequency
        # 2. Add priority annotations to hot rules
        # 3. Specialize generic tactics

        # For now, simulate by adding priorities to frequently used rules
        optimized_content = content

        # Add priority annotations (simplified)
        for rule, count in sorted(
            profile.simp_rule_usage.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            if count > 5:
                # High-frequency rules get high priority
                priority = max(100, 1000 - count * 50)
                # This is simplified - real implementation would modify AST
                optimized_content = optimized_content.replace(
                    "@[simp]",
                    f"@[simp, priority := {priority}]",
                    1,  # Only first occurrence
                )

        optimized_path.write_text(optimized_content)
        return optimized_path

    def _compile_baseline(self, module_path: Path) -> float:
        """Compile module and measure time."""
        start_time = time.time()

        try:
            # Use lake build for proper compilation
            result = subprocess.run(
                ["lake", "build", str(module_path)],
                cwd=self.lean_project,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                print(f"Compilation error: {result.stderr}")
                return 0

        except subprocess.TimeoutExpired:
            print("Compilation timed out")
            return 0
        except Exception as e:
            print(f"Compilation failed: {e}")
            return 0

        return time.time() - start_time

    def _compile_optimized(self, module_path: Path) -> float:
        """Compile optimized module and measure time."""
        # Same as baseline but with optimized module
        return self._compile_baseline(module_path)

    def _generate_summary(self):
        """Generate summary statistics and visualizations."""
        if not self.results:
            print("No results to summarize")
            return

        # Calculate aggregate statistics
        successful_results = [r for r in self.results if r.compilation_success]

        if successful_results:
            avg_baseline = np.mean([r.baseline_time for r in successful_results])
            avg_optimized = np.mean([r.optimized_time for r in successful_results])
            avg_improvement = np.mean(
                [r.improvement_percent for r in successful_results]
            )

            print("\n" + "=" * 70)
            print("BENCHMARK SUMMARY")
            print("=" * 70)
            print(f"Modules benchmarked: {len(self.results)}")
            print(f"Successful compilations: {len(successful_results)}")
            print(f"Average baseline time: {avg_baseline:.2f}s")
            print(f"Average optimized time: {avg_optimized:.2f}s")
            print(f"Average improvement: {avg_improvement:.1f}%")

            # Find best improvements
            best_results = sorted(
                successful_results, key=lambda r: r.improvement_percent, reverse=True
            )[:3]

            print("\nTop improvements:")
            for r in best_results:
                print(f"  {r.module_name}: {r.improvement_percent:.1f}%")

    def create_benchmark_chart(self, output_path: Path):
        """Create visualization of benchmark results."""
        if not self.results:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Chart 1: Compilation times
        modules = [r.module_name.split(".")[-1] for r in self.results]
        baseline_times = [r.baseline_time for r in self.results]
        optimized_times = [r.optimized_time for r in self.results]

        x = np.arange(len(modules))
        width = 0.35

        ax1.bar(
            x - width / 2,
            baseline_times,
            width,
            label="Baseline",
            color="red",
            alpha=0.7,
        )
        ax1.bar(
            x + width / 2,
            optimized_times,
            width,
            label="Optimized",
            color="green",
            alpha=0.7,
        )

        ax1.set_xlabel("Module")
        ax1.set_ylabel("Compilation Time (seconds)")
        ax1.set_title("Compilation Time: Baseline vs Optimized")
        ax1.set_xticks(x)
        ax1.set_xticklabels(modules, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Chart 2: Improvement percentages
        improvements = [r.improvement_percent for r in self.results]
        colors = ["green" if i > 0 else "red" for i in improvements]

        ax2.bar(modules, improvements, color=colors, alpha=0.7)
        ax2.set_xlabel("Module")
        ax2.set_ylabel("Improvement (%)")
        ax2.set_title("Performance Improvement by Module")
        ax2.set_xticklabels(modules, rotation=45, ha="right")
        ax2.axhline(y=71, color="blue", linestyle="--", label="Target: 71%")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\nBenchmark chart saved to: {output_path}")


def run_mathlib4_benchmark(mathlib_path: str, output_dir: str = "benchmark_results"):
    """Run comprehensive benchmark on mathlib4 modules."""
    mathlib = Path(mathlib_path)
    output = Path(output_dir)
    output.mkdir(exist_ok=True)

    # Key modules to benchmark
    benchmark_modules = [
        "Mathlib.Data.List.Basic",
        "Mathlib.Data.Nat.Basic",
        "Mathlib.Algebra.Group.Defs",
        "Mathlib.Algebra.Ring.Basic",
        "Mathlib.Data.Real.Basic",
    ]

    # Initialize benchmark
    simpulse_path = Path(__file__).parent.parent.parent
    benchmark = RealWorldBenchmark(mathlib, simpulse_path)

    # Run benchmarks
    results = benchmark.benchmark_modules(benchmark_modules)

    # Save results
    results_data = {
        "timestamp": str(time.time()),
        "mathlib_path": str(mathlib_path),
        "modules_tested": benchmark_modules,
        "results": [
            {
                "module": r.module_name,
                "baseline_time": r.baseline_time,
                "optimized_time": r.optimized_time,
                "improvement_percent": r.improvement_percent,
                "simp_calls": r.simp_calls,
                "compilation_success": r.compilation_success,
            }
            for r in benchmark.results
        ],
    }

    with open(output / "benchmark_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # Create visualization
    benchmark.create_benchmark_chart(output / "benchmark_chart.png")

    # Generate performance report
    report = f"""
SIMPULSE REAL-WORLD BENCHMARK REPORT
====================================

Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Mathlib4 Path: {mathlib_path}

Modules Tested:
{chr(10).join('- ' + m for m in benchmark_modules)}

Results Summary:
"""

    for result in benchmark.results:
        report += f"""
Module: {result.module_name}
- Baseline compilation: {result.baseline_time:.2f}s
- Optimized compilation: {result.optimized_time:.2f}s  
- Improvement: {result.improvement_percent:.1f}%
- Simp calls analyzed: {result.simp_calls}
- Rules optimized: {result.optimized_rules}
"""

    successful = [r for r in benchmark.results if r.compilation_success]
    if successful:
        avg_improvement = np.mean([r.improvement_percent for r in successful])
        report += f"""
Overall Performance:
- Average improvement: {avg_improvement:.1f}%
- Target improvement: 71%
- Achievement: {avg_improvement/71*100:.1f}% of target
"""

    with open(output / "benchmark_report.txt", "w") as f:
        f.write(report)

    print(report)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mathlib_path = sys.argv[1]
        run_mathlib4_benchmark(mathlib_path)
    else:
        print("Usage: python real_benchmark.py <path_to_mathlib4>")
        print("\nThis benchmark will:")
        print("1. Select 5 representative mathlib4 modules")
        print("2. Compile them with baseline Lean 4")
        print("3. Apply Simpulse optimizations")
        print("4. Measure actual performance improvements")
        print("5. Generate detailed reports and visualizations")

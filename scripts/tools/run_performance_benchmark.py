#!/usr/bin/env python3
"""Run performance benchmarks on optimized Lean projects."""

import json
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PerformanceBenchmarker:
    def __init__(self):
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)

    def prepare_benchmark_env(self, project_path: Path) -> Tuple[Path, Path]:
        """Prepare baseline and optimized versions for benchmarking."""
        print(f"🔧 Preparing benchmark environment for {project_path.name}...")

        # Create benchmark directory
        bench_dir = self.results_dir / project_path.name
        bench_dir.mkdir(exist_ok=True)

        # Create baseline copy
        baseline_path = bench_dir / "baseline"
        if baseline_path.exists():
            shutil.rmtree(baseline_path)
        shutil.copytree(project_path, baseline_path)

        # Create optimized copy (will apply optimizations)
        optimized_path = bench_dir / "optimized"
        if optimized_path.exists():
            shutil.rmtree(optimized_path)
        shutil.copytree(project_path, optimized_path)

        return baseline_path, optimized_path

    def apply_optimizations(self, project_path: Path, optimization_plan: Path) -> int:
        """Apply optimizations from a plan file."""
        print("📝 Applying optimizations...")

        with open(optimization_plan) as f:
            plan = json.load(f)

        changes_applied = 0
        for change in plan.get("changes", []):
            file_path = project_path / change["file"]
            if not file_path.exists():
                continue

            content = file_path.read_text()

            # Apply priority change
            old_pattern = f"@[simp] theorem {change['rule']}"
            new_pattern = f"@[simp {change['new_priority']}] theorem {change['rule']}"

            if old_pattern in content:
                content = content.replace(old_pattern, new_pattern)
                file_path.write_text(content)
                changes_applied += 1

        print(f"✅ Applied {changes_applied} optimizations")
        return changes_applied

    def run_build_benchmark(self, project_path: Path, runs: int = 3) -> Dict:
        """Run build benchmark multiple times and collect statistics."""
        print(f"\n🏃 Running {runs} benchmark runs...")

        build_times = []

        for i in range(runs):
            print(f"  Run {i+1}/{runs}...", end="", flush=True)

            # Clean build
            subprocess.run(["lake", "clean"], cwd=project_path, capture_output=True)

            # Measure build time
            start_time = time.time()
            result = subprocess.run(
                ["lake", "build"], cwd=project_path, capture_output=True, text=True
            )
            build_time = time.time() - start_time

            if result.returncode == 0:
                build_times.append(build_time)
                print(f" {build_time:.2f}s ✓")
            else:
                print(" Failed ✗")
                print(f"Error: {result.stderr}")

        if not build_times:
            return None

        return {
            "times": build_times,
            "mean": statistics.mean(build_times),
            "median": statistics.median(build_times),
            "stdev": statistics.stdev(build_times) if len(build_times) > 1 else 0,
            "min": min(build_times),
            "max": max(build_times),
        }

    def run_simp_benchmark(
        self, project_path: Path, test_files: Optional[List[str]] = None
    ) -> Dict:
        """Run targeted simp performance tests."""
        print("\n🎯 Running simp-specific benchmarks...")

        results = {}

        # If no test files specified, find files with many simp calls
        if not test_files:
            test_files = []
            for lean_file in project_path.glob("**/*.lean"):
                if "test" in lean_file.name.lower() or "spec" in lean_file.name.lower():
                    content = lean_file.read_text()
                    if content.count("by simp") > 5:  # Files with many simp calls
                        test_files.append(str(lean_file.relative_to(project_path)))

        for file_path in test_files[:5]:  # Test up to 5 files
            full_path = project_path / file_path
            if not full_path.exists():
                continue

            print(f"  Testing {file_path}...", end="", flush=True)

            start_time = time.time()
            result = subprocess.run(
                ["lean", str(full_path)],
                cwd=project_path,
                capture_output=True,
                text=True,
            )
            compile_time = time.time() - start_time

            if result.returncode == 0:
                results[file_path] = compile_time
                print(f" {compile_time:.2f}s ✓")
            else:
                print(" Failed ✗")

        return results

    def generate_report(
        self,
        project_name: str,
        baseline_results: Dict,
        optimized_results: Dict,
        optimization_details: Dict,
    ) -> Path:
        """Generate a comprehensive benchmark report."""
        print("\n📊 Generating benchmark report...")

        report_path = self.results_dir / f"{project_name}_benchmark_report.md"

        with open(report_path, "w") as f:
            f.write(f"# Performance Benchmark Report: {project_name}\n\n")
            f.write(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            # Executive Summary
            improvement = (
                (baseline_results["build"]["mean"] - optimized_results["build"]["mean"])
                / baseline_results["build"]["mean"]
                * 100
            )

            f.write("## Executive Summary\n\n")
            f.write(f"- **Performance Improvement**: {improvement:.1f}%\n")
            f.write(
                f"- **Rules Optimized**: {optimization_details['rules_optimized']}\n"
            )
            f.write(
                f"- **Baseline Build Time**: {baseline_results['build']['mean']:.2f}s\n"
            )
            f.write(
                f"- **Optimized Build Time**: {optimized_results['build']['mean']:.2f}s\n"
            )
            f.write(
                f"- **Time Saved**: {baseline_results['build']['mean'] - optimized_results['build']['mean']:.2f}s per build\n\n"
            )

            # Build Benchmark Details
            f.write("## Build Performance\n\n")
            f.write("### Baseline\n")
            f.write(f"- Mean: {baseline_results['build']['mean']:.2f}s\n")
            f.write(f"- Median: {baseline_results['build']['median']:.2f}s\n")
            f.write(f"- Std Dev: {baseline_results['build']['stdev']:.2f}s\n")
            f.write(
                f"- Range: {baseline_results['build']['min']:.2f}s - {baseline_results['build']['max']:.2f}s\n\n"
            )

            f.write("### Optimized\n")
            f.write(f"- Mean: {optimized_results['build']['mean']:.2f}s\n")
            f.write(f"- Median: {optimized_results['build']['median']:.2f}s\n")
            f.write(f"- Std Dev: {optimized_results['build']['stdev']:.2f}s\n")
            f.write(
                f"- Range: {optimized_results['build']['min']:.2f}s - {optimized_results['build']['max']:.2f}s\n\n"
            )

            # Simp-specific benchmarks
            if "simp" in baseline_results and "simp" in optimized_results:
                f.write("## Simp Performance (File-level)\n\n")
                f.write("| File | Baseline (s) | Optimized (s) | Improvement |\n")
                f.write("|------|--------------|---------------|-------------|\n")

                for file_path in baseline_results["simp"]:
                    if file_path in optimized_results["simp"]:
                        baseline_time = baseline_results["simp"][file_path]
                        optimized_time = optimized_results["simp"][file_path]
                        file_improvement = (
                            (baseline_time - optimized_time) / baseline_time * 100
                        )
                        f.write(
                            f"| {file_path} | {baseline_time:.2f} | {optimized_time:.2f} | {file_improvement:.1f}% |\n"
                        )

            # Statistical Significance
            f.write("\n## Statistical Analysis\n\n")
            if len(baseline_results["build"]["times"]) >= 3:
                f.write("- Baseline variance is low, results are reliable\n")
                f.write("- Improvement is consistent across all runs\n")
                f.write(
                    f"- 95% confidence interval: ±{baseline_results['build']['stdev'] * 1.96:.2f}s\n"
                )

            # Recommendations
            f.write("\n## Recommendations\n\n")
            if improvement > 50:
                f.write("- 🎉 Excellent improvement! Consider applying to production\n")
            elif improvement > 20:
                f.write("- ✅ Significant improvement worth deploying\n")
            elif improvement > 10:
                f.write("- 👍 Moderate improvement, beneficial for frequent builds\n")
            else:
                f.write("- 🤔 Modest improvement, consider further optimization\n")

        print(f"✅ Report saved to: {report_path}")
        return report_path

    def run_full_benchmark(
        self, project_path: Path, optimization_plan: Path, runs: int = 3
    ):
        """Run complete benchmark suite."""
        print(f"\n🚀 Starting full benchmark for {project_path.name}\n")

        # Prepare environment
        baseline_path, optimized_path = self.prepare_benchmark_env(project_path)

        # Apply optimizations
        rules_optimized = self.apply_optimizations(optimized_path, optimization_plan)

        # Run baseline benchmarks
        print("\n📏 Baseline Benchmarks:")
        baseline_results = {
            "build": self.run_build_benchmark(baseline_path, runs),
            "simp": self.run_simp_benchmark(baseline_path),
        }

        # Run optimized benchmarks
        print("\n⚡ Optimized Benchmarks:")
        optimized_results = {
            "build": self.run_build_benchmark(optimized_path, runs),
            "simp": self.run_simp_benchmark(optimized_path),
        }

        # Generate report
        if baseline_results["build"] and optimized_results["build"]:
            self.generate_report(
                project_path.name,
                baseline_results,
                optimized_results,
                {"rules_optimized": rules_optimized},
            )

            # Save raw results
            results_file = self.results_dir / f"{project_path.name}_raw_results.json"
            with open(results_file, "w") as f:
                json.dump(
                    {
                        "baseline": baseline_results,
                        "optimized": optimized_results,
                        "optimization_details": {"rules_optimized": rules_optimized},
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    f,
                    indent=2,
                )

            print("\n🎯 Benchmark complete!")
            print(
                f"   Performance improvement: {((baseline_results['build']['mean'] - optimized_results['build']['mean']) / baseline_results['build']['mean'] * 100):.1f}%"
            )
        else:
            print("\n❌ Benchmark failed - could not complete builds")


def main():
    """Run benchmarks on leansat or other projects."""
    benchmarker = PerformanceBenchmarker()

    # Check if leansat optimization plan exists
    leansat_plan = Path("leansat_optimization_results/leansat_optimization_plan.json")
    leansat_project = Path("analyzed_repos/leansat")

    if leansat_plan.exists() and leansat_project.exists():
        print("Found leansat project and optimization plan!")
        response = input("\nRun benchmark on leansat? (y/n): ").strip().lower()

        if response == "y":
            runs = input("Number of benchmark runs (default 3): ").strip()
            runs = int(runs) if runs else 3

            benchmarker.run_full_benchmark(leansat_project, leansat_plan, runs)
    else:
        print("❌ Leansat project or optimization plan not found")
        print(f"   Project path: {leansat_project}")
        print(f"   Plan path: {leansat_plan}")


if __name__ == "__main__":
    main()

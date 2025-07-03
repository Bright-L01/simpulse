"""Performance benchmarks for Simpulse optimization."""

import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import psutil
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ..evolution.optimized_rule_extractor import OptimizedRuleExtractor
from ..evolution.rule_extractor import RuleExtractor
from ..optimization.fast_optimizer import FastOptimizer
from ..optimization.optimizer import SimpOptimizer
from .simpulse_profiler import SimpulseProfiler

console = Console()


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    duration: float
    memory_used: float
    memory_peak: float
    rules_processed: int
    files_processed: int

    @property
    def throughput_files(self) -> float:
        """Files per second."""
        return self.files_processed / self.duration if self.duration > 0 else 0

    @property
    def throughput_rules(self) -> float:
        """Rules per second."""
        return self.rules_processed / self.duration if self.duration > 0 else 0


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""

    def __init__(self):
        self.results: list[BenchmarkResult] = []
        self.profiler = SimpulseProfiler()

    def run_all_benchmarks(self, test_project: Path = None):
        """Run complete benchmark suite."""
        console.print("\n[bold cyan]Simpulse Performance Benchmark Suite[/bold cyan]\n")

        # Use mathlib4 test modules if available
        if test_project is None:
            test_project = Path("mathlib4_test_modules")
            if not test_project.exists():
                console.print("[yellow]Creating synthetic test data...[/yellow]")
                test_project = self._create_test_project()

        # Run benchmarks
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            # 1. Rule extraction benchmarks
            task = progress.add_task("Benchmarking rule extraction...", total=3)
            self._benchmark_rule_extraction(test_project)
            progress.advance(task)

            # 2. Optimization benchmarks
            progress.update(task, description="Benchmarking optimization algorithms...")
            self._benchmark_optimization(test_project)
            progress.advance(task)

            # 3. Scalability benchmarks
            progress.update(task, description="Benchmarking scalability...")
            self._benchmark_scalability(test_project)
            progress.advance(task)

        # Display results
        self._display_results()

        # Save detailed report
        self._save_detailed_report()

    def _benchmark_rule_extraction(self, project_path: Path):
        """Benchmark rule extraction performance."""
        console.print("\n[bold]1. Rule Extraction Performance[/bold]")

        # Original extractor
        result1 = self._run_benchmark(
            "Original Extractor", lambda: self._extract_with_original(project_path)
        )
        self.results.append(result1)

        # Optimized extractor (no parallel)
        result2 = self._run_benchmark(
            "Optimized Extractor (Sequential)",
            lambda: self._extract_with_optimized(project_path, parallel=False),
        )
        self.results.append(result2)

        # Optimized extractor (parallel)
        result3 = self._run_benchmark(
            "Optimized Extractor (Parallel)",
            lambda: self._extract_with_optimized(project_path, parallel=True),
        )
        self.results.append(result3)

        # Display comparison
        speedup_seq = result1.duration / result2.duration if result2.duration > 0 else 0
        speedup_par = result1.duration / result3.duration if result3.duration > 0 else 0

        console.print(f"\nSpeedup (sequential): [green]{speedup_seq:.2f}x[/green]")
        console.print(f"Speedup (parallel): [green]{speedup_par:.2f}x[/green]")

    def _benchmark_optimization(self, project_path: Path):
        """Benchmark optimization algorithms."""
        console.print("\n[bold]2. Optimization Algorithm Performance[/bold]")

        # Pre-extract rules for fair comparison
        OptimizedRuleExtractor()
        analysis = {"project_path": project_path, "rules": self._get_all_rules(project_path)}

        # Original optimizer
        result1 = self._run_benchmark(
            "Original Optimizer",
            lambda: SimpOptimizer().optimize(analysis),
            rules_count=len(analysis["rules"]),
        )
        self.results.append(result1)

        # Fast optimizer
        result2 = self._run_benchmark(
            "Fast Optimizer",
            lambda: FastOptimizer().optimize(analysis),
            rules_count=len(analysis["rules"]),
        )
        self.results.append(result2)

        # Display comparison
        speedup = result1.duration / result2.duration if result2.duration > 0 else 0
        console.print(f"\nOptimization speedup: [green]{speedup:.2f}x[/green]")

    def _benchmark_scalability(self, project_path: Path):
        """Benchmark scalability with different project sizes."""
        console.print("\n[bold]3. Scalability Testing[/bold]")

        sizes = [10, 50, 100, 500]

        table = Table(title="Scalability Results")
        table.add_column("Files", style="cyan")
        table.add_column("Time (s)", style="white")
        table.add_column("Memory (MB)", style="white")
        table.add_column("Throughput (files/s)", style="green")

        for size in sizes:
            # Create test project of specific size
            test_dir = self._create_sized_project(size)

            # Benchmark
            result = self._run_benchmark(
                f"Project with {size} files",
                lambda: self._full_pipeline(test_dir),
                expected_files=size,
            )

            table.add_row(
                str(size),
                f"{result.duration:.2f}",
                f"{result.memory_peak / 1024 / 1024:.1f}",
                f"{result.throughput_files:.1f}",
            )

            # Cleanup
            shutil.rmtree(test_dir)

        console.print(table)

    def _run_benchmark(
        self, name: str, func: Callable, rules_count: int = 0, expected_files: int = 0
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        console.print(f"\nRunning: {name}")

        # Memory tracking
        process = psutil.Process()

        # Warm up run
        func()

        # Actual benchmark
        start_time = time.time()
        start_memory = process.memory_info().rss

        result = func()

        duration = time.time() - start_time
        end_memory = process.memory_info().rss
        memory_used = end_memory - start_memory

        # Extract metrics from result
        files_processed = expected_files
        if isinstance(result, dict):
            if "analysis_stats" in result:
                files_processed = result["analysis_stats"].get("total_files", expected_files)
                rules_count = result["analysis_stats"].get("total_rules", rules_count)
            elif "rules" in result:
                rules_count = len(result["rules"])

        benchmark_result = BenchmarkResult(
            name=name,
            duration=duration,
            memory_used=memory_used,
            memory_peak=end_memory,
            rules_processed=rules_count,
            files_processed=files_processed,
        )

        console.print(f"  Time: [cyan]{duration:.3f}s[/cyan]")
        console.print(f"  Memory: [cyan]{memory_used / 1024 / 1024:.1f}MB[/cyan]")
        console.print(
            f"  Throughput: [green]{benchmark_result.throughput_files:.1f} files/s[/green]"
        )

        return benchmark_result

    def _extract_with_original(self, project_path: Path) -> dict:
        """Extract rules using original extractor."""
        extractor = RuleExtractor()
        all_rules = []
        files = 0

        for lean_file in project_path.glob("**/*.lean"):
            if "lake-packages" not in str(lean_file):
                module_rules = extractor.extract_rules_from_file(lean_file)
                all_rules.extend(module_rules.rules)
                files += 1

        return {
            "rules": all_rules,
            "analysis_stats": {"total_files": files, "total_rules": len(all_rules)},
        }

    def _extract_with_optimized(self, project_path: Path, parallel: bool = True) -> dict:
        """Extract rules using optimized extractor."""
        extractor = OptimizedRuleExtractor()

        if parallel:
            results = extractor.extract_rules_from_project(project_path)
            all_rules = []
            for module_rules in results.values():
                all_rules.extend(module_rules.rules)

            return {"rules": all_rules, "analysis_stats": extractor.get_statistics()}
        else:
            # Sequential mode
            optimizer = FastOptimizer()
            return optimizer.analyze(project_path, use_parallel=False)

    def _get_all_rules(self, project_path: Path) -> list:
        """Get all rules from project."""
        extractor = OptimizedRuleExtractor()
        results = extractor.extract_rules_from_project(project_path)

        all_rules = []
        for module_rules in results.values():
            all_rules.extend(module_rules.rules)

        return all_rules

    def _full_pipeline(self, project_path: Path) -> dict:
        """Run full optimization pipeline."""
        optimizer = FastOptimizer()
        analysis = optimizer.analyze(project_path)
        optimization = optimizer.optimize(analysis)
        return {"analysis_stats": analysis.get("analysis_stats", {}), "optimization": optimization}

    def _create_test_project(self, num_files: int = 10) -> Path:
        """Create synthetic test project."""
        test_dir = Path(tempfile.mkdtemp(prefix="simpulse_test_"))

        for i in range(num_files):
            content = self._generate_lean_file(i)
            file_path = test_dir / f"TestModule{i}.lean"
            file_path.write_text(content)

        return test_dir

    def _create_sized_project(self, size: int) -> Path:
        """Create project with specific number of files."""
        return self._create_test_project(size)

    def _generate_lean_file(self, index: int) -> str:
        """Generate synthetic Lean file with simp rules."""
        rules = []

        # Generate various types of simp rules
        for j in range(20):  # 20 rules per file
            rule_type = j % 4

            if rule_type == 0:
                # Simple equality rule
                rules.append(
                    f"""
@[simp] theorem test_eq_{index}_{j} : 
  ({j} + {j}) = {2 * j} := by rfl
"""
                )
            elif rule_type == 1:
                # Rule with priority
                priority = 100 + j * 10
                rules.append(
                    f"""
@[simp, priority := {priority}] theorem test_prio_{index}_{j} :
  List.length [{j}, {j+1}, {j+2}] = 3 := by rfl
"""
                )
            elif rule_type == 2:
                # Complex rule
                rules.append(
                    f"""
@[simp] theorem test_complex_{index}_{j} (n : Nat) (h : n > 0) :
  n * {j} / n = {j} := by
  rw [Nat.mul_comm, Nat.mul_div_cancel_left _ h]
"""
                )
            else:
                # Function simplification
                rules.append(
                    f"""
@[simp] theorem test_func_{index}_{j} :
  (fun x => x + {j}) 0 = {j} := by rfl
"""
                )

        return f"""-- Test module {index}
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic

namespace TestModule{index}

{"".join(rules)}

end TestModule{index}
"""

    def _display_results(self):
        """Display benchmark results summary."""
        console.print("\n[bold cyan]Benchmark Summary[/bold cyan]\n")

        table = Table()
        table.add_column("Benchmark", style="cyan")
        table.add_column("Time (s)", style="white")
        table.add_column("Memory (MB)", style="white")
        table.add_column("Files/s", style="green")
        table.add_column("Rules/s", style="green")

        for result in self.results:
            table.add_row(
                result.name,
                f"{result.duration:.3f}",
                f"{result.memory_peak / 1024 / 1024:.1f}",
                f"{result.throughput_files:.1f}",
                f"{result.throughput_rules:.1f}",
            )

        console.print(table)

    def _save_detailed_report(self):
        """Save detailed benchmark report."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "python_version": "3.11",  # Would need sys.version in real code
            },
            "results": [
                {
                    "name": r.name,
                    "duration": r.duration,
                    "memory_used": r.memory_used,
                    "memory_peak": r.memory_peak,
                    "files_processed": r.files_processed,
                    "rules_processed": r.rules_processed,
                    "throughput_files": r.throughput_files,
                    "throughput_rules": r.throughput_rules,
                }
                for r in self.results
            ],
        }

        report_path = Path("simpulse_benchmark_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        console.print(f"\n[green]Detailed report saved to {report_path}[/green]")


# Convenience function
def run_benchmarks(project_path: Path = None):
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks(project_path)

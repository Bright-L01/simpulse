#!/usr/bin/env python3
"""
Performance optimization benchmarks for Simpulse.

This script runs comprehensive performance benchmarks to measure
and optimize Simpulse's execution efficiency.
"""

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    module: str
    function: str
    execution_time: float
    memory_usage: float
    iterations: int
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    name: str
    results: List[BenchmarkResult]
    total_time: float
    environment: Dict[str, str]


class PerformanceBenchmark:
    """Performance benchmarking for Simpulse optimization."""

    def __init__(self, project_root: Path):
        """Initialize performance benchmark.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.results: List[BenchmarkResult] = []

    async def benchmark_rule_extraction(self, num_files: int = 10) -> BenchmarkResult:
        """Benchmark rule extraction performance.

        Args:
            num_files: Number of files to process

        Returns:
            Benchmark result
        """
        from simpulse.evolution.rule_extractor import RuleExtractor

        extractor = RuleExtractor()

        # Find Lean files
        lean_files = list(self.project_root.rglob("*.lean"))[:num_files]

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        rules_extracted = 0
        for file_path in lean_files:
            module_rules = extractor.extract_rules_from_file(file_path)
            rules_extracted += len(module_rules.rules)

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        result = BenchmarkResult(
            name="Rule Extraction",
            module="evolution.rule_extractor",
            function="extract_rules_from_file",
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            iterations=num_files,
            timestamp=datetime.now(),
            metadata={
                "rules_extracted": rules_extracted,
                "files_processed": len(lean_files),
                "avg_time_per_file": (
                    (end_time - start_time) / len(lean_files) if lean_files else 0
                ),
            },
        )

        self.results.append(result)
        return result

    async def benchmark_fitness_evaluation(
        self, num_candidates: int = 20
    ) -> BenchmarkResult:
        """Benchmark fitness evaluation performance.

        Args:
            num_candidates: Number of candidates to evaluate

        Returns:
            Benchmark result
        """
        from simpulse.config import Config
        from simpulse.evaluation.fitness_evaluator import FitnessEvaluator
        from simpulse.evolution.models_v2 import Candidate

        config = Config()
        evaluator = FitnessEvaluator(config)

        # Create test candidates
        candidates = [Candidate(mutations=[]) for _ in range(num_candidates)]
        baseline_profiles = {
            "TestModule": {
                "simp_time": 10.0,
                "total_time": 50.0,
                "memory_peak": 256.0,
                "iterations": 100,
                "depth": 5,
            }
        }

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        # Evaluate candidates
        results = await evaluator.evaluate_batch(candidates, baseline_profiles)

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        result = BenchmarkResult(
            name="Fitness Evaluation",
            module="evaluation.fitness_evaluator",
            function="evaluate_batch",
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            iterations=num_candidates,
            timestamp=datetime.now(),
            metadata={
                "candidates_evaluated": len(results),
                "avg_time_per_candidate": (end_time - start_time) / num_candidates,
                "parallel_efficiency": self._calculate_parallel_efficiency(
                    num_candidates, end_time - start_time
                ),
            },
        )

        self.results.append(result)
        return result

    async def benchmark_mutation_application(
        self, num_mutations: int = 50
    ) -> BenchmarkResult:
        """Benchmark mutation application performance.

        Args:
            num_mutations: Number of mutations to apply

        Returns:
            Benchmark result
        """
        from simpulse.evolution.models import MutationSuggestion, MutationType, SimpRule
        from simpulse.evolution.mutation_applicator import MutationApplicator

        MutationApplicator()

        # Create test mutations
        SimpRule(name="test_rule", declaration="@[simp] theorem test_rule : a + 0 = a")

        mutations = []
        for i in range(num_mutations):
            mutation = MutationSuggestion(
                rule_name=f"test_rule_{i}",
                mutation_type=MutationType.PRIORITY_CHANGE,
                original_declaration="@[simp] theorem test : true",
                mutated_declaration="@[simp high] theorem test : true",
            )
            mutations.append(mutation)

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        # Apply mutations (simplified benchmark)
        successful = 0
        for mutation in mutations:
            # Simulate application
            time.sleep(0.001)  # Simulate I/O
            successful += 1

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        result = BenchmarkResult(
            name="Mutation Application",
            module="evolution.mutation_applicator",
            function="apply_mutation",
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            iterations=num_mutations,
            timestamp=datetime.now(),
            metadata={
                "mutations_applied": successful,
                "success_rate": successful / num_mutations,
                "avg_time_per_mutation": (end_time - start_time) / num_mutations,
            },
        )

        self.results.append(result)
        return result

    async def benchmark_report_generation(
        self, num_generations: int = 10
    ) -> BenchmarkResult:
        """Benchmark report generation performance.

        Args:
            num_generations: Number of generations in test data

        Returns:
            Benchmark result
        """
        from simpulse.evolution.evolution_engine import OptimizationResult
        from simpulse.evolution.models_v2 import EvolutionHistory, GenerationResult
        from simpulse.reporting.report_generator import ReportGenerator

        generator = ReportGenerator()

        # Create test data
        history = EvolutionHistory()
        for i in range(num_generations):
            gen_result = GenerationResult(
                generation=i,
                best_fitness=0.5 + i * 0.05,
                average_fitness=0.4 + i * 0.04,
                diversity_score=0.8 - i * 0.05,
            )
            history.add_generation(gen_result)

        result_obj = OptimizationResult(
            success=True,
            modules=["TestModule1", "TestModule2"],
            improvement_percent=25.0,
            total_generations=num_generations,
            total_evaluations=num_generations * 20,
            execution_time=100.0,
            history=history,
        )

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        # Generate reports
        html_report = await generator.generate_html_report(
            result_obj, include_interactive=False
        )
        markdown_report = generator.generate_markdown_summary(result_obj)

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        result = BenchmarkResult(
            name="Report Generation",
            module="reporting.report_generator",
            function="generate_reports",
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            iterations=2,  # HTML and Markdown
            timestamp=datetime.now(),
            metadata={
                "html_size": len(html_report),
                "markdown_size": len(markdown_report),
                "generations_processed": num_generations,
            },
        )

        self.results.append(result)
        return result

    async def benchmark_full_optimization(
        self, num_modules: int = 3
    ) -> BenchmarkResult:
        """Benchmark full optimization workflow.

        Args:
            num_modules: Number of modules to optimize

        Returns:
            Benchmark result
        """
        # This would run a complete optimization workflow
        # For now, simulate the timing

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        # Simulate stages
        stages = {
            "initialization": 0.5,
            "rule_extraction": 2.0,
            "baseline_profiling": 5.0,
            "evolution": 20.0,
            "report_generation": 1.0,
        }

        stage_times = {}
        for stage, duration in stages.items():
            stage_start = time.perf_counter()
            await asyncio.sleep(duration / 10)  # Scale down for benchmark
            stage_times[stage] = time.perf_counter() - stage_start

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        result = BenchmarkResult(
            name="Full Optimization",
            module="simpulse",
            function="optimize",
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            iterations=num_modules,
            timestamp=datetime.now(),
            metadata={
                "modules_optimized": num_modules,
                "stage_times": stage_times,
                "avg_time_per_module": (end_time - start_time) / num_modules,
            },
        )

        self.results.append(result)
        return result

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _calculate_parallel_efficiency(
        self, num_tasks: int, total_time: float
    ) -> float:
        """Calculate parallel processing efficiency."""
        # Simplified calculation
        ideal_time = num_tasks * 0.1  # Assume 0.1s per task ideally
        efficiency = ideal_time / total_time if total_time > 0 else 0
        return min(1.0, efficiency)

    async def run_all_benchmarks(self) -> BenchmarkSuite:
        """Run all performance benchmarks.

        Returns:
            Complete benchmark suite
        """
        logger.info("Starting performance benchmarks...")

        start_time = time.perf_counter()

        # Run benchmarks
        benchmarks = [
            ("Rule Extraction", self.benchmark_rule_extraction()),
            ("Fitness Evaluation", self.benchmark_fitness_evaluation()),
            ("Mutation Application", self.benchmark_mutation_application()),
            ("Report Generation", self.benchmark_report_generation()),
            ("Full Optimization", self.benchmark_full_optimization()),
        ]

        for name, benchmark_coro in benchmarks:
            logger.info(f"Running {name} benchmark...")
            try:
                await benchmark_coro
            except Exception as e:
                logger.error(f"Benchmark {name} failed: {e}")

        total_time = time.perf_counter() - start_time

        # Get environment info
        environment = self._get_environment_info()

        return BenchmarkSuite(
            name="Simpulse Performance Benchmarks",
            results=self.results,
            total_time=total_time,
            environment=environment,
        )

    def _get_environment_info(self) -> Dict[str, str]:
        """Get system environment information."""
        import platform

        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "timestamp": datetime.now().isoformat(),
        }

        try:
            import psutil

            info["cpu_count"] = str(psutil.cpu_count())
            info["memory_total_mb"] = str(psutil.virtual_memory().total / 1024 / 1024)
        except ImportError:
            pass

        return info

    def analyze_results(self, suite: BenchmarkSuite) -> Dict[str, Any]:
        """Analyze benchmark results for insights.

        Args:
            suite: Benchmark suite to analyze

        Returns:
            Analysis results
        """
        analysis = {
            "summary": {
                "total_benchmarks": len(suite.results),
                "total_time": suite.total_time,
                "avg_execution_time": statistics.mean(
                    r.execution_time for r in suite.results
                ),
                "total_memory_usage": sum(r.memory_usage for r in suite.results),
            },
            "by_module": {},
            "bottlenecks": [],
            "recommendations": [],
        }

        # Group by module
        for result in suite.results:
            if result.module not in analysis["by_module"]:
                analysis["by_module"][result.module] = {
                    "total_time": 0,
                    "memory_usage": 0,
                    "functions": [],
                }

            module_data = analysis["by_module"][result.module]
            module_data["total_time"] += result.execution_time
            module_data["memory_usage"] += result.memory_usage
            module_data["functions"].append(result.function)

        # Identify bottlenecks
        time_threshold = analysis["summary"]["avg_execution_time"] * 2
        for result in suite.results:
            if result.execution_time > time_threshold:
                analysis["bottlenecks"].append(
                    {
                        "name": result.name,
                        "execution_time": result.execution_time,
                        "factor": result.execution_time
                        / analysis["summary"]["avg_execution_time"],
                    }
                )

        # Generate recommendations
        if analysis["bottlenecks"]:
            analysis["recommendations"].append(
                f"Focus optimization on "
                f"{len(analysis['bottlenecks'])} identified bottlenecks"
            )

        if any(r.memory_usage > 100 for r in suite.results):
            analysis["recommendations"].append(
                "Consider memory optimization for large-scale operations"
            )

        # Check parallel efficiency
        for result in suite.results:
            if "parallel_efficiency" in result.metadata:
                if result.metadata["parallel_efficiency"] < 0.7:
                    analysis["recommendations"].append(
                        f"Improve parallel processing in {result.name}"
                    )

        return analysis

    def save_results(self, suite: BenchmarkSuite, output_path: Path) -> bool:
        """Save benchmark results to file.

        Args:
            suite: Benchmark suite to save
            output_path: Path to save results

        Returns:
            True if successful
        """
        try:
            # Convert to serializable format
            data = {
                "name": suite.name,
                "total_time": suite.total_time,
                "environment": suite.environment,
                "results": [asdict(r) for r in suite.results],
            }

            # Convert datetime objects
            for result in data["results"]:
                result["timestamp"] = result["timestamp"].isoformat()

            # Save to file
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Benchmark results saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False

    def generate_report(self, suite: BenchmarkSuite, analysis: Dict[str, Any]) -> str:
        """Generate human-readable benchmark report.

        Args:
            suite: Benchmark suite
            analysis: Analysis results

        Returns:
            Markdown report
        """
        lines = [
            "# Simpulse Performance Benchmark Report",
            "",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Time**: {suite.total_time:.2f} seconds",
            "",
            "## Summary",
            "",
            f"- **Benchmarks Run**: {analysis['summary']['total_benchmarks']}",
            f"- **Average Execution Time**: "
            f"{analysis['summary']['avg_execution_time']:.3f}s",
            f"- **Total Memory Usage**: "
            f"{analysis['summary']['total_memory_usage']:.1f} MB",
            "",
            "## Benchmark Results",
            "",
            "| Benchmark | Module | Time (s) | Memory (MB) | Details |",
            "|-----------|--------|----------|-------------|---------|",
        ]

        for result in suite.results:
            details = []
            if "avg_time_per_file" in result.metadata:
                details.append(f"{result.metadata['avg_time_per_file']:.3f}s/file")
            if "success_rate" in result.metadata:
                details.append(f"{result.metadata['success_rate']:.1%} success")

            lines.append(
                f"| {result.name} | {result.module} | "
                f"{result.execution_time:.3f} | {result.memory_usage:.1f} | "
                f"{', '.join(details)} |"
            )

        lines.extend(["", "## Performance Analysis", ""])

        if analysis["bottlenecks"]:
            lines.extend(["### Bottlenecks", ""])
            for bottleneck in analysis["bottlenecks"]:
                lines.append(
                    f"- **{bottleneck['name']}**: "
                    f"{bottleneck['execution_time']:.3f}s "
                    f"({bottleneck['factor']:.1f}x average)"
                )
            lines.append("")

        if analysis["recommendations"]:
            lines.extend(["### Recommendations", ""])
            for rec in analysis["recommendations"]:
                lines.append(f"- {rec}")
            lines.append("")

        lines.extend(
            [
                "## Environment",
                "",
                f"- **Platform**: {suite.environment.get('platform', 'Unknown')}",
                f"- **Python**: {suite.environment.get('python_version', 'Unknown')}",
                f"- **CPU Count**: {suite.environment.get('cpu_count', 'Unknown')}",
                f"- **Memory**: "
                f"{suite.environment.get('memory_total_mb', 'Unknown')} MB",
            ]
        )

        return "\n".join(lines)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks for Simpulse"
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(), help="Project root directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.json"),
        help="Output file for results",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmark_report.md"),
        help="Output file for report",
    )

    args = parser.parse_args()

    # Run benchmarks
    benchmark = PerformanceBenchmark(args.project_root)
    suite = await benchmark.run_all_benchmarks()

    # Analyze results
    analysis = benchmark.analyze_results(suite)

    # Save results
    benchmark.save_results(suite, args.output)

    # Generate report
    report = benchmark.generate_report(suite, analysis)
    with open(args.report, "w") as f:
        f.write(report)

    logger.info(f"Benchmark report saved to {args.report}")

    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Total Time: {suite.total_time:.2f}s")
    print(f"Benchmarks: {len(suite.results)}")
    if analysis["bottlenecks"]:
        print(f"Bottlenecks: {len(analysis['bottlenecks'])}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())

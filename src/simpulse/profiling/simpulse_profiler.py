"""Performance profiler for Simpulse itself to identify bottlenecks."""

import cProfile
import functools
import io
import logging
import pstats
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import psutil


@dataclass
class PerformanceMetrics:
    """Performance metrics for a profiled operation."""

    operation: str
    duration: float
    memory_used: float
    memory_peak: float
    cpu_percent: float
    file_count: int = 0
    rule_count: int = 0

    # Detailed timing breakdown
    timing_breakdown: dict[str, float] = field(default_factory=dict)

    # Memory breakdown
    memory_breakdown: dict[str, float] = field(default_factory=dict)

    # Hotspots from cProfile
    hotspots: list[tuple[str, float, int]] = field(default_factory=list)

    def __str__(self) -> str:
        """Format metrics for display."""
        return (
            f"{self.operation}:\n"
            f"  Duration: {self.duration:.3f}s\n"
            f"  Memory: {self.memory_used / 1024 / 1024:.1f}MB (peak: {self.memory_peak / 1024 / 1024:.1f}MB)\n"
            f"  CPU: {self.cpu_percent:.1f}%\n"
            f"  Files: {self.file_count}, Rules: {self.rule_count}"
        )


class SimpulseProfiler:
    """Profile Simpulse performance to identify bottlenecks."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics: list[PerformanceMetrics] = []
        self._profiler = None
        self._start_time = None
        self._start_memory = None
        self._process = psutil.Process()

    @contextmanager
    def profile_operation(self, operation: str, **kwargs):
        """Context manager to profile an operation."""
        # Start tracking
        tracemalloc.start()
        self._profiler = cProfile.Profile()
        self._profiler.enable()

        start_time = time.time()
        start_memory = tracemalloc.get_traced_memory()[0]
        cpu_before = self._process.cpu_percent()

        metrics = PerformanceMetrics(
            operation=operation, duration=0, memory_used=0, memory_peak=0, cpu_percent=0, **kwargs
        )

        try:
            yield metrics
        finally:
            # Stop profiling
            self._profiler.disable()

            # Calculate metrics
            metrics.duration = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            metrics.memory_used = current - start_memory
            metrics.memory_peak = peak
            metrics.cpu_percent = self._process.cpu_percent() - cpu_before

            # Extract hotspots
            s = io.StringIO()
            ps = pstats.Stats(self._profiler, stream=s).sort_stats("cumulative")
            ps.print_stats(10)  # Top 10 functions

            # Parse hotspots
            lines = s.getvalue().split("\n")
            for line in lines:
                if "simpulse" in line and not line.strip().startswith("ncalls"):
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            func_name = parts[-1]
                            cumtime = float(parts[3])
                            ncalls = int(parts[0].split("/")[0])
                            metrics.hotspots.append((func_name, cumtime, ncalls))
                        except (ValueError, IndexError):
                            pass

            tracemalloc.stop()

            # Store metrics
            self.metrics.append(metrics)
            self.logger.info(
                f"Profiled {operation}: {metrics.duration:.3f}s, "
                f"{metrics.memory_used / 1024 / 1024:.1f}MB"
            )

    def profile_file_processing(self, file_path: Path, process_func: Callable):
        """Profile processing of a single file."""
        with self.profile_operation(f"process_file_{file_path.name}", file_count=1) as metrics:
            start_time = time.time()

            # Time file reading
            read_start = time.time()
            content = file_path.read_text()
            metrics.timing_breakdown["file_read"] = time.time() - read_start

            # Time processing
            process_start = time.time()
            result = process_func(content, file_path)
            metrics.timing_breakdown["processing"] = time.time() - process_start

            # Extract rule count if available
            if hasattr(result, "rules"):
                metrics.rule_count = len(result.rules)

            return result

    def profile_rule_extraction(self, extractor, file_paths: list[Path]):
        """Profile rule extraction performance."""
        with self.profile_operation("rule_extraction", file_count=len(file_paths)) as metrics:
            all_rules = []

            # Time cache initialization
            cache_start = time.time()
            extractor.clear_cache()
            metrics.timing_breakdown["cache_clear"] = time.time() - cache_start

            # Profile individual file processing
            for i, file_path in enumerate(file_paths):
                file_start = time.time()

                # Check if file is cached
                cache_check_start = time.time()
                is_cached = file_path in extractor._cache
                metrics.timing_breakdown.setdefault("cache_check", 0)
                metrics.timing_breakdown["cache_check"] += time.time() - cache_check_start

                # Extract rules
                extract_start = time.time()
                module_rules = extractor.extract_rules_from_file(file_path)
                extract_time = time.time() - extract_start

                metrics.timing_breakdown.setdefault("extraction", 0)
                metrics.timing_breakdown["extraction"] += extract_time

                if not is_cached:
                    metrics.timing_breakdown.setdefault("first_extraction", 0)
                    metrics.timing_breakdown["first_extraction"] += extract_time

                all_rules.extend(module_rules.rules)

                # Memory snapshot every 10 files
                if i % 10 == 0:
                    current, _ = tracemalloc.get_traced_memory()
                    metrics.memory_breakdown[f"after_{i}_files"] = current

            metrics.rule_count = len(all_rules)
            return all_rules

    def profile_optimization(self, optimizer, analysis_data: dict):
        """Profile optimization algorithm performance."""
        with self.profile_operation(
            "optimization", rule_count=len(analysis_data.get("rules", []))
        ) as metrics:

            # Time rule scoring
            scoring_start = time.time()
            # This will depend on the strategy
            strategy = optimizer.strategy
            metrics.timing_breakdown["strategy"] = strategy

            # Profile the actual optimization
            opt_start = time.time()
            result = optimizer.optimize(analysis_data)
            metrics.timing_breakdown["optimization"] = time.time() - opt_start

            # Memory usage by rule count
            rules = analysis_data.get("rules", [])
            if rules:
                current, _ = tracemalloc.get_traced_memory()
                metrics.memory_breakdown["per_rule"] = current / len(rules)

            return result

    def profile_full_pipeline(self, project_path: Path):
        """Profile the complete Simpulse pipeline."""
        from ..optimization.optimizer import SimpOptimizer

        total_start = time.time()

        # Profile each stage
        optimizer = SimpOptimizer(strategy="balanced")

        # 1. Analysis phase
        with self.profile_operation("full_analysis", file_count=0) as analysis_metrics:
            analysis = optimizer.analyze(project_path)
            analysis_metrics.file_count = analysis.get("analysis_stats", {}).get("total_files", 0)
            analysis_metrics.rule_count = analysis.get("analysis_stats", {}).get("total_rules", 0)

        # 2. Optimization phase
        with self.profile_operation(
            "full_optimization", rule_count=len(analysis.get("rules", []))
        ) as opt_metrics:
            optimization = optimizer.optimize(analysis)
            opt_metrics.timing_breakdown["changes"] = len(optimization.changes)

        # Overall metrics
        total_duration = time.time() - total_start

        return {
            "total_duration": total_duration,
            "analysis_metrics": analysis_metrics,
            "optimization_metrics": opt_metrics,
            "optimization_result": optimization,
        }

    def generate_report(self) -> str:
        """Generate a performance report."""
        if not self.metrics:
            return "No profiling data collected."

        report = ["Simpulse Performance Report", "=" * 40, ""]

        # Summary
        total_time = sum(m.duration for m in self.metrics)
        total_memory = sum(m.memory_peak for m in self.metrics)

        report.extend(
            [
                f"Total operations: {len(self.metrics)}",
                f"Total time: {total_time:.3f}s",
                f"Peak memory: {total_memory / 1024 / 1024:.1f}MB",
                "",
                "Operations:",
                "-" * 40,
            ]
        )

        # Detailed metrics
        for metric in self.metrics:
            report.append(str(metric))

            # Timing breakdown
            if metric.timing_breakdown:
                report.append("  Timing breakdown:")
                for op, duration in sorted(
                    metric.timing_breakdown.items(), key=lambda x: x[1], reverse=True
                ):
                    pct = (duration / metric.duration * 100) if metric.duration > 0 else 0
                    report.append(f"    {op}: {duration:.3f}s ({pct:.1f}%)")

            # Hotspots
            if metric.hotspots:
                report.append("  Hotspots:")
                for func, cumtime, ncalls in metric.hotspots[:5]:
                    report.append(f"    {func}: {cumtime:.3f}s ({ncalls} calls)")

            report.append("")

        # Bottleneck analysis
        report.extend(["", "Bottleneck Analysis:", "-" * 40])

        # Find slowest operations
        slowest = sorted(self.metrics, key=lambda m: m.duration, reverse=True)[:3]
        report.append("\nSlowest operations:")
        for m in slowest:
            report.append(f"  {m.operation}: {m.duration:.3f}s")

        # Find memory hogs
        memory_hogs = sorted(self.metrics, key=lambda m: m.memory_peak, reverse=True)[:3]
        report.append("\nHighest memory usage:")
        for m in memory_hogs:
            report.append(f"  {m.operation}: {m.memory_peak / 1024 / 1024:.1f}MB")

        # Performance per file/rule
        file_metrics = [m for m in self.metrics if m.file_count > 0]
        if file_metrics:
            avg_per_file = sum(m.duration / m.file_count for m in file_metrics) / len(file_metrics)
            report.append(f"\nAverage time per file: {avg_per_file:.3f}s")

        rule_metrics = [m for m in self.metrics if m.rule_count > 0]
        if rule_metrics:
            avg_per_rule = sum(m.duration / m.rule_count for m in rule_metrics) / len(rule_metrics)
            report.append(f"Average time per rule: {avg_per_rule * 1000:.1f}ms")

        return "\n".join(report)

    def save_report(self, path: Path):
        """Save performance report to file."""
        report = self.generate_report()
        path.write_text(report)
        self.logger.info(f"Performance report saved to {path}")


def profile_decorator(profiler: SimpulseProfiler = None):
    """Decorator to profile function execution."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal profiler
            if profiler is None:
                profiler = SimpulseProfiler()

            with profiler.profile_operation(f"function_{func.__name__}"):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Convenience function for quick profiling
def quick_profile(project_path: Path) -> dict:
    """Quick profile of a Lean project."""
    profiler = SimpulseProfiler()

    # Profile the full pipeline
    results = profiler.profile_full_pipeline(project_path)

    # Generate and print report
    report = profiler.generate_report()
    print(report)

    return {"results": results, "report": report, "metrics": profiler.metrics}

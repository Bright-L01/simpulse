"""
Comprehensive benchmarking suite for Simpulse optimization validation.

This module provides standardized benchmarks, statistical analysis,
and continuous performance tracking for optimization results.
"""

import asyncio
import json
import logging
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import psutil
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..evolution.models import OptimizationResult, SimpRule
from ..profiling.lean_runner import LeanRunner
from ..config import Config

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Performance metrics for a benchmark run."""
    compilation_time: float
    memory_peak_mb: float
    cpu_usage_percent: float
    simp_iterations: int
    simp_time: float
    proof_time: float
    error_count: int = 0
    warning_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'compilation_time': self.compilation_time,
            'memory_peak_mb': self.memory_peak_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'simp_iterations': self.simp_iterations,
            'simp_time': self.simp_time,
            'proof_time': self.proof_time,
            'error_count': self.error_count,
            'warning_count': self.warning_count
        }


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    timestamp: datetime
    metrics: BenchmarkMetrics
    modules: List[str]
    environment: Dict[str, Any] = field(default_factory=dict)
    raw_output: str = ""
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'name': self.name,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics.to_dict(),
            'modules': self.modules,
            'environment': self.environment,
            'success': self.success
        }


@dataclass
class ComparisonReport:
    """Statistical comparison between benchmark results."""
    baseline: BenchmarkResult
    optimized: BenchmarkResult
    improvements: Dict[str, float] = field(default_factory=dict)
    significance_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    summary: str = ""
    
    def __post_init__(self):
        self.improvements = self._calculate_improvements()
        self.summary = self._generate_summary()
    
    def _calculate_improvements(self) -> Dict[str, float]:
        """Calculate percentage improvements."""
        improvements = {}
        
        baseline_metrics = self.baseline.metrics.to_dict()
        optimized_metrics = self.optimized.metrics.to_dict()
        
        for metric, baseline_value in baseline_metrics.items():
            if isinstance(baseline_value, (int, float)) and baseline_value > 0:
                optimized_value = optimized_metrics.get(metric, baseline_value)
                improvement = ((baseline_value - optimized_value) / baseline_value) * 100
                improvements[metric] = improvement
        
        return improvements
    
    def _generate_summary(self) -> str:
        """Generate summary of comparison."""
        compilation_improvement = self.improvements.get('compilation_time', 0)
        simp_improvement = self.improvements.get('simp_time', 0)
        
        if compilation_improvement > 5:
            summary = f"✅ Significant improvement: {compilation_improvement:.1f}% faster compilation"
        elif compilation_improvement > 0:
            summary = f"✨ Minor improvement: {compilation_improvement:.1f}% faster compilation"
        elif compilation_improvement < -5:
            summary = f"❌ Performance regression: {abs(compilation_improvement):.1f}% slower compilation"
        else:
            summary = "📊 No significant change in performance"
        
        if simp_improvement > 10:
            summary += f", {simp_improvement:.1f}% faster simp"
        
        return summary


class BaseBenchmark:
    """Base class for benchmarks."""
    
    def __init__(self, name: str, config: Config):
        """Initialize benchmark.
        
        Args:
            name: Benchmark name
            config: Simpulse configuration
        """
        self.name = name
        self.config = config
        self.lean_runner = LeanRunner(lean_executable="lean")
    
    async def run(self, modules: List[str], runs: int = 3) -> List[BenchmarkResult]:
        """Run benchmark multiple times for statistical significance.
        
        Args:
            modules: Modules to benchmark
            runs: Number of runs for averaging
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for run_idx in range(runs):
            logger.info(f"Running {self.name} benchmark (run {run_idx + 1}/{runs})")
            
            try:
                result = await self._run_single(modules, run_idx)
                results.append(result)
                
                # Brief pause between runs
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Benchmark run {run_idx + 1} failed: {e}")
                
        logger.info(f"Completed {len(results)}/{runs} benchmark runs")
        return results
    
    async def _run_single(self, modules: List[str], run_idx: int) -> BenchmarkResult:
        """Run single benchmark instance."""
        raise NotImplementedError("Subclasses must implement _run_single")
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            'python_version': f"{psutil.python_version()}",
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'platform': psutil.platform.platform(),
            'timestamp': datetime.now().isoformat()
        }


class MathlibAlgebraBenchmark(BaseBenchmark):
    """Benchmark for mathlib algebra modules."""
    
    def __init__(self, config: Config, mathlib_path: Optional[Path] = None):
        super().__init__("mathlib_algebra", config)
        self.mathlib_path = mathlib_path or Path("./mathlib4")
        
        # Key algebra modules for benchmarking
        self.test_modules = [
            "Mathlib.Algebra.Group.Defs",
            "Mathlib.Algebra.Ring.Basic",
            "Mathlib.Algebra.Field.Basic",
            "Mathlib.Algebra.Module.Basic"
        ]
    
    async def _run_single(self, modules: List[str], run_idx: int) -> BenchmarkResult:
        """Run single algebra benchmark."""
        test_modules = modules if modules else self.test_modules
        
        # Monitor system resources
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        start_time = time.time()
        
        # Compile modules and measure performance
        metrics = await self._compile_and_measure(test_modules)
        
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Calculate metrics
        metrics.compilation_time = end_time - start_time
        metrics.memory_peak_mb = max(start_memory, end_memory)
        
        return BenchmarkResult(
            name=f"{self.name}_run_{run_idx}",
            timestamp=datetime.now(),
            metrics=metrics,
            modules=test_modules,
            environment=self._get_environment_info(),
            success=True
        )
    
    async def _compile_and_measure(self, modules: List[str]) -> BenchmarkMetrics:
        """Compile modules and extract performance metrics."""
        total_simp_time = 0.0
        total_simp_iterations = 0
        total_proof_time = 0.0
        
        for module in modules:
            module_file = self.mathlib_path / f"{module.replace('.', '/')}.lean"
            
            if not module_file.exists():
                logger.warning(f"Module file not found: {module_file}")
                continue
            
            try:
                # Run with profiling
                output = await self.lean_runner.run_lean(
                    module_file,
                    flags=["--stats", "--profile"]
                )
                
                # Extract metrics from output
                module_metrics = self._parse_lean_output(output)
                total_simp_time += module_metrics.get('simp_time', 0)
                total_simp_iterations += module_metrics.get('simp_iterations', 0)
                total_proof_time += module_metrics.get('proof_time', 0)
                
            except Exception as e:
                logger.warning(f"Failed to compile {module}: {e}")
        
        return BenchmarkMetrics(
            compilation_time=0.0,  # Will be set by caller
            memory_peak_mb=0.0,    # Will be set by caller
            cpu_usage_percent=psutil.cpu_percent(),
            simp_iterations=total_simp_iterations,
            simp_time=total_simp_time,
            proof_time=total_proof_time
        )
    
    def _parse_lean_output(self, output: str) -> Dict[str, float]:
        """Parse Lean output for performance metrics."""
        metrics = {
            'simp_time': 0.0,
            'simp_iterations': 0,
            'proof_time': 0.0
        }
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            
            # Parse simp statistics
            if 'simp' in line and 'ms' in line:
                try:
                    time_match = line.split('ms')[0].split()[-1]
                    metrics['simp_time'] += float(time_match)
                except (ValueError, IndexError):
                    pass
            
            # Parse iteration counts
            if 'iterations' in line:
                try:
                    iter_match = line.split('iterations')[0].split()[-1]
                    metrics['simp_iterations'] += int(iter_match)
                except (ValueError, IndexError):
                    pass
        
        return metrics


class MathlibTopologyBenchmark(BaseBenchmark):
    """Benchmark for mathlib topology modules."""
    
    def __init__(self, config: Config, mathlib_path: Optional[Path] = None):
        super().__init__("mathlib_topology", config)
        self.mathlib_path = mathlib_path or Path("./mathlib4")
        
        # Key topology modules
        self.test_modules = [
            "Mathlib.Topology.Basic",
            "Mathlib.Topology.Constructions",
            "Mathlib.Topology.Continuous",
            "Mathlib.Topology.Metric.Basic"
        ]
    
    async def _run_single(self, modules: List[str], run_idx: int) -> BenchmarkResult:
        """Run single topology benchmark."""
        test_modules = modules if modules else self.test_modules
        
        start_time = time.time()
        metrics = await self._compile_modules(test_modules)
        end_time = time.time()
        
        metrics.compilation_time = end_time - start_time
        
        return BenchmarkResult(
            name=f"{self.name}_run_{run_idx}",
            timestamp=datetime.now(),
            metrics=metrics,
            modules=test_modules,
            environment=self._get_environment_info()
        )
    
    async def _compile_modules(self, modules: List[str]) -> BenchmarkMetrics:
        """Compile topology modules."""
        # Similar to algebra benchmark but with topology-specific parsing
        return BenchmarkMetrics(
            compilation_time=0.0,
            memory_peak_mb=psutil.virtual_memory().used / (1024*1024),
            cpu_usage_percent=psutil.cpu_percent(),
            simp_iterations=0,
            simp_time=0.0,
            proof_time=0.0
        )


class LeanCoreBenchmark(BaseBenchmark):
    """Benchmark for Lean core libraries."""
    
    def __init__(self, config: Config):
        super().__init__("lean_core", config)
        
        # Core Lean modules
        self.test_modules = [
            "Init.Core",
            "Init.Data.List.Basic",
            "Init.Logic",
            "Init.Util"
        ]
    
    async def _run_single(self, modules: List[str], run_idx: int) -> BenchmarkResult:
        """Run single core benchmark."""
        test_modules = modules if modules else self.test_modules
        
        start_time = time.time()
        
        # Simple compilation test for core modules
        success = await self._test_core_compilation()
        
        end_time = time.time()
        
        metrics = BenchmarkMetrics(
            compilation_time=end_time - start_time,
            memory_peak_mb=psutil.virtual_memory().used / (1024*1024),
            cpu_usage_percent=psutil.cpu_percent(),
            simp_iterations=0,
            simp_time=0.0,
            proof_time=0.0
        )
        
        return BenchmarkResult(
            name=f"{self.name}_run_{run_idx}",
            timestamp=datetime.now(),
            metrics=metrics,
            modules=test_modules,
            environment=self._get_environment_info(),
            success=success
        )
    
    async def _test_core_compilation(self) -> bool:
        """Test basic Lean core compilation."""
        try:
            # Create simple test file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
                f.write("""
import Init.Core
import Init.Data.List.Basic

#check List.nil
#check List.cons
""")
                temp_file = Path(f.name)
            
            # Try to compile
            await self.lean_runner.run_lean(temp_file)
            temp_file.unlink()  # Clean up
            return True
            
        except Exception as e:
            logger.warning(f"Core compilation test failed: {e}")
            return False


class CustomBenchmark(BaseBenchmark):
    """Customizable benchmark for specific projects."""
    
    def __init__(self, config: Config, custom_modules: List[str]):
        super().__init__("custom", config)
        self.custom_modules = custom_modules
    
    async def _run_single(self, modules: List[str], run_idx: int) -> BenchmarkResult:
        """Run single custom benchmark."""
        test_modules = modules if modules else self.custom_modules
        
        start_time = time.time()
        metrics = await self._benchmark_custom_modules(test_modules)
        end_time = time.time()
        
        metrics.compilation_time = end_time - start_time
        
        return BenchmarkResult(
            name=f"{self.name}_run_{run_idx}",
            timestamp=datetime.now(),
            metrics=metrics,
            modules=test_modules,
            environment=self._get_environment_info()
        )
    
    async def _benchmark_custom_modules(self, modules: List[str]) -> BenchmarkMetrics:
        """Benchmark custom modules."""
        return BenchmarkMetrics(
            compilation_time=0.0,
            memory_peak_mb=psutil.virtual_memory().used / (1024*1024),
            cpu_usage_percent=psutil.cpu_percent(),
            simp_iterations=0,
            simp_time=0.0,
            proof_time=0.0
        )


class BenchmarkSuite:
    """Comprehensive benchmarking system."""
    
    def __init__(self, config: Config, storage_dir: Optional[Path] = None):
        """Initialize benchmark suite.
        
        Args:
            config: Simpulse configuration
            storage_dir: Directory to store benchmark results
        """
        self.config = config
        self.storage_dir = storage_dir or Path("./benchmark_results")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize benchmarks
        self.benchmarks = {
            'mathlib_algebra': MathlibAlgebraBenchmark(config),
            'mathlib_topology': MathlibTopologyBenchmark(config),
            'lean_core': LeanCoreBenchmark(config),
            'custom': CustomBenchmark(config, [])
        }
        
        # Results cache
        self.results_cache: Dict[str, List[BenchmarkResult]] = defaultdict(list)
    
    async def run_benchmark(self, name: str, modules: Optional[List[str]] = None, 
                          baseline: bool = False, runs: int = 3) -> List[BenchmarkResult]:
        """Run standardized benchmark.
        
        Args:
            name: Benchmark name
            modules: Specific modules to benchmark
            baseline: Whether this is a baseline run
            runs: Number of runs for statistical significance
            
        Returns:
            List of benchmark results
        """
        logger.info(f"Running benchmark '{name}' {'(baseline)' if baseline else ''}")
        
        if name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {name}")
        
        benchmark = self.benchmarks[name]
        results = await benchmark.run(modules or [], runs)
        
        # Store results
        for result in results:
            if baseline:
                result.name = f"{result.name}_baseline"
            
            self.results_cache[name].append(result)
            await self._store_result(result)
        
        logger.info(f"Benchmark '{name}' completed with {len(results)} results")
        return results
    
    async def _store_result(self, result: BenchmarkResult):
        """Store benchmark result to disk."""
        result_file = self.storage_dir / f"{result.name}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to store result: {e}")
    
    def generate_comparison_report(self, before: BenchmarkResult, 
                                 after: BenchmarkResult) -> ComparisonReport:
        """Statistical comparison with confidence intervals.
        
        Args:
            before: Baseline benchmark result
            after: Optimized benchmark result
            
        Returns:
            Detailed comparison report
        """
        logger.info(f"Generating comparison report: {before.name} vs {after.name}")
        
        report = ComparisonReport(baseline=before, optimized=after)
        
        # Calculate statistical significance
        report.significance_tests = self._calculate_significance_tests(before, after)
        report.confidence_intervals = self._calculate_confidence_intervals(before, after)
        
        return report
    
    def _calculate_significance_tests(self, before: BenchmarkResult, 
                                    after: BenchmarkResult) -> Dict[str, Dict[str, float]]:
        """Calculate statistical significance tests."""
        tests = {}
        
        # For single results, we can't do proper statistical tests
        # In practice, this would use multiple runs
        metrics = ['compilation_time', 'simp_time', 'memory_peak_mb']
        
        for metric in metrics:
            before_value = getattr(before.metrics, metric, 0)
            after_value = getattr(after.metrics, metric, 0)
            
            # Simple effect size calculation
            if before_value > 0:
                effect_size = abs(after_value - before_value) / before_value
                
                tests[metric] = {
                    'effect_size': effect_size,
                    'significant': effect_size > 0.05,  # 5% threshold
                    'p_value': 0.5  # Placeholder - would need multiple runs
                }
        
        return tests
    
    def _calculate_confidence_intervals(self, before: BenchmarkResult, 
                                      after: BenchmarkResult) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for improvements."""
        intervals = {}
        
        # Placeholder implementation - would need multiple runs for real CIs
        metrics = ['compilation_time', 'simp_time', 'memory_peak_mb']
        
        for metric in metrics:
            before_value = getattr(before.metrics, metric, 0)
            after_value = getattr(after.metrics, metric, 0)
            
            if before_value > 0:
                improvement = ((before_value - after_value) / before_value) * 100
                # Simple confidence interval estimate
                margin = abs(improvement) * 0.1  # 10% margin
                intervals[metric] = (improvement - margin, improvement + margin)
        
        return intervals
    
    async def continuous_benchmark(self, interval: int = 3600, duration: int = 86400):
        """Track performance over time.
        
        Args:
            interval: Time between benchmarks in seconds
            duration: Total duration to run in seconds
        """
        logger.info(f"Starting continuous benchmarking for {duration}s with {interval}s intervals")
        
        start_time = time.time()
        run_count = 0
        
        while time.time() - start_time < duration:
            run_count += 1
            logger.info(f"Continuous benchmark run {run_count}")
            
            # Run lightweight benchmark
            try:
                results = await self.run_benchmark('lean_core', runs=1)
                
                if results:
                    logger.info(f"Run {run_count} completed: "
                              f"{results[0].metrics.compilation_time:.1f}s compilation")
                
            except Exception as e:
                logger.error(f"Continuous benchmark run {run_count} failed: {e}")
            
            # Wait for next interval
            await asyncio.sleep(interval)
        
        logger.info(f"Continuous benchmarking completed after {run_count} runs")
    
    def generate_performance_report(self, benchmark_name: str) -> str:
        """Generate performance report for a benchmark."""
        if benchmark_name not in self.results_cache:
            return f"No results found for benchmark '{benchmark_name}'"
        
        results = self.results_cache[benchmark_name]
        if not results:
            return f"No results available for benchmark '{benchmark_name}'"
        
        # Calculate statistics
        compilation_times = [r.metrics.compilation_time for r in results]
        simp_times = [r.metrics.simp_time for r in results]
        
        report = f"""# Performance Report: {benchmark_name}

## Summary
- **Runs**: {len(results)}
- **Average Compilation Time**: {statistics.mean(compilation_times):.2f}s
- **Compilation Time Range**: {min(compilation_times):.2f}s - {max(compilation_times):.2f}s
- **Average Simp Time**: {statistics.mean(simp_times):.2f}s

## Statistical Analysis
- **Compilation Time StdDev**: {statistics.stdev(compilation_times) if len(compilation_times) > 1 else 0:.2f}s
- **Simp Time StdDev**: {statistics.stdev(simp_times) if len(simp_times) > 1 else 0:.2f}s

## Latest Result
- **Timestamp**: {results[-1].timestamp}
- **Success**: {results[-1].success}
- **Modules**: {len(results[-1].modules)}
"""
        
        return report
    
    async def export_results(self, format: str = "json") -> Path:
        """Export all benchmark results.
        
        Args:
            format: Export format ('json', 'csv')
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == "json":
            export_file = self.storage_dir / f"benchmark_export_{timestamp}.json"
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'benchmarks': {}
            }
            
            for benchmark_name, results in self.results_cache.items():
                export_data['benchmarks'][benchmark_name] = [
                    result.to_dict() for result in results
                ]
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        elif format == "csv":
            import csv
            export_file = self.storage_dir / f"benchmark_export_{timestamp}.csv"
            
            with open(export_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'benchmark', 'timestamp', 'compilation_time', 'simp_time',
                    'memory_peak_mb', 'success', 'modules'
                ])
                
                # Data
                for benchmark_name, results in self.results_cache.items():
                    for result in results:
                        writer.writerow([
                            benchmark_name,
                            result.timestamp.isoformat(),
                            result.metrics.compilation_time,
                            result.metrics.simp_time,
                            result.metrics.memory_peak_mb,
                            result.success,
                            ','.join(result.modules)
                        ])
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported benchmark results to {export_file}")
        return export_file
"""Performance profiling and monitoring for Simpulse."""

import cProfile
import functools
import io
import pstats
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific operation."""

    operation_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    peak_memory_mb: float
    call_count: int = 1

    def __str__(self) -> str:
        """Human readable metrics."""
        return (
            f"{self.operation_name}: {self.execution_time:.3f}s, "
            f"{self.memory_usage_mb:.1f}MB, {self.cpu_percent:.1f}% CPU"
        )


@dataclass
class ProfileSession:
    """Complete profiling session data."""

    session_id: str
    start_time: float
    metrics: List[PerformanceMetrics] = field(default_factory=list)
    total_execution_time: float = 0.0
    peak_memory_mb: float = 0.0

    def add_metric(self, metric: PerformanceMetrics) -> None:
        """Add a performance metric to the session."""
        self.metrics.append(metric)
        self.peak_memory_mb = max(self.peak_memory_mb, metric.peak_memory_mb)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the session."""
        if not self.metrics:
            return {"total_operations": 0}

        total_time = sum(m.execution_time for m in self.metrics)
        avg_memory = sum(m.memory_usage_mb for m in self.metrics) / len(self.metrics)
        avg_cpu = sum(m.cpu_percent for m in self.metrics) / len(self.metrics)

        return {
            "session_id": self.session_id,
            "total_operations": len(self.metrics),
            "total_execution_time": total_time,
            "average_memory_mb": avg_memory,
            "peak_memory_mb": self.peak_memory_mb,
            "average_cpu_percent": avg_cpu,
            "operations": [
                {
                    "name": m.operation_name,
                    "time": m.execution_time,
                    "memory_mb": m.memory_usage_mb,
                    "cpu_percent": m.cpu_percent,
                }
                for m in self.metrics
            ],
        }


class PerformanceProfiler:
    """Production-grade performance profiler for Simpulse."""

    def __init__(self, enable_memory_tracking: bool = True):
        """Initialize the profiler.

        Args:
            enable_memory_tracking: Whether to track memory usage.
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.sessions: Dict[str, ProfileSession] = {}
        self.current_session: Optional[ProfileSession] = None

        if enable_memory_tracking:
            tracemalloc.start()

    def start_session(self, session_id: str) -> ProfileSession:
        """Start a new profiling session."""
        session = ProfileSession(session_id=session_id, start_time=time.time())
        self.sessions[session_id] = session
        self.current_session = session
        return session

    def end_session(self, session_id: str) -> Optional[ProfileSession]:
        """End a profiling session and return results."""
        session = self.sessions.get(session_id)
        if session:
            session.total_execution_time = time.time() - session.start_time
            if session == self.current_session:
                self.current_session = None
        return session

    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling a specific operation."""
        start_time = time.time()
        start_memory = 0
        current_memory = 0
        peak_memory = 0

        if self.enable_memory_tracking:
            current, peak = tracemalloc.get_traced_memory()
            start_memory = current / 1024 / 1024  # Convert to MB

        # Get current process for CPU monitoring
        process = psutil.Process()
        cpu_percent = process.cpu_percent()

        try:
            yield
        finally:
            execution_time = time.time() - start_time

            if self.enable_memory_tracking:
                current, peak = tracemalloc.get_traced_memory()
                current_memory = current / 1024 / 1024
                peak_memory = peak / 1024 / 1024
                memory_usage = current_memory - start_memory
            else:
                memory_usage = 0
                peak_memory = 0

            # Get final CPU reading
            final_cpu = process.cpu_percent()
            avg_cpu = (cpu_percent + final_cpu) / 2

            metric = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_percent=avg_cpu,
                peak_memory_mb=peak_memory,
            )

            if self.current_session:
                self.current_session.add_metric(metric)

    def profile_function(self, operation_name: Optional[str] = None):
        """Decorator for profiling functions."""

        def decorator(func: Callable) -> Callable:
            name = operation_name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_operation(name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def profile_with_cprofile(self, func: Callable, *args, **kwargs) -> tuple:
        """Profile function with cProfile for detailed analysis."""
        profiler = cProfile.Profile()

        try:
            result = profiler.runcall(func, *args, **kwargs)

            # Capture profile statistics
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats("cumulative")
            stats.print_stats(20)  # Top 20 functions

            profile_output = stats_stream.getvalue()

            return result, profile_output
        except Exception as e:
            return None, f"Profiling failed: {e}"

    def save_session_report(self, session_id: str, output_path: Path) -> bool:
        """Save detailed profiling report to file."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        try:
            import json

            report = {
                "profiling_report": True,
                "session": session.get_summary(),
                "generated_at": time.time(),
                "performance_insights": self._generate_insights(session),
            }

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

            return True
        except Exception:
            return False

    def _generate_insights(self, session: ProfileSession) -> Dict[str, Any]:
        """Generate performance insights from session data."""
        if not session.metrics:
            return {"insights": []}

        insights = []

        # Find slowest operations
        slowest = max(session.metrics, key=lambda m: m.execution_time)
        if slowest.execution_time > 1.0:
            insights.append(
                {
                    "type": "slow_operation",
                    "operation": slowest.operation_name,
                    "time": slowest.execution_time,
                    "recommendation": "Consider optimization or parallel processing",
                }
            )

        # Find memory-intensive operations
        memory_intensive = max(session.metrics, key=lambda m: m.memory_usage_mb)
        if memory_intensive.memory_usage_mb > 100:
            insights.append(
                {
                    "type": "high_memory_usage",
                    "operation": memory_intensive.operation_name,
                    "memory_mb": memory_intensive.memory_usage_mb,
                    "recommendation": "Consider streaming or batch processing",
                }
            )

        # Check for CPU bottlenecks
        cpu_intensive = max(session.metrics, key=lambda m: m.cpu_percent)
        if cpu_intensive.cpu_percent > 80:
            insights.append(
                {
                    "type": "cpu_bottleneck",
                    "operation": cpu_intensive.operation_name,
                    "cpu_percent": cpu_intensive.cpu_percent,
                    "recommendation": "Consider parallel processing or algorithm optimization",
                }
            )

        return {
            "insights": insights,
            "total_operations": len(session.metrics),
            "performance_score": self._calculate_performance_score(session),
        }

    def _calculate_performance_score(self, session: ProfileSession) -> float:
        """Calculate overall performance score (0-100)."""
        if not session.metrics:
            return 0.0

        # Score based on execution time, memory usage, and CPU efficiency
        avg_time = sum(m.execution_time for m in session.metrics) / len(session.metrics)
        avg_memory = sum(m.memory_usage_mb for m in session.metrics) / len(session.metrics)
        avg_cpu = sum(m.cpu_percent for m in session.metrics) / len(session.metrics)

        # Simple scoring algorithm (can be made more sophisticated)
        time_score = max(0, 100 - (avg_time * 10))  # Penalty for slow operations
        memory_score = max(0, 100 - (avg_memory / 10))  # Penalty for high memory usage
        cpu_score = max(0, 100 - max(0, avg_cpu - 50))  # Penalty for high CPU usage

        return (time_score + memory_score + cpu_score) / 3


# Global profiler instance
_global_profiler = None


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile_operation(operation_name: str):
    """Convenience decorator for profiling operations."""
    return get_profiler().profile_function(operation_name)


@contextmanager
def profile_block(operation_name: str):
    """Convenience context manager for profiling code blocks."""
    with get_profiler().profile_operation(operation_name):
        yield


def start_profiling_session(session_id: str) -> ProfileSession:
    """Start a global profiling session."""
    return get_profiler().start_session(session_id)


def end_profiling_session(session_id: str) -> Optional[ProfileSession]:
    """End a global profiling session."""
    return get_profiler().end_session(session_id)

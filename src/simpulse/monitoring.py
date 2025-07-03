"""
Performance monitoring and effectiveness tracking for Simpulse optimizations.

Tracks optimization effectiveness over time, providing insights into which strategies work best.
"""

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .errors import ErrorCategory, ErrorContext, ErrorHandler, ErrorSeverity


@dataclass
class OptimizationMetrics:
    """Metrics for a single optimization run."""

    timestamp: float
    strategy: str
    project_path: str
    rules_analyzed: int
    optimizations_applied: int

    # Performance metrics
    baseline_time: Optional[float] = None
    optimized_time: Optional[float] = None
    improvement_percent: Optional[float] = None

    # Quality metrics
    compilation_success: bool = True
    test_success: bool = True

    # Effectiveness metrics
    actual_speedup: Optional[float] = None
    estimated_speedup: Optional[float] = None
    accuracy_score: Optional[float] = None  # How close estimate was to actual

    # Additional data
    errors_encountered: int = 0
    warnings_encountered: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Calculate accuracy score if we have both estimates
        if self.actual_speedup is not None and self.estimated_speedup is not None:
            if self.estimated_speedup > 0:
                error_ratio = (
                    abs(self.actual_speedup - self.estimated_speedup) / self.estimated_speedup
                )
                self.accuracy_score = max(0.0, 1.0 - error_ratio)
            else:
                self.accuracy_score = 0.0


@dataclass
class StrategyEffectiveness:
    """Effectiveness metrics for an optimization strategy."""

    strategy_name: str
    total_runs: int = 0
    successful_runs: int = 0

    # Performance statistics
    avg_improvement: float = 0.0
    max_improvement: float = 0.0
    min_improvement: float = 0.0
    std_improvement: float = 0.0

    # Accuracy statistics
    avg_accuracy: float = 0.0
    prediction_reliability: float = 0.0  # How often predictions are within 20% of actual

    # Quality statistics
    compilation_success_rate: float = 0.0
    test_success_rate: float = 0.0

    # Efficiency metrics
    avg_rules_processed: float = 0.0
    avg_optimizations_applied: float = 0.0
    optimization_rate: float = 0.0  # optimizations / rules analyzed

    last_updated: float = field(default_factory=time.time)


class PerformanceMonitor:
    """Monitor and track optimization performance over time."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.cwd() / ".simpulse" / "metrics.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.metrics_history: list[OptimizationMetrics] = []
        self.strategy_stats: dict[str, StrategyEffectiveness] = {}

        self.error_handler = ErrorHandler()

        # Load existing data
        self._load_data()

    def record_optimization(self, metrics: OptimizationMetrics):
        """Record metrics from an optimization run."""
        try:
            self.metrics_history.append(metrics)
            self._update_strategy_stats(metrics)
            self._save_data()

        except Exception as e:
            self.error_handler.handle_error(
                category=ErrorCategory.PERFORMANCE,
                severity=ErrorSeverity.MEDIUM,
                message="Failed to record optimization metrics",
                context=ErrorContext(operation="record_optimization"),
                exception=e,
            )

    def _update_strategy_stats(self, metrics: OptimizationMetrics):
        """Update strategy effectiveness statistics."""
        strategy = metrics.strategy

        if strategy not in self.strategy_stats:
            self.strategy_stats[strategy] = StrategyEffectiveness(strategy_name=strategy)

        stats = self.strategy_stats[strategy]
        stats.total_runs += 1
        stats.last_updated = time.time()

        # Update success metrics
        if metrics.compilation_success and metrics.test_success:
            stats.successful_runs += 1

        # Update performance statistics if we have improvement data
        if metrics.improvement_percent is not None:
            improvements = [
                m.improvement_percent
                for m in self.metrics_history
                if m.strategy == strategy and m.improvement_percent is not None
            ]

            if improvements:
                stats.avg_improvement = statistics.mean(improvements)
                stats.max_improvement = max(improvements)
                stats.min_improvement = min(improvements)
                if len(improvements) > 1:
                    stats.std_improvement = statistics.stdev(improvements)

        # Update accuracy statistics
        accuracies = [
            m.accuracy_score
            for m in self.metrics_history
            if m.strategy == strategy and m.accuracy_score is not None
        ]

        if accuracies:
            stats.avg_accuracy = statistics.mean(accuracies)
            # Prediction reliability: percentage within 20% of actual
            reliable_predictions = sum(1 for acc in accuracies if acc >= 0.8)
            stats.prediction_reliability = reliable_predictions / len(accuracies)

        # Update quality statistics
        strategy_metrics = [m for m in self.metrics_history if m.strategy == strategy]
        if strategy_metrics:
            stats.compilation_success_rate = sum(
                m.compilation_success for m in strategy_metrics
            ) / len(strategy_metrics)
            stats.test_success_rate = sum(m.test_success for m in strategy_metrics) / len(
                strategy_metrics
            )

        # Update efficiency metrics
        rules_processed = [m.rules_analyzed for m in strategy_metrics if m.rules_analyzed > 0]
        optimizations_applied = [m.optimizations_applied for m in strategy_metrics]

        if rules_processed:
            stats.avg_rules_processed = statistics.mean(rules_processed)
        if optimizations_applied:
            stats.avg_optimizations_applied = statistics.mean(optimizations_applied)
        if rules_processed and optimizations_applied:
            total_rules = sum(rules_processed)
            total_optimizations = sum(optimizations_applied)
            stats.optimization_rate = total_optimizations / total_rules if total_rules > 0 else 0.0

    def get_strategy_ranking(self) -> list[tuple[str, StrategyEffectiveness]]:
        """Get strategies ranked by overall effectiveness."""

        def effectiveness_score(stats: StrategyEffectiveness) -> float:
            """Calculate overall effectiveness score (0-1)."""
            if stats.total_runs == 0:
                return 0.0

            # Weight different factors
            improvement_score = min(stats.avg_improvement / 50, 1.0)  # Normalize to 50% max
            reliability_score = stats.prediction_reliability
            quality_score = (stats.compilation_success_rate + stats.test_success_rate) / 2
            efficiency_score = min(stats.optimization_rate * 10, 1.0)  # Normalize to 10% rate

            # Weighted average
            return (
                0.4 * improvement_score
                + 0.2 * reliability_score
                + 0.2 * quality_score
                + 0.2 * efficiency_score
            )

        ranked = [(name, stats) for name, stats in self.strategy_stats.items()]
        ranked.sort(key=lambda x: effectiveness_score(x[1]), reverse=True)

        return ranked

    def get_performance_summary(self, days: int = 30) -> dict[str, Any]:
        """Get performance summary for the last N days."""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {"error": "No recent data available"}

        # Overall statistics
        improvements = [
            m.improvement_percent for m in recent_metrics if m.improvement_percent is not None
        ]

        summary = {
            "period_days": days,
            "total_optimizations": len(recent_metrics),
            "successful_optimizations": sum(
                1 for m in recent_metrics if m.compilation_success and m.test_success
            ),
            "success_rate": sum(
                1 for m in recent_metrics if m.compilation_success and m.test_success
            )
            / len(recent_metrics),
        }

        if improvements:
            summary.update(
                {
                    "avg_improvement": statistics.mean(improvements),
                    "max_improvement": max(improvements),
                    "min_improvement": min(improvements),
                    "total_projects_optimized": len({m.project_path for m in recent_metrics}),
                }
            )

        # Strategy performance
        strategy_performance = {}
        for strategy_name, stats in self.strategy_stats.items():
            recent_strategy_metrics = [m for m in recent_metrics if m.strategy == strategy_name]
            if recent_strategy_metrics:
                strategy_improvements = [
                    m.improvement_percent
                    for m in recent_strategy_metrics
                    if m.improvement_percent is not None
                ]

                strategy_performance[strategy_name] = {
                    "runs": len(recent_strategy_metrics),
                    "avg_improvement": (
                        statistics.mean(strategy_improvements) if strategy_improvements else 0
                    ),
                    "success_rate": sum(
                        1
                        for m in recent_strategy_metrics
                        if m.compilation_success and m.test_success
                    )
                    / len(recent_strategy_metrics),
                }

        summary["by_strategy"] = strategy_performance

        # Trends
        if len(recent_metrics) >= 5:
            # Split into first and second half to detect trends
            mid_point = len(recent_metrics) // 2
            first_half = recent_metrics[:mid_point]
            second_half = recent_metrics[mid_point:]

            first_half_improvements = [
                m.improvement_percent for m in first_half if m.improvement_percent is not None
            ]
            second_half_improvements = [
                m.improvement_percent for m in second_half if m.improvement_percent is not None
            ]

            if first_half_improvements and second_half_improvements:
                trend = statistics.mean(second_half_improvements) - statistics.mean(
                    first_half_improvements
                )
                summary["improvement_trend"] = trend

        return summary

    def get_recommendations(self) -> list[str]:
        """Get recommendations based on monitoring data."""
        recommendations = []

        if not self.strategy_stats:
            recommendations.append("Run more optimizations to gather performance data")
            return recommendations

        # Get strategy ranking
        ranked_strategies = self.get_strategy_ranking()

        if ranked_strategies:
            best_strategy = ranked_strategies[0]
            worst_strategy = ranked_strategies[-1]

            recommendations.append(
                f"Best performing strategy: '{best_strategy[0]}' "
                f"(avg improvement: {best_strategy[1].avg_improvement:.1f}%)"
            )

            if len(ranked_strategies) > 1:
                recommendations.append(
                    f"Consider avoiding '{worst_strategy[0]}' strategy "
                    f"(avg improvement: {worst_strategy[1].avg_improvement:.1f}%)"
                )

        # Check for strategies with low accuracy
        for name, stats in self.strategy_stats.items():
            if stats.avg_accuracy < 0.5 and stats.total_runs >= 3:
                recommendations.append(
                    f"Strategy '{name}' has low prediction accuracy ({stats.avg_accuracy:.1%}) - "
                    "consider tuning estimation algorithms"
                )

        # Check for strategies with low success rates
        for name, stats in self.strategy_stats.items():
            if stats.compilation_success_rate < 0.8 and stats.total_runs >= 3:
                recommendations.append(
                    f"Strategy '{name}' has low compilation success rate ({stats.compilation_success_rate:.1%}) - "
                    "consider more conservative optimization parameters"
                )

        # General recommendations based on data patterns
        total_runs = sum(stats.total_runs for stats in self.strategy_stats.values())
        if total_runs < 10:
            recommendations.append("Run more optimizations to improve recommendation accuracy")

        return recommendations

    def _save_data(self):
        """Save monitoring data to disk."""
        try:
            data = {
                "metrics_history": [
                    {
                        "timestamp": m.timestamp,
                        "strategy": m.strategy,
                        "project_path": m.project_path,
                        "rules_analyzed": m.rules_analyzed,
                        "optimizations_applied": m.optimizations_applied,
                        "baseline_time": m.baseline_time,
                        "optimized_time": m.optimized_time,
                        "improvement_percent": m.improvement_percent,
                        "compilation_success": m.compilation_success,
                        "test_success": m.test_success,
                        "actual_speedup": m.actual_speedup,
                        "estimated_speedup": m.estimated_speedup,
                        "accuracy_score": m.accuracy_score,
                        "errors_encountered": m.errors_encountered,
                        "warnings_encountered": m.warnings_encountered,
                        "metadata": m.metadata,
                    }
                    for m in self.metrics_history
                ],
                "strategy_stats": {
                    name: {
                        "strategy_name": stats.strategy_name,
                        "total_runs": stats.total_runs,
                        "successful_runs": stats.successful_runs,
                        "avg_improvement": stats.avg_improvement,
                        "max_improvement": stats.max_improvement,
                        "min_improvement": stats.min_improvement,
                        "std_improvement": stats.std_improvement,
                        "avg_accuracy": stats.avg_accuracy,
                        "prediction_reliability": stats.prediction_reliability,
                        "compilation_success_rate": stats.compilation_success_rate,
                        "test_success_rate": stats.test_success_rate,
                        "avg_rules_processed": stats.avg_rules_processed,
                        "avg_optimizations_applied": stats.avg_optimizations_applied,
                        "optimization_rate": stats.optimization_rate,
                        "last_updated": stats.last_updated,
                    }
                    for name, stats in self.strategy_stats.items()
                },
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.error_handler.handle_error(
                category=ErrorCategory.FILE_ACCESS,
                severity=ErrorSeverity.LOW,
                message="Failed to save monitoring data",
                context=ErrorContext(operation="save_monitoring_data", file_path=self.storage_path),
                exception=e,
            )

    def _load_data(self):
        """Load monitoring data from disk."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            # Load metrics history
            for m_data in data.get("metrics_history", []):
                metrics = OptimizationMetrics(**m_data)
                self.metrics_history.append(metrics)

            # Load strategy stats
            for name, s_data in data.get("strategy_stats", {}).items():
                stats = StrategyEffectiveness(**s_data)
                self.strategy_stats[name] = stats

        except Exception as e:
            self.error_handler.handle_error(
                category=ErrorCategory.FILE_ACCESS,
                severity=ErrorSeverity.MEDIUM,
                message="Failed to load monitoring data",
                context=ErrorContext(operation="load_monitoring_data", file_path=self.storage_path),
                exception=e,
            )

    def export_data(self, output_path: Path, format: str = "json") -> bool:
        """Export monitoring data in various formats."""
        try:
            if format == "json":
                summary = self.get_performance_summary(days=365)  # Full year
                summary["strategy_ranking"] = [
                    {"strategy": name, "effectiveness_data": stats.__dict__}
                    for name, stats in self.get_strategy_ranking()
                ]

                with open(output_path, "w") as f:
                    json.dump(summary, f, indent=2)

            elif format == "csv":
                import csv

                with open(output_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "timestamp",
                            "strategy",
                            "project_path",
                            "rules_analyzed",
                            "optimizations_applied",
                            "improvement_percent",
                            "compilation_success",
                            "test_success",
                            "accuracy_score",
                        ]
                    )

                    for m in self.metrics_history:
                        writer.writerow(
                            [
                                m.timestamp,
                                m.strategy,
                                m.project_path,
                                m.rules_analyzed,
                                m.optimizations_applied,
                                m.improvement_percent,
                                m.compilation_success,
                                m.test_success,
                                m.accuracy_score,
                            ]
                        )

            return True

        except Exception as e:
            self.error_handler.handle_error(
                category=ErrorCategory.FILE_ACCESS,
                severity=ErrorSeverity.MEDIUM,
                message=f"Failed to export data in {format} format",
                context=ErrorContext(operation="export_data", file_path=output_path),
                exception=e,
            )
            return False

    def clear_old_data(self, days: int = 90):
        """Clear monitoring data older than specified days."""
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        original_count = len(self.metrics_history)
        self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        removed_count = original_count - len(self.metrics_history)
        if removed_count > 0:
            # Recalculate strategy stats with remaining data
            self.strategy_stats.clear()
            for metrics in self.metrics_history:
                self._update_strategy_stats(metrics)

            self._save_data()

        return removed_count


def monitor_operation(operation_name: str):
    """Decorator to monitor operation performance.

    Simple decorator that tracks operation execution time.
    For more comprehensive monitoring, use PerformanceMonitor class.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                # Simple logging - could be enhanced to record to PerformanceMonitor
                if elapsed > 1.0:  # Log slow operations
                    print(f"Operation '{operation_name}' took {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"Operation '{operation_name}' failed after {elapsed:.2f}s: {e}")
                raise

        return wrapper

    return decorator

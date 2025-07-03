"""
Comprehensive monitoring and alerting system for Simpulse.

Provides real-time monitoring of system health, performance metrics,
error patterns, and automated alerting for production environments.
"""

import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from ..errors import ErrorHandler


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics we track."""

    COUNTER = "counter"  # Always increasing
    GAUGE = "gauge"  # Current value
    HISTOGRAM = "histogram"  # Distribution of values
    RATE = "rate"  # Rate per time unit


@dataclass
class Metric:
    """Represents a single metric measurement."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class Alert:
    """Represents an alert condition."""

    level: AlertLevel
    message: str
    source: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None


@dataclass
class HealthCheck:
    """Represents a health check."""

    name: str
    check_func: Callable[[], bool]
    interval: float
    timeout: float
    last_run: float = 0.0
    last_result: Optional[bool] = None
    consecutive_failures: int = 0
    enabled: bool = True


class ComprehensiveMonitor:
    """Comprehensive monitoring system for Simpulse operations."""

    def __init__(
        self,
        error_handler: ErrorHandler,
        logger: Optional[logging.Logger] = None,
        db_path: Optional[Path] = None,
        retention_days: int = 30,
    ):
        self.error_handler = error_handler
        self.logger = logger or logging.getLogger(__name__)
        self.db_path = db_path or Path("simpulse_monitoring.db")
        self.retention_days = retention_days

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)

        # Alerts
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.alert_history: deque = deque(maxlen=1000)

        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}

        # Thresholds for automatic alerts
        self.alert_thresholds = {
            "error_rate": {"warning": 0.05, "critical": 0.1},  # 5% warning, 10% critical
            "memory_usage": {"warning": 85.0, "critical": 95.0},
            "disk_usage": {"warning": 90.0, "critical": 95.0},
            "response_time": {"warning": 5.0, "critical": 10.0},  # seconds
            "consecutive_failures": {"warning": 3, "critical": 5},
        }

        # Performance tracking
        self.operation_metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "total_time": 0.0,
                "errors": 0,
                "last_execution": 0.0,
                "avg_time": 0.0,
                "error_rate": 0.0,
            }
        )

        # System metrics
        self.system_metrics_interval = 10.0  # seconds
        self.last_system_check = 0.0

        # Initialize database
        self._init_database()

        # Setup default health checks
        self._setup_default_health_checks()

    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()

                # Metrics table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        value REAL NOT NULL,
                        metric_type TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        tags TEXT,
                        unit TEXT,
                        UNIQUE(name, timestamp, tags)
                    )
                """
                )

                # Alerts table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        source TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        details TEXT,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolved_at REAL
                    )
                """
                )

                # Performance table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation TEXT NOT NULL,
                        duration REAL NOT NULL,
                        success BOOLEAN NOT NULL,
                        timestamp REAL NOT NULL,
                        details TEXT
                    )
                """
                )

                # Create indexes
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON metrics(name, timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_alerts_level_time ON alerts(level, timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_performance_op_time ON performance(operation, timestamp)"
                )

                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring database: {e}")

    def _setup_default_health_checks(self):
        """Setup default health checks."""

        if HAS_PSUTIL:
            self.register_health_check(
                "memory_check",
                lambda: psutil.virtual_memory().percent < 95.0,
                interval=30.0,
                timeout=5.0,
            )

            self.register_health_check(
                "disk_check",
                lambda: psutil.disk_usage(".").percent < 95.0,
                interval=60.0,
                timeout=5.0,
            )

        self.register_health_check(
            "error_handler_check",
            lambda: not self.error_handler.has_fatal_errors(),
            interval=15.0,
            timeout=2.0,
        )

    def start_monitoring(self):
        """Start the monitoring system."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Started comprehensive monitoring")

    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("Stopped comprehensive monitoring")

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None,
    ):
        """Record a metric measurement."""
        timestamp = time.time()
        tags = tags or {}

        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=timestamp,
            tags=tags,
            unit=unit,
        )

        # Store in memory
        self.metrics[name].append(metric)

        # Update aggregated metrics
        if metric_type == MetricType.COUNTER:
            self.counters[name] += value
        elif metric_type == MetricType.GAUGE:
            self.gauges[name] = value
        elif metric_type == MetricType.HISTOGRAM:
            self.histograms[name].append(value)
            # Keep only recent values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]

        # Store in database
        self._store_metric_to_db(metric)

        # Check for alert conditions
        self._check_metric_alerts(name, value, metric_type)

    def record_operation(
        self,
        operation_name: str,
        duration: float,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Record operation performance metrics."""
        timestamp = time.time()

        # Update operation metrics
        metrics = self.operation_metrics[operation_name]
        metrics["count"] += 1
        metrics["total_time"] += duration
        metrics["last_execution"] = timestamp
        metrics["avg_time"] = metrics["total_time"] / metrics["count"]

        if not success:
            metrics["errors"] += 1

        metrics["error_rate"] = metrics["errors"] / metrics["count"]

        # Record as metrics
        self.record_metric(
            f"operation.{operation_name}.duration", duration, MetricType.HISTOGRAM, unit="seconds"
        )
        self.record_metric(f"operation.{operation_name}.count", 1, MetricType.COUNTER)

        if not success:
            self.record_metric(f"operation.{operation_name}.errors", 1, MetricType.COUNTER)

        # Store in database
        self._store_performance_to_db(operation_name, duration, success, timestamp, details)

        # Check for performance alerts
        self._check_performance_alerts(operation_name, metrics)

    def register_health_check(
        self,
        name: str,
        check_func: Callable[[], bool],
        interval: float = 60.0,
        timeout: float = 5.0,
    ):
        """Register a health check."""
        health_check = HealthCheck(
            name=name, check_func=check_func, interval=interval, timeout=timeout
        )

        self.health_checks[name] = health_check
        self.logger.info(f"Registered health check: {name}")

    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """Register callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def create_alert(
        self, level: AlertLevel, message: str, source: str, details: Optional[Dict[str, Any]] = None
    ):
        """Create and process an alert."""
        alert_id = f"{source}_{hash(message)}"

        # Check if this alert is already active
        if alert_id in self.active_alerts and not self.active_alerts[alert_id].resolved:
            return  # Don't duplicate active alerts

        alert = Alert(
            level=level,
            message=message,
            source=source,
            timestamp=time.time(),
            details=details or {},
        )

        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Store in database
        self._store_alert_to_db(alert)

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

        # Log alert
        log_func = {
            AlertLevel.INFO: self.logger.info,
            AlertLevel.WARNING: self.logger.warning,
            AlertLevel.ERROR: self.logger.error,
            AlertLevel.CRITICAL: self.logger.critical,
        }.get(level, self.logger.info)

        log_func(f"ALERT [{level.value.upper()}] {source}: {message}")

    def resolve_alert(self, source: str, message: str):
        """Resolve an active alert."""
        alert_id = f"{source}_{hash(message)}"

        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            if not alert.resolved:
                alert.resolved = True
                alert.resolved_at = time.time()

                # Update in database
                self._update_alert_resolution(alert)

                self.logger.info(f"Resolved alert: {source} - {message}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health_status = {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "health_checks": {},
            "active_alerts": len(self.active_alerts),
            "critical_alerts": len(
                [a for a in self.active_alerts.values() if a.level == AlertLevel.CRITICAL]
            ),
            "system_metrics": {},
            "performance_summary": {},
        }

        # Health check status
        failing_checks = 0
        for name, check in self.health_checks.items():
            if check.enabled:
                status = {
                    "status": "passing" if check.last_result else "failing",
                    "last_run": check.last_run,
                    "consecutive_failures": check.consecutive_failures,
                }
                health_status["health_checks"][name] = status

                if not check.last_result:
                    failing_checks += 1

        # Overall status determination
        if health_status["critical_alerts"] > 0 or failing_checks > 0:
            health_status["overall_status"] = "critical"
        elif len(self.active_alerts) > 0:
            health_status["overall_status"] = "warning"

        # System metrics
        if HAS_PSUTIL:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(".")

            health_status["system_metrics"] = {
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / 1024 / 1024 / 1024,
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / 1024 / 1024 / 1024,
                "cpu_percent": psutil.cpu_percent(),
                "load_average": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None,
            }

        # Performance summary
        for op_name, metrics in self.operation_metrics.items():
            if metrics["count"] > 0:
                health_status["performance_summary"][op_name] = {
                    "avg_duration": metrics["avg_time"],
                    "error_rate": metrics["error_rate"],
                    "total_count": metrics["count"],
                }

        return health_status

    def get_metrics_summary(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get summary of metrics within time window."""
        cutoff_time = time.time() - (time_window or 3600)  # Default 1 hour

        summary = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {},
            "recent_metrics": {},
        }

        # Histogram summaries
        for name, values in self.histograms.items():
            if values:
                summary["histograms"][name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "recent_avg": sum(values[-10:]) / min(len(values), 10),
                }

        # Recent metrics
        for name, metric_deque in self.metrics.items():
            recent_metrics = [m for m in metric_deque if m.timestamp > cutoff_time]
            if recent_metrics:
                summary["recent_metrics"][name] = {
                    "count": len(recent_metrics),
                    "latest_value": recent_metrics[-1].value,
                    "avg_value": sum(m.value for m in recent_metrics) / len(recent_metrics),
                }

        return summary

    def export_monitoring_data(
        self, output_path: Path, time_window: Optional[float] = None
    ) -> bool:
        """Export monitoring data to JSON file."""
        try:
            data = {
                "timestamp": time.time(),
                "time_window": time_window,
                "system_health": self.get_system_health(),
                "metrics_summary": self.get_metrics_summary(time_window),
                "active_alerts": [
                    {
                        "level": alert.level.value,
                        "message": alert.message,
                        "source": alert.source,
                        "timestamp": alert.timestamp,
                        "details": alert.details,
                    }
                    for alert in self.active_alerts.values()
                ],
                "operation_metrics": dict(self.operation_metrics),
                "alert_thresholds": self.alert_thresholds,
            }

            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            self.logger.info(f"Exported monitoring data to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export monitoring data: {e}")
            return False

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                current_time = time.time()

                # Run health checks
                self._run_health_checks(current_time)

                # Collect system metrics
                if current_time - self.last_system_check > self.system_metrics_interval:
                    self._collect_system_metrics()
                    self.last_system_check = current_time

                # Clean up old data
                if current_time % 3600 < 10:  # Once per hour
                    self._cleanup_old_data()

                time.sleep(5.0)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)

    def _run_health_checks(self, current_time: float):
        """Run due health checks."""
        for check in self.health_checks.values():
            if not check.enabled:
                continue

            if current_time - check.last_run >= check.interval:
                try:
                    # Run health check with timeout
                    result = check.check_func()
                    check.last_result = result
                    check.last_run = current_time

                    if result:
                        # Health check passed
                        if check.consecutive_failures > 0:
                            self.resolve_alert(
                                "health_check", f"Health check '{check.name}' recovered"
                            )
                        check.consecutive_failures = 0
                    else:
                        # Health check failed
                        check.consecutive_failures += 1

                        # Create alert based on failure count
                        if (
                            check.consecutive_failures
                            >= self.alert_thresholds["consecutive_failures"]["critical"]
                        ):
                            self.create_alert(
                                AlertLevel.CRITICAL,
                                f"Health check '{check.name}' failed {check.consecutive_failures} times",
                                "health_check",
                                {
                                    "check_name": check.name,
                                    "consecutive_failures": check.consecutive_failures,
                                },
                            )
                        elif (
                            check.consecutive_failures
                            >= self.alert_thresholds["consecutive_failures"]["warning"]
                        ):
                            self.create_alert(
                                AlertLevel.WARNING,
                                f"Health check '{check.name}' failed {check.consecutive_failures} times",
                                "health_check",
                                {
                                    "check_name": check.name,
                                    "consecutive_failures": check.consecutive_failures,
                                },
                            )

                    # Record metric
                    self.record_metric(
                        f"health_check.{check.name}", 1.0 if result else 0.0, MetricType.GAUGE
                    )

                except Exception as e:
                    self.logger.error(f"Health check '{check.name}' failed with exception: {e}")
                    check.last_result = False
                    check.consecutive_failures += 1

    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        if not HAS_PSUTIL:
            return

        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric(
                "system.memory.percent", memory.percent, MetricType.GAUGE, unit="percent"
            )
            self.record_metric(
                "system.memory.available_gb",
                memory.available / 1024 / 1024 / 1024,
                MetricType.GAUGE,
                unit="GB",
            )

            # Disk metrics
            disk = psutil.disk_usage(".")
            self.record_metric(
                "system.disk.percent", disk.percent, MetricType.GAUGE, unit="percent"
            )
            self.record_metric(
                "system.disk.free_gb", disk.free / 1024 / 1024 / 1024, MetricType.GAUGE, unit="GB"
            )

            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            self.record_metric("system.cpu.percent", cpu_percent, MetricType.GAUGE, unit="percent")

            # Process metrics
            process = psutil.Process()
            process_info = process.memory_info()
            self.record_metric(
                "process.memory.rss_mb", process_info.rss / 1024 / 1024, MetricType.GAUGE, unit="MB"
            )
            self.record_metric(
                "process.memory.vms_mb", process_info.vms / 1024 / 1024, MetricType.GAUGE, unit="MB"
            )

        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")

    def _check_metric_alerts(self, name: str, value: float, metric_type: MetricType):
        """Check if metric value triggers any alerts."""
        # Check memory usage
        if name == "system.memory.percent":
            if value >= self.alert_thresholds["memory_usage"]["critical"]:
                self.create_alert(
                    AlertLevel.CRITICAL, f"Critical memory usage: {value:.1f}%", "system_metrics"
                )
            elif value >= self.alert_thresholds["memory_usage"]["warning"]:
                self.create_alert(
                    AlertLevel.WARNING, f"High memory usage: {value:.1f}%", "system_metrics"
                )

        # Check disk usage
        elif name == "system.disk.percent":
            if value >= self.alert_thresholds["disk_usage"]["critical"]:
                self.create_alert(
                    AlertLevel.CRITICAL, f"Critical disk usage: {value:.1f}%", "system_metrics"
                )
            elif value >= self.alert_thresholds["disk_usage"]["warning"]:
                self.create_alert(
                    AlertLevel.WARNING, f"High disk usage: {value:.1f}%", "system_metrics"
                )

    def _check_performance_alerts(self, operation_name: str, metrics: Dict[str, Any]):
        """Check operation performance for alert conditions."""
        # Check error rate
        if metrics["error_rate"] >= self.alert_thresholds["error_rate"]["critical"]:
            self.create_alert(
                AlertLevel.CRITICAL,
                f"Critical error rate for {operation_name}: {metrics['error_rate']:.1%}",
                "performance",
                {"operation": operation_name, "error_rate": metrics["error_rate"]},
            )
        elif metrics["error_rate"] >= self.alert_thresholds["error_rate"]["warning"]:
            self.create_alert(
                AlertLevel.WARNING,
                f"High error rate for {operation_name}: {metrics['error_rate']:.1%}",
                "performance",
                {"operation": operation_name, "error_rate": metrics["error_rate"]},
            )

        # Check response time
        if metrics["avg_time"] >= self.alert_thresholds["response_time"]["critical"]:
            self.create_alert(
                AlertLevel.CRITICAL,
                f"Critical response time for {operation_name}: {metrics['avg_time']:.2f}s",
                "performance",
                {"operation": operation_name, "avg_time": metrics["avg_time"]},
            )
        elif metrics["avg_time"] >= self.alert_thresholds["response_time"]["warning"]:
            self.create_alert(
                AlertLevel.WARNING,
                f"High response time for {operation_name}: {metrics['avg_time']:.2f}s",
                "performance",
                {"operation": operation_name, "avg_time": metrics["avg_time"]},
            )

    def _store_metric_to_db(self, metric: Metric):
        """Store metric to database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR IGNORE INTO metrics (name, value, metric_type, timestamp, tags, unit) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        metric.name,
                        metric.value,
                        metric.metric_type.value,
                        metric.timestamp,
                        json.dumps(metric.tags),
                        metric.unit,
                    ),
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store metric to database: {e}")

    def _store_alert_to_db(self, alert: Alert):
        """Store alert to database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO alerts (level, message, source, timestamp, details) VALUES (?, ?, ?, ?, ?)",
                    (
                        alert.level.value,
                        alert.message,
                        alert.source,
                        alert.timestamp,
                        json.dumps(alert.details),
                    ),
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store alert to database: {e}")

    def _store_performance_to_db(
        self,
        operation: str,
        duration: float,
        success: bool,
        timestamp: float,
        details: Optional[Dict[str, Any]],
    ):
        """Store performance data to database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO performance (operation, duration, success, timestamp, details) VALUES (?, ?, ?, ?, ?)",
                    (
                        operation,
                        duration,
                        success,
                        timestamp,
                        json.dumps(details) if details else None,
                    ),
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store performance data to database: {e}")

    def _update_alert_resolution(self, alert: Alert):
        """Update alert resolution in database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE alerts SET resolved = ?, resolved_at = ? WHERE message = ? AND source = ? AND timestamp = ?",
                    (True, alert.resolved_at, alert.message, alert.source, alert.timestamp),
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to update alert resolution: {e}")

    def _cleanup_old_data(self):
        """Clean up old data beyond retention period."""
        cutoff_time = time.time() - (self.retention_days * 24 * 3600)

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()

                # Clean up old metrics
                cursor.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_time,))

                # Clean up old resolved alerts
                cursor.execute(
                    "DELETE FROM alerts WHERE resolved = 1 AND resolved_at < ?", (cutoff_time,)
                )

                # Clean up old performance data
                cursor.execute("DELETE FROM performance WHERE timestamp < ?", (cutoff_time,))

                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")


# Convenience functions
def create_default_monitor(
    error_handler: ErrorHandler, db_path: Optional[Path] = None
) -> ComprehensiveMonitor:
    """Create monitor with sensible defaults."""
    monitor = ComprehensiveMonitor(error_handler, db_path=db_path)

    # Register default alert callback
    def log_alert(alert: Alert):
        logger = logging.getLogger("simpulse.alerts")
        log_func = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }.get(alert.level, logger.info)

        log_func(f"{alert.source}: {alert.message}")

    monitor.register_alert_callback(log_alert)

    return monitor

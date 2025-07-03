"""
Production-ready logging and reporting system for Simpulse.

Provides structured logging, centralized error reporting, performance metrics,
audit trails, and integration with external monitoring systems.
"""

import gzip
import json
import logging
import logging.handlers
import os
import platform
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    pass

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from ..errors import ErrorSeverity, SimpulseError


class LogLevel(Enum):
    """Extended log levels for production use."""

    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60  # Security-related events
    AUDIT = 70  # Audit trail events


class EventType(Enum):
    """Types of events we track."""

    OPERATION_START = "operation_start"
    OPERATION_SUCCESS = "operation_success"
    OPERATION_FAILURE = "operation_failure"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"
    SECURITY_EVENT = "security_event"
    AUDIT_TRAIL = "audit_trail"
    SYSTEM_STATUS = "system_status"
    USER_ACTION = "user_action"
    CONFIGURATION_CHANGE = "configuration_change"


@dataclass
class LogContext:
    """Context information for log entries."""

    session_id: str
    request_id: str
    user_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    version: Optional[str] = None
    environment: Optional[str] = None
    correlation_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    span_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class StructuredLogEntry:
    """Structured log entry with comprehensive metadata."""

    timestamp: str
    level: str
    message: str
    event_type: EventType
    context: LogContext
    details: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    hostname: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["context"] = self.context.to_dict()
        data["event_type"] = self.event_type.value
        return data


class LogFormatter:
    """Custom formatter for production logs."""

    def __init__(self, format_type: str = "json"):
        self.format_type = format_type

    def format(self, entry: StructuredLogEntry) -> str:
        """Format log entry for output."""
        if self.format_type == "json":
            return json.dumps(entry.to_dict(), default=str, separators=(",", ":"))

        elif self.format_type == "human":
            context_str = f"[{entry.context.session_id[:8]}]"
            if entry.context.operation:
                context_str += f"[{entry.context.operation}]"

            message = f"{entry.timestamp} {entry.level:<8} {context_str} {entry.message}"

            if entry.performance_metrics:
                metrics_str = ", ".join(f"{k}={v}" for k, v in entry.performance_metrics.items())
                message += f" | Metrics: {metrics_str}"

            if entry.error_info:
                message += f" | Error: {entry.error_info.get('type', 'Unknown')}"

            return message

        else:
            return str(entry.message)


class ProductionLogger:
    """Production-ready logger with structured logging and external integrations."""

    def __init__(
        self,
        name: str = "simpulse",
        log_level: LogLevel = LogLevel.INFO,
        log_dir: Optional[Path] = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 10,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_structured: bool = True,
        external_handlers: Optional[List[logging.Handler]] = None,
    ):
        self.name = name
        self.log_level = log_level
        self.log_dir = log_dir or Path("logs")
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_structured = enable_structured
        self.external_handlers = external_handlers or []

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Session context
        self.session_id = str(uuid.uuid4())
        self.default_context = LogContext(
            session_id=self.session_id,
            request_id=str(uuid.uuid4()),
            version=self._get_version(),
            environment=os.getenv("SIMPULSE_ENV", "development"),
            hostname=platform.node(),
        )

        # Performance tracking
        self.operation_timers: Dict[str, float] = {}
        self.metrics_buffer: deque = deque(maxlen=1000)

        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.recent_errors: deque = deque(maxlen=100)

        # Thread safety
        self._lock = threading.Lock()

        # Setup loggers
        self._setup_loggers()

        # Start background tasks
        self._start_background_tasks()

    def _setup_loggers(self):
        """Setup various logger handlers."""
        # Main structured logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.log_level.value)
        self.logger.handlers.clear()

        # JSON formatter for structured logs
        LogFormatter("json")
        LogFormatter("human")

        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(console_handler)

        # File handlers
        if self.enable_file:
            # Main log file (human readable)
            main_file = self.log_dir / f"{self.name}.log"
            main_handler = logging.handlers.RotatingFileHandler(
                main_file, maxBytes=self.max_file_size, backupCount=self.backup_count
            )
            main_handler.setLevel(logging.DEBUG)
            main_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
                )
            )
            self.logger.addHandler(main_handler)

            # Error log file
            error_file = self.log_dir / f"{self.name}.error.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_file, maxBytes=self.max_file_size, backupCount=self.backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\\n%(exc_info)s"
                )
            )
            self.logger.addHandler(error_handler)

            # Structured JSON log file
            if self.enable_structured:
                json_file = self.log_dir / f"{self.name}.json.log"
                json_handler = logging.handlers.RotatingFileHandler(
                    json_file, maxBytes=self.max_file_size, backupCount=self.backup_count
                )
                json_handler.setLevel(logging.DEBUG)
                self.logger.addHandler(json_handler)

        # Add external handlers
        for handler in self.external_handlers:
            self.logger.addHandler(handler)

        # Setup specialized loggers
        self._setup_specialized_loggers()

    def _setup_specialized_loggers(self):
        """Setup specialized loggers for different purposes."""
        # Performance logger
        self.perf_logger = logging.getLogger(f"{self.name}.performance")
        perf_file = self.log_dir / f"{self.name}.performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_file, maxBytes=self.max_file_size, backupCount=self.backup_count
        )
        perf_handler.setFormatter(logging.Formatter("%(asctime)s - PERFORMANCE - %(message)s"))
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)

        # Security logger
        self.security_logger = logging.getLogger(f"{self.name}.security")
        security_file = self.log_dir / f"{self.name}.security.log"
        security_handler = logging.handlers.RotatingFileHandler(
            security_file, maxBytes=self.max_file_size, backupCount=self.backup_count
        )
        security_handler.setFormatter(logging.Formatter("%(asctime)s - SECURITY - %(message)s"))
        self.security_logger.addHandler(security_handler)
        self.security_logger.setLevel(logging.WARNING)

        # Audit logger
        self.audit_logger = logging.getLogger(f"{self.name}.audit")
        audit_file = self.log_dir / f"{self.name}.audit.log"
        audit_handler = logging.handlers.RotatingFileHandler(
            audit_file, maxBytes=self.max_file_size, backupCount=self.backup_count
        )
        audit_handler.setFormatter(logging.Formatter("%(asctime)s - AUDIT - %(message)s"))
        self.audit_logger.addHandler(audit_handler)
        self.audit_logger.setLevel(logging.INFO)

    def _get_version(self) -> str:
        """Get application version."""
        try:
            # Try to get version from package
            import pkg_resources

            return pkg_resources.get_distribution("simpulse").version
        except:
            return "unknown"

    def _start_background_tasks(self):
        """Start background tasks for log management."""
        # Metrics aggregation thread
        metrics_thread = threading.Thread(target=self._metrics_aggregator, daemon=True)
        metrics_thread.start()

        # Log compression thread
        compression_thread = threading.Thread(target=self._log_compressor, daemon=True)
        compression_thread.start()

    def _metrics_aggregator(self):
        """Background thread for aggregating metrics."""
        while True:
            try:
                time.sleep(60)  # Aggregate every minute
                self._aggregate_metrics()
            except Exception as e:
                self.logger.error(f"Metrics aggregation failed: {e}")

    def _log_compressor(self):
        """Background thread for compressing old log files."""
        while True:
            try:
                time.sleep(3600)  # Check every hour
                self._compress_old_logs()
            except Exception as e:
                self.logger.error(f"Log compression failed: {e}")

    def _aggregate_metrics(self):
        """Aggregate performance metrics."""
        if not self.metrics_buffer:
            return

        with self._lock:
            metrics = list(self.metrics_buffer)
            self.metrics_buffer.clear()

        if metrics:
            aggregated = self._calculate_aggregated_metrics(metrics)
            self.log_performance("metrics_aggregated", aggregated)

    def _calculate_aggregated_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregated metrics."""
        operations = defaultdict(list)

        for metric in metrics:
            if "operation" in metric and "duration" in metric:
                operations[metric["operation"]].append(metric["duration"])

        aggregated = {}
        for operation, durations in operations.items():
            if durations:
                aggregated[f"{operation}_count"] = len(durations)
                aggregated[f"{operation}_avg"] = sum(durations) / len(durations)
                aggregated[f"{operation}_min"] = min(durations)
                aggregated[f"{operation}_max"] = max(durations)
                aggregated[f"{operation}_p95"] = (
                    sorted(durations)[int(len(durations) * 0.95)]
                    if len(durations) > 1
                    else durations[0]
                )

        return aggregated

    def _compress_old_logs(self):
        """Compress old log files to save space."""
        for log_file in self.log_dir.glob("*.log.*"):
            if log_file.suffix.isdigit() and not str(log_file).endswith(".gz"):
                try:
                    compressed_file = log_file.with_suffix(log_file.suffix + ".gz")
                    with open(log_file, "rb") as f_in:
                        with gzip.open(compressed_file, "wb") as f_out:
                            f_out.writelines(f_in)
                    log_file.unlink()
                except Exception as e:
                    self.logger.error(f"Failed to compress {log_file}: {e}")

    def create_context(
        self,
        operation: Optional[str] = None,
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs,
    ) -> LogContext:
        """Create a new log context."""
        return LogContext(
            session_id=self.default_context.session_id,
            request_id=str(uuid.uuid4()),
            user_id=user_id,
            operation=operation,
            component=kwargs.get("component"),
            version=self.default_context.version,
            environment=self.default_context.environment,
            correlation_id=correlation_id,
            hostname=self.default_context.hostname,
        )

    def log_structured(
        self,
        level: LogLevel,
        message: str,
        event_type: EventType,
        context: Optional[LogContext] = None,
        **details,
    ):
        """Log a structured message."""
        if level.value < self.log_level.value:
            return

        # Get caller information
        frame = sys._getframe(1)
        source_file = frame.f_code.co_filename
        source_line = frame.f_lineno

        entry = StructuredLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.name,
            message=message,
            event_type=event_type,
            context=context or self.default_context,
            details=details,
            source_file=source_file,
            source_line=source_line,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            hostname=platform.node(),
        )

        # Add stack trace for errors
        if level.value >= LogLevel.ERROR.value:
            entry.stack_trace = "".join(traceback.format_stack())

        # Log to appropriate logger
        if event_type == EventType.SECURITY_EVENT:
            self.security_logger.log(level.value, json.dumps(entry.to_dict()))
        elif event_type == EventType.AUDIT_TRAIL:
            self.audit_logger.log(level.value, json.dumps(entry.to_dict()))
        elif event_type == EventType.PERFORMANCE_METRIC:
            self.perf_logger.log(level.value, json.dumps(entry.to_dict()))
        else:
            self.logger.log(level.value, json.dumps(entry.to_dict()))

    def log_operation_start(self, operation: str, context: Optional[LogContext] = None, **details):
        """Log the start of an operation."""
        ctx = context or self.create_context(operation=operation)
        self.operation_timers[f"{ctx.session_id}_{operation}"] = time.time()

        self.log_structured(
            LogLevel.INFO,
            f"Operation started: {operation}",
            EventType.OPERATION_START,
            ctx,
            **details,
        )

    def log_operation_success(
        self,
        operation: str,
        context: Optional[LogContext] = None,
        duration: Optional[float] = None,
        **details,
    ):
        """Log successful completion of an operation."""
        ctx = context or self.create_context(operation=operation)

        # Calculate duration if not provided
        timer_key = f"{ctx.session_id}_{operation}"
        if duration is None and timer_key in self.operation_timers:
            duration = time.time() - self.operation_timers.pop(timer_key)

        performance_metrics = {}
        if duration is not None:
            performance_metrics["duration"] = duration
            performance_metrics["operation"] = operation

            # Add to metrics buffer
            with self._lock:
                self.metrics_buffer.append(
                    {
                        "operation": operation,
                        "duration": duration,
                        "success": True,
                        "timestamp": time.time(),
                    }
                )

        entry = StructuredLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=LogLevel.INFO.name,
            message=f"Operation completed successfully: {operation}",
            event_type=EventType.OPERATION_SUCCESS,
            context=ctx,
            details=details,
            performance_metrics=performance_metrics,
        )

        self.logger.info(json.dumps(entry.to_dict()))

    def log_operation_failure(
        self,
        operation: str,
        error: Exception,
        context: Optional[LogContext] = None,
        duration: Optional[float] = None,
        **details,
    ):
        """Log failed operation."""
        ctx = context or self.create_context(operation=operation)

        # Calculate duration if not provided
        timer_key = f"{ctx.session_id}_{operation}"
        if duration is None and timer_key in self.operation_timers:
            duration = time.time() - self.operation_timers.pop(timer_key)

        # Track error
        error_key = f"{operation}_{type(error).__name__}"
        self.error_counts[error_key] += 1

        error_info = {"type": type(error).__name__, "message": str(error), "operation": operation}

        performance_metrics = {}
        if duration is not None:
            performance_metrics["duration"] = duration
            performance_metrics["operation"] = operation

            # Add to metrics buffer
            with self._lock:
                self.metrics_buffer.append(
                    {
                        "operation": operation,
                        "duration": duration,
                        "success": False,
                        "timestamp": time.time(),
                        "error_type": type(error).__name__,
                    }
                )

        entry = StructuredLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=LogLevel.ERROR.name,
            message=f"Operation failed: {operation}",
            event_type=EventType.OPERATION_FAILURE,
            context=ctx,
            details=details,
            performance_metrics=performance_metrics,
            error_info=error_info,
            stack_trace=traceback.format_exc(),
        )

        # Add to recent errors
        with self._lock:
            self.recent_errors.append(entry.to_dict())

        self.logger.error(json.dumps(entry.to_dict()))

    def log_performance(
        self,
        metric_name: str,
        value: Union[float, Dict[str, float]],
        context: Optional[LogContext] = None,
        **details,
    ):
        """Log performance metrics."""
        ctx = context or self.default_context

        metrics = value if isinstance(value, dict) else {metric_name: value}

        entry = StructuredLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=LogLevel.INFO.name,
            message=f"Performance metric: {metric_name}",
            event_type=EventType.PERFORMANCE_METRIC,
            context=ctx,
            details=details,
            performance_metrics=metrics,
        )

        self.perf_logger.info(json.dumps(entry.to_dict()))

    def log_security_event(
        self,
        event: str,
        severity: LogLevel = LogLevel.WARNING,
        context: Optional[LogContext] = None,
        **details,
    ):
        """Log security-related events."""
        ctx = context or self.default_context

        self.log_structured(
            severity, f"Security event: {event}", EventType.SECURITY_EVENT, ctx, **details
        )

    def log_audit_trail(self, action: str, context: Optional[LogContext] = None, **details):
        """Log audit trail events."""
        ctx = context or self.default_context

        self.log_structured(
            LogLevel.AUDIT, f"Audit: {action}", EventType.AUDIT_TRAIL, ctx, **details
        )

    def log_simpulse_error(self, error: SimpulseError, context: Optional[LogContext] = None):
        """Log a Simpulse error with full context."""
        ctx = context or self.create_context(operation=error.context.operation)

        error_details = {
            "error_id": error.error_id,
            "category": error.category.value,
            "severity": error.severity.value,
            "recoverable": error.recoverable,
            "retry_count": error.retry_count,
            "context": {
                "operation": error.context.operation,
                "file_path": str(error.context.file_path) if error.context.file_path else None,
                "rule_name": error.context.rule_name,
                "strategy": error.context.strategy,
            },
            "recovery_suggestions": error.recovery_suggestions,
        }

        entry = StructuredLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=error.severity.name,
            message=error.message,
            event_type=EventType.ERROR_OCCURRED,
            context=ctx,
            details=error_details,
            error_info={
                "type": (
                    type(error.original_exception).__name__
                    if error.original_exception
                    else "SimpulseError"
                ),
                "message": (
                    str(error.original_exception) if error.original_exception else error.message
                ),
            },
            stack_trace=error.stack_trace,
        )

        log_level_map = {
            ErrorSeverity.LOW: LogLevel.INFO,
            ErrorSeverity.MEDIUM: LogLevel.WARNING,
            ErrorSeverity.HIGH: LogLevel.ERROR,
            ErrorSeverity.FATAL: LogLevel.CRITICAL,
        }

        log_level = log_level_map.get(error.severity, LogLevel.ERROR)
        self.logger.log(log_level.value, json.dumps(entry.to_dict()))

    @contextmanager
    def operation_context(self, operation: str, user_id: Optional[str] = None, **details):
        """Context manager for logging operation lifecycle."""
        ctx = self.create_context(operation=operation, user_id=user_id)

        self.log_operation_start(operation, ctx, **details)
        start_time = time.time()

        try:
            yield ctx
            duration = time.time() - start_time
            self.log_operation_success(operation, ctx, duration, **details)
        except Exception as e:
            duration = time.time() - start_time
            self.log_operation_failure(operation, e, ctx, duration, **details)
            raise

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        with self._lock:
            recent_errors = list(self.recent_errors)

        return {
            "total_errors": len(recent_errors),
            "error_counts": dict(self.error_counts),
            "recent_errors": recent_errors[-10:],  # Last 10 errors
            "error_rate": (
                len(recent_errors) / max(1, len(self.metrics_buffer)) if self.metrics_buffer else 0
            ),
        }

    def export_logs(
        self,
        output_path: Path,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> bool:
        """Export logs to a file for analysis."""
        try:
            logs_to_export = []

            # Read from JSON log file
            json_log_file = self.log_dir / f"{self.name}.json.log"
            if json_log_file.exists():
                with open(json_log_file) as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line)
                            entry_time = datetime.fromisoformat(
                                log_entry["timestamp"].replace("Z", "+00:00")
                            )

                            if start_time and entry_time < start_time:
                                continue
                            if end_time and entry_time > end_time:
                                continue

                            logs_to_export.append(log_entry)
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue

            # Export to file
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "export_metadata": {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "start_time": start_time.isoformat() if start_time else None,
                            "end_time": end_time.isoformat() if end_time else None,
                            "total_entries": len(logs_to_export),
                        },
                        "logs": logs_to_export,
                    },
                    f,
                    indent=2,
                )

            return True

        except Exception as e:
            self.logger.error(f"Failed to export logs: {e}")
            return False

    def send_to_external_service(self, webhook_url: str, alert_data: Dict[str, Any]) -> bool:
        """Send alert data to external monitoring service."""
        if not HAS_REQUESTS:
            self.logger.warning("Requests library not available for external service integration")
            return False

        try:
            response = requests.post(
                webhook_url,
                json=alert_data,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.error(f"Failed to send data to external service: {e}")
            return False


# Global logger instance
_global_logger: Optional[ProductionLogger] = None


def get_production_logger(name: str = "simpulse", **kwargs) -> ProductionLogger:
    """Get or create global production logger."""
    global _global_logger

    if _global_logger is None:
        _global_logger = ProductionLogger(name, **kwargs)

    return _global_logger


def setup_production_logging(
    log_dir: Optional[Path] = None,
    log_level: LogLevel = LogLevel.INFO,
    external_webhook: Optional[str] = None,
    **kwargs,
) -> ProductionLogger:
    """Setup production logging with sensible defaults."""
    logger = ProductionLogger(log_dir=log_dir, log_level=log_level, **kwargs)

    # Setup external integration if provided
    if external_webhook:

        def error_webhook_handler(record):
            if record.levelno >= logging.ERROR.value:
                alert_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "logger": record.name,
                    "source": f"{record.filename}:{record.lineno}",
                    "environment": os.getenv("SIMPULSE_ENV", "development"),
                }
                logger.send_to_external_service(external_webhook, alert_data)

        webhook_handler = logging.Handler()
        webhook_handler.emit = error_webhook_handler
        webhook_handler.setLevel(logging.ERROR)
        logger.logger.addHandler(webhook_handler)

    return logger


# Context managers and decorators
@contextmanager
def log_operation(operation: str, logger: Optional[ProductionLogger] = None, **details):
    """Context manager for logging operations."""
    log = logger or get_production_logger()

    with log.operation_context(operation, **details) as ctx:
        yield ctx


def logged_operation(operation: str, logger: Optional[ProductionLogger] = None):
    """Decorator for logging function operations."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            log = logger or get_production_logger()

            with log.operation_context(operation) as ctx:
                return func(*args, **kwargs)

        return wrapper

    return decorator

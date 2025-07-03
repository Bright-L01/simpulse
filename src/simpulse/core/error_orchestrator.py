"""
Error handling orchestrator for Simpulse.

Integrates all error handling, monitoring, and recovery systems into a
unified, production-ready error management solution.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..errors import ErrorCategory, ErrorHandler, ErrorSeverity, SimpulseError
from .comprehensive_monitor import AlertLevel, ComprehensiveMonitor
from .graceful_degradation import GracefulDegradationManager, OperationMode, PartialResult
from .memory_manager import create_default_memory_monitor
from .production_logging import LogLevel, setup_production_logging
from .retry import RetryConfig, RetryManager, create_file_retry_config, create_lean_retry_config
from .robust_file_handler import RobustFileHandler


class ErrorHandlingOrchestrator:
    """
    Orchestrates all error handling systems for production-ready operation.

    This class integrates:
    - Error detection and categorization
    - Retry mechanisms with circuit breakers
    - Graceful degradation and partial results
    - Memory management and cleanup
    - Comprehensive monitoring and alerting
    - Production logging and audit trails
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        enable_monitoring: bool = True,
        enable_memory_management: bool = True,
        max_memory_mb: int = 4096,
        retention_days: int = 30,
    ):
        """Initialize the error handling orchestrator."""

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize core error handler
        self.error_handler = ErrorHandler()

        # Initialize production logging
        self.logger = setup_production_logging(
            log_dir=log_dir,
            log_level=LogLevel.INFO,
            external_webhook=self.config.get("external_webhook"),
        )

        # Initialize retry manager
        self.retry_manager = RetryManager(self.error_handler, self.logger.logger)

        # Initialize graceful degradation
        self.degradation_manager = GracefulDegradationManager(
            self.error_handler, self.logger.logger
        )

        # Initialize file handler
        self.file_handler = RobustFileHandler(
            self.error_handler,
            self.degradation_manager,
            max_file_size_mb=self.config.get("max_file_size_mb", 100),
        )

        # Initialize memory monitoring
        if enable_memory_management:
            self.memory_monitor = create_default_memory_monitor(self.error_handler)
            self.memory_monitor.max_memory_mb = max_memory_mb

            # Register memory cleanup callbacks
            self._setup_memory_cleanup()
        else:
            self.memory_monitor = None

        # Initialize comprehensive monitoring
        if enable_monitoring:
            self.monitor = ComprehensiveMonitor(
                self.error_handler, logger=self.logger.logger, retention_days=retention_days
            )
            self._setup_monitoring_callbacks()
        else:
            self.monitor = None

        # Setup degradation fallbacks
        self._setup_fallback_strategies()

        # Register error callbacks
        self._setup_error_callbacks()

        # Track orchestrator state
        self.is_running = False
        self.startup_time = time.time()
        self.health_status = "initializing"

        self.logger.log_audit_trail(
            "error_orchestrator_initialized",
            config=self.config,
            memory_management=enable_memory_management,
            monitoring=enable_monitoring,
        )

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "max_file_size_mb": 100,
            "retry_max_attempts": 3,
            "memory_warning_threshold": 85.0,
            "memory_critical_threshold": 95.0,
            "circuit_breaker_enabled": True,
            "degradation_enabled": True,
            "external_webhook": None,
            "log_compression": True,
            "audit_trail": True,
        }

        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logging.warning(f"Failed to load config from {config_path}: {e}")

        return default_config

    def _setup_memory_cleanup(self):
        """Setup memory cleanup callbacks and tasks."""
        if not self.memory_monitor:
            return

        # Register cleanup for file handler cache
        self.memory_monitor.register_cleanup_task(
            "file_handler_cache", lambda: self._cleanup_file_cache(), estimated_memory_freed_mb=10.0
        )

        # Register cleanup for degradation manager cache
        self.memory_monitor.register_cleanup_task(
            "degradation_cache",
            lambda: self._cleanup_degradation_cache(),
            estimated_memory_freed_mb=5.0,
        )

        # Register cleanup for error handler
        self.memory_monitor.register_cleanup_task(
            "error_handler_cleanup",
            lambda: self._cleanup_error_handler(),
            estimated_memory_freed_mb=2.0,
        )

    def _cleanup_file_cache(self) -> float:
        """Cleanup file handler cache."""
        before_count = len(self.file_handler.file_info_cache)
        self.file_handler.clear_cache()
        after_count = len(self.file_handler.file_info_cache)
        freed_items = before_count - after_count
        return freed_items * 0.1  # Estimate 0.1MB per cached file info

    def _cleanup_degradation_cache(self) -> float:
        """Cleanup degradation manager cache."""
        before_count = len(self.degradation_manager.cached_results)
        self.degradation_manager.cached_results.clear()
        after_count = len(self.degradation_manager.cached_results)
        freed_items = before_count - after_count
        return freed_items * 0.5  # Estimate 0.5MB per cached result

    def _cleanup_error_handler(self) -> float:
        """Cleanup error handler old data."""
        before_count = len(self.error_handler.errors)
        # Keep only recent errors (last 100)
        if before_count > 100:
            self.error_handler.errors = self.error_handler.errors[-100:]
        after_count = len(self.error_handler.errors)
        freed_items = before_count - after_count
        return freed_items * 0.01  # Estimate 0.01MB per error record

    def _setup_monitoring_callbacks(self):
        """Setup monitoring callbacks and alerts."""
        if not self.monitor:
            return

        # Register alert callback for critical issues
        def handle_critical_alert(alert):
            if alert.level == AlertLevel.CRITICAL:
                self.logger.log_security_event(
                    f"Critical alert: {alert.message}",
                    LogLevel.CRITICAL,
                    source=alert.source,
                    details=alert.details,
                )

                # Trigger emergency degradation for certain alerts
                if "memory" in alert.message.lower():
                    self.degradation_manager._degrade_to_mode(
                        OperationMode.EMERGENCY, f"Critical memory alert: {alert.message}"
                    )

        self.monitor.register_alert_callback(handle_critical_alert)

        # Register health checks
        self.monitor.register_health_check(
            "error_handler_health", lambda: not self.error_handler.has_fatal_errors(), interval=30.0
        )

        self.monitor.register_health_check(
            "degradation_manager_health",
            lambda: self.degradation_manager.current_mode != OperationMode.EMERGENCY,
            interval=60.0,
        )

        if self.memory_monitor:
            self.monitor.register_health_check(
                "memory_health",
                lambda: self.memory_monitor.get_memory_info().pressure_level.value != "critical",
                interval=30.0,
            )

    def _setup_fallback_strategies(self):
        """Setup fallback strategies for common operations."""

        # File reading fallback
        def file_read_fallback(file_path: Path, **kwargs):
            """Fallback for file reading - try basic read."""
            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    return f.read()
            except:
                return f"# Failed to read {file_path}"

        self.degradation_manager.register_fallback("file_read", file_read_fallback)

        # Lean compilation fallback
        def lean_compile_fallback(**kwargs):
            """Fallback for Lean compilation - skip compilation."""
            return {"status": "skipped", "reason": "compilation_disabled_due_to_failures"}

        self.degradation_manager.register_fallback("lean_compile", lean_compile_fallback)

        # Optimization fallback
        def optimization_fallback(rules, **kwargs):
            """Fallback for optimization - return original rules."""
            return rules

        self.degradation_manager.register_fallback("optimization", optimization_fallback)

    def _setup_error_callbacks(self):
        """Setup error callbacks for logging and monitoring."""

        # Register callback to log errors with production logger
        def log_error(error: SimpulseError):
            self.logger.log_simpulse_error(error)

            # Record metrics
            if self.monitor:
                self.monitor.record_metric(
                    f"error.{error.category.value}", 1.0, self.monitor.MetricType.COUNTER
                )

        # Note: We would add this to ErrorHandler if it supported callbacks
        # For now, errors are logged when handled

    def start(self):
        """Start all monitoring and management systems."""
        if self.is_running:
            return

        self.is_running = True

        # Start memory monitoring
        if self.memory_monitor:
            self.memory_monitor.start_monitoring()

        # Start comprehensive monitoring
        if self.monitor:
            self.monitor.start_monitoring()

        self.health_status = "running"

        self.logger.log_audit_trail(
            "orchestrator_started",
            startup_time=self.startup_time,
            components={
                "memory_monitor": self.memory_monitor is not None,
                "comprehensive_monitor": self.monitor is not None,
            },
        )

    def stop(self):
        """Stop all monitoring and management systems."""
        if not self.is_running:
            return

        self.is_running = False

        # Stop monitoring systems
        if self.memory_monitor:
            self.memory_monitor.stop_monitoring()

        if self.monitor:
            self.monitor.stop_monitoring()

        self.health_status = "stopped"

        self.logger.log_audit_trail("orchestrator_stopped", uptime=time.time() - self.startup_time)

    def handle_operation(
        self,
        operation_name: str,
        operation_func: Callable,
        *args,
        retry_config: Optional[RetryConfig] = None,
        enable_fallback: bool = True,
        enable_retry: bool = True,
        enable_monitoring: bool = True,
        **kwargs,
    ) -> PartialResult:
        """
        Handle an operation with full error management.

        This is the main entry point for executing operations with
        comprehensive error handling, retry, fallback, and monitoring.
        """

        with self.logger.operation_context(operation_name) as context:

            # Record operation start
            if enable_monitoring and self.monitor:
                self.monitor.record_metric(
                    f"operation.{operation_name}.started", 1.0, self.monitor.MetricType.COUNTER
                )

            # Choose retry config based on operation type
            if enable_retry and retry_config is None:
                if "file" in operation_name.lower():
                    retry_config = create_file_retry_config()
                elif "lean" in operation_name.lower():
                    retry_config = create_lean_retry_config()
                else:
                    retry_config = RetryConfig()

            # Execute operation with degradation management
            try:
                if enable_retry and retry_config:
                    # Execute with retry
                    def retry_wrapper():
                        return self.degradation_manager.execute_with_degradation(
                            operation_name,
                            operation_func,
                            *args,
                            enable_fallback=enable_fallback,
                            **kwargs,
                        )

                    result = self.retry_manager.execute_with_retry(
                        retry_wrapper, (), {}, operation_name, retry_config
                    )

                    # Wrap retry result in PartialResult if needed
                    if not isinstance(result, PartialResult):
                        result = PartialResult(data=result)
                else:
                    # Execute with degradation only
                    result = self.degradation_manager.execute_with_degradation(
                        operation_name,
                        operation_func,
                        *args,
                        enable_fallback=enable_fallback,
                        **kwargs,
                    )

                # Record success metrics
                if enable_monitoring and self.monitor:
                    self.monitor.record_operation(
                        operation_name,
                        result.processing_time,
                        result.is_usable(),
                        {
                            "status": result.status.value,
                            "fallback_used": result.fallback_used,
                            "cache_hit": result.cache_hit,
                        },
                    )

                return result

            except Exception as e:
                # Handle unrecoverable errors
                context_info = {
                    "operation": operation_name,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                    "retry_enabled": enable_retry,
                    "fallback_enabled": enable_fallback,
                }

                self.error_handler.handle_error(
                    category=self._categorize_operation_error(e, operation_name),
                    severity=ErrorSeverity.HIGH,
                    message=f"Operation {operation_name} failed completely",
                    context=self.error_handler.ErrorContext(
                        operation=operation_name, additional_info=context_info
                    ),
                    exception=e,
                )

                # Record failure metrics
                if enable_monitoring and self.monitor:
                    self.monitor.record_operation(
                        operation_name,
                        0.0,  # No timing info available
                        False,
                        {"error_type": type(e).__name__, "error_message": str(e)},
                    )

                # Return failed result
                return PartialResult(
                    data=None,
                    status=self.degradation_manager.ResultStatus.FAILED,
                    success_rate=0.0,
                    errors=[str(e)],
                )

    def _categorize_operation_error(
        self, exception: Exception, operation_name: str
    ) -> ErrorCategory:
        """Categorize an operation error."""
        if isinstance(exception, FileNotFoundError):
            return ErrorCategory.FILE_ACCESS
        elif isinstance(exception, PermissionError):
            return ErrorCategory.SECURITY
        elif isinstance(exception, MemoryError):
            return ErrorCategory.MEMORY
        elif isinstance(exception, TimeoutError):
            return ErrorCategory.TIMEOUT
        elif "lean" in operation_name.lower():
            return ErrorCategory.LEAN_EXECUTION
        elif "network" in operation_name.lower() or isinstance(exception, ConnectionError):
            return ErrorCategory.NETWORK
        else:
            return ErrorCategory.PERFORMANCE

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        health = {
            "orchestrator": {
                "status": self.health_status,
                "running": self.is_running,
                "uptime": time.time() - self.startup_time if self.is_running else 0,
            },
            "error_handler": {
                "total_errors": len(self.error_handler.errors),
                "fatal_errors": self.error_handler.has_fatal_errors(),
                "high_severity_errors": self.error_handler.has_high_severity_errors(),
                "recovery_attempts": self.error_handler.recovery_attempts,
            },
            "degradation": {
                "current_mode": self.degradation_manager.current_mode.value,
                "degradation_count": len(self.degradation_manager.degradation_history),
                "cached_results": len(self.degradation_manager.cached_results),
                "fallback_strategies": len(self.degradation_manager.fallback_strategies),
            },
        }

        # Add memory monitor status
        if self.memory_monitor:
            memory_info = self.memory_monitor.get_memory_info()
            health["memory"] = {
                "pressure_level": memory_info.pressure_level.value,
                "used_percent": memory_info.used_percent,
                "process_rss_mb": memory_info.process_rss_mb,
                "cleanup_tasks": len(self.memory_monitor.cleanup_tasks),
            }

        # Add comprehensive monitor status
        if self.monitor:
            system_health = self.monitor.get_system_health()
            health["monitoring"] = {
                "overall_status": system_health["overall_status"],
                "active_alerts": system_health["active_alerts"],
                "critical_alerts": system_health["critical_alerts"],
                "health_checks_passing": sum(
                    1
                    for check in system_health["health_checks"].values()
                    if check["status"] == "passing"
                ),
            }

        return health

    def export_comprehensive_report(self, output_path: Path) -> bool:
        """Export comprehensive error and monitoring report."""
        try:
            report = {
                "metadata": {
                    "timestamp": time.time(),
                    "orchestrator_uptime": time.time() - self.startup_time,
                    "config": self.config,
                },
                "health_status": self.get_health_status(),
                "error_summary": self.error_handler.get_user_friendly_summary(),
                "degradation_summary": self.degradation_manager.get_degradation_summary(),
                "error_statistics": self.error_handler.get_error_statistics(),
                "recovery_plan": self.error_handler.get_recovery_plan(),
            }

            # Add memory statistics if available
            if self.memory_monitor:
                report["memory_statistics"] = self.memory_monitor.get_memory_statistics()

            # Add monitoring data if available
            if self.monitor:
                report["monitoring_data"] = {
                    "system_health": self.monitor.get_system_health(),
                    "metrics_summary": self.monitor.get_metrics_summary(),
                    "operation_stats": self.monitor.operation_metrics,
                }

            # Export file handler statistics
            report["file_handler_stats"] = self.file_handler.get_file_stats()

            # Export retry manager statistics
            report["retry_statistics"] = {
                "operation_stats": self.retry_manager.get_operation_stats(),
                "circuit_breakers": self.retry_manager.get_circuit_status(),
            }

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.log_audit_trail(
                "comprehensive_report_exported",
                output_path=str(output_path),
                report_size_kb=output_path.stat().st_size / 1024,
            )

            return True

        except Exception as e:
            self.logger.log_operation_failure("export_report", e)
            return False

    def emergency_shutdown(self, reason: str):
        """Emergency shutdown with state preservation."""
        self.logger.log_security_event(
            f"Emergency shutdown initiated: {reason}", LogLevel.CRITICAL, reason=reason
        )

        # Save state before shutdown
        try:
            state_file = Path("emergency_state.json")
            self.export_comprehensive_report(state_file)
        except Exception as e:
            self.logger.logger.error(f"Failed to save emergency state: {e}")

        # Stop all systems
        self.stop()

        self.health_status = "emergency_shutdown"

        self.logger.log_audit_trail("emergency_shutdown_completed", reason=reason)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type:
            self.logger.log_operation_failure(
                "orchestrator_context",
                exc_val,
                details={"exc_type": exc_type.__name__ if exc_type else None},
            )

        self.stop()


# Global orchestrator instance
_global_orchestrator: Optional[ErrorHandlingOrchestrator] = None


def get_error_orchestrator(**kwargs) -> ErrorHandlingOrchestrator:
    """Get or create global error orchestrator."""
    global _global_orchestrator

    if _global_orchestrator is None:
        _global_orchestrator = ErrorHandlingOrchestrator(**kwargs)

    return _global_orchestrator


def with_error_handling(
    operation_name: str,
    retry_config: Optional[RetryConfig] = None,
    enable_fallback: bool = True,
    enable_retry: bool = True,
):
    """Decorator for adding comprehensive error handling to functions."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            orchestrator = get_error_orchestrator()
            return orchestrator.handle_operation(
                operation_name,
                func,
                *args,
                retry_config=retry_config,
                enable_fallback=enable_fallback,
                enable_retry=enable_retry,
                **kwargs,
            )

        return wrapper

    return decorator

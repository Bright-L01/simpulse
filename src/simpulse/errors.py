"""
Robust error handling and recovery for Simpulse.

Provides comprehensive error handling, recovery mechanisms, and user-friendly error reporting.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"  # Warning, operation can continue
    MEDIUM = "medium"  # Error, but recoverable
    HIGH = "high"  # Critical error, operation fails
    FATAL = "fatal"  # System-level failure


class ErrorCategory(Enum):
    """Categories of errors that can occur."""

    FILE_ACCESS = "file_access"
    LEAN_EXECUTION = "lean_execution"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    DEPENDENCY = "dependency"
    PERFORMANCE = "performance"


@dataclass
class ErrorContext:
    """Context information for an error."""

    operation: str
    file_path: Optional[Path] = None
    rule_name: Optional[str] = None
    strategy: Optional[str] = None
    additional_info: dict[str, Any] = None


@dataclass
class SimpulseError:
    """Comprehensive error information."""

    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: ErrorContext
    original_exception: Optional[Exception] = None
    recovery_suggestions: list[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            import time

            self.timestamp = time.time()

        if self.recovery_suggestions is None:
            self.recovery_suggestions = []


class ErrorHandler:
    """Robust error handling and recovery system."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.errors: list[SimpulseError] = []
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3

    def handle_error(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        message: str,
        context: ErrorContext,
        exception: Optional[Exception] = None,
    ) -> SimpulseError:
        """Handle an error with comprehensive logging and recovery suggestions."""

        error = SimpulseError(
            category=category,
            severity=severity,
            message=message,
            context=context,
            original_exception=exception,
            recovery_suggestions=self._generate_recovery_suggestions(category, exception),
        )

        self.errors.append(error)
        self._log_error(error)

        # Attempt recovery for medium severity errors
        if severity == ErrorSeverity.MEDIUM and self.recovery_attempts < self.max_recovery_attempts:
            self._attempt_recovery(error)

        return error

    def _generate_recovery_suggestions(
        self, category: ErrorCategory, exception: Optional[Exception]
    ) -> list[str]:
        """Generate context-specific recovery suggestions."""
        suggestions = []

        if category == ErrorCategory.FILE_ACCESS:
            suggestions.extend(
                [
                    "Check that the file path exists and is accessible",
                    "Verify file permissions (read/write access)",
                    "Ensure the file is not being used by another process",
                    "Try running with elevated privileges if necessary",
                ]
            )

        elif category == ErrorCategory.LEAN_EXECUTION:
            suggestions.extend(
                [
                    "Verify Lean 4 is installed and in PATH",
                    "Check that lake is available and working",
                    "Ensure the project has a valid lakefile.lean",
                    "Try 'lake clean' followed by 'lake build'",
                    "Check for syntax errors in Lean files",
                ]
            )

        elif category == ErrorCategory.OPTIMIZATION:
            suggestions.extend(
                [
                    "Try a more conservative optimization strategy",
                    "Reduce the number of rules being optimized",
                    "Check that the project analysis completed successfully",
                    "Verify rule extraction found valid simp rules",
                ]
            )

        elif category == ErrorCategory.VALIDATION:
            suggestions.extend(
                [
                    "Check that the optimized files compile correctly",
                    "Verify backup files exist before applying changes",
                    "Try validating with a smaller subset of changes",
                    "Ensure Lean dependencies are up to date",
                ]
            )

        elif category == ErrorCategory.CONFIGURATION:
            suggestions.extend(
                [
                    "Check configuration file syntax and format",
                    "Verify all required configuration fields are present",
                    "Try using default configuration values",
                    "Check for typos in strategy names or paths",
                ]
            )

        elif category == ErrorCategory.DEPENDENCY:
            suggestions.extend(
                [
                    "Install missing dependencies with 'pip install -r requirements.txt'",
                    "Check Python version compatibility (requires Python 3.10+)",
                    "Verify torch and sentence-transformers are properly installed",
                    "Try reinstalling dependencies in a fresh virtual environment",
                ]
            )

        # Add exception-specific suggestions
        if exception:
            if isinstance(exception, FileNotFoundError):
                suggestions.append(f"File not found: {str(exception)}")
            elif isinstance(exception, PermissionError):
                suggestions.append("Permission denied - check file/directory permissions")
            elif isinstance(exception, TimeoutError):
                suggestions.append("Operation timed out - try increasing timeout values")
            elif "No such file or directory" in str(exception):
                suggestions.append("Check that all required files and directories exist")

        return suggestions

    def _log_error(self, error: SimpulseError):
        """Log error with appropriate severity level."""
        log_message = self._format_error_message(error)

        if error.severity == ErrorSeverity.FATAL:
            self.logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

        # Log full traceback for debugging
        if error.original_exception and error.severity in [ErrorSeverity.HIGH, ErrorSeverity.FATAL]:
            self.logger.debug("Full traceback:", exc_info=error.original_exception)

    def _format_error_message(self, error: SimpulseError) -> str:
        """Format error message for logging."""
        message_parts = [
            f"[{error.category.value.upper()}]",
            f"[{error.severity.value.upper()}]",
            error.message,
        ]

        if error.context.operation:
            message_parts.append(f"Operation: {error.context.operation}")

        if error.context.file_path:
            message_parts.append(f"File: {error.context.file_path}")

        if error.context.rule_name:
            message_parts.append(f"Rule: {error.context.rule_name}")

        return " | ".join(message_parts)

    def _attempt_recovery(self, error: SimpulseError):
        """Attempt automatic recovery from recoverable errors."""
        self.recovery_attempts += 1
        self.logger.info(
            f"Attempting recovery ({self.recovery_attempts}/{self.max_recovery_attempts})"
        )

        try:
            if error.category == ErrorCategory.FILE_ACCESS:
                self._recover_file_access(error)
            elif error.category == ErrorCategory.LEAN_EXECUTION:
                self._recover_lean_execution(error)
            elif error.category == ErrorCategory.OPTIMIZATION:
                self._recover_optimization(error)

        except Exception as recovery_exception:
            self.logger.warning(f"Recovery attempt failed: {recovery_exception}")

    def _recover_file_access(self, error: SimpulseError):
        """Attempt to recover from file access errors."""
        if error.context.file_path:
            # Try to create parent directory if it doesn't exist
            if not error.context.file_path.parent.exists():
                error.context.file_path.parent.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created missing directory: {error.context.file_path.parent}")

    def _recover_lean_execution(self, error: SimpulseError):
        """Attempt to recover from Lean execution errors."""
        # Try lake clean if build fails
        if "build" in error.message.lower():
            import subprocess

            try:
                subprocess.run(["lake", "clean"], capture_output=True, check=True)
                self.logger.info("Executed 'lake clean' for recovery")
            except Exception:
                pass  # Recovery attempt failed, but don't crash

    def _recover_optimization(self, error: SimpulseError):
        """Attempt to recover from optimization errors."""
        # Could implement strategy switching or rule filtering

    def get_user_friendly_summary(self) -> dict[str, Any]:
        """Get a user-friendly summary of all errors."""
        if not self.errors:
            return {"status": "success", "errors": []}

        error_summary = {
            "status": (
                "error"
                if any(e.severity in [ErrorSeverity.HIGH, ErrorSeverity.FATAL] for e in self.errors)
                else "warning"
            ),
            "total_errors": len(self.errors),
            "by_severity": {},
            "by_category": {},
            "recent_errors": [],
            "recovery_suggestions": [],
        }

        # Count by severity and category
        for error in self.errors:
            severity_key = error.severity.value
            category_key = error.category.value

            error_summary["by_severity"][severity_key] = (
                error_summary["by_severity"].get(severity_key, 0) + 1
            )
            error_summary["by_category"][category_key] = (
                error_summary["by_category"].get(category_key, 0) + 1
            )

        # Add recent errors (last 5)
        for error in self.errors[-5:]:
            error_summary["recent_errors"].append(
                {
                    "message": error.message,
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "operation": error.context.operation,
                }
            )

        # Collect unique recovery suggestions
        all_suggestions = set()
        for error in self.errors:
            all_suggestions.update(error.recovery_suggestions)
        error_summary["recovery_suggestions"] = list(all_suggestions)[:10]  # Top 10

        return error_summary

    def clear_errors(self):
        """Clear all recorded errors."""
        self.errors.clear()
        self.recovery_attempts = 0

    def has_fatal_errors(self) -> bool:
        """Check if any fatal errors have occurred."""
        return any(error.severity == ErrorSeverity.FATAL for error in self.errors)

    def has_high_severity_errors(self) -> bool:
        """Check if any high severity errors have occurred."""
        return any(
            error.severity in [ErrorSeverity.HIGH, ErrorSeverity.FATAL] for error in self.errors
        )


# Convenience functions for common error patterns
def handle_file_error(
    handler: ErrorHandler, operation: str, file_path: Path, exception: Exception
) -> SimpulseError:
    """Handle file-related errors."""
    context = ErrorContext(operation=operation, file_path=file_path)
    severity = (
        ErrorSeverity.HIGH if isinstance(exception, PermissionError) else ErrorSeverity.MEDIUM
    )

    return handler.handle_error(
        category=ErrorCategory.FILE_ACCESS,
        severity=severity,
        message=f"File operation failed: {str(exception)}",
        context=context,
        exception=exception,
    )


def handle_lean_error(
    handler: ErrorHandler, operation: str, exception: Exception, file_path: Optional[Path] = None
) -> SimpulseError:
    """Handle Lean execution errors."""
    context = ErrorContext(operation=operation, file_path=file_path)

    return handler.handle_error(
        category=ErrorCategory.LEAN_EXECUTION,
        severity=ErrorSeverity.HIGH,
        message=f"Lean execution failed: {str(exception)}",
        context=context,
        exception=exception,
    )


def handle_optimization_error(
    handler: ErrorHandler,
    operation: str,
    exception: Exception,
    strategy: Optional[str] = None,
    rule_name: Optional[str] = None,
) -> SimpulseError:
    """Handle optimization errors."""
    context = ErrorContext(operation=operation, strategy=strategy, rule_name=rule_name)

    return handler.handle_error(
        category=ErrorCategory.OPTIMIZATION,
        severity=ErrorSeverity.MEDIUM,
        message=f"Optimization failed: {str(exception)}",
        context=context,
        exception=exception,
    )

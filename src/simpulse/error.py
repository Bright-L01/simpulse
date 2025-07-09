"""
Simple, clear error handling for Simpulse.
No frameworks, just helpful error messages.
"""

import logging
import os
import signal
from contextlib import contextmanager
from pathlib import Path

# Import safety limits from config
from .config import MAX_FILE_SIZE, MAX_MEMORY_USAGE
from .config import OPTIMIZATION_TIMEOUT as DEFAULT_TIMEOUT

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class SimpulseError(Exception):
    """Base class for all Simpulse errors."""

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class OptimizationError(SimpulseError):
    """Error during optimization process."""

    def __init__(self, message: str, file: Path | None = None, details: str | None = None):
        self.file = file
        if file:
            message = f"{message} (file: {file})"
        super().__init__(message, details)


class ConfigurationError(SimpulseError):
    """Error in configuration or environment setup."""


class LeanNotFoundError(SimpulseError):
    """Lean executable not found or not working."""

    def __init__(self, lean_path: str):
        message = f"Lean executable not found or not working: {lean_path}"
        details = "Set LEAN_PATH environment variable to the correct path, or ensure 'lean' is in your PATH"
        super().__init__(message, details)


class FileError(SimpulseError):
    """Error reading or writing files."""

    def __init__(self, message: str, file: Path, details: str | None = None):
        self.file = file
        message = f"{message}: {file}"
        super().__init__(message, details)


class FileTooLargeError(FileError):
    """File exceeds safe processing size limit."""

    def __init__(self, file: Path, size: int):
        self.size = size
        size_mb = size / 1_000_000
        message = f"File too large to process safely ({size_mb:.1f}MB)"
        details = (
            f"Maximum file size: {MAX_FILE_SIZE / 1_000_000:.1f}MB. Consider splitting the file."
        )
        super().__init__(message, file, details)


class TimeoutError(SimpulseError):
    """Operation exceeded time limit."""

    def __init__(self, operation: str, timeout: int):
        message = f"Operation '{operation}' timed out after {timeout} seconds"
        details = "The operation took too long and was cancelled for safety"
        super().__init__(message, details)


class MemoryError(SimpulseError):
    """Memory usage exceeded safe limit."""

    def __init__(self, current_usage: int):
        usage_mb = current_usage / 1_000_000
        limit_mb = MAX_MEMORY_USAGE / 1_000_000
        message = f"Memory usage too high: {usage_mb:.0f}MB (limit: {limit_mb:.0f}MB)"
        details = "Consider processing fewer files at once or using a machine with more memory"
        super().__init__(message, details)


def handle_error(error: Exception, file: Path | None = None, debug: bool = False) -> str:
    """Convert any error to a user-friendly message.

    Args:
        error: The exception that occurred
        file: Optional file context
        debug: Whether to include full stack trace

    Returns:
        User-friendly error message
    """
    if isinstance(error, SimpulseError):
        logger.error(f"Simpulse error: {error}")
        if debug:
            logger.exception("Full stack trace:")
        return str(error)

    # Convert common exceptions to friendly messages
    if isinstance(error, FileNotFoundError):
        file_context = f" {file}" if file else ""
        message = f"File not found{file_context}"
        if debug:
            logger.exception("Full stack trace:")
        return message

    if isinstance(error, PermissionError):
        file_context = f" {file}" if file else ""
        message = f"Permission denied{file_context}"
        details = "Check file permissions and try again"
        if debug:
            logger.exception("Full stack trace:")
        return f"{message}\nDetails: {details}"

    if isinstance(error, UnicodeDecodeError):
        file_context = f" {file}" if file else ""
        message = f"Unable to read file - invalid encoding{file_context}"
        details = "File may be binary or use unsupported encoding"
        if debug:
            logger.exception("Full stack trace:")
        return f"{message}\nDetails: {details}"

    # Generic error handling
    if debug:
        logger.exception(f"Unexpected error: {error}")
        return f"Unexpected error: {error}\nRun with --debug for full details"
    else:
        logger.error(f"Unexpected error: {error}")
        return f"Unexpected error: {type(error).__name__}\nRun with --debug for full details"


def safe_file_read(file_path: Path, debug: bool = False) -> str | None:
    """Safely read a file with proper error handling and size limits.

    Returns:
        File content or None if error occurred
    """
    try:
        # Check file size first
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            raise FileTooLargeError(file_path, file_size)

        return file_path.read_text()
    except Exception as e:
        error_msg = handle_error(e, file_path, debug)
        logger.error(f"Failed to read file: {error_msg}")
        return None


def safe_file_write(file_path: Path, content: str, debug: bool = False) -> bool:
    """Safely write a file with proper error handling.

    Returns:
        True if successful, False if error occurred
    """
    try:
        file_path.write_text(content)
        return True
    except Exception as e:
        error_msg = handle_error(e, file_path, debug)
        logger.error(f"Failed to write file: {error_msg}")
        return False


class TimeoutHandler:
    """Handler for timeout context."""

    def __init__(self, seconds: int, operation: str):
        self.seconds = seconds
        self.operation = operation
        self.timed_out = False

    def handle_timeout(self, signum, frame):
        self.timed_out = True
        raise TimeoutError(self.operation, self.seconds)


@contextmanager
def timeout(seconds: int = DEFAULT_TIMEOUT, operation: str = "optimization"):
    """Context manager for timeout protection.

    Usage:
        with timeout(30, "processing"):
            long_running_operation()
    """
    # Check if we can use signal-based timeout (Unix-like systems)
    if hasattr(signal, "SIGALRM"):
        handler = TimeoutHandler(seconds, operation)

        # Set the signal handler
        old_handler = signal.signal(signal.SIGALRM, handler.handle_timeout)
        signal.alarm(seconds)

        try:
            yield
        finally:
            # Disable the alarm
            signal.alarm(0)
            # Restore the old handler
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # On Windows or systems without SIGALRM, just log a warning and proceed
        logger.warning(f"Timeout protection not available on this platform for {operation}")
        yield


def check_memory_usage(operation: str | None = None) -> None:
    """Check current memory usage and raise if too high.

    Args:
        operation: Optional description of current operation

    Raises:
        MemoryError: If memory usage exceeds limit
    """
    if not HAS_PSUTIL:
        # Can't check memory without psutil
        return

    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        current_usage = memory_info.rss  # Resident Set Size

        if current_usage > MAX_MEMORY_USAGE:
            if operation:
                logger.error(f"Memory limit exceeded during {operation}")
            raise MemoryError(current_usage)

        # Log warning if getting close (80% of limit)
        if current_usage > MAX_MEMORY_USAGE * 0.8:
            usage_mb = current_usage / 1_000_000
            limit_mb = MAX_MEMORY_USAGE / 1_000_000
            logger.warning(f"High memory usage: {usage_mb:.0f}MB of {limit_mb:.0f}MB limit")

    except ImportError:
        # psutil not available
        pass
    except Exception as e:
        # Don't crash on memory check failure
        logger.debug(f"Failed to check memory: {e}")


def with_safety_limits(func):
    """Decorator to add safety limits to a function.

    Adds timeout and memory checking to any function.
    """

    def wrapper(*args, **kwargs):
        operation_name = func.__name__

        # Check memory before starting
        check_memory_usage(f"before {operation_name}")

        # Run with timeout
        with timeout(DEFAULT_TIMEOUT, operation_name):
            result = func(*args, **kwargs)

        # Check memory after completion
        check_memory_usage(f"after {operation_name}")

        return result

    return wrapper

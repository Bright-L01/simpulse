"""
Robust retry mechanisms with exponential backoff for Simpulse operations.

Provides comprehensive retry strategies for different types of operations
with intelligent backoff and failure analysis.
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Type

from ..errors import ErrorCategory, ErrorContext, ErrorHandler, ErrorSeverity


class RetryStrategy(Enum):
    """Different retry strategies available."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"
    LINEAR_BACKOFF = "linear_backoff"
    JITTERED_EXPONENTIAL = "jittered_exponential"
    ADAPTIVE = "adaptive"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.JITTERED_EXPONENTIAL
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    stop_on_exceptions: Tuple[Type[Exception], ...] = (KeyboardInterrupt, SystemExit)
    timeout_per_attempt: Optional[float] = None
    total_timeout: Optional[float] = None


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    monitor_window: float = 300.0  # 5 minutes


class RetryManager:
    """Advanced retry manager with circuit breaker and adaptive strategies."""

    def __init__(self, error_handler: ErrorHandler, logger: Optional[logging.Logger] = None):
        self.error_handler = error_handler
        self.logger = logger or logging.getLogger(__name__)
        self.operation_stats: Dict[str, Dict[str, Any]] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}

    def retry(
        self,
        operation_name: str,
        config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ):
        """Decorator for adding retry behavior to functions."""
        if config is None:
            config = RetryConfig()

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_with_retry(
                    func, args, kwargs, operation_name, config, circuit_breaker_config
                )

            return wrapper

        return decorator

    def execute_with_retry(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        operation_name: str,
        config: RetryConfig,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ) -> Any:
        """Execute a function with retry logic."""

        # Check circuit breaker
        if circuit_breaker_config and self._is_circuit_open(operation_name, circuit_breaker_config):
            raise Exception(f"Circuit breaker is open for operation: {operation_name}")

        start_time = time.time()
        last_exception = None

        for attempt in range(1, config.max_attempts + 1):
            try:
                # Check total timeout
                if config.total_timeout and (time.time() - start_time) > config.total_timeout:
                    raise TimeoutError(f"Total timeout exceeded for {operation_name}")

                # Execute with per-attempt timeout
                if config.timeout_per_attempt:
                    result = self._execute_with_timeout(
                        func, args, kwargs, config.timeout_per_attempt
                    )
                else:
                    result = func(*args, **kwargs)

                # Success - update stats and circuit breaker
                self._record_success(operation_name, attempt, time.time() - start_time)
                if circuit_breaker_config:
                    self._record_circuit_success(operation_name, circuit_breaker_config)

                return result

            except config.stop_on_exceptions:
                # Don't retry on these exceptions
                raise

            except config.retry_on_exceptions as e:
                last_exception = e

                # Record failure
                self._record_failure(operation_name, attempt, e)
                if circuit_breaker_config:
                    self._record_circuit_failure(operation_name, circuit_breaker_config)

                # Log attempt
                context = ErrorContext(
                    operation=f"{operation_name}_attempt_{attempt}",
                    additional_info={"attempt": attempt, "max_attempts": config.max_attempts},
                )

                self.error_handler.handle_error(
                    category=self._categorize_exception(e),
                    severity=(
                        ErrorSeverity.MEDIUM
                        if attempt < config.max_attempts
                        else ErrorSeverity.HIGH
                    ),
                    message=f"Attempt {attempt}/{config.max_attempts} failed: {str(e)}",
                    context=context,
                    exception=e,
                )

                # If this was the last attempt, don't sleep
                if attempt == config.max_attempts:
                    break

                # Calculate delay and sleep
                delay = self._calculate_delay(attempt, config, operation_name)
                self.logger.info(
                    f"Retrying {operation_name} in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts})"
                )
                time.sleep(delay)

        # All attempts failed
        total_time = time.time() - start_time
        self.logger.error(f"All retry attempts failed for {operation_name} after {total_time:.2f}s")
        raise last_exception

    async def execute_with_retry_async(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        operation_name: str,
        config: RetryConfig,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ) -> Any:
        """Async version of execute_with_retry."""

        if circuit_breaker_config and self._is_circuit_open(operation_name, circuit_breaker_config):
            raise Exception(f"Circuit breaker is open for operation: {operation_name}")

        start_time = time.time()
        last_exception = None

        for attempt in range(1, config.max_attempts + 1):
            try:
                if config.total_timeout and (time.time() - start_time) > config.total_timeout:
                    raise TimeoutError(f"Total timeout exceeded for {operation_name}")

                if config.timeout_per_attempt:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs), timeout=config.timeout_per_attempt
                    )
                else:
                    result = await func(*args, **kwargs)

                self._record_success(operation_name, attempt, time.time() - start_time)
                if circuit_breaker_config:
                    self._record_circuit_success(operation_name, circuit_breaker_config)

                return result

            except config.stop_on_exceptions:
                raise

            except config.retry_on_exceptions as e:
                last_exception = e

                self._record_failure(operation_name, attempt, e)
                if circuit_breaker_config:
                    self._record_circuit_failure(operation_name, circuit_breaker_config)

                context = ErrorContext(
                    operation=f"{operation_name}_async_attempt_{attempt}",
                    additional_info={"attempt": attempt, "max_attempts": config.max_attempts},
                )

                self.error_handler.handle_error(
                    category=self._categorize_exception(e),
                    severity=(
                        ErrorSeverity.MEDIUM
                        if attempt < config.max_attempts
                        else ErrorSeverity.HIGH
                    ),
                    message=f"Async attempt {attempt}/{config.max_attempts} failed: {str(e)}",
                    context=context,
                    exception=e,
                )

                if attempt == config.max_attempts:
                    break

                delay = self._calculate_delay(attempt, config, operation_name)
                self.logger.info(
                    f"Retrying async {operation_name} in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts})"
                )
                await asyncio.sleep(delay)

        total_time = time.time() - start_time
        self.logger.error(
            f"All async retry attempts failed for {operation_name} after {total_time:.2f}s"
        )
        raise last_exception

    def _calculate_delay(self, attempt: int, config: RetryConfig, operation_name: str) -> float:
        """Calculate delay before next retry attempt."""

        if config.strategy == RetryStrategy.FIXED_INTERVAL:
            delay = config.base_delay

        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * attempt

        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_factor ** (attempt - 1))

        elif config.strategy == RetryStrategy.JITTERED_EXPONENTIAL:
            base_delay = config.base_delay * (config.backoff_factor ** (attempt - 1))
            jitter = random.uniform(0.5, 1.5) if config.jitter else 1.0
            delay = base_delay * jitter

        elif config.strategy == RetryStrategy.ADAPTIVE:
            delay = self._adaptive_delay(operation_name, attempt, config)

        else:
            delay = config.base_delay

        # Apply max delay limit
        return min(delay, config.max_delay)

    def _adaptive_delay(self, operation_name: str, attempt: int, config: RetryConfig) -> float:
        """Calculate adaptive delay based on operation history."""
        stats = self.operation_stats.get(operation_name, {})

        # If we have historical data, use it to adapt
        if "avg_success_time" in stats and "failure_rate" in stats:
            base_delay = config.base_delay * (config.backoff_factor ** (attempt - 1))

            # Adjust based on failure rate
            failure_penalty = 1 + (stats["failure_rate"] * 2)

            # Adjust based on average success time
            time_factor = max(0.5, min(2.0, stats["avg_success_time"] / 5.0))

            delay = base_delay * failure_penalty * time_factor
        else:
            # Fall back to jittered exponential
            base_delay = config.base_delay * (config.backoff_factor ** (attempt - 1))
            delay = base_delay * random.uniform(0.8, 1.2)

        return delay

    def _execute_with_timeout(
        self, func: Callable, args: tuple, kwargs: dict, timeout: float
    ) -> Any:
        """Execute function with timeout (for sync functions)."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function execution exceeded {timeout}s timeout")

        # Set up timeout signal (Unix only)
        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Fallback for Windows - just execute without timeout
            return func(*args, **kwargs)

    def _categorize_exception(self, exception: Exception) -> ErrorCategory:
        """Categorize exception for error handling."""
        if isinstance(exception, FileNotFoundError):
            return ErrorCategory.FILE_ACCESS
        elif isinstance(exception, PermissionError):
            return ErrorCategory.SECURITY
        elif isinstance(exception, MemoryError):
            return ErrorCategory.MEMORY
        elif isinstance(exception, TimeoutError):
            return ErrorCategory.TIMEOUT
        elif isinstance(exception, UnicodeDecodeError):
            return ErrorCategory.ENCODING
        elif "network" in str(exception).lower() or "connection" in str(exception).lower():
            return ErrorCategory.NETWORK
        elif "lean" in str(exception).lower():
            return ErrorCategory.LEAN_EXECUTION
        else:
            return ErrorCategory.PERFORMANCE

    def _record_success(self, operation_name: str, attempts: int, duration: float):
        """Record successful operation for statistics."""
        if operation_name not in self.operation_stats:
            self.operation_stats[operation_name] = {
                "total_attempts": 0,
                "successful_attempts": 0,
                "failed_attempts": 0,
                "total_duration": 0.0,
                "avg_success_time": 0.0,
                "failure_rate": 0.0,
                "last_success": time.time(),
            }

        stats = self.operation_stats[operation_name]
        stats["total_attempts"] += attempts
        stats["successful_attempts"] += 1
        stats["total_duration"] += duration
        stats["avg_success_time"] = stats["total_duration"] / stats["successful_attempts"]
        stats["failure_rate"] = stats["failed_attempts"] / stats["total_attempts"]
        stats["last_success"] = time.time()

    def _record_failure(self, operation_name: str, attempt: int, exception: Exception):
        """Record failed operation for statistics."""
        if operation_name not in self.operation_stats:
            self.operation_stats[operation_name] = {
                "total_attempts": 0,
                "successful_attempts": 0,
                "failed_attempts": 0,
                "total_duration": 0.0,
                "avg_success_time": 0.0,
                "failure_rate": 0.0,
                "last_failure": time.time(),
                "recent_exceptions": [],
            }

        stats = self.operation_stats[operation_name]
        stats["total_attempts"] += 1
        stats["failed_attempts"] += 1
        stats["failure_rate"] = stats["failed_attempts"] / stats["total_attempts"]
        stats["last_failure"] = time.time()

        # Keep track of recent exceptions
        if "recent_exceptions" not in stats:
            stats["recent_exceptions"] = []

        stats["recent_exceptions"].append(
            {
                "timestamp": time.time(),
                "attempt": attempt,
                "exception_type": type(exception).__name__,
                "message": str(exception),
            }
        )

        # Keep only last 10 exceptions
        stats["recent_exceptions"] = stats["recent_exceptions"][-10:]

    def _is_circuit_open(self, operation_name: str, config: CircuitBreakerConfig) -> bool:
        """Check if circuit breaker is open for this operation."""
        circuit = self.circuit_breakers.get(operation_name)
        if not circuit:
            return False

        current_time = time.time()

        if circuit["state"] == CircuitBreakerState.OPEN.value:
            # Check if recovery timeout has passed
            if current_time - circuit["opened_at"] > config.recovery_timeout:
                circuit["state"] = CircuitBreakerState.HALF_OPEN.value
                circuit["successes_in_half_open"] = 0
                self.logger.info(f"Circuit breaker for {operation_name} moved to HALF_OPEN")
                return False
            return True

        return False

    def _record_circuit_success(self, operation_name: str, config: CircuitBreakerConfig):
        """Record success for circuit breaker."""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = {
                "state": CircuitBreakerState.CLOSED.value,
                "failures": 0,
                "successes_in_half_open": 0,
                "opened_at": None,
            }

        circuit = self.circuit_breakers[operation_name]

        if circuit["state"] == CircuitBreakerState.HALF_OPEN.value:
            circuit["successes_in_half_open"] += 1
            if circuit["successes_in_half_open"] >= config.success_threshold:
                circuit["state"] = CircuitBreakerState.CLOSED.value
                circuit["failures"] = 0
                self.logger.info(f"Circuit breaker for {operation_name} moved to CLOSED")

        elif circuit["state"] == CircuitBreakerState.CLOSED.value:
            circuit["failures"] = max(0, circuit["failures"] - 1)  # Gradual recovery

    def _record_circuit_failure(self, operation_name: str, config: CircuitBreakerConfig):
        """Record failure for circuit breaker."""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = {
                "state": CircuitBreakerState.CLOSED.value,
                "failures": 0,
                "successes_in_half_open": 0,
                "opened_at": None,
            }

        circuit = self.circuit_breakers[operation_name]
        circuit["failures"] += 1

        if circuit["state"] == CircuitBreakerState.CLOSED.value:
            if circuit["failures"] >= config.failure_threshold:
                circuit["state"] = CircuitBreakerState.OPEN.value
                circuit["opened_at"] = time.time()
                self.logger.warning(f"Circuit breaker for {operation_name} moved to OPEN")

        elif circuit["state"] == CircuitBreakerState.HALF_OPEN.value:
            circuit["state"] = CircuitBreakerState.OPEN.value
            circuit["opened_at"] = time.time()
            self.logger.warning(f"Circuit breaker for {operation_name} moved back to OPEN")

    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for operations."""
        if operation_name:
            return self.operation_stats.get(operation_name, {})
        return self.operation_stats

    def get_circuit_status(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get circuit breaker status."""
        if operation_name:
            return self.circuit_breakers.get(operation_name, {})
        return self.circuit_breakers

    def reset_operation_stats(self, operation_name: Optional[str] = None):
        """Reset statistics for operations."""
        if operation_name:
            self.operation_stats.pop(operation_name, None)
        else:
            self.operation_stats.clear()

    def reset_circuit_breaker(self, operation_name: str):
        """Reset circuit breaker to closed state."""
        if operation_name in self.circuit_breakers:
            self.circuit_breakers[operation_name] = {
                "state": CircuitBreakerState.CLOSED.value,
                "failures": 0,
                "successes_in_half_open": 0,
                "opened_at": None,
            }
            self.logger.info(f"Reset circuit breaker for {operation_name}")


# Convenience functions for common retry patterns
def create_file_retry_config() -> RetryConfig:
    """Create retry config optimized for file operations."""
    return RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        max_delay=5.0,
        strategy=RetryStrategy.JITTERED_EXPONENTIAL,
        retry_on_exceptions=(FileNotFoundError, PermissionError, OSError),
        stop_on_exceptions=(KeyboardInterrupt, SystemExit),
        timeout_per_attempt=30.0,
    )


def create_lean_retry_config() -> RetryConfig:
    """Create retry config optimized for Lean operations."""
    return RetryConfig(
        max_attempts=5,
        base_delay=2.0,
        max_delay=30.0,
        strategy=RetryStrategy.ADAPTIVE,
        retry_on_exceptions=(subprocess.SubprocessError, TimeoutError, OSError),
        stop_on_exceptions=(KeyboardInterrupt, SystemExit),
        timeout_per_attempt=120.0,
        total_timeout=600.0,
    )


def create_network_retry_config() -> RetryConfig:
    """Create retry config optimized for network operations."""
    return RetryConfig(
        max_attempts=4,
        base_delay=1.0,
        max_delay=60.0,
        strategy=RetryStrategy.JITTERED_EXPONENTIAL,
        retry_on_exceptions=(ConnectionError, TimeoutError, OSError),
        stop_on_exceptions=(KeyboardInterrupt, SystemExit),
        timeout_per_attempt=60.0,
        total_timeout=300.0,
    )


def create_memory_retry_config() -> RetryConfig:
    """Create retry config for memory-sensitive operations."""
    return RetryConfig(
        max_attempts=2,
        base_delay=5.0,
        max_delay=30.0,
        strategy=RetryStrategy.FIXED_INTERVAL,
        retry_on_exceptions=(MemoryError,),
        stop_on_exceptions=(KeyboardInterrupt, SystemExit, MemoryError),
        timeout_per_attempt=300.0,
    )


def create_circuit_breaker_config(operation_type: str = "default") -> CircuitBreakerConfig:
    """Create circuit breaker config based on operation type."""
    configs = {
        "file": CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout=30.0, success_threshold=2
        ),
        "lean": CircuitBreakerConfig(
            failure_threshold=5, recovery_timeout=120.0, success_threshold=3
        ),
        "network": CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=60.0, success_threshold=1
        ),
        "default": CircuitBreakerConfig(),
    }

    return configs.get(operation_type, configs["default"])

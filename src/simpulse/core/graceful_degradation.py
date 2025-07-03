"""
Graceful degradation and partial result handling for Simpulse.

Provides mechanisms to handle partial failures gracefully and continue
processing with reduced functionality when errors occur.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from ..errors import ErrorCategory, ErrorContext, ErrorHandler, ErrorSeverity

T = TypeVar("T")


class OperationMode(Enum):
    """Different operation modes for graceful degradation."""

    FULL = "full"  # All features enabled
    REDUCED = "reduced"  # Some features disabled
    MINIMAL = "minimal"  # Only core features
    EMERGENCY = "emergency"  # Bare minimum functionality
    OFFLINE = "offline"  # No external dependencies


class ResultStatus(Enum):
    """Status of operation results."""

    COMPLETE = "complete"  # Operation completed successfully
    PARTIAL = "partial"  # Operation partially completed
    FALLBACK = "fallback"  # Fallback method used
    CACHED = "cached"  # Result from cache
    DEGRADED = "degraded"  # Reduced quality result
    FAILED = "failed"  # Operation failed


@dataclass
class PartialResult(Generic[T]):
    """Container for partial results with metadata."""

    data: Optional[T] = None
    status: ResultStatus = ResultStatus.COMPLETE
    success_rate: float = 1.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    operation_mode: OperationMode = OperationMode.FULL
    fallback_used: bool = False
    cache_hit: bool = False
    processing_time: float = 0.0

    def is_usable(self, min_success_rate: float = 0.5) -> bool:
        """Check if result is usable based on success rate."""
        return (
            self.status != ResultStatus.FAILED
            and self.success_rate >= min_success_rate
            and self.data is not None
        )

    def get_quality_score(self) -> float:
        """Get overall quality score (0.0 to 1.0)."""
        status_weights = {
            ResultStatus.COMPLETE: 1.0,
            ResultStatus.PARTIAL: 0.8,
            ResultStatus.FALLBACK: 0.6,
            ResultStatus.CACHED: 0.7,
            ResultStatus.DEGRADED: 0.4,
            ResultStatus.FAILED: 0.0,
        }

        base_score = status_weights.get(self.status, 0.0)
        return base_score * self.success_rate


class GracefulDegradationManager:
    """Manages graceful degradation strategies and partial results."""

    def __init__(self, error_handler: ErrorHandler, logger: Optional[logging.Logger] = None):
        self.error_handler = error_handler
        self.logger = logger or logging.getLogger(__name__)
        self.current_mode = OperationMode.FULL
        self.degradation_history: List[Dict[str, Any]] = []
        self.fallback_strategies: Dict[str, Callable] = {}
        self.cached_results: Dict[str, PartialResult] = {}
        self.operation_stats: Dict[str, Dict[str, Any]] = {}
        self.degradation_triggers: Dict[ErrorCategory, OperationMode] = {
            ErrorCategory.MEMORY: OperationMode.REDUCED,
            ErrorCategory.TIMEOUT: OperationMode.REDUCED,
            ErrorCategory.NETWORK: OperationMode.OFFLINE,
            ErrorCategory.RESOURCE_EXHAUSTION: OperationMode.MINIMAL,
            ErrorCategory.CORRUPTION: OperationMode.EMERGENCY,
        }

    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register a fallback function for an operation."""
        self.fallback_strategies[operation_name] = fallback_func
        self.logger.info(f"Registered fallback strategy for {operation_name}")

    def execute_with_degradation(
        self,
        operation_name: str,
        primary_func: Callable,
        *args,
        min_success_rate: float = 0.5,
        cache_key: Optional[str] = None,
        enable_fallback: bool = True,
        **kwargs,
    ) -> PartialResult:
        """Execute operation with graceful degradation support."""

        start_time = time.time()
        result = PartialResult(operation_mode=self.current_mode)

        # Check cache first
        if cache_key and cache_key in self.cached_results:
            cached_result = self.cached_results[cache_key]
            if self._is_cache_valid(cached_result):
                cached_result.cache_hit = True
                self.logger.info(f"Using cached result for {operation_name}")
                return cached_result

        # Try primary operation
        try:
            result.data = primary_func(*args, **kwargs)
            result.status = ResultStatus.COMPLETE
            result.success_rate = 1.0

        except Exception as e:
            self.logger.warning(f"Primary operation {operation_name} failed: {e}")
            result.errors.append(str(e))

            # Handle error and potentially degrade mode
            self._handle_operation_error(operation_name, e)

            # Try fallback if enabled and available
            if enable_fallback and operation_name in self.fallback_strategies:
                try:
                    fallback_func = self.fallback_strategies[operation_name]
                    result.data = fallback_func(*args, **kwargs)
                    result.status = ResultStatus.FALLBACK
                    result.fallback_used = True
                    result.success_rate = 0.7  # Fallback gets lower success rate
                    self.logger.info(f"Used fallback strategy for {operation_name}")

                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback for {operation_name} also failed: {fallback_error}"
                    )
                    result.errors.append(f"Fallback failed: {fallback_error}")
                    result.status = ResultStatus.FAILED
                    result.success_rate = 0.0
            else:
                result.status = ResultStatus.FAILED
                result.success_rate = 0.0

        result.processing_time = time.time() - start_time

        # Cache successful results
        if cache_key and result.is_usable(min_success_rate):
            self.cached_results[cache_key] = result

        # Update operation statistics
        self._update_operation_stats(operation_name, result)

        return result

    def batch_execute_with_degradation(
        self,
        operation_name: str,
        items: List[Any],
        item_processor: Callable,
        min_success_rate: float = 0.5,
        fail_fast: bool = False,
        max_failures: Optional[int] = None,
    ) -> PartialResult[List[Any]]:
        """Execute batch operation with partial success handling."""

        start_time = time.time()
        results = []
        errors = []
        warnings = []
        successful_items = 0
        failed_items = 0

        for i, item in enumerate(items):
            try:
                item_result = item_processor(item)
                results.append(item_result)
                successful_items += 1

            except Exception as e:
                error_msg = f"Item {i} failed: {e}"
                errors.append(error_msg)
                failed_items += 1

                self.logger.warning(error_msg)

                # Check if we should fail fast
                if fail_fast:
                    break

                # Check if we've hit max failures
                if max_failures and failed_items >= max_failures:
                    warnings.append(f"Stopping batch after {max_failures} failures")
                    break

                # Add placeholder for failed item to maintain indexing
                results.append(None)

        # Calculate overall success rate
        total_items = successful_items + failed_items
        success_rate = successful_items / total_items if total_items > 0 else 0.0

        # Determine result status
        if success_rate == 1.0:
            status = ResultStatus.COMPLETE
        elif success_rate >= min_success_rate:
            status = ResultStatus.PARTIAL
        elif success_rate > 0:
            status = ResultStatus.DEGRADED
        else:
            status = ResultStatus.FAILED

        result = PartialResult(
            data=results,
            status=status,
            success_rate=success_rate,
            errors=errors,
            warnings=warnings,
            metadata={
                "total_items": len(items),
                "successful_items": successful_items,
                "failed_items": failed_items,
                "processing_stopped_early": failed_items < len(items) - successful_items,
            },
            processing_time=time.time() - start_time,
            operation_mode=self.current_mode,
        )

        self._update_operation_stats(f"{operation_name}_batch", result)

        return result

    def _handle_operation_error(self, operation_name: str, exception: Exception):
        """Handle operation error and potentially trigger degradation."""

        # Categorize the error
        error_category = self._categorize_error(exception)

        # Check if this error should trigger degradation
        if error_category in self.degradation_triggers:
            target_mode = self.degradation_triggers[error_category]

            # Only degrade if target mode is more restrictive
            if self._mode_priority(target_mode) > self._mode_priority(self.current_mode):
                self._degrade_to_mode(target_mode, f"Error in {operation_name}: {exception}")

        # Record error with context
        context = ErrorContext(
            operation=operation_name, additional_info={"current_mode": self.current_mode.value}
        )

        self.error_handler.handle_error(
            category=error_category,
            severity=ErrorSeverity.MEDIUM,
            message=f"Operation {operation_name} failed",
            context=context,
            exception=exception,
        )

    def _categorize_error(self, exception: Exception) -> ErrorCategory:
        """Categorize exception for degradation logic."""
        if isinstance(exception, MemoryError):
            return ErrorCategory.MEMORY
        elif isinstance(exception, TimeoutError):
            return ErrorCategory.TIMEOUT
        elif (
            isinstance(exception, (ConnectionError, OSError))
            and "network" in str(exception).lower()
        ):
            return ErrorCategory.NETWORK
        elif isinstance(exception, FileNotFoundError):
            return ErrorCategory.FILE_ACCESS
        elif "encoding" in str(exception).lower():
            return ErrorCategory.ENCODING
        else:
            return ErrorCategory.PERFORMANCE

    def _mode_priority(self, mode: OperationMode) -> int:
        """Get priority of operation mode (higher = more restrictive)."""
        priorities = {
            OperationMode.FULL: 0,
            OperationMode.REDUCED: 1,
            OperationMode.MINIMAL: 2,
            OperationMode.OFFLINE: 3,
            OperationMode.EMERGENCY: 4,
        }
        return priorities.get(mode, 0)

    def _degrade_to_mode(self, target_mode: OperationMode, reason: str):
        """Degrade operation mode."""
        previous_mode = self.current_mode
        self.current_mode = target_mode

        degradation_event = {
            "timestamp": time.time(),
            "from_mode": previous_mode.value,
            "to_mode": target_mode.value,
            "reason": reason,
        }

        self.degradation_history.append(degradation_event)

        self.logger.warning(
            f"Degraded operation mode from {previous_mode.value} to {target_mode.value}: {reason}"
        )

        # Clear cache when degrading as results may no longer be valid
        if self._mode_priority(target_mode) > self._mode_priority(previous_mode):
            self._clear_cache()

    def _is_cache_valid(self, cached_result: PartialResult, max_age: float = 3600.0) -> bool:
        """Check if cached result is still valid."""
        age = time.time() - cached_result.timestamp

        # Check age
        if age > max_age:
            return False

        # Check if cache is compatible with current mode
        if self._mode_priority(self.current_mode) > self._mode_priority(
            cached_result.operation_mode
        ):
            return False

        # Check success rate
        if cached_result.success_rate < 0.5:
            return False

        return True

    def _clear_cache(self):
        """Clear cached results."""
        self.cached_results.clear()
        self.logger.info("Cleared result cache due to mode degradation")

    def _update_operation_stats(self, operation_name: str, result: PartialResult):
        """Update statistics for operation."""
        if operation_name not in self.operation_stats:
            self.operation_stats[operation_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "partial_executions": 0,
                "failed_executions": 0,
                "average_success_rate": 0.0,
                "average_processing_time": 0.0,
                "fallback_usage_count": 0,
                "cache_hit_count": 0,
            }

        stats = self.operation_stats[operation_name]
        stats["total_executions"] += 1

        if result.status == ResultStatus.COMPLETE:
            stats["successful_executions"] += 1
        elif result.status in [ResultStatus.PARTIAL, ResultStatus.FALLBACK, ResultStatus.DEGRADED]:
            stats["partial_executions"] += 1
        else:
            stats["failed_executions"] += 1

        if result.fallback_used:
            stats["fallback_usage_count"] += 1

        if result.cache_hit:
            stats["cache_hit_count"] += 1

        # Update averages
        total_success_rate = (
            stats["average_success_rate"] * (stats["total_executions"] - 1) + result.success_rate
        )
        stats["average_success_rate"] = total_success_rate / stats["total_executions"]

        total_time = (
            stats["average_processing_time"] * (stats["total_executions"] - 1)
            + result.processing_time
        )
        stats["average_processing_time"] = total_time / stats["total_executions"]

    def upgrade_mode(self, target_mode: OperationMode, reason: str = "Manual upgrade") -> bool:
        """Attempt to upgrade operation mode."""
        if self._mode_priority(target_mode) >= self._mode_priority(self.current_mode):
            self.logger.info(
                f"Cannot upgrade to {target_mode.value} - already at same or more restrictive mode"
            )
            return False

        # Test if upgrade is possible by checking system health
        if self._can_upgrade_to_mode(target_mode):
            previous_mode = self.current_mode
            self.current_mode = target_mode

            upgrade_event = {
                "timestamp": time.time(),
                "from_mode": previous_mode.value,
                "to_mode": target_mode.value,
                "reason": reason,
            }

            self.degradation_history.append(upgrade_event)

            self.logger.info(
                f"Upgraded operation mode from {previous_mode.value} to {target_mode.value}: {reason}"
            )
            return True
        else:
            self.logger.warning(
                f"Cannot upgrade to {target_mode.value} - system constraints prevent upgrade"
            )
            return False

    def _can_upgrade_to_mode(self, target_mode: OperationMode) -> bool:
        """Check if system can support upgraded mode."""
        try:
            import psutil

            # Check memory availability
            memory = psutil.virtual_memory()
            if target_mode == OperationMode.FULL and memory.percent > 85:
                return False

            # Check disk space
            disk = psutil.disk_usage(".")
            if target_mode == OperationMode.FULL and disk.percent > 90:
                return False

            # Could add more sophisticated checks here
            return True

        except Exception:
            # If we can't check, assume we can't upgrade safely
            return False

    def get_degradation_summary(self) -> Dict[str, Any]:
        """Get summary of degradation events and current state."""
        return {
            "current_mode": self.current_mode.value,
            "degradation_history": self.degradation_history[-10:],  # Last 10 events
            "total_degradations": len(self.degradation_history),
            "operation_stats": self.operation_stats,
            "cached_results_count": len(self.cached_results),
            "fallback_strategies_count": len(self.fallback_strategies),
        }

    def save_state(self, file_path: Path) -> bool:
        """Save current state to file for recovery."""
        try:
            state = {
                "timestamp": time.time(),
                "current_mode": self.current_mode.value,
                "degradation_history": self.degradation_history,
                "operation_stats": self.operation_stats,
                "cached_results": {
                    k: {
                        "data": v.data,
                        "status": v.status.value,
                        "success_rate": v.success_rate,
                        "metadata": v.metadata,
                        "timestamp": v.timestamp,
                    }
                    for k, v in self.cached_results.items()
                    if v.is_usable()
                },
            }

            with open(file_path, "w") as f:
                json.dump(state, f, indent=2, default=str)

            self.logger.info(f"Saved degradation state to {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            return False

    def load_state(self, file_path: Path) -> bool:
        """Load state from file for recovery."""
        try:
            if not file_path.exists():
                return False

            with open(file_path) as f:
                state = json.load(f)

            self.current_mode = OperationMode(state.get("current_mode", OperationMode.FULL.value))
            self.degradation_history = state.get("degradation_history", [])
            self.operation_stats = state.get("operation_stats", {})

            # Restore cached results (if they're still valid)
            cached_data = state.get("cached_results", {})
            for key, cached_info in cached_data.items():
                if time.time() - cached_info["timestamp"] < 3600:  # 1 hour max age
                    result = PartialResult(
                        data=cached_info["data"],
                        status=ResultStatus(cached_info["status"]),
                        success_rate=cached_info["success_rate"],
                        metadata=cached_info["metadata"],
                        timestamp=cached_info["timestamp"],
                    )
                    self.cached_results[key] = result

            self.logger.info(f"Loaded degradation state from {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False


# Convenience functions
def create_simple_fallback(default_value: Any) -> Callable:
    """Create a simple fallback that returns a default value."""

    def fallback(*args, **kwargs):
        return default_value

    return fallback


def create_cached_fallback(cache_file: Path) -> Callable:
    """Create a fallback that uses cached data from file."""

    def fallback(*args, **kwargs):
        try:
            if cache_file.exists():
                with open(cache_file) as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    return fallback


def create_reduced_functionality_fallback(simplified_func: Callable) -> Callable:
    """Create a fallback that uses a simplified version of functionality."""

    def fallback(*args, **kwargs):
        # Remove complex parameters and use simplified processing
        simplified_kwargs = {
            k: v for k, v in kwargs.items() if k in ["path", "content", "basic_mode"]
        }
        return simplified_func(*args, **simplified_kwargs)

    return fallback

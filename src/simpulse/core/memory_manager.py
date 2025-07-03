"""
Memory management and resource monitoring for Simpulse.

Provides comprehensive memory monitoring, cleanup, and protection against
memory exhaustion scenarios.
"""

import gc
import logging
import sys
import threading
import time
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil

from ..errors import ErrorCategory, ErrorContext, ErrorHandler, ErrorSeverity


class MemoryPressure(Enum):
    """Memory pressure levels."""

    LOW = "low"  # < 70% memory usage
    MODERATE = "moderate"  # 70-85% memory usage
    HIGH = "high"  # 85-95% memory usage
    CRITICAL = "critical"  # > 95% memory usage


class CleanupPriority(Enum):
    """Priority levels for cleanup operations."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MemoryInfo:
    """Current memory information."""

    total_mb: float
    available_mb: float
    used_mb: float
    used_percent: float
    pressure_level: MemoryPressure
    process_rss_mb: float
    process_vms_mb: float
    swap_used_mb: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CleanupTask:
    """Represents a cleanup task."""

    name: str
    cleanup_func: Callable
    priority: CleanupPriority
    estimated_memory_freed_mb: float
    last_executed: float = 0.0
    execution_count: int = 0
    average_time: float = 0.0


class MemoryMonitor:
    """Monitors memory usage and triggers cleanup when needed."""

    def __init__(
        self,
        error_handler: ErrorHandler,
        logger: Optional[logging.Logger] = None,
        check_interval: float = 5.0,
        warning_threshold: float = 85.0,
        critical_threshold: float = 95.0,
    ):
        self.error_handler = error_handler
        self.logger = logger or logging.getLogger(__name__)
        self.check_interval = check_interval
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self.cleanup_tasks: Dict[str, CleanupTask] = {}
        self.memory_history: deque = deque(maxlen=100)
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.cleanup_callbacks: List[Callable[[MemoryPressure], None]] = []

        # Object tracking for cleanup
        self.tracked_objects: Dict[str, weakref.WeakSet] = defaultdict(weakref.WeakSet)
        self.large_allocations: Dict[int, Dict[str, Any]] = {}

        # Statistics
        self.stats = {
            "cleanup_executions": 0,
            "memory_warnings": 0,
            "critical_events": 0,
            "bytes_freed": 0,
            "peak_memory_mb": 0.0,
        }

    def start_monitoring(self):
        """Start continuous memory monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Started memory monitoring")

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Stopped memory monitoring")

    def get_memory_info(self) -> MemoryInfo:
        """Get current memory information."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Process memory
            process = psutil.Process()
            process_info = process.memory_info()

            # Determine pressure level
            if memory.percent < 70:
                pressure = MemoryPressure.LOW
            elif memory.percent < self.warning_threshold:
                pressure = MemoryPressure.MODERATE
            elif memory.percent < self.critical_threshold:
                pressure = MemoryPressure.HIGH
            else:
                pressure = MemoryPressure.CRITICAL

            info = MemoryInfo(
                total_mb=memory.total / 1024 / 1024,
                available_mb=memory.available / 1024 / 1024,
                used_mb=memory.used / 1024 / 1024,
                used_percent=memory.percent,
                pressure_level=pressure,
                process_rss_mb=process_info.rss / 1024 / 1024,
                process_vms_mb=process_info.vms / 1024 / 1024,
                swap_used_mb=swap.used / 1024 / 1024,
            )

            # Update peak memory
            if info.process_rss_mb > self.stats["peak_memory_mb"]:
                self.stats["peak_memory_mb"] = info.process_rss_mb

            return info

        except Exception as e:
            self.logger.error(f"Failed to get memory info: {e}")
            # Return safe defaults
            return MemoryInfo(
                total_mb=0.0,
                available_mb=0.0,
                used_mb=0.0,
                used_percent=0.0,
                pressure_level=MemoryPressure.LOW,
                process_rss_mb=0.0,
                process_vms_mb=0.0,
                swap_used_mb=0.0,
            )

    def register_cleanup_task(
        self,
        name: str,
        cleanup_func: Callable,
        priority: CleanupPriority = CleanupPriority.NORMAL,
        estimated_memory_freed_mb: float = 1.0,
    ):
        """Register a cleanup task."""
        task = CleanupTask(
            name=name,
            cleanup_func=cleanup_func,
            priority=priority,
            estimated_memory_freed_mb=estimated_memory_freed_mb,
        )

        self.cleanup_tasks[name] = task
        self.logger.info(f"Registered cleanup task: {name}")

    def register_cleanup_callback(self, callback: Callable[[MemoryPressure], None]):
        """Register callback for memory pressure events."""
        self.cleanup_callbacks.append(callback)

    def track_object(self, obj: Any, category: str = "general"):
        """Track object for potential cleanup."""
        self.tracked_objects[category].add(obj)

    def track_large_allocation(self, size_mb: float, description: str = "unknown") -> int:
        """Track large memory allocation."""
        allocation_id = id(object())  # Unique ID
        self.large_allocations[allocation_id] = {
            "size_mb": size_mb,
            "description": description,
            "timestamp": time.time(),
        }
        return allocation_id

    def untrack_large_allocation(self, allocation_id: int):
        """Remove tracking for large allocation."""
        self.large_allocations.pop(allocation_id, None)

    def execute_cleanup(self, pressure_level: MemoryPressure) -> float:
        """Execute cleanup tasks based on pressure level."""
        total_freed = 0.0
        executed_tasks = []

        # Sort tasks by priority (high priority first)
        tasks_to_run = []

        for task in self.cleanup_tasks.values():
            if pressure_level == MemoryPressure.CRITICAL:
                tasks_to_run.append(task)  # Run all tasks in critical situation
            elif (
                pressure_level == MemoryPressure.HIGH
                and task.priority.value >= CleanupPriority.NORMAL.value
            ):
                tasks_to_run.append(task)
            elif (
                pressure_level == MemoryPressure.MODERATE
                and task.priority.value >= CleanupPriority.HIGH.value
            ):
                tasks_to_run.append(task)

        # Sort by priority, then by estimated benefit
        tasks_to_run.sort(key=lambda t: (-t.priority.value, -t.estimated_memory_freed_mb))

        for task in tasks_to_run:
            try:
                start_time = time.time()

                # Execute cleanup function
                freed_mb = task.cleanup_func()
                if freed_mb is None:
                    freed_mb = task.estimated_memory_freed_mb

                execution_time = time.time() - start_time

                # Update task statistics
                task.last_executed = time.time()
                task.execution_count += 1
                task.average_time = (
                    task.average_time * (task.execution_count - 1) + execution_time
                ) / task.execution_count

                total_freed += freed_mb
                executed_tasks.append(task.name)

                self.logger.info(
                    f"Cleanup task '{task.name}' freed {freed_mb:.1f}MB in {execution_time:.3f}s"
                )

            except Exception as e:
                context = ErrorContext(
                    operation="memory_cleanup",
                    additional_info={
                        "task_name": task.name,
                        "pressure_level": pressure_level.value,
                    },
                )

                self.error_handler.handle_error(
                    category=ErrorCategory.MEMORY,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Cleanup task '{task.name}' failed: {e}",
                    context=context,
                    exception=e,
                )

        if executed_tasks:
            self.logger.info(
                f"Executed {len(executed_tasks)} cleanup tasks, freed {total_freed:.1f}MB total"
            )

        self.stats["cleanup_executions"] += 1
        self.stats["bytes_freed"] += total_freed * 1024 * 1024

        return total_freed

    def force_cleanup(self, pressure_level: MemoryPressure = MemoryPressure.CRITICAL) -> float:
        """Force immediate cleanup."""
        self.logger.info(f"Forcing cleanup at {pressure_level.value} pressure level")
        return self.execute_cleanup(pressure_level)

    def _monitor_loop(self):
        """Main monitoring loop."""
        last_warning_time = 0
        last_critical_time = 0

        while self.is_monitoring:
            try:
                info = self.get_memory_info()
                self.memory_history.append(info)

                # Handle memory pressure
                if info.pressure_level == MemoryPressure.CRITICAL:
                    if time.time() - last_critical_time > 30:  # Don't spam
                        self.logger.critical(f"Critical memory pressure: {info.used_percent:.1f}%")
                        self.stats["critical_events"] += 1
                        last_critical_time = time.time()

                        # Execute critical cleanup
                        self.execute_cleanup(MemoryPressure.CRITICAL)

                        # Notify callbacks
                        for callback in self.cleanup_callbacks:
                            try:
                                callback(MemoryPressure.CRITICAL)
                            except Exception as e:
                                self.logger.error(f"Cleanup callback failed: {e}")

                elif info.pressure_level == MemoryPressure.HIGH:
                    if time.time() - last_warning_time > 60:  # Less frequent than critical
                        self.logger.warning(f"High memory pressure: {info.used_percent:.1f}%")
                        self.stats["memory_warnings"] += 1
                        last_warning_time = time.time()

                        # Execute high-priority cleanup
                        self.execute_cleanup(MemoryPressure.HIGH)

                        # Notify callbacks
                        for callback in self.cleanup_callbacks:
                            try:
                                callback(MemoryPressure.HIGH)
                            except Exception as e:
                                self.logger.error(f"Cleanup callback failed: {e}")

                elif info.pressure_level == MemoryPressure.MODERATE:
                    # Only execute preventive cleanup
                    self.execute_cleanup(MemoryPressure.MODERATE)

                time.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.check_interval)

    def get_tracked_objects_count(self) -> Dict[str, int]:
        """Get count of tracked objects by category."""
        return {category: len(objects) for category, objects in self.tracked_objects.items()}

    def get_large_allocations_summary(self) -> Dict[str, Any]:
        """Get summary of large allocations."""
        if not self.large_allocations:
            return {"total_count": 0, "total_size_mb": 0.0}

        total_size = sum(alloc["size_mb"] for alloc in self.large_allocations.values())
        oldest_allocation = min(self.large_allocations.values(), key=lambda x: x["timestamp"])

        return {
            "total_count": len(self.large_allocations),
            "total_size_mb": total_size,
            "oldest_timestamp": oldest_allocation["timestamp"],
            "allocations": list(self.large_allocations.values()),
        }

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_info = self.get_memory_info()

        # Calculate trends if we have history
        trends = {}
        if len(self.memory_history) > 1:
            recent = list(self.memory_history)[-10:]  # Last 10 measurements

            if len(recent) >= 2:
                usage_trend = recent[-1].used_percent - recent[0].used_percent
                trends["usage_trend_percent"] = usage_trend
                trends["trend_direction"] = (
                    "increasing"
                    if usage_trend > 1
                    else "decreasing" if usage_trend < -1 else "stable"
                )

        return {
            "current": {
                "total_mb": current_info.total_mb,
                "available_mb": current_info.available_mb,
                "used_percent": current_info.used_percent,
                "pressure_level": current_info.pressure_level.value,
                "process_rss_mb": current_info.process_rss_mb,
                "swap_used_mb": current_info.swap_used_mb,
            },
            "statistics": self.stats,
            "trends": trends,
            "tracked_objects": self.get_tracked_objects_count(),
            "large_allocations": self.get_large_allocations_summary(),
            "cleanup_tasks": {
                name: {
                    "priority": task.priority.value,
                    "estimated_freed_mb": task.estimated_memory_freed_mb,
                    "execution_count": task.execution_count,
                    "average_time": task.average_time,
                }
                for name, task in self.cleanup_tasks.items()
            },
        }


class MemoryGuard:
    """Context manager for memory-safe operations."""

    def __init__(
        self,
        memory_monitor: MemoryMonitor,
        max_memory_mb: Optional[float] = None,
        cleanup_on_exit: bool = True,
    ):
        self.memory_monitor = memory_monitor
        self.max_memory_mb = max_memory_mb
        self.cleanup_on_exit = cleanup_on_exit
        self.start_memory = None
        self.allocations_tracked = []

    def __enter__(self):
        self.start_memory = self.memory_monitor.get_memory_info()

        # Check if we can safely proceed
        if self.max_memory_mb and self.start_memory.used_percent > 90:
            raise MemoryError(
                f"Cannot proceed: memory usage at {self.start_memory.used_percent:.1f}%"
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_on_exit:
            # Force garbage collection
            gc.collect()

            # Clean up tracked allocations
            for allocation_id in self.allocations_tracked:
                self.memory_monitor.untrack_large_allocation(allocation_id)

        # Log memory usage change
        end_memory = self.memory_monitor.get_memory_info()
        memory_diff = end_memory.process_rss_mb - self.start_memory.process_rss_mb

        if abs(memory_diff) > 10:  # Log significant changes
            self.memory_monitor.logger.info(
                f"Memory change in guarded operation: {memory_diff:+.1f}MB "
                f"({self.start_memory.process_rss_mb:.1f} -> {end_memory.process_rss_mb:.1f}MB)"
            )

    def track_allocation(self, size_mb: float, description: str = "guarded_operation"):
        """Track allocation within this guard."""
        allocation_id = self.memory_monitor.track_large_allocation(size_mb, description)
        self.allocations_tracked.append(allocation_id)
        return allocation_id

    def check_memory_limit(self):
        """Check if memory limit is exceeded."""
        if self.max_memory_mb:
            current = self.memory_monitor.get_memory_info()
            if current.process_rss_mb > self.max_memory_mb:
                raise MemoryError(
                    f"Memory limit exceeded: {current.process_rss_mb:.1f}MB > {self.max_memory_mb:.1f}MB"
                )


# Default cleanup functions
def cleanup_garbage_collection() -> float:
    """Basic garbage collection cleanup."""
    before = psutil.Process().memory_info().rss

    # Multiple GC passes for better cleanup
    collected = 0
    for generation in range(3):
        collected += gc.collect()

    after = psutil.Process().memory_info().rss
    freed_mb = (before - after) / 1024 / 1024

    return max(0, freed_mb)


def cleanup_import_cache() -> float:
    """Clean up import cache."""
    before = psutil.Process().memory_info().rss

    # Clear import cache
    if hasattr(sys, "modules"):
        # Don't clear critical modules
        critical_modules = {"sys", "os", "gc", "logging", "__main__"}
        modules_to_clear = [
            name
            for name in list(sys.modules.keys())
            if not any(critical in name for critical in critical_modules)
            and not name.startswith("simpulse")  # Don't clear our own modules
        ]

        for module_name in modules_to_clear[:50]:  # Limit to avoid breaking things
            sys.modules.pop(module_name, None)

    after = psutil.Process().memory_info().rss
    freed_mb = (before - after) / 1024 / 1024

    return max(0, freed_mb)


def create_default_memory_monitor(error_handler: ErrorHandler) -> MemoryMonitor:
    """Create memory monitor with default cleanup tasks."""
    monitor = MemoryMonitor(error_handler)

    # Register default cleanup tasks
    monitor.register_cleanup_task(
        "garbage_collection",
        cleanup_garbage_collection,
        CleanupPriority.HIGH,
        estimated_memory_freed_mb=10.0,
    )

    monitor.register_cleanup_task(
        "import_cache_cleanup",
        cleanup_import_cache,
        CleanupPriority.NORMAL,
        estimated_memory_freed_mb=5.0,
    )

    return monitor


@contextmanager
def memory_limit(max_memory_mb: float, error_handler: ErrorHandler):
    """Context manager that enforces memory limits."""
    monitor = MemoryMonitor(error_handler)

    with MemoryGuard(monitor, max_memory_mb=max_memory_mb) as guard:
        yield guard

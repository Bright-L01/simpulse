"""
Comprehensive test suite for deliberate error scenarios in Simpulse.

Tests all error handling, recovery mechanisms, and graceful degradation
with deliberately broken inputs and failure conditions.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simpulse.core.comprehensive_monitor import AlertLevel, ComprehensiveMonitor, MetricType
from simpulse.core.graceful_degradation import (
    GracefulDegradationManager,
    OperationMode,
    PartialResult,
    ResultStatus,
)
from simpulse.core.memory_manager import CleanupPriority, MemoryGuard, MemoryMonitor, MemoryPressure
from simpulse.core.retry import (
    CircuitBreakerConfig,
    RetryConfig,
    RetryManager,
    RetryStrategy,
)
from simpulse.core.robust_file_handler import EncodingConfidence, FileType, RobustFileHandler
from simpulse.errors import (
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
    handle_file_error,
    handle_lean_error,
    handle_optimization_error,
)


class TestErrorHandling:
    """Test comprehensive error handling framework."""

    def setup_method(self):
        """Setup for each test."""
        self.error_handler = ErrorHandler()

    def test_file_not_found_error(self):
        """Test handling of file not found errors."""
        non_existent_path = Path("/non/existent/file.lean")
        exception = FileNotFoundError("File not found")

        error = handle_file_error(self.error_handler, "read_file", non_existent_path, exception)

        assert error.category == ErrorCategory.FILE_ACCESS
        assert error.severity == ErrorSeverity.MEDIUM
        assert "File operation failed" in error.message
        assert error.context.file_path == non_existent_path
        assert len(error.recovery_suggestions) > 0
        assert any("file path exists" in suggestion for suggestion in error.recovery_suggestions)

    def test_permission_error(self):
        """Test handling of permission errors."""
        test_path = Path("/root/secure_file.lean")
        exception = PermissionError("Permission denied")

        error = handle_file_error(self.error_handler, "write_file", test_path, exception)

        assert error.category == ErrorCategory.FILE_ACCESS
        assert error.severity == ErrorSeverity.HIGH
        assert "accessible" in error.recovery_suggestions[0].lower()

    def test_memory_error_handling(self):
        """Test handling of memory errors."""
        exception = MemoryError("Out of memory")

        error = handle_optimization_error(
            self.error_handler, "optimize_rules", exception, strategy="aggressive"
        )

        assert error.category == ErrorCategory.MEMORY
        assert error.severity == ErrorSeverity.HIGH
        assert not error.recoverable  # Memory errors are typically not recoverable

    def test_encoding_error_handling(self):
        """Test handling of encoding errors."""
        test_path = Path("test_file.lean")
        exception = UnicodeDecodeError("utf-8", b"\\xff", 0, 1, "invalid start byte")

        error = handle_file_error(self.error_handler, "read_text_file", test_path, exception)

        assert error.category == ErrorCategory.ENCODING
        assert error.severity == ErrorSeverity.MEDIUM
        assert any("encoding" in suggestion.lower() for suggestion in error.recovery_suggestions)

    def test_lean_execution_error(self):
        """Test handling of Lean execution errors."""
        exception = subprocess.CalledProcessError(1, "lean", "compilation failed")

        error = handle_lean_error(
            self.error_handler, "compile_lean_file", exception, file_path=Path("test.lean")
        )

        assert error.category == ErrorCategory.LEAN_EXECUTION
        assert error.severity == ErrorSeverity.HIGH
        assert any("lean" in suggestion.lower() for suggestion in error.recovery_suggestions)

    def test_error_recovery_plan(self):
        """Test generation of error recovery plans."""
        # Create multiple errors of different categories
        handle_file_error(self.error_handler, "op1", Path("file1"), FileNotFoundError())
        handle_file_error(self.error_handler, "op2", Path("file2"), PermissionError())
        handle_optimization_error(self.error_handler, "op3", MemoryError())

        recovery_plan = self.error_handler.get_recovery_plan()

        assert recovery_plan["estimated_success_rate"] > 0
        assert len(recovery_plan["immediate_actions"]) > 0
        assert "Check file permissions" in recovery_plan["immediate_actions"]

    def test_error_statistics(self):
        """Test error statistics collection."""
        # Generate various errors
        for i in range(5):
            handle_file_error(self.error_handler, f"op_{i}", Path(f"file_{i}"), FileNotFoundError())

        for i in range(3):
            handle_optimization_error(self.error_handler, f"opt_{i}", ValueError("test"))

        stats = self.error_handler.get_error_statistics()

        assert stats["total"] == 8
        assert stats["by_category"]["file_access"] == 5
        assert stats["by_category"]["optimization"] == 3
        assert stats["recovery_rate"] >= 0


class TestRetryMechanisms:
    """Test retry mechanisms with exponential backoff."""

    def setup_method(self):
        """Setup for each test."""
        self.error_handler = ErrorHandler()
        self.retry_manager = RetryManager(self.error_handler)

    def test_exponential_backoff_calculation(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0, backoff_factor=2.0, strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )

        # Test delay calculation for different attempts
        delay1 = self.retry_manager._calculate_delay(1, config, "test_op")
        delay2 = self.retry_manager._calculate_delay(2, config, "test_op")
        delay3 = self.retry_manager._calculate_delay(3, config, "test_op")

        assert delay1 == 1.0  # base_delay * 2^0
        assert delay2 == 2.0  # base_delay * 2^1
        assert delay3 == 4.0  # base_delay * 2^2

    def test_retry_with_eventual_success(self):
        """Test retry mechanism with eventual success."""
        call_count = 0

        def failing_then_succeeding_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        config = RetryConfig(max_attempts=5, base_delay=0.1)

        result = self.retry_manager.execute_with_retry(
            failing_then_succeeding_func, (), {}, "test_operation", config
        )

        assert result == "success"
        assert call_count == 3

    def test_retry_with_permanent_failure(self):
        """Test retry mechanism with permanent failure."""
        call_count = 0

        def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent failure")

        config = RetryConfig(max_attempts=3, base_delay=0.1)

        with pytest.raises(ValueError, match="Permanent failure"):
            self.retry_manager.execute_with_retry(
                always_failing_func, (), {}, "test_operation", config
            )

        assert call_count == 3

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after repeated failures."""

        def always_failing_func():
            raise ConnectionError("Service unavailable")

        config = RetryConfig(max_attempts=1, base_delay=0.1)
        circuit_config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)

        # First few failures should trigger circuit breaker
        for _ in range(3):
            try:
                self.retry_manager.execute_with_retry(
                    always_failing_func, (), {}, "circuit_test", config, circuit_config
                )
            except Exception:
                pass  # Expected failure for circuit breaker test

        # Circuit should now be open
        circuit_status = self.retry_manager.get_circuit_status("circuit_test")
        assert circuit_status["state"] == "open"

    def test_adaptive_delay_strategy(self):
        """Test adaptive delay strategy based on operation history."""
        config = RetryConfig(strategy=RetryStrategy.ADAPTIVE, base_delay=1.0)

        # Record some operation history
        self.retry_manager._record_failure("adaptive_test", 1, ValueError("test"))
        self.retry_manager._record_failure("adaptive_test", 2, ValueError("test"))

        # Delay should be adjusted based on failure rate
        delay = self.retry_manager._calculate_delay(1, config, "adaptive_test")
        assert delay > 1.0  # Should be higher due to failure history

    def test_timeout_handling(self):
        """Test timeout handling in retry mechanism."""

        def slow_function():
            time.sleep(2.0)
            return "slow_result"

        config = RetryConfig(max_attempts=2, timeout_per_attempt=0.5, base_delay=0.1)

        with pytest.raises(TimeoutError):
            self.retry_manager.execute_with_retry(slow_function, (), {}, "timeout_test", config)


class TestGracefulDegradation:
    """Test graceful degradation and partial results."""

    def setup_method(self):
        """Setup for each test."""
        self.error_handler = ErrorHandler()
        self.degradation_manager = GracefulDegradationManager(self.error_handler)

    def test_fallback_execution(self):
        """Test execution with fallback strategy."""

        def primary_func():
            raise ConnectionError("Service unavailable")

        def fallback_func():
            return "fallback_result"

        self.degradation_manager.register_fallback("test_operation", fallback_func)

        result = self.degradation_manager.execute_with_degradation("test_operation", primary_func)

        assert result.data == "fallback_result"
        assert result.status == ResultStatus.FALLBACK
        assert result.fallback_used
        assert result.success_rate == 0.7

    def test_batch_processing_with_partial_failures(self):
        """Test batch processing with some failures."""
        items = list(range(10))

        def item_processor(item):
            if item % 3 == 0:  # Fail every third item
                raise ValueError(f"Failed on item {item}")
            return item * 2

        result = self.degradation_manager.batch_execute_with_degradation(
            "batch_test", items, item_processor, min_success_rate=0.5
        )

        assert result.status == ResultStatus.PARTIAL
        assert result.success_rate > 0.5
        assert result.metadata["successful_items"] == 7  # 10 - 3 failures
        assert result.metadata["failed_items"] == 3

    def test_operation_mode_degradation(self):
        """Test automatic operation mode degradation."""
        # Simulate memory error triggering degradation
        exception = MemoryError("Out of memory")
        self.degradation_manager._handle_operation_error("memory_intensive_op", exception)

        assert self.degradation_manager.current_mode == OperationMode.REDUCED

    def test_cached_results(self):
        """Test cached result retrieval."""

        def expensive_operation():
            return "expensive_result"

        # First execution
        result1 = self.degradation_manager.execute_with_degradation(
            "cached_op", expensive_operation, cache_key="test_cache"
        )

        # Second execution should use cache
        result2 = self.degradation_manager.execute_with_degradation(
            "cached_op", lambda: "should_not_be_called", cache_key="test_cache"
        )

        assert result1.data == "expensive_result"
        assert result2.data == "expensive_result"
        assert result2.cache_hit

    def test_mode_upgrade_and_downgrade(self):
        """Test operation mode upgrade and downgrade."""
        # Start in full mode
        assert self.degradation_manager.current_mode == OperationMode.FULL

        # Force degradation
        self.degradation_manager._degrade_to_mode(OperationMode.MINIMAL, "Test degradation")
        assert self.degradation_manager.current_mode == OperationMode.MINIMAL

        # Try to upgrade (should work in test environment)
        upgraded = self.degradation_manager.upgrade_mode(OperationMode.REDUCED, "Test upgrade")
        assert upgraded
        assert self.degradation_manager.current_mode == OperationMode.REDUCED


class TestRobustFileHandling:
    """Test robust file handling with malformed inputs."""

    def setup_method(self):
        """Setup for each test."""
        self.error_handler = ErrorHandler()
        self.degradation_manager = GracefulDegradationManager(self.error_handler)
        self.file_handler = RobustFileHandler(self.error_handler, self.degradation_manager)
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_malformed_lean_file(self):
        """Test handling of malformed Lean file."""
        malformed_file = self.temp_dir / "malformed.lean"

        # Create file with mixed encoding and invalid syntax
        with open(malformed_file, "wb") as f:
            f.write(b"theorem test : 1 = 1 := \\xff\\xfe invalid_bytes \\x00\\x01")

        file_info = self.file_handler.analyze_file(malformed_file)

        assert file_info.file_type == FileType.LEAN
        assert file_info.potential_corruption
        assert file_info.encoding_confidence in [EncodingConfidence.LOW, EncodingConfidence.GUESS]

    def test_binary_data_in_text_file(self):
        """Test detection of binary data in text file."""
        binary_text_file = self.temp_dir / "binary_text.lean"

        with open(binary_text_file, "wb") as f:
            f.write(b"theorem valid_start := \\x00\\x01\\x02 invalid_binary_data")

        file_info = self.file_handler.analyze_file(binary_text_file)

        assert file_info.potential_corruption
        assert (
            "corruption" in str(file_info.potential_corruption).lower()
            or file_info.potential_corruption is True
        )

    def test_encoding_detection_and_fallback(self):
        """Test encoding detection with fallbacks."""
        # Create file with Latin-1 encoding
        latin1_file = self.temp_dir / "latin1.lean"
        content = 'theorem café : String := "résumé"'

        with open(latin1_file, "w", encoding="latin-1") as f:
            f.write(content)

        result = self.file_handler.read_file_robust(latin1_file)

        assert result.status in [ResultStatus.COMPLETE, ResultStatus.DEGRADED]
        assert result.data.success
        assert result.data.content is not None

    def test_large_file_streaming(self):
        """Test streaming of large files."""
        large_file = self.temp_dir / "large.lean"

        # Create a file larger than the default limit
        large_content = "-- Large file\\n" * 100000  # ~1.5MB
        with open(large_file, "w") as f:
            f.write(large_content)

        result = self.file_handler.read_file_robust(
            large_file, max_size_mb=1, enable_streaming=True  # Force streaming
        )

        assert result.status == ResultStatus.PARTIAL
        assert result.data.partial_read
        assert hasattr(result.data.content, "__iter__")  # Should be a generator

    def test_corrupted_file_recovery(self):
        """Test recovery from corrupted file."""
        corrupted_file = self.temp_dir / "corrupted.lean"

        # Create file with embedded null bytes
        with open(corrupted_file, "wb") as f:
            f.write(b"theorem start := \\x00\\x00 null bytes \\x00 more content")

        result = self.file_handler.read_file_robust(corrupted_file)

        # Should still read something, even if degraded
        assert result.status in [ResultStatus.DEGRADED, ResultStatus.PARTIAL]
        assert len(result.data.warnings) > 0

    def test_atomic_file_writing(self):
        """Test atomic file writing with backup."""
        test_file = self.temp_dir / "atomic_test.lean"
        original_content = "original content"
        new_content = "new content"

        # Write original content
        with open(test_file, "w") as f:
            f.write(original_content)

        # Atomic write with backup
        result = self.file_handler.write_file_robust(
            test_file, new_content, backup=True, atomic=True
        )

        assert result.status == ResultStatus.COMPLETE
        assert result.data  # Success

        # Check content was updated
        with open(test_file) as f:
            assert f.read() == new_content

        # Check backup was created
        backup_files = list(self.temp_dir.glob("atomic_test.lean.backup_*"))
        assert len(backup_files) > 0

    def test_file_permission_handling(self):
        """Test handling of file permission issues."""
        if os.name == "nt":  # Skip on Windows
            pytest.skip("File permission tests not reliable on Windows")

        readonly_file = self.temp_dir / "readonly.lean"

        # Create file and make it readonly
        with open(readonly_file, "w") as f:
            f.write("theorem test := sorry")

        os.chmod(readonly_file, 0o444)  # Read-only

        # Try to write to readonly file
        result = self.file_handler.write_file_robust(readonly_file, "new content")

        assert result.status == ResultStatus.FAILED
        assert len(result.errors) > 0


class TestMemoryManagement:
    """Test memory management and resource monitoring."""

    def setup_method(self):
        """Setup for each test."""
        self.error_handler = ErrorHandler()
        self.memory_monitor = MemoryMonitor(self.error_handler)

    def test_memory_info_collection(self):
        """Test memory information collection."""
        info = self.memory_monitor.get_memory_info()

        assert info.total_mb > 0
        assert info.used_percent >= 0
        assert info.used_percent <= 100
        assert info.pressure_level in list(MemoryPressure)

    def test_cleanup_task_registration(self):
        """Test cleanup task registration and execution."""
        cleanup_called = False

        def test_cleanup():
            nonlocal cleanup_called
            cleanup_called = True
            return 10.0  # Freed 10MB

        self.memory_monitor.register_cleanup_task(
            "test_cleanup", test_cleanup, CleanupPriority.HIGH, estimated_memory_freed_mb=10.0
        )

        freed_mb = self.memory_monitor.execute_cleanup(MemoryPressure.HIGH)

        assert cleanup_called
        assert freed_mb == 10.0

    def test_memory_guard_context_manager(self):
        """Test memory guard context manager."""
        with MemoryGuard(self.memory_monitor, max_memory_mb=100000) as guard:
            # Simulate allocation tracking
            allocation_id = guard.track_allocation(50.0, "test_allocation")
            assert allocation_id in guard.allocations_tracked

        # Allocation should be cleaned up after context exit
        assert allocation_id not in self.memory_monitor.large_allocations

    def test_memory_pressure_detection(self):
        """Test memory pressure level detection."""
        # Mock high memory usage
        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = Mock(
                percent=95.0,
                total=8 * 1024 * 1024 * 1024,
                available=1 * 1024 * 1024 * 1024,
                used=7 * 1024 * 1024 * 1024,
            )

            info = self.memory_monitor.get_memory_info()
            assert info.pressure_level == MemoryPressure.CRITICAL

    def test_large_allocation_tracking(self):
        """Test tracking of large allocations."""
        allocation_id = self.memory_monitor.track_large_allocation(500.0, "large_array")

        summary = self.memory_monitor.get_large_allocations_summary()

        assert summary["total_count"] == 1
        assert summary["total_size_mb"] == 500.0

        self.memory_monitor.untrack_large_allocation(allocation_id)

        summary = self.memory_monitor.get_large_allocations_summary()
        assert summary["total_count"] == 0


class TestComprehensiveMonitoring:
    """Test comprehensive monitoring and alerting."""

    def setup_method(self):
        """Setup for each test."""
        self.error_handler = ErrorHandler()
        self.temp_db = Path(tempfile.mkdtemp()) / "test.db"
        self.monitor = ComprehensiveMonitor(self.error_handler, db_path=self.temp_db)

    def teardown_method(self):
        """Cleanup after each test."""
        self.monitor.stop_monitoring()
        if self.temp_db.exists():
            self.temp_db.unlink()

    def test_metric_recording(self):
        """Test metric recording and retrieval."""
        self.monitor.record_metric("test.counter", 1.0, MetricType.COUNTER)
        self.monitor.record_metric("test.gauge", 50.0, MetricType.GAUGE, unit="percent")
        self.monitor.record_metric("test.histogram", 2.5, MetricType.HISTOGRAM)

        summary = self.monitor.get_metrics_summary()

        assert "test.counter" in summary["counters"]
        assert summary["counters"]["test.counter"] == 1.0
        assert summary["gauges"]["test.gauge"] == 50.0
        assert "test.histogram" in summary["histograms"]

    def test_alert_creation_and_resolution(self):
        """Test alert creation and resolution."""
        self.monitor.create_alert(
            AlertLevel.WARNING, "Test alert message", "test_source", {"detail": "test_detail"}
        )

        health = self.monitor.get_system_health()
        assert health["active_alerts"] == 1

        self.monitor.resolve_alert("test_source", "Test alert message")

        # Check that alert is marked as resolved
        alert_id = f"test_source_{hash('Test alert message')}"
        assert self.monitor.active_alerts[alert_id].resolved

    def test_operation_performance_tracking(self):
        """Test operation performance tracking."""
        # Record successful operations
        for i in range(5):
            self.monitor.record_operation("test_operation", 1.0 + i * 0.1, True)

        # Record one failure
        self.monitor.record_operation("test_operation", 2.0, False)

        health = self.monitor.get_system_health()
        perf_summary = health["performance_summary"]["test_operation"]

        assert perf_summary["total_count"] == 6
        assert perf_summary["error_rate"] == 1 / 6  # One failure out of 6
        assert perf_summary["avg_duration"] > 1.0

    def test_health_check_registration(self):
        """Test health check registration and execution."""
        check_called = False

        def test_health_check():
            nonlocal check_called
            check_called = True
            return True

        self.monitor.register_health_check("test_health_check", test_health_check, interval=1.0)

        # Manually run health checks
        self.monitor._run_health_checks(time.time())

        assert check_called
        assert self.monitor.health_checks["test_health_check"].last_result

    def test_automatic_alert_thresholds(self):
        """Test automatic alerts based on thresholds."""
        # Mock high memory usage to trigger alert
        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = Mock(percent=96.0)

            self.monitor._collect_system_metrics()

            # Should have created a critical memory alert
            health = self.monitor.get_system_health()
            assert health["critical_alerts"] > 0

    def test_monitoring_data_export(self):
        """Test export of monitoring data."""
        # Record some data
        self.monitor.record_metric("export.test", 42.0, MetricType.GAUGE)
        self.monitor.create_alert(AlertLevel.INFO, "Export test", "test")

        export_path = Path(tempfile.mkdtemp()) / "export.json"

        try:
            success = self.monitor.export_monitoring_data(export_path)
            assert success
            assert export_path.exists()

            # Verify exported data
            with open(export_path) as f:
                data = json.load(f)

            assert "system_health" in data
            assert "metrics_summary" in data
            assert "active_alerts" in data

        finally:
            if export_path.exists():
                export_path.unlink()


class TestIntegratedErrorScenarios:
    """Test integrated error scenarios combining multiple systems."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.error_handler = ErrorHandler()
        self.degradation_manager = GracefulDegradationManager(self.error_handler)
        self.file_handler = RobustFileHandler(self.error_handler, self.degradation_manager)
        self.memory_monitor = MemoryMonitor(self.error_handler)
        self.retry_manager = RetryManager(self.error_handler)
        self.monitor = ComprehensiveMonitor(self.error_handler)

    def teardown_method(self):
        """Cleanup after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.monitor.stop_monitoring()

    def test_cascading_failure_recovery(self):
        """Test recovery from cascading failures."""
        # Create a chain of failures: file error -> memory pressure -> degradation

        # 1. File access failure
        bad_file = self.temp_dir / "nonexistent.lean"

        def file_operation():
            result = self.file_handler.read_file_robust(bad_file)
            if result.status == ResultStatus.FAILED:
                raise FileNotFoundError(f"Could not read {bad_file}")
            return result

        # 2. Register fallback
        def fallback_operation():
            return PartialResult(data="fallback_content", status=ResultStatus.FALLBACK)

        self.degradation_manager.register_fallback("file_operation", fallback_operation)

        # 3. Execute with full error handling
        result = self.degradation_manager.execute_with_degradation("file_operation", file_operation)

        assert result.status == ResultStatus.FALLBACK
        assert result.fallback_used
        assert result.data == "fallback_content"

    def test_memory_exhaustion_scenario(self):
        """Test handling of memory exhaustion scenario."""

        # Simulate memory exhaustion
        def memory_intensive_operation():
            # Simulate memory allocation
            self.memory_monitor.track_large_allocation(1000.0, "test_allocation")
            raise MemoryError("Simulated memory exhaustion")

        # Setup cleanup
        cleanup_executed = False

        def emergency_cleanup():
            nonlocal cleanup_executed
            cleanup_executed = True
            return 500.0  # Freed 500MB

        self.memory_monitor.register_cleanup_task(
            "emergency_cleanup", emergency_cleanup, CleanupPriority.CRITICAL
        )

        # Execute with retry and memory handling
        config = RetryConfig(max_attempts=3, base_delay=0.1)

        with pytest.raises(MemoryError):
            self.retry_manager.execute_with_retry(
                memory_intensive_operation, (), {}, "memory_test", config
            )

        # Force cleanup
        freed = self.memory_monitor.execute_cleanup(MemoryPressure.CRITICAL)
        assert freed == 500.0
        assert cleanup_executed

    def test_network_failure_with_offline_mode(self):
        """Test network failure with graceful degradation to offline mode."""
        network_call_count = 0

        def network_operation():
            nonlocal network_call_count
            network_call_count += 1
            raise ConnectionError("Network unreachable")

        def offline_fallback():
            return "offline_result"

        self.degradation_manager.register_fallback("network_op", offline_fallback)

        # Should trigger degradation to offline mode
        result = self.degradation_manager.execute_with_degradation("network_op", network_operation)

        assert result.status == ResultStatus.FALLBACK
        assert result.data == "offline_result"
        # Should have degraded to offline mode due to network error
        assert self.degradation_manager.current_mode in [
            OperationMode.OFFLINE,
            OperationMode.REDUCED,
        ]

    def test_complete_system_stress_test(self):
        """Test complete system under stress conditions."""
        # Start monitoring
        self.monitor.start_monitoring()

        # Create multiple types of failures simultaneously
        failures = []

        # File system stress
        for i in range(10):
            bad_file = self.temp_dir / f"stress_{i}.lean"
            try:
                self.file_handler.read_file_robust(bad_file)
            except Exception as e:
                failures.append(("file", e))

        # Memory stress simulation
        for i in range(5):
            self.memory_monitor.track_large_allocation(100.0, f"stress_allocation_{i}")

        # Network stress simulation
        def create_failing_network_op(index):
            def failing_network_op():
                raise ConnectionError(f"Network failure {index}")

            return failing_network_op

        for i in range(3):
            try:
                self.degradation_manager.execute_with_degradation(
                    f"network_stress_{i}", create_failing_network_op(i)
                )
            except Exception as e:
                failures.append(("network", e))

        # Check system health after stress
        health = self.monitor.get_system_health()
        error_summary = self.error_handler.get_user_friendly_summary()

        # System should still be functional
        assert health["overall_status"] in ["healthy", "warning", "critical"]
        assert error_summary["status"] in ["success", "warning", "error"]

        # Should have recorded the failures
        assert len(failures) > 0
        assert error_summary["total_errors"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

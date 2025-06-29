"""
Tests for performance monitoring functionality.
"""

import asyncio
import json
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from simpulse.monitoring.performance_monitor import (
    PerformanceMonitor, ResourceUsage, PerformanceAlert, AlertSeverity
)


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor."""
    
    @pytest.fixture
    def monitor(self):
        """Create a PerformanceMonitor instance for testing."""
        return PerformanceMonitor(
            collection_interval=1.0,  # 1 second for testing
            history_size=100
        )
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping monitoring."""
        # Start monitoring
        await monitor.start()
        assert monitor.is_running == True
        
        # Let it collect some data
        await asyncio.sleep(2)
        
        # Stop monitoring
        await monitor.stop()
        assert monitor.is_running == False
        
        # Should have collected some metrics
        assert len(monitor.metrics_history) > 0
    
    def test_collect_resource_usage(self, monitor):
        """Test resource usage collection."""
        with patch('psutil.cpu_percent', return_value=45.5):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value = Mock(percent=60.0, used=1024*1024*1024)
                
                with patch('psutil.disk_usage') as mock_disk:
                    mock_disk.return_value = Mock(percent=75.0)
                    
                    usage = monitor._collect_resource_usage()
                    
                    assert isinstance(usage, ResourceUsage)
                    assert usage.cpu_percent == 45.5
                    assert usage.memory_percent == 60.0
                    assert usage.memory_mb == 1024.0
                    assert usage.disk_percent == 75.0
    
    def test_check_alerts(self, monitor):
        """Test alert checking."""
        # Set alert thresholds
        monitor.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 90.0,
            "disk_percent": 95.0
        }
        
        # Test high CPU usage
        usage = ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=85.0,
            memory_percent=50.0,
            memory_mb=1024.0,
            disk_percent=50.0
        )
        
        alerts = monitor._check_alerts(usage)
        
        assert len(alerts) == 1
        assert alerts[0].metric == "cpu_percent"
        assert alerts[0].severity == AlertSeverity.WARNING
        assert alerts[0].current_value == 85.0
    
    def test_multiple_alerts(self, monitor):
        """Test multiple simultaneous alerts."""
        monitor.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 70.0,
            "disk_percent": 85.0
        }
        
        # All metrics exceeding thresholds
        usage = ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=90.0,
            memory_percent=85.0,
            memory_mb=2048.0,
            disk_percent=95.0
        )
        
        alerts = monitor._check_alerts(usage)
        
        assert len(alerts) == 3
        metric_names = [alert.metric for alert in alerts]
        assert "cpu_percent" in metric_names
        assert "memory_percent" in metric_names
        assert "disk_percent" in metric_names
    
    def test_alert_severity_levels(self, monitor):
        """Test different alert severity levels."""
        monitor.alert_thresholds = {
            "cpu_percent": 70.0,
            "critical_cpu_percent": 90.0
        }
        
        # Warning level
        usage1 = ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=75.0,
            memory_percent=50.0,
            memory_mb=1024.0,
            disk_percent=50.0
        )
        
        alerts1 = monitor._check_alerts(usage1)
        assert len(alerts1) == 1
        assert alerts1[0].severity == AlertSeverity.WARNING
        
        # Critical level
        usage2 = ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=95.0,
            memory_percent=50.0,
            memory_mb=1024.0,
            disk_percent=50.0
        )
        
        alerts2 = monitor._check_alerts(usage2)
        assert len(alerts2) >= 1
        # Should have at least one critical alert
        critical_alerts = [a for a in alerts2 if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, monitor):
        """Test performance metric tracking over time."""
        # Add some metrics
        for i in range(5):
            usage = ResourceUsage(
                timestamp=datetime.now() + timedelta(seconds=i),
                cpu_percent=40.0 + i * 5,
                memory_percent=50.0 + i * 2,
                memory_mb=1024.0,
                disk_percent=60.0
            )
            monitor.metrics_history.append(usage)
        
        # Get statistics
        stats = monitor.get_performance_stats()
        
        assert "average_cpu" in stats
        assert "peak_cpu" in stats
        assert "average_memory" in stats
        assert "peak_memory" in stats
        
        # Verify calculations
        assert stats["average_cpu"] == 50.0  # (40+45+50+55+60)/5
        assert stats["peak_cpu"] == 60.0
        assert stats["average_memory"] == 54.0  # (50+52+54+56+58)/5
        assert stats["peak_memory"] == 58.0
    
    def test_export_metrics(self, monitor, temp_dir):
        """Test exporting metrics to file."""
        # Add sample metrics
        for i in range(3):
            usage = ResourceUsage(
                timestamp=datetime.now() + timedelta(seconds=i),
                cpu_percent=45.0,
                memory_percent=60.0,
                memory_mb=1024.0,
                disk_percent=70.0
            )
            monitor.metrics_history.append(usage)
        
        # Export to JSON
        output_file = temp_dir / "metrics.json"
        result = monitor.export_metrics(output_file, format="json")
        
        assert result == True
        assert output_file.exists()
        
        # Verify content
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert "metrics" in data
        assert len(data["metrics"]) == 3
        assert "statistics" in data
    
    def test_export_csv(self, monitor, temp_dir):
        """Test exporting metrics to CSV."""
        # Add sample metrics
        usage = ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=45.0,
            memory_percent=60.0,
            memory_mb=1024.0,
            disk_percent=70.0
        )
        monitor.metrics_history.append(usage)
        
        # Export to CSV
        output_file = temp_dir / "metrics.csv"
        result = monitor.export_metrics(output_file, format="csv")
        
        assert result == True
        assert output_file.exists()
        
        # Verify content
        content = output_file.read_text()
        assert "timestamp,cpu_percent,memory_percent" in content
        assert "45.0" in content
        assert "60.0" in content
    
    @pytest.mark.asyncio
    async def test_alert_callbacks(self, monitor):
        """Test alert callback functionality."""
        callback_called = False
        alert_received = None
        
        def alert_callback(alert: PerformanceAlert):
            nonlocal callback_called, alert_received
            callback_called = True
            alert_received = alert
        
        # Register callback
        monitor.register_alert_callback(alert_callback)
        
        # Set low threshold to trigger alert
        monitor.alert_thresholds = {"cpu_percent": 10.0}
        
        # Simulate high CPU usage
        with patch.object(monitor, '_collect_resource_usage') as mock_collect:
            mock_collect.return_value = ResourceUsage(
                timestamp=datetime.now(),
                cpu_percent=90.0,
                memory_percent=50.0,
                memory_mb=1024.0,
                disk_percent=50.0
            )
            
            # Start monitoring briefly
            await monitor.start()
            await asyncio.sleep(1.5)
            await monitor.stop()
        
        # Verify callback was called
        assert callback_called == True
        assert alert_received is not None
        assert alert_received.metric == "cpu_percent"
        assert alert_received.current_value == 90.0
    
    def test_metrics_aggregation(self, monitor):
        """Test metrics aggregation over time windows."""
        # Add metrics over time
        base_time = datetime.now()
        for i in range(60):  # 1 minute of data
            usage = ResourceUsage(
                timestamp=base_time + timedelta(seconds=i),
                cpu_percent=50.0 + (i % 10),  # Oscillating CPU
                memory_percent=60.0,
                memory_mb=1024.0,
                disk_percent=70.0
            )
            monitor.metrics_history.append(usage)
        
        # Get aggregated stats
        stats = monitor.get_aggregated_stats(window_seconds=30)
        
        assert "windows" in stats
        assert len(stats["windows"]) == 2  # Two 30-second windows
        
        # Check window stats
        for window in stats["windows"]:
            assert "start_time" in window
            assert "end_time" in window
            assert "average_cpu" in window
            assert "peak_cpu" in window
    
    def test_clear_old_metrics(self, monitor):
        """Test clearing old metrics from history."""
        monitor.history_size = 10
        
        # Add more metrics than history size
        for i in range(20):
            usage = ResourceUsage(
                timestamp=datetime.now() + timedelta(seconds=i),
                cpu_percent=50.0,
                memory_percent=60.0,
                memory_mb=1024.0,
                disk_percent=70.0
            )
            monitor.metrics_history.append(usage)
        
        # Clear old metrics
        monitor._clear_old_metrics()
        
        # Should only keep the most recent metrics
        assert len(monitor.metrics_history) == 10


class TestResourceUsage:
    """Test suite for ResourceUsage dataclass."""
    
    def test_resource_usage_creation(self):
        """Test creating ResourceUsage instances."""
        usage = ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=45.5,
            memory_percent=60.0,
            memory_mb=1024.0,
            disk_percent=75.0,
            network_mb_sent=10.5,
            network_mb_recv=20.3
        )
        
        assert usage.cpu_percent == 45.5
        assert usage.memory_percent == 60.0
        assert usage.memory_mb == 1024.0
        assert usage.disk_percent == 75.0
        assert usage.network_mb_sent == 10.5
        assert usage.network_mb_recv == 20.3
    
    def test_resource_usage_to_dict(self):
        """Test converting ResourceUsage to dictionary."""
        timestamp = datetime.now()
        usage = ResourceUsage(
            timestamp=timestamp,
            cpu_percent=45.5,
            memory_percent=60.0,
            memory_mb=1024.0,
            disk_percent=75.0
        )
        
        data = usage.to_dict()
        
        assert data["cpu_percent"] == 45.5
        assert data["memory_percent"] == 60.0
        assert data["memory_mb"] == 1024.0
        assert data["disk_percent"] == 75.0
        assert "timestamp" in data


class TestPerformanceAlert:
    """Test suite for PerformanceAlert dataclass."""
    
    def test_alert_creation(self):
        """Test creating PerformanceAlert instances."""
        alert = PerformanceAlert(
            timestamp=datetime.now(),
            metric="cpu_percent",
            current_value=85.0,
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            message="CPU usage high: 85.0% (threshold: 80.0%)"
        )
        
        assert alert.metric == "cpu_percent"
        assert alert.current_value == 85.0
        assert alert.threshold == 80.0
        assert alert.severity == AlertSeverity.WARNING
        assert "CPU usage high" in alert.message
    
    def test_alert_severity_enum(self):
        """Test AlertSeverity enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"


@pytest.mark.integration
class TestPerformanceMonitorIntegration:
    """Integration tests for performance monitor."""
    
    @pytest.mark.asyncio
    async def test_real_system_monitoring(self):
        """Test monitoring real system metrics."""
        monitor = PerformanceMonitor(collection_interval=0.5)
        
        # Monitor for a short time
        await monitor.start()
        await asyncio.sleep(2)
        await monitor.stop()
        
        # Should have collected real metrics
        assert len(monitor.metrics_history) > 0
        
        # Verify metrics are reasonable
        for usage in monitor.metrics_history:
            assert 0 <= usage.cpu_percent <= 100
            assert 0 <= usage.memory_percent <= 100
            assert usage.memory_mb > 0
            assert 0 <= usage.disk_percent <= 100
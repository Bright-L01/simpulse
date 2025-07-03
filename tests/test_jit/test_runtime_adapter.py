"""
Tests for JIT runtime adapter.
"""

import json
import os
import tempfile
import time

import pytest

from simpulse.jit import AdapterConfig, RuleStatistics, RuntimeAdapter


class TestRuleStatistics:
    """Test RuleStatistics class."""

    def test_success_rate(self):
        """Test success rate calculation."""
        stats = RuleStatistics("test_rule", attempts=10, successes=7)
        assert stats.success_rate == 0.7

        # Test with no attempts
        stats_empty = RuleStatistics("empty_rule")
        assert stats_empty.success_rate == 0.0

    def test_avg_time(self):
        """Test average time calculation."""
        stats = RuleStatistics("test_rule", attempts=5, total_time=0.5)
        assert stats.avg_time == 0.1

        # Test with no attempts
        stats_empty = RuleStatistics("empty_rule")
        assert stats_empty.avg_time == 0.0

    def test_apply_decay(self):
        """Test exponential decay application."""
        stats = RuleStatistics(
            "test_rule",
            attempts=100,
            successes=80,
            total_time=1.0,
            last_used=time.time() - 60,  # 1 minute ago
        )

        original_attempts = stats.attempts
        stats.apply_decay(0.9, time.time())

        # Should have decayed
        assert stats.attempts < original_attempts
        assert stats.successes < 80


class TestRuntimeAdapter:
    """Test RuntimeAdapter class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def adapter(self, temp_dir):
        """Create adapter with test configuration."""
        config = AdapterConfig(
            stats_file=os.path.join(temp_dir, "stats.json"),
            priority_file=os.path.join(temp_dir, "priorities.json"),
            log_file=os.path.join(temp_dir, "adapter.log"),
            adaptation_interval=10,
            min_samples=5,
        )
        return RuntimeAdapter(config)

    def test_update_statistics(self, adapter):
        """Test updating rule statistics."""
        # Update statistics
        adapter.update_statistics("add_zero", True, 0.001)
        adapter.update_statistics("add_zero", True, 0.002)
        adapter.update_statistics("add_zero", False, 0.003)

        # Check statistics
        assert "add_zero" in adapter.statistics
        stats = adapter.statistics["add_zero"]
        assert stats.attempts == 3
        assert stats.successes == 2
        assert stats.total_time == 0.006

    def test_calculate_priority(self, adapter):
        """Test priority calculation."""
        # Create rule with good performance
        good_stats = RuleStatistics("good_rule", attempts=20, successes=18, total_time=0.02)

        # Create rule with poor performance
        poor_stats = RuleStatistics("poor_rule", attempts=20, successes=2, total_time=0.5)

        good_priority = adapter.calculate_priority(good_stats)
        poor_priority = adapter.calculate_priority(poor_stats)

        # Good rule should have higher priority
        assert good_priority > poor_priority

        # Check priority bounds
        assert adapter.config.priority_range[0] <= good_priority <= adapter.config.priority_range[1]
        assert adapter.config.priority_range[0] <= poor_priority <= adapter.config.priority_range[1]

    def test_optimize_priorities(self, adapter):
        """Test priority optimization."""
        import time

        # Add statistics for several rules
        rules = [
            ("add_zero", 50, 45, 0.05),  # High success, fast
            ("mul_one", 40, 35, 0.04),  # High success, fast
            ("complex_rule", 30, 5, 0.5),  # Low success, slow
            ("rare_rule", 5, 3, 0.01),  # Too few samples
        ]

        current_time = time.time()
        for rule, attempts, successes, total_time in rules:
            adapter.statistics[rule] = RuleStatistics(
                rule,
                attempts=attempts,
                successes=successes,
                total_time=total_time,
                last_used=current_time,
            )

        # Optimize
        priorities = adapter.optimize_priorities()

        # Check priorities assigned
        assert "add_zero" in priorities
        assert "mul_one" in priorities
        assert "complex_rule" in priorities
        assert "rare_rule" not in priorities  # Too few samples

        # Check ordering
        assert priorities["add_zero"] > priorities["complex_rule"]
        assert priorities["mul_one"] > priorities["complex_rule"]

    def test_save_load_statistics(self, adapter):
        """Test saving and loading statistics."""
        # Add some statistics
        adapter.update_statistics("test_rule", True, 0.001)
        adapter.update_statistics("test_rule", False, 0.002)

        # Save
        adapter.save_statistics()

        # Create new adapter and load
        new_adapter = RuntimeAdapter(adapter.config)
        new_adapter.load_statistics()

        # Check loaded statistics
        assert "test_rule" in new_adapter.statistics
        stats = new_adapter.statistics["test_rule"]
        assert stats.attempts == 2
        assert stats.successes == 1

    def test_save_load_priorities(self, adapter):
        """Test saving and loading priorities."""
        priorities = {"add_zero": 2000, "mul_one": 1800, "complex_rule": 500}

        # Save
        adapter.save_priorities(priorities)

        # Load
        loaded = adapter.load_priorities()
        assert loaded == priorities

    def test_adaptation_interval(self, adapter):
        """Test that optimization happens at correct intervals."""
        adapter.config.adaptation_interval = 5

        # Add statistics but not enough for optimization
        for i in range(4):
            adapter.update_statistics(f"rule_{i}", True, 0.001)

        # No optimization yet
        assert not os.path.exists(adapter.config.priority_file)

        # One more should trigger optimization
        adapter.update_statistics("rule_4", True, 0.001)

        # Should have optimized (though may not save due to min_samples)
        assert adapter.call_count == 5

    def test_get_statistics_summary(self, adapter):
        """Test statistics summary generation."""
        # Add various statistics
        adapter.update_statistics("add_zero", True, 0.001)
        adapter.update_statistics("add_zero", True, 0.001)
        adapter.update_statistics("mul_one", True, 0.002)
        adapter.update_statistics("mul_one", False, 0.002)

        summary = adapter.get_statistics_summary()

        assert "Total rules tracked: 2" in summary
        assert "Total attempts: 4" in summary
        assert "add_zero" in summary
        assert "mul_one" in summary

    def test_export_analysis(self, adapter, temp_dir):
        """Test analysis export."""
        import time

        # Reset call count to avoid triggering optimization
        adapter.call_count = 0

        # Add statistics manually instead of using update_statistics to avoid decay
        adapter.statistics["test_rule"] = RuleStatistics(
            rule_name="test_rule", attempts=10, successes=5, total_time=0.01, last_used=time.time()
        )

        # Export
        export_path = os.path.join(temp_dir, "analysis.json")
        adapter.export_analysis(export_path)

        # Load and check
        with open(export_path) as f:
            analysis = json.load(f)

        assert "metadata" in analysis
        assert "rules" in analysis
        assert "test_rule" in analysis["rules"]

        rule_data = analysis["rules"]["test_rule"]
        assert rule_data["attempts"] == 10
        assert rule_data["success_rate"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

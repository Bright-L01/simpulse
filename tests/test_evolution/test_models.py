"""Tests for evolution models."""

from simpulse.evolution.models import (
    MutationType,
    OptimizationGoal,
    SimpDirection,
    SimpPriority,
)


class TestModels:
    """Test evolution models."""

    def test_simp_priority_enum(self):
        """Test SimpPriority enum values."""
        assert SimpPriority.HIGH.value == "high"
        assert SimpPriority.DEFAULT.value == "default"
        assert SimpPriority.LOW.value == "low"

    def test_simp_direction_enum(self):
        """Test SimpDirection enum values."""
        assert SimpDirection.FORWARD.value == "forward"
        assert SimpDirection.BACKWARD.value == "backward"

    def test_mutation_type_enum(self):
        """Test MutationType enum values."""
        assert MutationType.PRIORITY_CHANGE.value == "priority_change"
        assert MutationType.RULE_DISABLE.value == "rule_disable"

    def test_optimization_goal_enum(self):
        """Test OptimizationGoal enum values."""
        assert OptimizationGoal.MINIMIZE_TIME.value == "minimize_time"
        assert OptimizationGoal.MINIMIZE_MEMORY.value == "minimize_memory"

"""Core functionality tests for the simplified Simpulse codebase."""

import tempfile
from pathlib import Path

from simpulse.unified_optimizer import UnifiedOptimizer


def test_unified_optimizer_basic():
    """Test basic UnifiedOptimizer functionality."""
    optimizer = UnifiedOptimizer()
    assert optimizer is not None


def test_unified_optimizer_with_simple_file():
    """Test UnifiedOptimizer with a simple file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "test.lean"
        test_file.write_text(
            """
@[simp] theorem test_rule : true = true := rfl
@[simp] theorem another_rule : false = false := rfl
"""
        )

        optimizer = UnifiedOptimizer()
        results = optimizer.optimize(tmp_dir, apply=False)

        assert "total_rules" in results
        assert "rules_changed" in results
        assert "estimated_improvement" in results


def test_unified_optimizer_handles_empty_directory():
    """Test UnifiedOptimizer with empty directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        optimizer = UnifiedOptimizer()
        results = optimizer.optimize(tmp_dir, apply=False)

        assert results["total_rules"] == 0
        assert results["rules_changed"] == 0
        assert results["estimated_improvement"] == 0

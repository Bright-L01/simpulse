"""Integration tests for Simpulse."""

from simpulse.analysis.health_checker import HealthChecker
from simpulse.optimization.optimizer import SimpOptimizer


class TestIntegration:
    """Integration tests."""

    def test_full_optimization_workflow(self, tmp_path):
        """Test complete optimization workflow."""
        # Create test Lean project
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create lakefile
        lakefile = project_dir / "lakefile.toml"
        lakefile.write_text(
            """
name = "test"
version = "0.1.0"
"""
        )

        # Create Lean file with simp rules
        lean_file = project_dir / "Test.lean"
        lean_file.write_text(
            """
@[simp] theorem test1 : 1 + 1 = 2 := rfl
@[simp] theorem test2 : 2 + 2 = 4 := rfl
@[simp 2000] theorem test3 : 0 + n = n := rfl
"""
        )

        # Test health check
        checker = HealthChecker()
        health = checker.check_project(project_dir)
        assert health.total_rules == 3
        assert health.default_priority_percentage > 0

        # Test optimization
        optimizer = SimpOptimizer()
        analysis = optimizer.analyze(project_dir)
        optimization = optimizer.optimize(analysis)

        # Verify optimization makes sense
        assert optimization.rules_changed >= 0
        assert optimization.estimated_improvement >= 0

    def test_empty_project(self, tmp_path):
        """Test handling of empty project."""
        checker = HealthChecker()
        result = checker.check_project(tmp_path)
        assert result.total_rules == 0
        assert result.score == 0

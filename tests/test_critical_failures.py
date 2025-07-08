#!/usr/bin/env python3
"""
Critical failure test suite
Tests for failure modes that must be prevented in production
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from simpulse.optimization.safe_optimizer import SafeOptimizer


class TestCriticalFailures:
    """Test suite for critical failure modes."""

    def setup_method(self):
        """Setup for each test."""
        self.optimizer = SafeOptimizer(safe_mode=True)

    def test_large_file_protection(self):
        """Test that large files are blocked to prevent stack overflow."""
        # Generate a file that would cause stack overflow
        large_content = "\n".join(
            [f"theorem test_{i} : {i} + 0 = {i} := by simp" for i in range(2000)]
        )

        analysis = self.optimizer.analyze_file(large_content)
        should_optimize, reasons = self.optimizer.should_optimize(analysis)

        # Should be blocked due to size
        assert not should_optimize
        assert any("large" in reason.lower() or "size" in reason.lower() for reason in reasons)

    def test_custom_simp_priority_detection(self):
        """Test detection of custom simp priorities that would conflict."""
        custom_priority_content = """
        @[simp 2000] theorem high_priority : 2 + 2 = 4 := rfl
        @[simp 500] theorem low_priority : 3 + 3 = 6 := rfl
        theorem uses_both : (2 + 2) + (3 + 3) = 10 := by simp
        """

        analysis = self.optimizer.analyze_file(custom_priority_content)
        should_optimize, reasons = self.optimizer.should_optimize(analysis)

        # Should be blocked due to custom simp lemmas
        assert not should_optimize
        assert any("custom simp" in reason.lower() for reason in reasons)

    def test_recursive_simp_definition_detection(self):
        """Test detection of recursive simp definitions."""
        recursive_content = """
        inductive Expr
          | Const : Nat → Expr
          | Add : Expr → Expr → Expr
        
        @[simp] def eval : Expr → Nat
          | Expr.Const n => n
          | Expr.Add e1 e2 => eval e1 + eval e2
        """

        analysis = self.optimizer.analyze_file(recursive_content)

        # Should detect patterns that cause regressions
        assert analysis.has_forbidden_patterns or analysis.custom_simp_lemmas > 0

    def test_stack_overflow_prevention(self):
        """Test that we don't attempt to compile files that would cause stack overflow."""
        # This is a unit test - we don't actually want to cause a stack overflow
        huge_content = "\n".join([f"theorem huge_{i} : {i} = {i} := rfl" for i in range(5000)])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(huge_content)
            temp_file = Path(f.name)

        try:
            should_optimize, _, reasons = self.optimizer.optimize_file(temp_file)

            # Should refuse to optimize
            assert not should_optimize
            assert any("size" in reason.lower() or "large" in reason.lower() for reason in reasons)
        finally:
            temp_file.unlink(missing_ok=True)

    def test_compilation_error_handling(self):
        """Test handling of code that fails to compile."""
        broken_content = """
        -- Intentionally broken syntax
        theorem broken_syntax : invalid syntax here := by simp
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(broken_content)
            temp_file = Path(f.name)

        try:
            should_optimize, _, reasons = self.optimizer.optimize_file(temp_file)

            # Should handle gracefully, not crash
            assert isinstance(should_optimize, bool)
            assert isinstance(reasons, list)
        finally:
            temp_file.unlink(missing_ok=True)

    def test_domain_specific_pattern_detection(self):
        """Test detection of domain-specific patterns that typically regress."""
        compiler_content = """
        inductive Token
          | NUMBER : Nat → Token
          | PLUS : Token
        
        @[simp] def tokenValue : Token → Nat
          | Token.NUMBER n => n
          | Token.PLUS => 0
        """

        analysis = self.optimizer.analyze_file(compiler_content)

        # Should detect this as risky
        assert analysis.custom_simp_lemmas > 0 or analysis.has_forbidden_patterns

    def test_empty_file_handling(self):
        """Test handling of empty or tiny files."""
        tiny_content = "-- Just a comment"

        analysis = self.optimizer.analyze_file(tiny_content)
        should_optimize, reasons = self.optimizer.should_optimize(analysis)

        # Should refuse to optimize tiny files
        assert not should_optimize
        assert any("small" in reason.lower() or "size" in reason.lower() for reason in reasons)

    def test_safe_mode_vs_extended_mode(self):
        """Test that safe mode is more conservative than extended mode."""
        borderline_content = """
        theorem arith1 : 5 + 0 = 5 := by simp
        theorem arith2 : 10 * 1 = 10 := by simp
        theorem list1 (l : List Nat) : l ++ [] = l := by simp
        """

        safe_optimizer = SafeOptimizer(safe_mode=True)
        extended_optimizer = SafeOptimizer(safe_mode=False)

        safe_analysis = safe_optimizer.analyze_file(borderline_content)
        extended_analysis = extended_optimizer.analyze_file(borderline_content)

        safe_should, _ = safe_optimizer.should_optimize(safe_analysis)
        extended_should, _ = extended_optimizer.should_optimize(extended_analysis)

        # Safe mode should be more conservative
        if not safe_should:
            # If safe mode rejects, extended mode decision doesn't matter
            pass
        else:
            # If safe mode accepts, extended mode should also accept
            assert extended_should

    def test_performance_regression_bounds(self):
        """Test that we can detect patterns likely to cause severe regressions."""
        high_regression_content = """
        -- Pattern known to cause 40%+ regression
        inductive Expr
          | Var : String → Expr
          | App : Expr → Expr → Expr
        
        @[simp] def size : Expr → Nat
          | Expr.Var _ => 1
          | Expr.App e1 e2 => size e1 + size e2 + 1
        """

        analysis = self.optimizer.analyze_file(high_regression_content)
        should_optimize, reasons = self.optimizer.should_optimize(analysis)

        # Should be blocked
        assert not should_optimize
        assert len(reasons) > 0


# Integration tests for critical failures
class TestCriticalFailuresIntegration:
    """Integration tests for critical failure prevention."""

    def test_cli_rejects_dangerous_files(self):
        """Test that CLI properly rejects files that would cause failures."""
        dangerous_content = """
        @[simp 2000] theorem custom_high : 1 = 1 := rfl
        @[simp 500] theorem custom_low : 2 = 2 := rfl
        theorem mixed : 1 + 2 = 3 := by simp
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(dangerous_content)
            temp_file = Path(f.name)

        try:
            # Test CLI analysis
            result = subprocess.run(
                ["python", "src/simpulse/cli_safe.py", str(temp_file), "--analyze"],
                capture_output=True,
                text=True,
            )

            # Should complete without error and recommend against optimization
            assert result.returncode == 0
            assert "not recommended" in result.stdout.lower() or "skip" in result.stdout.lower()
        finally:
            temp_file.unlink(missing_ok=True)

    def test_forced_optimization_warnings(self):
        """Test that forced optimization shows appropriate warnings."""
        risky_content = """
        theorem tiny : 1 = 1 := rfl
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(risky_content)
            temp_file = Path(f.name)

        try:
            # Test forced optimization
            result = subprocess.run(
                ["python", "src/simpulse/cli_safe.py", str(temp_file), "--force", "-v"],
                capture_output=True,
                text=True,
            )

            # Should show warnings about forcing optimization
            assert "forcing" in result.stdout.lower() or "warning" in result.stdout.lower()
        finally:
            temp_file.unlink(missing_ok=True)
            # Clean up generated file
            optimized_file = temp_file.parent / f"{temp_file.stem}_optimized.lean"
            optimized_file.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

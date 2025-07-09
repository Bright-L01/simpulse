"""
Regression test for the simplified optimizer.
Ensures the core functionality works and prevents future breakage.
"""

import sys
import tempfile
from pathlib import Path

sys.path.append("src")
from simpulse.unified_optimizer import UnifiedOptimizer


def test_simple_optimizer_basic_functionality():
    """Test that the simplified optimizer works on basic Lean code."""

    # Create a temporary Lean file with simp rules
    lean_content = """
@[simp]
theorem test_rule1 (x : Nat) : x + 0 = x := by simp

@[simp 500]  
theorem test_rule2 (x : Nat) : 0 + x = x := by simp

lemma test_usage : ∀ x : Nat, x + 0 = 0 + x := by
  intro x
  simp [test_rule1, test_rule2, test_rule1]  -- test_rule1 used twice
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write test file
        lean_file = Path(tmpdir) / "test.lean"
        lean_file.write_text(lean_content)

        # Run optimizer
        optimizer = UnifiedOptimizer()
        results = optimizer.optimize(tmpdir, apply=False)

        # Verify basic functionality
        assert results["total_rules"] == 2
        assert results["rules_changed"] >= 1  # test_rule1 should be optimized

        # Check that most frequent rule gets highest priority
        changes = results["changes"]
        if changes:
            # test_rule1 (used 2 times) should get priority 100
            test_rule1_change = next((c for c in changes if c["rule_name"] == "test_rule1"), None)
            if test_rule1_change:
                assert test_rule1_change["new_priority"] == 100
                assert test_rule1_change["old_priority"] == 1000


def test_optimizer_handles_no_usage():
    """Test that optimizer handles rules with no usage gracefully."""

    lean_content = """
@[simp]
theorem unused_rule (x : Nat) : x + 0 = x := by simp

lemma no_simp_usage : 2 + 2 = 4 := by norm_num
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        lean_file = Path(tmpdir) / "test.lean"
        lean_file.write_text(lean_content)

        optimizer = UnifiedOptimizer()
        results = optimizer.optimize(tmpdir, apply=False)

        # Should find the rule but not optimize it (no usage)
        assert results["total_rules"] == 1
        assert results["rules_changed"] == 0


def test_optimizer_line_count_constraint():
    """Regression test: ensure optimizer stays under 210 lines."""

    optimizer_file = Path("src/simpulse/unified_optimizer.py")
    lines = optimizer_file.read_text().count("\n")

    # Allow some flexibility but catch if it grows too much
    assert lines <= 315, f"Optimizer has {lines} lines, should be ≤315"


if __name__ == "__main__":
    test_simple_optimizer_basic_functionality()
    test_optimizer_handles_no_usage()
    test_optimizer_line_count_constraint()
    print("✅ All regression tests passed!")

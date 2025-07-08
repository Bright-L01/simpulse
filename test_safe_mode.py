#!/usr/bin/env python3
"""
Test safe mode optimization guards
"""

import tempfile
from pathlib import Path

from simpulse.optimization.safe_optimizer import SafeOptimizer

# Test cases
TEST_CASES = {
    "good_arithmetic": """
-- Good candidate for optimization
theorem test1 : ∀ n : Nat, n + 0 = n := by simp
theorem test2 : ∀ n : Nat, 0 + n = n := by simp
theorem test3 : ∀ n : Nat, n * 1 = n := by simp
theorem test4 : ∀ n : Nat, 1 * n = n := by simp
theorem test5 : ∀ n m : Nat, (n + 0) * (m * 1) = n * m := by simp
""",
    "bad_lists": """
-- Bad candidate - list heavy
theorem list1 : ∀ (l : List Nat), l.reverse.reverse = l := by simp
theorem list2 : ∀ (l1 l2 : List Nat), (l1 ++ l2).reverse = l2.reverse ++ l1.reverse := by simp
theorem list3 : ∀ (l : List Nat), l ++ [] = l := by simp
""",
    "bad_custom_simp": """
-- Bad candidate - has custom simp lemmas
@[simp] theorem my_lemma : ∀ n : Nat, n.succ.pred = n := sorry
theorem uses_custom : ∀ n : Nat, (n + 1).pred = n := by simp
""",
    "bad_too_small": """
-- Too small
theorem tiny : 5 = 5 := rfl
""",
    "mixed_content": """
-- Mixed content
theorem arith1 : ∀ n : Nat, n + 0 = n := by simp
theorem arith2 : ∀ n : Nat, n * 1 = n := by simp
theorem list1 : ∀ (l : List Nat), l ++ [] = l := by simp
theorem logic1 : ∀ (p : Prop), p ∧ True = p := by simp
""",
}


def test_safe_optimizer():
    """Test the safe optimizer on various cases."""
    print("🧪 Testing Safe Optimizer")
    print("=" * 60)

    # Test in safe mode
    print("\n📋 SAFE MODE TESTS")
    print("-" * 40)
    safe_optimizer = SafeOptimizer(safe_mode=True)

    for name, content in TEST_CASES.items():
        print(f"\nTesting: {name}")
        analysis = safe_optimizer.analyze_file(content)
        should_opt, reasons = safe_optimizer.should_optimize(analysis)

        print(f"  Arithmetic ratio: {analysis.arithmetic_ratio:.1%}")
        print(f"  List ratio: {analysis.list_ratio:.1%}")
        print(f"  Should optimize: {'✅ YES' if should_opt else '❌ NO'}")
        if reasons:
            print("  Reasons:")
            for reason in reasons:
                print(f"    - {reason}")

    # Test in extended mode
    print("\n\n📋 EXTENDED MODE TESTS")
    print("-" * 40)
    extended_optimizer = SafeOptimizer(safe_mode=False)

    for name, content in TEST_CASES.items():
        print(f"\nTesting: {name}")
        analysis = extended_optimizer.analyze_file(content)
        should_opt, reasons = extended_optimizer.should_optimize(analysis)

        print(f"  Should optimize: {'✅ YES' if should_opt else '❌ NO'}")
        if reasons:
            print("  Reasons:")
            for reason in reasons:
                print(f"    - {reason}")


def test_cli_integration():
    """Test CLI integration."""
    print("\n\n🔧 TESTING CLI INTEGRATION")
    print("=" * 60)

    import subprocess

    # Create test file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(TEST_CASES["good_arithmetic"])
        test_file = f.name

    try:
        # Test analyze mode
        print("\nTesting --analyze mode:")
        result = subprocess.run(
            ["python", "src/simpulse/cli_safe.py", test_file, "--analyze"],
            capture_output=True,
            text=True,
        )
        print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)

        # Test safe optimization
        print("\nTesting safe optimization:")
        result = subprocess.run(
            ["python", "src/simpulse/cli_safe.py", test_file, "-v"], capture_output=True, text=True
        )
        print(result.stdout)

    finally:
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
        Path(test_file.replace(".lean", "_optimized.lean")).unlink(missing_ok=True)


if __name__ == "__main__":
    test_safe_optimizer()
    test_cli_integration()

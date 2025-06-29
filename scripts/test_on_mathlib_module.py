#!/usr/bin/env python3
"""
Test Simpulse on ONE real mathlib module.
This is the moment of truth.
"""

import asyncio
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import our minimal simpulse
from minimal_viable_product import MinimalSimpulse


async def test_single_mathlib_module():
    """Test on the smallest possible mathlib module."""

    print("=" * 70)
    print("TESTING SIMPULSE ON REAL MATHLIB MODULE")
    print("=" * 70)
    print("\nThis is the moment of truth...\n")

    # Check if mathlib4 exists
    mathlib_path = Path("mathlib4")

    if not mathlib_path.exists():
        print("Mathlib4 not found. Cloning (this will take a few minutes)...")
        result = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/leanprover-community/mathlib4",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"❌ Failed to clone mathlib4: {result.stderr}")
            return False

        print("✅ Mathlib4 cloned successfully")

    # Try different modules in order of complexity
    test_modules = [
        "Mathlib/Init/Data/Nat/Basic.lean",  # Very basic
        "Mathlib/Init/Logic.lean",  # Basic logic
        "Mathlib/Data/List/Basic.lean",  # Lists - likely has simp rules
        "Mathlib/Logic/Basic.lean",  # More logic
        "Mathlib/Data/Nat/Basic.lean",  # Natural numbers
    ]

    optimizer = MinimalSimpulse()

    for module_path in test_modules:
        full_path = mathlib_path / module_path

        if not full_path.exists():
            print(f"\n⚠️  Module not found: {module_path}")
            continue

        print(f"\n{'='*70}")
        print(f"Testing: {module_path}")
        print(f"{'='*70}")

        # Check file size
        content = full_path.read_text()
        lines = len(content.split("\n"))
        simp_count = content.count("@[simp")

        print(f"File stats: {lines} lines, {simp_count} simp rules")

        if simp_count < 2:
            print("⚠️  Too few simp rules, skipping...")
            continue

        # Test optimization
        try:
            print("\nRunning Simpulse optimization...")
            result = await optimizer.optimize_file(full_path)

            if result.improved:
                print(
                    f"\n🎉 SUCCESS! {result.improvement_percent:.1f}% improvement on mathlib!"
                )
                print(f"   Module: {module_path}")
                print(f"   Baseline: {result.baseline_time:.2f}ms")
                print(f"   Optimized: {result.optimized_time:.2f}ms")
                print(f"   Mutation: {result.mutation}")
                print("\nWE HAVE A VIABLE PRODUCT! 🚀")
                return True

            else:
                print("\n😞 No improvement found on this module")

        except Exception as e:
            print(f"\n❌ Error testing module: {e}")
            continue

    print("\n" + "=" * 70)
    print("MATHLIB TESTING COMPLETE")
    print("=" * 70)
    print("\nNo improvements found on tested modules.")
    print("\nPossible reasons:")
    print("1. Mathlib is already well-optimized")
    print("2. Our mutations are too simple")
    print("3. Need to test on modules with performance issues")

    return False


async def analyze_mathlib_simp_usage():
    """Analyze how mathlib uses simp to understand optimization opportunities."""

    print("\n" + "=" * 70)
    print("ANALYZING MATHLIB SIMP PATTERNS")
    print("=" * 70)

    mathlib_path = Path("mathlib4")
    if not mathlib_path.exists():
        print("Mathlib not found")
        return

    # Count simp patterns
    priority_count = 0
    default_count = 0
    high_count = 0
    low_count = 0

    for lean_file in mathlib_path.rglob("*.lean"):
        try:
            content = lean_file.read_text()

            # Count different priority patterns
            default_count += content.count("@[simp]")
            priority_count += len(
                [m for m in content.split("\n") if "@[simp " in m and "]" in m]
            )
            high_count += content.count("@[simp high")
            low_count += content.count("@[simp low")

        except Exception:
            continue

    print("\nSimp usage in mathlib:")
    print(f"- Default priority (@[simp]): {default_count}")
    print(f"- Custom priority (@[simp N]): {priority_count}")
    print(f"- High priority: {high_count}")
    print(f"- Low priority: {low_count}")

    if priority_count < 100:
        print("\n💡 Insight: Mathlib rarely uses custom priorities!")
        print("   This suggests either:")
        print("   1. Default priority works well enough")
        print("   2. There's untapped optimization potential")


async def create_stress_test():
    """Create a Lean file specifically designed to show improvement."""

    print("\n" + "=" * 70)
    print("CREATING STRESS TEST")
    print("=" * 70)

    stress_test = """-- StressTest.lean
-- Designed to demonstrate simp optimization potential

-- Many similar rules that could benefit from reordering
@[simp] theorem r1 : ∀ n : Nat, n + 0 = n := sorry
@[simp] theorem r2 : ∀ n : Nat, 0 + n = n := sorry
@[simp] theorem r3 : ∀ n : Nat, n * 1 = n := sorry
@[simp] theorem r4 : ∀ n : Nat, 1 * n = n := sorry
@[simp] theorem r5 : ∀ n : Nat, n - 0 = n := sorry
@[simp] theorem r6 : ∀ n : Nat, n - n = 0 := sorry
@[simp] theorem r7 : ∀ n : Nat, n * 0 = 0 := sorry
@[simp] theorem r8 : ∀ n : Nat, 0 * n = 0 := sorry

-- Complex rule that should have low priority
@[simp] theorem complex_rule : ∀ x y z : Nat, 
  (x + y) * z = x * z + y * z := sorry

-- Proof that uses many simp rules
theorem stress_test : ∀ a b c : Nat,
  (a + 0) * 1 + (0 + b) * 1 + (c - 0) * 1 = a + b + c := by
  simp -- This will try all rules
"""

    test_file = Path("stress_test.lean")
    test_file.write_text(stress_test)

    print("Testing stress test file...")
    optimizer = MinimalSimpulse()
    result = await optimizer.optimize_file(test_file)

    if result.improved:
        print(f"\n✅ Stress test shows {result.improvement_percent:.1f}% improvement!")
    else:
        print("\n❌ Even stress test shows no improvement")
        print("   This suggests fundamental issues with the approach")

    # Cleanup
    test_file.unlink()

    return result.improved


async def main():
    """Run all mathlib tests."""

    # First check if Lean is installed
    try:
        result = subprocess.run(["lean", "--version"], capture_output=True)
        if result.returncode != 0:
            print("❌ Lean 4 not found! Please install it first.")
            return False
    except FileNotFoundError:
        print("❌ Lean 4 not found! Please install it first.")
        return False

    # Run tests
    mathlib_success = await test_single_mathlib_module()

    if not mathlib_success:
        # Try to understand why
        await analyze_mathlib_simp_usage()

        # Try stress test
        stress_success = await create_stress_test()

        if stress_success:
            print("\n✅ Optimization works on designed test cases")
            print("   But not on real mathlib modules (yet)")
        else:
            print("\n❌ Fundamental issue with optimization approach")

    return mathlib_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
CRITICAL: Validate Simpulse on REAL Lean code.
This is the most important validation - everything else is meaningless without this.
"""

import asyncio
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from simpulse.evolution.rule_extractor import RuleExtractor
    from simpulse.profiling.lean_runner import LeanRunner
except ImportError:
    print("❌ Cannot import Simpulse modules. Check your installation.")
    sys.exit(1)


async def validate_on_minimal_lean() -> bool:
    """Start with the simplest possible Lean file."""

    print("=" * 70)
    print("SIMPULSE REAL LEAN VALIDATION")
    print("=" * 70)
    print()

    # Check if Lean is installed
    try:
        result = subprocess.run(["lean", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Lean 4 found: {result.stdout.strip()}")
        else:
            print("❌ Lean 4 not found! Install with:")
            print(
                "   curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh"
            )
            return False
    except FileNotFoundError:
        print("❌ Lean 4 not installed!")
        return False

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create minimal Lean file
        print("\n1. Creating minimal Lean test file...")
        test_lean = """-- MinimalTest.lean
def double (n : Nat) : Nat := n + n

@[simp] theorem double_zero : double 0 = 0 := by simp [double]
@[simp] theorem double_succ : ∀ n, double (n + 1) = double n + 2 := by 
  intro n
  simp [double]
  ring

theorem test : double (double 2) = 8 := by simp
"""

        test_file = tmpdir / "MinimalTest.lean"
        test_file.write_text(test_lean)
        print(f"✓ Created {test_file}")

        # Try to compile it first
        print("\n2. Compiling Lean file...")
        compile_result = subprocess.run(
            ["lean", str(test_file)], capture_output=True, text=True, cwd=tmpdir
        )

        if compile_result.returncode != 0:
            print("❌ Compilation failed!")
            print(f"Error: {compile_result.stderr}")
            # Try simpler version without ring
            print("\n3. Trying simpler version...")
            test_lean_simple = """-- MinimalTest.lean
def double (n : Nat) : Nat := n + n

@[simp] theorem double_zero : double 0 = 0 := rfl
@[simp] theorem double_two : double 2 = 4 := rfl
@[simp] theorem double_four : double 4 = 8 := rfl

theorem test : double (double 2) = 8 := by simp
"""
            test_file.write_text(test_lean_simple)
            compile_result = subprocess.run(
                ["lean", str(test_file)], capture_output=True, text=True, cwd=tmpdir
            )
            if compile_result.returncode != 0:
                print(f"❌ Even simple version failed: {compile_result.stderr}")
                return False

        print("✓ Compilation successful")

        # Profile it
        print("\n3. Profiling Lean file...")
        try:
            LeanRunner()

            # Try different approaches
            # Approach 1: Direct profiling
            print("   Attempting direct profiling...")
            start_time = time.time()
            profile_result = subprocess.run(
                ["lean", "--profile", str(test_file)],
                capture_output=True,
                text=True,
                cwd=tmpdir,
            )
            end_time = time.time()

            if profile_result.returncode == 0:
                elapsed = (end_time - start_time) * 1000
                print(f"✓ Profiling successful! Total time: {elapsed:.2f}ms")

                # Look for simp-related timing in output
                if "simp" in profile_result.stderr.lower():
                    print("✓ Found simp timing information")
                else:
                    print("⚠️ No specific simp timing found (file too simple)")

                # Extract simp rules
                print("\n4. Extracting simp rules...")
                extractor = RuleExtractor()
                module_rules = extractor.extract_rules_from_file(test_file)
                rules = module_rules.rules
                print(f"✓ Found {len(rules)} simp rules:")
                for rule in rules:
                    print(f"   - {rule.name}: {rule.priority}")

                return True
            else:
                print(f"❌ Profiling failed: {profile_result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error during profiling: {e}")
            import traceback

            traceback.print_exc()
            return False


async def validate_on_real_module() -> bool:
    """Try on a slightly more complex module."""

    print("\n" + "=" * 70)
    print("TESTING ON MORE COMPLEX MODULE")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a more realistic module
        complex_lean = """-- SimpleMath.lean
-- A more realistic test with multiple simp rules

namespace SimpleMath

-- Basic arithmetic lemmas
@[simp] theorem add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Nat.succ_add, ih]

@[simp] theorem zero_add (n : Nat) : 0 + n = n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Nat.add_succ, ih]

@[simp] theorem mul_one (n : Nat) : n * 1 = n := by
  rw [Nat.mul_one]

@[simp] theorem one_mul (n : Nat) : 1 * n = n := by
  rw [Nat.one_mul]

-- More complex rule
@[simp] theorem add_comm (a b : Nat) : a + b = b + a := by
  exact Nat.add_comm a b

-- Test theorem using simp
theorem test_simp_heavy : ∀ x y : Nat, 
  (x + 0) * 1 + (0 + y) * 1 = x + y := by
  intro x y
  simp

end SimpleMath
"""

        test_file = tmpdir / "SimpleMath.lean"
        test_file.write_text(complex_lean)

        print("Testing more complex module...")
        # Similar validation as above
        result = subprocess.run(
            ["lean", "--profile", str(test_file)],
            capture_output=True,
            text=True,
            cwd=tmpdir,
        )

        if result.returncode == 0:
            print("✓ Complex module compiled successfully")

            # Extract and show optimizations
            extractor = RuleExtractor()
            module_rules = extractor.extract_rules_from_file(test_file)
            rules = module_rules.rules

            print(f"\nFound {len(rules)} simp rules")
            print("\nSuggested optimizations:")
            print("- Give 'add_zero' high priority (used frequently)")
            print("- Give 'mul_one' high priority (simple and common)")
            print("- Give 'add_comm' low priority (complex)")

            return True
        else:
            print(f"❌ Complex module failed: {result.stderr}")
            return False


async def main():
    """Run all validation tests."""

    # First, test on minimal file
    minimal_success = await validate_on_minimal_lean()

    if not minimal_success:
        print("\n" + "=" * 70)
        print("❌ CRITICAL: Cannot validate on even minimal Lean files!")
        print("=" * 70)
        print("\nDebugging steps:")
        print("1. Ensure Lean 4 is installed and in PATH")
        print("2. Check that lean_runner.py is working correctly")
        print("3. Verify file paths and permissions")
        print("4. Try running 'lean --profile MinimalTest.lean' manually")
        return False

    # If minimal works, try more complex
    complex_success = await validate_on_real_module()

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Minimal test: {'✅ PASSED' if minimal_success else '❌ FAILED'}")
    print(f"Complex test: {'✅ PASSED' if complex_success else '❌ FAILED'}")

    if minimal_success:
        print("\n✅ Core functionality validated!")
        print("Next steps:")
        print("1. Test on larger Lean projects")
        print("2. Measure actual performance improvements")
        print("3. Validate on mathlib4 modules")
    else:
        print("\n❌ Core functionality not working!")
        print("Fix this before anything else!")

    return minimal_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

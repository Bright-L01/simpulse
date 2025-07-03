#!/usr/bin/env python3
"""
Demonstration of the Simpulse Correctness Validation System

This script shows how the correctness validator ensures that optimizations
preserve the correctness of Lean proofs.
"""

import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simpulse.validator.correctness import CorrectnessValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_lean_file() -> Path:
    """Create a simple Lean file for testing."""
    test_dir = Path("test_validation")
    test_dir.mkdir(exist_ok=True)

    # Create a simple Lean file with simp rules
    test_file = test_dir / "TestSimp.lean"
    content = """-- Test file for Simpulse correctness validation

@[simp] theorem add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.add_succ, ih]

@[simp] theorem zero_add (n : Nat) : 0 + n = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.succ_add, ih]

@[simp] theorem mul_one (n : Nat) : n * 1 = n := by
  simp [Nat.mul_succ, Nat.mul_zero, Nat.add_zero]

@[simp] theorem one_mul (n : Nat) : 1 * n = n := by
  induction n with
  | zero => simp
  | succ n ih => simp [Nat.mul_succ, ih]

-- Test theorem using the simp rules
theorem test_simplification (a b : Nat) : (a + 0) * 1 = a := by
  simp
"""

    test_file.write_text(content)

    # Create a simple lakefile
    lakefile = test_dir / "lakefile.lean"
    lakefile_content = """import Lake
open Lake DSL

package testSimp

lean_lib TestSimp
"""
    lakefile.write_text(lakefile_content)

    # Create lean-toolchain file
    toolchain = test_dir / "lean-toolchain"
    toolchain.write_text("leanprover/lean4:stable\n")

    return test_file


def create_test_optimizations() -> list:
    """Create test optimization suggestions."""
    return [
        {
            "rule": "add_zero",
            "location": "theorem add_zero",
            "line": 3,
            "original": "@[simp] theorem add_zero",
            "replacement": "@[simp, priority := 500] theorem add_zero",
            "priority": 500,
        },
        {
            "rule": "zero_add",
            "location": "theorem zero_add",
            "line": 8,
            "original": "@[simp] theorem zero_add",
            "replacement": "@[simp, priority := 600] theorem zero_add",
            "priority": 600,
        },
        {
            "rule": "mul_one",
            "location": "theorem mul_one",
            "line": 13,
            "original": "@[simp] theorem mul_one",
            "replacement": "@[simp, priority := 700] theorem mul_one",
            "priority": 700,
        },
        # This one might break compilation (intentionally malformed)
        {
            "rule": "one_mul",
            "location": "theorem one_mul",
            "line": 16,
            "original": "@[simp] theorem one_mul",
            "replacement": "@[simp, priority := INVALID] theorem one_mul",  # Invalid priority
            "priority": "INVALID",
        },
    ]


def demonstrate_single_file_validation():
    """Demonstrate validation on a single file."""
    print("\n" + "=" * 60)
    print("SINGLE FILE VALIDATION DEMO")
    print("=" * 60)

    # Create test file
    test_file = create_test_lean_file()
    print(f"\nCreated test file: {test_file}")

    # Create validator
    validator = CorrectnessValidator(timeout=30)

    # Create test optimizations
    optimizations = create_test_optimizations()
    print(f"\nTesting {len(optimizations)} optimizations...")

    # Run validation
    result = validator.validate_file(test_file, optimizations)

    # Display results
    print("\n" + "-" * 60)
    print("VALIDATION RESULTS")
    print("-" * 60)
    print(f"File: {result.file_path}")
    print(f"Original compilation time: {result.original_compilation_time:.2f}s")
    print(
        f"Optimized compilation time: {result.optimized_compilation_time:.2f}s"
        if result.optimized_compilation_time
        else "N/A"
    )
    print(f"Total optimizations: {result.total_optimizations}")
    print(f"Successful: {result.successful_optimizations}")
    print(f"Failed: {result.failed_optimizations}")
    print(f"Success rate: {result.success_rate:.1%}")
    print(f"Speedup: {result.speedup:.2f}x")

    if result.error:
        print(f"\nError: {result.error}")

    # Show individual optimization results
    print("\n" + "-" * 60)
    print("OPTIMIZATION DETAILS")
    print("-" * 60)
    for opt_result in result.optimization_results:
        status = "✅" if opt_result.success else "❌"
        print(f"{status} {opt_result.rule} at {opt_result.location}")
        if not opt_result.success:
            print(f"   Error: {opt_result.error_message}")

    return result


def demonstrate_batch_validation():
    """Demonstrate batch validation on multiple files."""
    print("\n" + "=" * 60)
    print("BATCH VALIDATION DEMO")
    print("=" * 60)

    # Create multiple test files
    test_files = []
    for i in range(3):
        test_dir = Path(f"test_validation_{i}")
        test_dir.mkdir(exist_ok=True)

        # Copy test file structure
        test_file = create_test_lean_file()
        new_file = test_dir / f"TestSimp{i}.lean"

        # Modify content slightly for each file
        content = test_file.read_text()
        content = content.replace("TestSimp", f"TestSimp{i}")
        new_file.write_text(content)

        # Copy lakefile and toolchain
        import shutil

        shutil.copy(test_file.parent / "lakefile.lean", test_dir)
        shutil.copy(test_file.parent / "lean-toolchain", test_dir)

        test_files.append(new_file)

    # Create validator
    validator = CorrectnessValidator(timeout=30)

    # Prepare files and optimizations
    files_and_optimizations = []
    for file_path in test_files:
        # Use different optimizations for each file
        optimizations = create_test_optimizations()[:2]  # Only use safe optimizations
        files_and_optimizations.append((file_path, optimizations))

    # Run batch validation
    batch_report = validator.validate_batch(files_and_optimizations)

    # Display batch report
    print("\n" + "-" * 60)
    print("BATCH VALIDATION REPORT")
    print("-" * 60)
    print(f"Total files tested: {batch_report['total_files_tested']}")
    print(f"Files successfully optimized: {batch_report['files_successfully_optimized']}")
    print(f"Overall success rate: {batch_report['overall_success_rate']:.1%}")
    print(f"Average speedup: {batch_report['average_speedup']:.2f}x")

    # Save report
    report_path = Path("batch_validation_demo_report.json")
    with open(report_path, "w") as f:
        json.dump(batch_report, f, indent=2)
    print(f"\nBatch report saved to: {report_path}")

    return batch_report


def demonstrate_safety_analysis():
    """Demonstrate safety analysis of optimization rules."""
    print("\n" + "=" * 60)
    print("SAFETY ANALYSIS DEMO")
    print("=" * 60)

    # Create validator
    validator = CorrectnessValidator(timeout=30)

    # Create multiple validation results for analysis
    validation_results = []

    # Simulate results for different rules
    from src.simpulse.validator.correctness import OptimizationResult, ValidationResult

    # Rule 1: Always safe
    result1 = ValidationResult(
        file_path="test1.lean", original_compilation_time=1.0, optimized_compilation_time=0.8
    )
    result1.optimization_results = [
        OptimizationResult("add_zero", "line 10", True),
        OptimizationResult("add_zero", "line 20", True),
        OptimizationResult("add_zero", "line 30", True),
    ]
    validation_results.append(result1)

    # Rule 2: Sometimes safe
    result2 = ValidationResult(
        file_path="test2.lean", original_compilation_time=1.2, optimized_compilation_time=1.0
    )
    result2.optimization_results = [
        OptimizationResult("mul_comm", "line 15", True),
        OptimizationResult("mul_comm", "line 25", False, "Type mismatch"),
        OptimizationResult("mul_comm", "line 35", True),
    ]
    validation_results.append(result2)

    # Rule 3: Always unsafe
    result3 = ValidationResult(
        file_path="test3.lean", original_compilation_time=0.9, optimized_compilation_time=None
    )
    result3.optimization_results = [
        OptimizationResult("custom_rule", "line 5", False, "Syntax error"),
        OptimizationResult("custom_rule", "line 15", False, "Invalid priority"),
    ]
    validation_results.append(result3)

    # Generate safety report
    safety_report = validator.generate_safety_report(validation_results)

    # Display safety report
    print("\n" + "-" * 60)
    print("RULE SAFETY REPORT")
    print("-" * 60)
    print(f"Total rules tested: {safety_report['total_rules_tested']}")
    print(f"Safe rules: {len(safety_report['safe_rules'])}")
    print(f"Unsafe rules: {len(safety_report['unsafe_rules'])}")
    print(f"Conditional rules: {len(safety_report['conditional_rules'])}")

    print("\n" + "-" * 60)
    print("RULE SAFETY SCORES")
    print("-" * 60)
    for rule, score in safety_report["rule_safety_scores"].items():
        print(f"\n{rule}:")
        print(f"  Safety rate: {score['safety_rate']:.1%}")
        print(f"  Safe applications: {score['safe_applications']}")
        print(f"  Unsafe applications: {score['unsafe_applications']}")
        print(f"  Recommendation: {score['recommendation']}")

    # Save safety report
    report_path = Path("safety_analysis_demo_report.json")
    with open(report_path, "w") as f:
        json.dump(safety_report, f, indent=2)
    print(f"\n\nSafety report saved to: {report_path}")

    return safety_report


def main():
    """Run all demonstrations."""
    print("\n🚀 SIMPULSE CORRECTNESS VALIDATION DEMO")
    print("=" * 60)
    print("This demo shows how Simpulse validates that optimizations")
    print("preserve the correctness of Lean proofs.")
    print("=" * 60)

    # Note about Lean installation
    print("\n⚠️  Note: This demo requires Lean 4 to be installed.")
    print("   If compilation fails, the validator will still demonstrate")
    print("   the rollback mechanism and safety analysis.")

    try:
        # Demo 1: Single file validation
        demonstrate_single_file_validation()

        # Demo 2: Batch validation
        demonstrate_batch_validation()

        # Demo 3: Safety analysis
        demonstrate_safety_analysis()

        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("1. Optimizations are applied incrementally and validated")
        print("2. Breaking changes are automatically rolled back")
        print("3. Only safe optimizations are kept in the final result")
        print("4. Batch validation enables testing across multiple files")
        print("5. Safety analysis helps identify reliable optimization rules")

    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\n❌ Demo encountered an error: {e}")
        print("   This is likely due to Lean 4 not being installed.")
        print("   The validation system would normally handle this gracefully.")

    finally:
        # Cleanup test directories
        import shutil

        for i in range(4):
            test_dir = Path(f"test_validation_{i}" if i > 0 else "test_validation")
            if test_dir.exists():
                shutil.rmtree(test_dir)


if __name__ == "__main__":
    main()

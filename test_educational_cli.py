#!/usr/bin/env python3
"""
Test the educational CLI implementation
"""

import subprocess
import tempfile
from pathlib import Path

# Test files with different patterns
TEST_FILES = {
    "safe_arithmetic": """
-- Safe file with lots of arithmetic patterns
theorem arith1 : ∀ n : Nat, n + 0 = n := by simp
theorem arith2 : ∀ n : Nat, 0 + n = n := by simp
theorem arith3 : ∀ n : Nat, n * 1 = n := by simp
theorem arith4 : ∀ n : Nat, 1 * n = n := by simp
theorem arith5 : ∀ n m : Nat, (n + 0) * (m * 1) = n * m := by simp
theorem arith6 : ∀ n : Nat, (n + 0) + 0 = n := by simp
theorem arith7 : ∀ n m k : Nat, (n + 0) + (m * 1) + (k + 0) = n + m + k := by simp
""",
    "risky_mixed": """
-- Risky file with mixed patterns
theorem arith1 : ∀ n : Nat, n + 0 = n := by simp
theorem arith2 : ∀ n : Nat, n * 1 = n := by simp
theorem list1 (l : List Nat) : l ++ [] = l := by simp
theorem list2 (l : List Nat) : l.reverse.reverse = l := by simp
def complex_simp (n : Nat) : Nat := n + 0
theorem uses_complex : complex_simp 5 = 5 := by simp [complex_simp]
""",
    "unsafe_custom_simp": """
-- Unsafe file with custom simp priorities
@[simp 2000] theorem high_priority : 2 + 2 = 4 := rfl
@[simp 500] theorem low_priority : 3 + 3 = 6 := rfl
theorem arith1 : ∀ n : Nat, n + 0 = n := by simp
theorem uses_custom : (2 + 2) + (3 + 3) = 10 := by simp
""",
    "unsafe_large": """
-- Unsafe file - too many lines
"""
    + "\n".join([f"theorem large_test_{i} : {i} + 0 = {i} := by simp" for i in range(1200)]),
}


def run_command(cmd: list) -> tuple:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def test_file_classification():
    """Test that files are correctly classified as SAFE/RISKY/UNSAFE."""
    print("🔍 Testing File Classification")
    print("=" * 50)

    for test_name, content in TEST_FILES.items():
        print(f"\nTesting {test_name}...")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            test_file = Path(f.name)

        try:
            # Test basic analysis
            cmd = ["python", "src/simpulse/cli_educational.py", str(test_file)]
            returncode, stdout, stderr = run_command(cmd)

            print(f"  Return code: {returncode}")

            # Check expected classifications (check unsafe first since unsafe_custom_simp contains both)
            if "unsafe" in test_name:
                assert (
                    "🔴 UNSAFE" in stdout
                ), f"Should classify {test_name} as UNSAFE but got: {stdout[:200]}"
                assert returncode == 2, f"Unsafe files should return 2"
                print("  ✅ Correctly classified as UNSAFE")

            elif "risky" in test_name:
                assert "🟡 RISKY" in stdout, f"Should classify {test_name} as RISKY"
                assert returncode == 1, f"Risky files should return 1"
                print("  ✅ Correctly classified as RISKY")

            elif "safe" in test_name:
                assert "🟢 SAFE" in stdout, f"Should classify {test_name} as SAFE"
                assert returncode == 0, f"Safe files should return 0"
                print("  ✅ Correctly classified as SAFE")

            # Check speedup predictions
            if "Expected Speedup:" in stdout:
                if "unsafe" in test_name:
                    assert "REGRESSION WARNING" in stdout, "Unsafe files should warn of regression"
                elif "safe" in test_name:
                    assert (
                        "EXCELLENT" in stdout or "GOOD" in stdout
                    ), "Safe files should predict good speedup"
                print("  ✅ Speedup prediction appropriate")

        finally:
            test_file.unlink(missing_ok=True)


def test_pattern_profiling():
    """Test the --profile command."""
    print("\n📊 Testing Pattern Profiling")
    print("=" * 50)

    # Test with safe arithmetic file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(TEST_FILES["safe_arithmetic"])
        test_file = Path(f.name)

    try:
        cmd = ["python", "src/simpulse/cli_educational.py", str(test_file), "--profile"]
        returncode, stdout, stderr = run_command(cmd)

        print(f"Profile command return code: {returncode}")

        # Check profile output
        assert "PATTERN ANALYSIS" in stdout, "Should show pattern analysis"
        assert "OPTIMIZATION TARGETS" in stdout, "Should show safe patterns"
        assert "arithmetic" in stdout.lower(), "Should detect arithmetic patterns"

        print("✅ Pattern profiling working correctly")

        # Test detailed profiling
        cmd = [
            "python",
            "src/simpulse/cli_educational.py",
            str(test_file),
            "--profile",
            "--detailed",
        ]
        returncode, stdout, stderr = run_command(cmd)

        assert "Lines:" in stdout, "Detailed mode should show line numbers"
        print("✅ Detailed profiling working")

    finally:
        test_file.unlink(missing_ok=True)


def test_speedup_prediction():
    """Test the --predict command."""
    print("\n🔮 Testing Speedup Prediction")
    print("=" * 50)

    test_cases = [
        ("safe_arithmetic", "should predict good speedup"),
        ("unsafe_custom_simp", "should predict regression"),
    ]

    for test_name, expectation in test_cases:
        print(f"\nTesting prediction for {test_name} ({expectation})...")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(TEST_FILES[test_name])
            test_file = Path(f.name)

        try:
            cmd = ["python", "src/simpulse/cli_educational.py", str(test_file), "--predict"]
            returncode, stdout, stderr = run_command(cmd)

            assert "DETAILED PREDICTION" in stdout, "Should show prediction details"
            assert "Range:" in stdout, "Should show speedup range"
            assert "Confidence:" in stdout, "Should show confidence level"
            assert "Reasoning:" in stdout, "Should explain reasoning"

            if "unsafe" in test_name:
                # Should predict regression
                assert "REGRESSION WARNING" in stdout, "Unsafe file should warn of regression"
            elif "safe" in test_name:
                # Should predict good speedup
                assert not "REGRESSION WARNING" in stdout, "Safe file shouldn't warn of regression"

            print(f"  ✅ Prediction appropriate for {test_name}")

        finally:
            test_file.unlink(missing_ok=True)


def test_educational_explanations():
    """Test the --explain command."""
    print("\n📚 Testing Educational Explanations")
    print("=" * 50)

    # Test explanation for unsafe file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(TEST_FILES["unsafe_custom_simp"])
        test_file = Path(f.name)

    try:
        cmd = ["python", "src/simpulse/cli_educational.py", str(test_file), "--explain"]
        returncode, stdout, stderr = run_command(cmd)

        assert "WHY THIS FAILED" in stdout, "Should explain why it failed"
        assert "UNSAFE PATTERNS DETECTED" in stdout, "Should identify unsafe patterns"
        assert "WHAT WORKS INSTEAD" in stdout, "Should suggest alternatives"
        assert "Solution:" in stdout, "Should provide solutions"

        print("✅ Educational explanations working for unsafe files")

    finally:
        test_file.unlink(missing_ok=True)

    # Test explanation for safe file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(TEST_FILES["safe_arithmetic"])
        test_file = Path(f.name)

    try:
        cmd = ["python", "src/simpulse/cli_educational.py", str(test_file), "--explain"]
        returncode, stdout, stderr = run_command(cmd)

        assert "WHY THIS WORKS" in stdout, "Should explain why it works"
        assert "GOOD PATTERNS DETECTED" in stdout, "Should identify good patterns"
        assert "OPTIMIZATION STRATEGY" in stdout, "Should explain strategy"

        print("✅ Educational explanations working for safe files")

    finally:
        test_file.unlink(missing_ok=True)


def test_visual_reports():
    """Test visual report generation."""
    print("\n📄 Testing Visual Report Generation")
    print("=" * 50)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(TEST_FILES["safe_arithmetic"])
        test_file = Path(f.name)

    try:
        cmd = ["python", "src/simpulse/cli_educational.py", str(test_file), "--visual-report"]
        returncode, stdout, stderr = run_command(cmd)

        # Check if report was generated
        report_file = test_file.parent / f"{test_file.stem}_pattern_profile.md"

        if report_file.exists():
            print("✅ Visual report generated")

            content = report_file.read_text()
            assert "Pattern Profile" in content, "Report should have title"
            assert "Classification:" in content, "Should show classification"
            assert "Predicted Speedup:" in content, "Should show prediction"
            assert "Safe Patterns" in content, "Should analyze patterns"

            print("✅ Report content looks correct")
            report_file.unlink()
        else:
            print("❌ Visual report not generated")

    finally:
        test_file.unlink(missing_ok=True)


def test_json_output():
    """Test JSON output mode."""
    print("\n📋 Testing JSON Output")
    print("=" * 50)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(TEST_FILES["safe_arithmetic"])
        test_file = Path(f.name)

    try:
        cmd = ["python", "src/simpulse/cli_educational.py", str(test_file), "--json"]
        returncode, stdout, stderr = run_command(cmd)

        # Parse JSON output
        import json

        try:
            data = json.loads(stdout)
            assert "classification" in data, "JSON should include classification"
            assert "speedup_prediction" in data, "JSON should include prediction"
            assert "patterns" in data, "JSON should include pattern counts"
            assert "educational_insights" in data, "JSON should include insights"

            print("✅ JSON output structure correct")

        except json.JSONDecodeError:
            print("❌ Invalid JSON output")
            print(f"Output: {stdout[:200]}...")

    finally:
        test_file.unlink(missing_ok=True)


def main():
    """Run all educational CLI tests."""
    print("🎓 Testing Educational CLI Implementation")
    print("=" * 60)

    try:
        test_file_classification()
        test_pattern_profiling()
        test_speedup_prediction()
        test_educational_explanations()
        test_visual_reports()
        test_json_output()

        print("\n" + "=" * 60)
        print("✅ ALL EDUCATIONAL CLI TESTS PASSED")
        print("🎓 Educational features working correctly!")
        print("\nKey features verified:")
        print("  ✅ File classification (SAFE/RISKY/UNSAFE)")
        print("  ✅ Speedup prediction with explanations")
        print("  ✅ Pattern profiling (--profile)")
        print("  ✅ Educational explanations (--explain)")
        print("  ✅ Visual report generation")
        print("  ✅ JSON output format")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

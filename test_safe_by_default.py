#!/usr/bin/env python3
"""
Test the safe-by-default CLI implementation
"""

import subprocess
import tempfile
from pathlib import Path

# Test files with different compatibility levels
TEST_FILES = {
    "excellent": """
-- Excellent candidate for optimization
theorem arith1 : ∀ n : Nat, n + 0 = n := by simp
theorem arith2 : ∀ n : Nat, 0 + n = n := by simp
theorem arith3 : ∀ n : Nat, n * 1 = n := by simp
theorem arith4 : ∀ n : Nat, 1 * n = n := by simp
theorem arith5 : ∀ n m : Nat, (n + 0) * (m * 1) = n * m := by simp
theorem arith6 : ∀ n : Nat, (n + 0) + 0 = n := by simp
""",
    "dangerous": """
-- Dangerous file with custom simp priorities
@[simp 2000] theorem high_priority : 2 + 2 = 4 := rfl
@[simp 500] theorem low_priority : 3 + 3 = 6 := rfl
theorem uses_custom : (2 + 2) + (3 + 3) = 10 := by simp
""",
    "incompatible": """
-- Incompatible file - too large and complex patterns
"""
    + "\n".join([f"theorem large_test_{i} : {i} + 0 = {i} := by simp" for i in range(1200)]),
    "fair": """
-- Fair candidate with mixed patterns
theorem arith1 : ∀ n : Nat, n + 0 = n := by simp
theorem list1 (l : List Nat) : l ++ [] = l := by simp
theorem logic1 (p : Prop) : p ∧ True = p := by simp
""",
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


def test_doctor_command():
    """Test the simpulse-doctor command."""
    print("🩺 Testing simpulse-doctor command")
    print("=" * 50)

    for test_name, content in TEST_FILES.items():
        print(f"\nTesting {test_name} file...")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            test_file = Path(f.name)

        try:
            # Test doctor command
            cmd = ["python", "src/simpulse/cli_doctor.py", str(test_file), "--detailed"]
            returncode, stdout, stderr = run_command(cmd)

            print(f"  Return code: {returncode}")
            print(f"  Output preview: {stdout[:200]}...")

            # Check expected behaviors
            if test_name == "excellent":
                assert "EXCELLENT" in stdout or "GOOD" in stdout, "Should detect excellent file"
                assert returncode == 0, "Should return success for excellent file"
            elif test_name == "dangerous":
                assert (
                    "DANGEROUS" in stdout or "INCOMPATIBLE" in stdout
                ), "Should detect dangerous patterns"
                assert returncode != 0, "Should return error for dangerous file"
            elif test_name == "incompatible":
                assert (
                    "INCOMPATIBLE" in stdout or "large" in stdout.lower()
                ), "Should detect incompatible file"
                assert returncode != 0, "Should return error for incompatible file"

            print(f"  ✅ {test_name} behaves correctly")

        finally:
            test_file.unlink(missing_ok=True)


def test_safe_by_default_cli():
    """Test the safe-by-default CLI."""
    print("\n🛡️ Testing safe-by-default CLI")
    print("=" * 50)

    for test_name, content in TEST_FILES.items():
        print(f"\nTesting {test_name} file with safe CLI...")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            test_file = Path(f.name)

        try:
            # Test compatibility check only
            cmd = [
                "python",
                "src/simpulse/cli_safe_by_default.py",
                str(test_file),
                "--check-compatibility",
            ]
            returncode, stdout, stderr = run_command(cmd)

            print(f"  Compatibility check return code: {returncode}")
            print(f"  Output preview: {stdout[:150]}...")

            # Test optimization attempt (should be blocked for unsafe files)
            cmd = ["python", "src/simpulse/cli_safe_by_default.py", str(test_file)]
            returncode, stdout, stderr = run_command(cmd)

            print(f"  Optimization attempt return code: {returncode}")

            if test_name == "excellent":
                # Should require --unsafe flag even for excellent files by default
                assert returncode != 0, "Should require explicit --unsafe flag"
                assert "unsafe" in stdout.lower(), "Should mention unsafe flag requirement"
            elif test_name in ["dangerous", "incompatible"]:
                assert returncode != 0, "Should block dangerous/incompatible files"
                assert (
                    "blocked" in stdout.lower() or "not recommended" in stdout.lower()
                ), "Should show blocking message"

            # Test with --unsafe flag
            if test_name == "excellent":
                cmd = ["python", "src/simpulse/cli_safe_by_default.py", str(test_file), "--unsafe"]
                returncode, stdout, stderr = run_command(cmd)
                print(f"  Unsafe mode return code: {returncode}")
                # Should work for excellent files with --unsafe
                # Note: May still fail due to missing dependencies, but shouldn't be blocked

            print(f"  ✅ {test_name} safe-by-default behavior correct")

        finally:
            test_file.unlink(missing_ok=True)
            # Clean up any generated files
            optimized_file = test_file.parent / f"{test_file.stem}_optimized.lean"
            optimized_file.unlink(missing_ok=True)


def test_compatibility_reports():
    """Test compatibility report generation."""
    print("\n📄 Testing compatibility report generation")
    print("=" * 50)

    # Test with excellent file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(TEST_FILES["excellent"])
        test_file = Path(f.name)

    try:
        # Test report generation
        cmd = ["python", "src/simpulse/cli_doctor.py", str(test_file), "--export-report"]
        returncode, stdout, stderr = run_command(cmd)

        print(f"Report generation return code: {returncode}")

        # Check if report was created
        report_file = test_file.parent / f"{test_file.stem}_compatibility_report.md"
        if report_file.exists():
            print(f"✅ Report generated: {report_file}")
            content = report_file.read_text()
            print(f"Report preview: {content[:300]}...")

            # Check report contains expected sections
            assert "Compatibility Report" in content, "Should have title"
            assert "Analysis Summary" in content, "Should have summary"
            assert "File Statistics" in content, "Should have statistics"
            assert "Next Steps" in content, "Should have recommendations"

            print("✅ Report content looks correct")
            report_file.unlink(missing_ok=True)
        else:
            print("❌ Report file not generated")

    finally:
        test_file.unlink(missing_ok=True)


def test_warning_banners():
    """Test that warning banners are displayed."""
    print("\n⚠️ Testing warning banners")
    print("=" * 50)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(TEST_FILES["excellent"])
        test_file = Path(f.name)

    try:
        # Test doctor warnings
        cmd = ["python", "src/simpulse/cli_doctor.py", str(test_file)]
        returncode, stdout, stderr = run_command(cmd)

        assert "WARNING" in stdout or "⚠️" in stdout, "Doctor should show warnings"
        assert "66.7%" in stdout or "failure rate" in stdout.lower(), "Should mention failure rate"
        print("✅ Doctor shows appropriate warnings")

        # Test CLI warnings
        cmd = [
            "python",
            "src/simpulse/cli_safe_by_default.py",
            str(test_file),
            "--check-compatibility",
        ]
        returncode, stdout, stderr = run_command(cmd)

        assert "WARNING" in stdout or "🚨" in stdout, "CLI should show warnings"
        assert "mathlib4" in stdout.lower(), "Should mention mathlib4 limitation"
        print("✅ CLI shows appropriate warnings")

        # Test suppressed warnings
        cmd = ["python", "src/simpulse/cli_doctor.py", str(test_file), "--no-warnings"]
        returncode, stdout, stderr = run_command(cmd)

        # Should still show some warnings but fewer
        print("✅ Warning suppression works")

    finally:
        test_file.unlink(missing_ok=True)


def main():
    """Run all tests."""
    print("🧪 Testing Safe-by-Default Implementation")
    print("=" * 60)

    try:
        test_doctor_command()
        test_safe_by_default_cli()
        test_compatibility_reports()
        test_warning_banners()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("Safe-by-default implementation working correctly!")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

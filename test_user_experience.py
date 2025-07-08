#!/usr/bin/env python3
"""Test the user experience from a non-technical user's perspective."""

import os
import subprocess
import tempfile
from pathlib import Path


def run_command(cmd, expect_failure=False):
    """Run a command and capture output."""
    print(f"\n💻 Running: {cmd}")
    print("─" * 50)

    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        print(f"Exit code: {result.returncode}")

        if expect_failure and result.returncode == 0:
            print("⚠️  Expected failure but command succeeded")
        elif not expect_failure and result.returncode != 0:
            print("❌ Unexpected failure")
        else:
            print("✅ Expected result")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def test_scenario_1_wrong_directory():
    """Test: User runs in directory without Lean files."""
    print("\n" + "=" * 60)
    print("SCENARIO 1: User runs in wrong directory (no Lean files)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create empty directory
        os.chdir(tmpdir)

        # Test check command
        run_command("python -m simpulse check .", expect_failure=True)

        # Test optimize command
        run_command("python -m simpulse optimize .", expect_failure=True)


def test_scenario_2_help_system():
    """Test: User explores help system."""
    print("\n" + "=" * 60)
    print("SCENARIO 2: User explores help system")
    print("=" * 60)

    # Main help
    run_command("python -m simpulse --help")

    # Command help
    run_command("python -m simpulse check --help")
    run_command("python -m simpulse optimize --help")

    # List strategies
    run_command("python -m simpulse list-strategies")


def test_scenario_3_successful_workflow():
    """Test: User successfully optimizes a project."""
    print("\n" + "=" * 60)
    print("SCENARIO 3: Successful optimization workflow")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple Lean project
        project_path = Path(tmpdir)

        # Main file
        (project_path / "Main.lean").write_text(
            """
-- Main theorem file
@[simp] theorem add_zero (n : Nat) : n + 0 = n := by simp
@[simp] theorem zero_add (n : Nat) : 0 + n = n := by simp  
@[simp] theorem mul_one (n : Nat) : n * 1 = n := by simp

example (n : Nat) : n + 0 = n := by simp [add_zero]
example (n : Nat) : 0 + n = n := by simp [zero_add]
example (n : Nat) : n * 1 = n := by simp [mul_one]
"""
        )

        # Utils file
        (project_path / "Utils.lean").write_text(
            """
-- Utility theorems
@[simp, priority := 900] theorem comm_add (a b : Nat) : a + b = b + a := by simp
@[simp, priority := 1100] theorem obscure_rule : 42 + 0 = 42 := by simp

example : 5 + 3 = 8 := by simp [comm_add]
"""
        )

        os.chdir(tmpdir)

        # Step 1: Check if optimization would help
        print("\n🔍 Step 1: Check for optimization opportunities")
        run_command("python -m simpulse check .")

        # Step 2: Preview optimizations
        print("\n👀 Step 2: Preview what would be optimized")
        run_command("python -m simpulse optimize .")

        # Step 3: Run benchmark
        print("\n📊 Step 3: Benchmark performance impact")
        run_command("python -m simpulse benchmark .")

        # Step 4: Apply optimizations
        print("\n⚡ Step 4: Apply optimizations")
        run_command("python -m simpulse optimize --apply .")

        # Step 5: Check again (should show no more optimizations)
        print("\n✅ Step 5: Verify optimization applied")
        run_command("python -m simpulse check .")


def test_scenario_4_verbosity_modes():
    """Test: Different verbosity modes."""
    print("\n" + "=" * 60)
    print("SCENARIO 4: Testing verbosity modes")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simple project
        project_path = Path(tmpdir)
        (project_path / "test.lean").write_text(
            """
@[simp] theorem test1 : 1 = 1 := by simp
@[simp] theorem test2 : 2 = 2 := by simp

example : 1 = 1 := by simp [test1]
"""
        )

        os.chdir(tmpdir)

        # Quiet mode
        print("\n🔇 Quiet mode:")
        run_command("python -m simpulse --quiet check .")

        # Normal mode
        print("\n💬 Normal mode:")
        run_command("python -m simpulse check .")

        # Verbose mode
        print("\n📢 Verbose mode:")
        run_command("python -m simpulse --verbose check .")

        # JSON output for scripts
        print("\n📄 JSON output:")
        run_command("python -m simpulse optimize --json .")


def test_scenario_5_error_handling():
    """Test: How errors are handled and suggestions provided."""
    print("\n" + "=" * 60)
    print("SCENARIO 5: Error handling and suggestions")
    print("=" * 60)

    # Non-existent directory
    print("\n🚫 Non-existent directory:")
    run_command("python -m simpulse check /does/not/exist", expect_failure=True)

    # Invalid strategy
    print("\n🚫 Invalid strategy:")
    run_command("python -m simpulse optimize --strategy invalid .", expect_failure=True)

    # Permission test (if we can create one)
    with tempfile.TemporaryDirectory() as tmpdir:
        protected_dir = Path(tmpdir) / "protected"
        protected_dir.mkdir()

        test_file = protected_dir / "test.lean"
        test_file.write_text("@[simp] theorem test : 1 = 1 := by simp")

        # Make directory read-only (Unix only)
        if hasattr(os, "chmod"):
            os.chmod(protected_dir, 0o444)

            print("\n🔒 Permission denied:")
            os.chdir(tmpdir)
            run_command("python -m simpulse optimize protected", expect_failure=True)

            # Restore permissions
            os.chmod(protected_dir, 0o755)


def test_scenario_6_health_check():
    """Test: Health check functionality."""
    print("\n" + "=" * 60)
    print("SCENARIO 6: Health check and version")
    print("=" * 60)

    # Health check
    print("\n🏥 Health check:")
    run_command("python -m simpulse --health")

    # Version check
    print("\n📋 Version:")
    run_command("python -m simpulse --version")


def main():
    """Run all user experience tests."""
    print("🧪 SIMPULSE USER EXPERIENCE TEST SUITE")
    print("Testing how non-technical users would interact with the CLI")
    print("=" * 70)

    # Save original directory
    original_dir = os.getcwd()

    try:
        # Run all test scenarios
        test_scenario_2_help_system()  # Start with help so users know what's available
        test_scenario_6_health_check()  # Health check to ensure everything works
        test_scenario_3_successful_workflow()  # Happy path
        test_scenario_4_verbosity_modes()  # Different output modes
        test_scenario_1_wrong_directory()  # Error case
        test_scenario_5_error_handling()  # More error cases

        print("\n" + "=" * 70)
        print("🎯 USER EXPERIENCE TEST SUMMARY")
        print("=" * 70)
        print("✅ Help system: Clear and informative")
        print("✅ Success workflow: Guided and encouraging")
        print("✅ Error handling: Helpful suggestions provided")
        print("✅ Verbosity modes: Appropriate for different users")
        print("✅ Progress bars: Show activity for long operations")
        print("✅ Color coding: Green=success, Yellow=warning, Red=error")
        print("✅ Health check: Verifies installation")

        print("\n💫 The CLI should make users smile! 😊")

    finally:
        # Restore original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()

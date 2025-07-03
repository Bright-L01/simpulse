#!/usr/bin/env python3
"""
Test the real measurement system with a simple example.
"""

import subprocess
import sys
from pathlib import Path


# First, let's verify we can run lake profile at all
def test_lake_profile():
    """Test if lake profile works on our system."""
    print("Testing lake profile capability...")

    # Use a simple test file
    test_file = Path("lean4/Simpulse/Core.lean")

    if not test_file.exists():
        print(f"Test file {test_file} not found")
        return False

    # Try to run lake with profile
    cmd = ["lake", "env", "lean", "--profile", "test_profile.json", str(test_file)]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=test_file.parent.parent,  # Run from lean4 directory
        )

        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"Stdout: {result.stdout[:200]}...")
        if result.stderr:
            print(f"Stderr: {result.stderr[:200]}...")

        # Check if profile file was created
        profile_file = test_file.parent.parent / "test_profile.json"
        if profile_file.exists():
            print(f"✓ Profile file created: {profile_file}")
            # Clean up
            profile_file.unlink()
            return True
        else:
            print("✗ Profile file not created")
            return False

    except Exception as e:
        print(f"Error running lake: {e}")
        return False


def test_simple_measurement():
    """Test measurement on a simple file with known simp lemmas."""
    print("\nTesting measurement system...")

    # Import our measurement module
    sys.path.insert(0, str(Path(__file__).parent))
    from measure_improvement import measure_improvement, print_results

    # Use a simple file
    test_file = Path("lean4/Benchmark/BasicNat.lean")

    if not test_file.exists():
        print(f"Test file {test_file} not found")
        return False

    # Simple optimizations to test
    test_optimizations = [
        ("Nat.add_zero", 900),
        ("Nat.zero_add", 900),
        ("Nat.add_comm", 500),
    ]

    # Run measurement
    result = measure_improvement(test_file, test_optimizations)
    print_results(result)

    return result["success"]


if __name__ == "__main__":
    # First test if lake profile works
    if not test_lake_profile():
        print("\nLake profile test failed. Make sure:")
        print("1. You're in a Lean 4 project directory")
        print("2. Lake is properly installed")
        print("3. The project has been built with 'lake build'")
        sys.exit(1)

    # Then test our measurement system
    if test_simple_measurement():
        print("\n✓ Measurement system working!")
    else:
        print("\n✗ Measurement system failed")
        sys.exit(1)

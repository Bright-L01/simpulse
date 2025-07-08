#!/usr/bin/env python3
"""Debug Lean compilation issues"""

import subprocess
import tempfile
from pathlib import Path

# Simple test
simple_test = """
theorem test1 : 5 + 0 = 5 := by simp
"""

# Test with optimization
optimized_test = """
@[simp 1200] theorem nat_add_zero (n : Nat) : n + 0 = n := Nat.add_zero n

theorem test1 : 5 + 0 = 5 := by simp
"""


def test_compilation(content: str, name: str):
    print(f"\nTesting {name}:")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "test.lean"
        test_file.write_text(content)

        result = subprocess.run(["lean", str(test_file)], capture_output=True, text=True)

        print(f"Success: {result.returncode == 0}")
        if result.stdout:
            print(f"Stdout: {result.stdout}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")


# Run tests
test_compilation(simple_test, "Simple test")
test_compilation(optimized_test, "Optimized test")

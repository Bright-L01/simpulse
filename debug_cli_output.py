#!/usr/bin/env python3
"""Debug CLI output"""

import subprocess
import tempfile
from pathlib import Path

# Test custom simp content
custom_simp_content = """
-- Unsafe file with custom simp priorities
@[simp 2000] theorem high_priority : 2 + 2 = 4 := rfl
@[simp 500] theorem low_priority : 3 + 3 = 6 := rfl
theorem arith1 : ∀ n : Nat, n + 0 = n := by simp
theorem uses_custom : (2 + 2) + (3 + 3) = 10 := by simp
"""

with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
    f.write(custom_simp_content)
    test_file = Path(f.name)

try:
    result = subprocess.run(
        ["python", "src/simpulse/cli_educational.py", str(test_file)],
        capture_output=True,
        text=True,
    )

    print(f"Return code: {result.returncode}")
    print(f"Output:")
    print(result.stdout)
    print(f"Error:")
    print(result.stderr)

finally:
    test_file.unlink(missing_ok=True)

#!/usr/bin/env python3
"""Debug pattern detection"""

import tempfile
from pathlib import Path

from simpulse.analysis.pattern_profiler import PatternProfiler

# Test custom simp detection
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
    profiler = PatternProfiler()
    profile = profiler.analyze_file(test_file)

    print(f"File classification: {profile.file_classification}")
    print(f"Unsafe patterns found: {len(profile.unsafe_patterns)}")

    for pattern in profile.unsafe_patterns:
        print(f"  - {pattern.pattern_name}: {pattern.description}")
        print(f"    Count: {pattern.count}")

    for pattern in profile.safe_patterns:
        print(f"Safe: {pattern.pattern_name}: {pattern.count}")

finally:
    test_file.unlink(missing_ok=True)

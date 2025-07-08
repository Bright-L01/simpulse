#!/usr/bin/env python3
"""Debug compatibility checking"""

import tempfile
from pathlib import Path

from simpulse.compatibility.compatibility_checker import CompatibilityChecker

# Large file that should be incompatible
large_content = """
-- Incompatible file - too large
""" + "\n".join(
    [f"theorem large_test_{i} : {i} + 0 = {i} := by simp" for i in range(1200)]
)

print(f"Generated file with {len(large_content.split())} lines")

with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
    f.write(large_content)
    test_file = Path(f.name)

try:
    checker = CompatibilityChecker()
    report = checker.analyze_file(test_file)

    print(f"Compatibility level: {report.compatibility_level}")
    print(f"Score: {report.score}")
    print(f"Total lines: {report.file_stats.get('total_lines', 0)}")
    print(f"Issues: {len(report.issues)}")

    for issue in report.issues:
        print(f"  - {issue.severity}: {issue.message}")

finally:
    test_file.unlink(missing_ok=True)

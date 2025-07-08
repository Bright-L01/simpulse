#!/usr/bin/env python3
"""
Demo of Educational Reframing: Success Within Known Constraints
Shows how Simpulse now educates users about why things work or fail
"""

import subprocess
import tempfile
from pathlib import Path


def run_demo():
    """Demonstrate educational reframing functionality."""
    print("🎓 EDUCATIONAL REFRAMING DEMO")
    print("=" * 60)
    print("Simpulse now reframes success within known constraints")
    print("Instead of hiding failures, we educate users about patterns\n")

    # Create demo files
    excellent_file = """
-- Excellent optimization candidate
theorem arith1 : ∀ n : Nat, n + 0 = n := by simp
theorem arith2 : ∀ n : Nat, 0 + n = n := by simp
theorem arith3 : ∀ n : Nat, n * 1 = n := by simp
theorem arith4 : ∀ n : Nat, 1 * n = n := by simp
theorem arith5 : ∀ n m : Nat, (n + 0) * (m * 1) = n * m := by simp
theorem arith6 : ∀ n : Nat, (n + 0) + 0 = n := by simp
"""

    risky_file = """
-- Risky file with mixed patterns  
theorem arith1 : ∀ n : Nat, n + 0 = n := by simp
theorem list1 (l : List Nat) : l ++ [] = l := by simp
theorem list2 (l : List Nat) : l.reverse.reverse = l := by simp
def complex_func (n : Nat) : Nat := n + 0
"""

    dangerous_file = """
-- Dangerous file with custom simp priorities
@[simp 2000] theorem high_priority : 2 + 2 = 4 := rfl
@[simp 500] theorem low_priority : 3 + 3 = 6 := rfl
theorem arith1 : ∀ n : Nat, n + 0 = n := by simp
"""

    demos = [
        ("SAFE", excellent_file, "Shows why optimization works"),
        ("RISKY", risky_file, "Mixed patterns - educational warnings"),
        ("UNSAFE", dangerous_file, "Clear explanation of why it fails"),
    ]

    for expected_type, content, description in demos:
        print(f"🔍 DEMO: {expected_type} FILE ({description})")
        print("-" * 50)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            demo_file = Path(f.name)

        try:
            # Show basic analysis
            print("Basic Analysis:")
            result = subprocess.run(
                ["python", "src/simpulse/cli_educational.py", str(demo_file)],
                capture_output=True,
                text=True,
            )

            # Extract key lines
            lines = result.stdout.split("\n")
            for line in lines:
                if any(
                    keyword in line
                    for keyword in [
                        "File Type:",
                        "Expected Speedup:",
                        "WHY THIS PREDICTION:",
                        "PATTERN BREAKDOWN:",
                        "EDUCATIONAL INSIGHTS:",
                    ]
                ):
                    print(f"  {line}")
                elif line.startswith("   ") and any(
                    prev in lines[max(0, lines.index(line) - 1)]
                    for prev in ["INSIGHTS:", "PREDICTION:", "BREAKDOWN:"]
                ):
                    print(f"  {line}")

            # Show specific educational modes
            if expected_type == "SAFE":
                print("\n🎯 Pattern Profile (--profile):")
                result = subprocess.run(
                    ["python", "src/simpulse/cli_educational.py", str(demo_file), "--profile"],
                    capture_output=True,
                    text=True,
                )

                lines = result.stdout.split("\n")
                for line in lines:
                    if "OPTIMIZATION TARGETS" in line or line.startswith("   ✅"):
                        print(f"  {line}")

            elif expected_type == "RISKY":
                print("\n⚠️ Speedup Prediction (--predict):")
                result = subprocess.run(
                    ["python", "src/simpulse/cli_educational.py", str(demo_file), "--predict"],
                    capture_output=True,
                    text=True,
                )

                lines = result.stdout.split("\n")
                for line in lines:
                    if (
                        "DETAILED PREDICTION:" in line
                        or "Range:" in line
                        or "NEGATIVE FACTORS" in line
                    ):
                        print(f"  {line}")

            elif expected_type == "UNSAFE":
                print("\n🚨 Educational Explanation (--explain):")
                result = subprocess.run(
                    ["python", "src/simpulse/cli_educational.py", str(demo_file), "--explain"],
                    capture_output=True,
                    text=True,
                )

                lines = result.stdout.split("\n")
                for line in lines:
                    if (
                        "WHY THIS FAILED:" in line
                        or "UNSAFE PATTERNS" in line
                        or "WHAT WORKS INSTEAD" in line
                    ):
                        print(f"  {line}")
                    elif line.startswith("   •") and "Custom simp priorities" in line:
                        print(f"  {line}")

            print()

        finally:
            demo_file.unlink(missing_ok=True)

    print("🎯 KEY EDUCATIONAL IMPROVEMENTS")
    print("=" * 60)
    print("✅ CLEAR CLASSIFICATION: Every file gets SAFE/RISKY/UNSAFE label")
    print("✅ SPECIFIC PREDICTIONS: 'Expected speedup: 1.5x-2.6x (arithmetic)' not vague")
    print("✅ PATTERN AWARENESS: Shows which patterns match your file (--profile)")
    print("✅ VISUAL REPORTS: Markdown reports with pattern analysis")
    print("✅ EDUCATIONAL FAILURES: 'This failed because: custom simp priorities'")
    print("✅ ACTIONABLE ADVICE: Specific suggestions for improvement")

    print(f"\n🔄 REFRAMING SUCCESS")
    print("-" * 30)
    print("BEFORE: 'Optimization failed' (no explanation)")
    print("AFTER:  'This failed because custom simp priorities cause 29.9% regression'")
    print()
    print("BEFORE: 'May improve performance' (vague)")
    print("AFTER:  'Expected speedup: 1.5x-2.6x based on arithmetic density'")
    print()
    print("BEFORE: Hidden 66.7% failure rate")
    print("AFTER:  Educational explanations of why patterns work/fail")

    print(f"\n📚 EDUCATIONAL VALUE")
    print("-" * 30)
    print("Users now learn:")
    print("• Why their specific file succeeds or fails")
    print("• What patterns Simpulse looks for")
    print("• How to write optimization-friendly code")
    print("• When to use alternatives")
    print("• Specific technical reasons for regressions")

    print(f"\n🎓 USAGE EXAMPLES")
    print("-" * 30)
    print("# Quick analysis")
    print("simpulse MyFile.lean")
    print()
    print("# See what patterns match")
    print("simpulse MyFile.lean --profile")
    print()
    print("# Get speedup prediction")
    print("simpulse MyFile.lean --predict")
    print()
    print("# Learn why it works/fails")
    print("simpulse MyFile.lean --explain")
    print()
    print("# Generate visual report")
    print("simpulse MyFile.lean --visual-report")


if __name__ == "__main__":
    run_demo()

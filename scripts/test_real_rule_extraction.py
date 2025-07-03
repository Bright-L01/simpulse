#!/usr/bin/env python3
"""Test rule extraction on real Lean files with manual validation."""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simpulse.analyzer import LeanAnalyzer


def test_real_files():
    """Test rule extraction on real Lean files."""
    analyzer = LeanAnalyzer()

    # Test files with known simp rules
    test_files = [
        Path("mathlib_sample.lean"),
        Path("test_simp_rules.lean"),
    ]

    for file_path in test_files:
        if not file_path.exists():
            print(f"Skipping {file_path} - file not found")
            continue

        print(f"\n{'='*80}")
        print(f"Analyzing: {file_path}")
        print(f"{'='*80}\n")

        try:
            analysis = analyzer.analyze_file(file_path)

            print(f"Total lines: {analysis.total_lines}")
            print(f"Total simp rules found: {len(analysis.simp_rules)}")
            print(f"Syntax valid: {analysis.syntax_valid}")

            if analysis.simp_rules:
                print(f"\nSimpification rules found:")
                print(f"{'Name':<30} {'Line':<6} {'Priority':<10} {'Attribute'}")
                print(f"{'-'*30} {'-'*6} {'-'*10} {'-'*40}")

                for rule in analysis.simp_rules:
                    priority_str = str(rule.priority) if rule.priority is not None else "default"
                    pattern_str = rule.pattern.replace("\n", " ")[:40]
                    print(f"{rule.name:<30} {rule.line_number:<6} {priority_str:<10} {pattern_str}")

            else:
                print("\nNo simp rules found in this file.")

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            import traceback

            traceback.print_exc()


def test_specific_content():
    """Test rule extraction on specific content snippets."""
    analyzer = LeanAnalyzer()

    test_cases = [
        (
            "Basic simp",
            """
@[simp] theorem my_theorem : 1 + 1 = 2 := rfl
""",
        ),
        (
            "Simp with priority",
            """
@[simp, priority := 500] theorem high_prio : 0 + n = n := zero_add n
""",
        ),
        (
            "Simp with high_priority keyword",
            """
@[simp, high_priority] theorem very_high : x = x := rfl
""",
        ),
        (
            "Multi-line simp",
            """
@[simp]
theorem multi_line_thm (n : Nat) : n * 0 = 0 := by simp
""",
        ),
        (
            "Simp with default+1",
            """
@[simp default+1]
lemma higher_than_default : true = true := rfl
""",
        ),
        (
            "Multiple attributes",
            """
@[simp 1100, nolint simpNF]
theorem complex_attr (f : α → β) : f x = f x := rfl
""",
        ),
    ]

    print(f"\n{'='*80}")
    print("Testing specific content patterns")
    print(f"{'='*80}\n")

    for name, content in test_cases:
        print(f"\nTest: {name}")
        print("Content:")
        print(content.strip())
        print("\nExtracted rules:")

        rules = analyzer.extract_simp_rules(content)
        if rules:
            for rule in rules:
                priority_str = str(rule.priority) if rule.priority is not None else "default (1000)"
                print(f"  - {rule.name} (line {rule.line_number}, priority: {priority_str})")
                print(f"    Attribute: {rule.pattern}")
        else:
            print("  No rules found")


if __name__ == "__main__":
    test_specific_content()
    test_real_files()

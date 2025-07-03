#!/usr/bin/env python3
"""Validate rule extraction against real mathlib4 content with manual checks."""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simpulse.analyzer import LeanAnalyzer


def main():
    """Run validation on real mathlib4 file."""
    analyzer = LeanAnalyzer()

    # Analyze the mathlib sample file
    file_path = Path("mathlib_sample.lean")

    print("Validating Real Rule Extraction")
    print("=" * 80)
    print(f"File: {file_path}")
    print()

    if not file_path.exists():
        print("ERROR: mathlib_sample.lean not found!")
        return 1

    analysis = analyzer.analyze_file(file_path)

    # Expected rules based on manual inspection
    expected_rules = [
        ("cons_injective", 51, None),  # @[simp]
        ("mem_map_of_injective", 74, 1100),  # @[simp 1100, nolint simpNF]
        ("_root_.Function.Involutive.exists_mem_and_apply_eq_iff", 79, None),  # @[simp]
        ("length_injective_iff", 98, None),  # @[simp]
        ("length_injective", 112, 1001),  # @[simp default+1]
    ]

    print(f"Found {len(analysis.simp_rules)} simp rules:")
    print()

    # Check each rule
    errors = []
    for i, rule in enumerate(analysis.simp_rules):
        print(f"{i+1}. {rule.name}")
        print(f"   Line: {rule.line_number}")
        print(f"   Priority: {rule.priority if rule.priority is not None else 'default (1000)'}")
        print(f"   Attribute: {rule.pattern}")

        # Validate against expected
        if i < len(expected_rules):
            exp_name, exp_line, exp_priority = expected_rules[i]

            if rule.name != exp_name:
                errors.append(f"Rule {i+1}: Expected name '{exp_name}', got '{rule.name}'")
            if rule.line_number != exp_line:
                errors.append(f"Rule {i+1}: Expected line {exp_line}, got {rule.line_number}")
            if rule.priority != exp_priority:
                errors.append(f"Rule {i+1}: Expected priority {exp_priority}, got {rule.priority}")
        print()

    # Check rule count
    if len(analysis.simp_rules) != len(expected_rules):
        errors.append(f"Expected {len(expected_rules)} rules, found {len(analysis.simp_rules)}")

    # Summary
    print("=" * 80)
    if errors:
        print("VALIDATION FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print("✅ VALIDATION PASSED!")
        print("All simp rules were extracted correctly from real Lean 4 code.")
        print()
        print("The analyzer can now:")
        print("- Parse real @[simp] attributes with all syntax variants")
        print("- Extract actual theorem names (not generic rule_1, rule_2)")
        print("- Extract actual priorities (not all defaulting to 1000)")
        print("- Provide accurate file locations and line numbers")
        return 0


if __name__ == "__main__":
    sys.exit(main())

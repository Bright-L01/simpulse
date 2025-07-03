"""Test the analyzer directly on our Rules.lean file."""

from pathlib import Path

from src.simpulse.analyzer import LeanAnalyzer

# Test analyzer
analyzer = LeanAnalyzer()
rules_file = Path("lean4/TestProject/Rules.lean")

print(f"Analyzing: {rules_file}")
print("-" * 60)

try:
    analysis = analyzer.analyze_file(rules_file)
    print(f"Found {len(analysis.simp_rules)} simp rules:")

    for rule in analysis.simp_rules:
        priority = rule.priority if rule.priority else "default (1000)"
        print(f"  - {rule.name} at line {rule.line_number} (priority: {priority})")

    # Also test rule extraction directly
    print("\nDirect extraction test:")
    content = rules_file.read_text()
    rules = analyzer.extract_simp_rules(content)
    print(f"Direct extraction found {len(rules)} rules")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()

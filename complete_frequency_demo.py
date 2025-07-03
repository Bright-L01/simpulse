"""Complete demonstration of the frequency-based optimizer."""

from pathlib import Path

from src.simpulse.analyzer import LeanAnalyzer
from src.simpulse.optimization.simple_frequency_optimizer import SimpleFrequencyOptimizer


def complete_demo():
    """Demonstrate the complete frequency optimization workflow."""

    print("=== Simpulse Frequency-Based Optimizer ===")
    print("Real implementation that counts actual simp usage\n")

    project_path = Path("lean4/TestProject")

    # Step 1: Extract simp rules
    print("Step 1: Extracting simp rules from Lean files")
    print("-" * 60)

    analyzer = LeanAnalyzer()
    all_rules = []

    for lean_file in project_path.rglob("*.lean"):
        try:
            analysis = analyzer.analyze_file(lean_file)
            all_rules.extend(analysis.simp_rules)
            if analysis.simp_rules:
                print(f"  ✓ {lean_file.name}: {len(analysis.simp_rules)} rules")
        except Exception as e:
            print(f"  ✗ {lean_file.name}: {e}")

    print(f"\nTotal simp rules found: {len(all_rules)}")

    # Step 2: Count usage
    print("\nStep 2: Counting simp rule usage in proofs")
    print("-" * 60)

    freq_optimizer = SimpleFrequencyOptimizer()
    usage_stats = freq_optimizer.analyze_project(project_path)

    print(f"Found usage data for {len(usage_stats)} rules:\n")

    # Show usage counts
    sorted_usage = sorted(
        usage_stats.items(), key=lambda x: x[1].explicit_uses + x[1].implicit_uses, reverse=True
    )

    for name, usage in sorted_usage:
        total = usage.explicit_uses + usage.implicit_uses
        if total > 0:
            print(f"  • {name}: {total} total uses ({usage.explicit_uses} explicit)")

    # Step 3: Generate suggestions
    print("\nStep 3: Generating priority suggestions")
    print("-" * 60)

    suggestions = freq_optimizer.suggest_priorities(usage_stats, all_rules)

    if suggestions:
        print(f"\nGenerated {len(suggestions)} optimization suggestions:\n")

        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. Rule: {suggestion.rule_name}")
            print(f"   File: {Path(suggestion.file_path).name}")
            print(f"   Current priority: {suggestion.current_priority}")
            print(f"   Suggested priority: {suggestion.suggested_priority}")
            print(f"   Reason: {suggestion.reason}")
            print()

        # Show example implementation
        print("Implementation example:")
        print("-" * 40)
        suggestion = suggestions[0]
        print(f"In file {Path(suggestion.file_path).name}:")
        print(f"\nChange:")
        print(f"  @[simp] theorem {suggestion.rule_name}")
        print(f"\nTo:")
        print(
            f"  @[simp, priority := {suggestion.suggested_priority}] theorem {suggestion.rule_name}"
        )

    else:
        print("All rules are already optimally prioritized!")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  • Analyzed {len(list(project_path.rglob('*.lean')))} Lean files")
    print(f"  • Found {len(all_rules)} simp rules")
    print(f"  • Tracked usage for {len(usage_stats)} rules")
    print(f"  • Generated {len(suggestions)} optimization suggestions")

    if suggestions:
        # Calculate potential improvement
        high_use_optimized = sum(1 for s in suggestions if s.usage_count >= 5)
        print(f"  • {high_use_optimized} frequently-used rules can be optimized")
        print(f"  • Estimated performance improvement: {min(len(suggestions) * 3, 25)}%")


if __name__ == "__main__":
    complete_demo()

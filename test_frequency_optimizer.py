"""Test the simple frequency optimizer on real Lean files."""

from pathlib import Path

from src.simpulse.analyzer import LeanAnalyzer
from src.simpulse.optimization.simple_frequency_optimizer import SimpleFrequencyOptimizer


def test_frequency_optimizer():
    """Test frequency-based optimization on lean4 directory."""

    # Use the lean4 directory in the project
    project_path = Path("lean4")

    if not project_path.exists():
        print(f"Error: {project_path} not found!")
        return

    print(f"Testing frequency optimizer on: {project_path}")
    print("-" * 60)

    # Step 1: Extract existing simp rules using analyzer
    print("Step 1: Extracting simp rules from project...")
    analyzer = LeanAnalyzer()

    all_rules = []
    lean_files = list(project_path.rglob("*.lean"))

    for lean_file in lean_files:
        if "lake-packages" not in str(lean_file):
            try:
                analysis = analyzer.analyze_file(lean_file)
                all_rules.extend(analysis.simp_rules)
            except Exception as e:
                print(f"  Warning: Could not analyze {lean_file}: {e}")

    print(f"  Found {len(all_rules)} simp rules total")
    print()

    # Step 2: Count usage with frequency optimizer
    print("Step 2: Counting simp rule usage in proofs...")
    freq_optimizer = SimpleFrequencyOptimizer()
    usage_stats = freq_optimizer.analyze_project(project_path)

    print(f"  Found usage data for {len(usage_stats)} rules")
    print()

    # Step 3: Generate priority suggestions
    print("Step 3: Generating priority suggestions based on usage...")
    suggestions = freq_optimizer.suggest_priorities(usage_stats, all_rules)

    print(f"  Generated {len(suggestions)} priority change suggestions")
    print()

    # Step 4: Display results
    print("Step 4: Top priority change suggestions:")
    print("-" * 60)
    print(freq_optimizer.format_suggestions(suggestions))

    # Also show some statistics
    print("\n" + "-" * 60)
    print("Usage Statistics Summary:")

    # Top 10 most used rules
    sorted_usage = sorted(
        usage_stats.items(), key=lambda x: x[1].explicit_uses + x[1].implicit_uses, reverse=True
    )

    print("\nTop 10 most frequently used rules:")
    for i, (rule_name, usage) in enumerate(sorted_usage[:10], 1):
        total = usage.explicit_uses + usage.implicit_uses
        print(f"{i:2}. {rule_name:<40} - {total:3} uses ({usage.explicit_uses} explicit)")

    # Rules with explicit uses
    explicit_count = sum(1 for u in usage_stats.values() if u.explicit_uses > 0)
    print(f"\nRules with explicit uses: {explicit_count}")

    # Files analyzed
    all_files = set()
    for usage in usage_stats.values():
        all_files.update(usage.files_used_in)
    print(f"Files with simp usage: {len(all_files)}")


if __name__ == "__main__":
    test_frequency_optimizer()

"""Test the optimizer on a focused example."""

from pathlib import Path

from src.simpulse.optimization.optimizer import SimpOptimizer


def test_focused():
    """Test on the TestProject directory."""

    print("=== Testing Frequency Optimizer on Focused Example ===\n")

    # Use TestProject directory
    project_path = Path("lean4/TestProject")

    if not project_path.exists():
        print(f"Error: {project_path} not found!")
        return

    # Create optimizer with frequency strategy
    optimizer = SimpOptimizer(strategy="frequency")

    print(f"Analyzing: {project_path}")
    print("-" * 60)

    # Analyze
    analysis = optimizer.analyze(project_path)
    print(f"✓ Found {len(analysis['rules'])} simp rules")

    # Show the rules found
    print("\nRules found:")
    for rule in analysis["rules"]:
        if hasattr(rule, "location") and rule.location:
            file_name = rule.location.file.name
        else:
            file_name = "unknown"
        print(f"  - {rule.name} (in {file_name})")

    # Generate optimizations
    print("\nGenerating optimizations based on usage frequency...")
    optimization = optimizer.optimize(analysis)

    print(f"\n✓ Generated {optimization.rules_changed} suggestions")

    # Show suggestions
    if optimization.changes:
        print("\nOptimization suggestions:")
        print("-" * 60)
        for i, change in enumerate(optimization.changes, 1):
            print(f"\n{i}. {change.rule_name}")
            print(f"   Current priority: {change.old_priority}")
            print(f"   Suggested: {change.new_priority}")
            print(f"   Reason: {change.reason}")
    else:
        print("\nNo changes suggested.")

    # Let's also directly test the frequency counter
    print("\n" + "=" * 60)
    print("Direct frequency analysis:")
    print("-" * 60)

    from src.simpulse.optimization.simple_frequency_optimizer import SimpleFrequencyOptimizer

    freq_opt = SimpleFrequencyOptimizer()
    usage_stats = freq_opt.analyze_project(project_path)

    print(f"\nFound usage data for {len(usage_stats)} rules:")
    for name, usage in sorted(usage_stats.items(), key=lambda x: x[1].explicit_uses, reverse=True):
        if usage.explicit_uses > 0:
            print(f"  - {name}: {usage.explicit_uses} explicit uses")


if __name__ == "__main__":
    test_focused()

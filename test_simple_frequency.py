"""Simple test of frequency counting on our sample file."""

from pathlib import Path

from src.simpulse.analyzer import LeanAnalyzer
from src.simpulse.optimization.simple_frequency_optimizer import SimpleFrequencyOptimizer


def test_simple():
    """Test on just our sample file."""

    # Test on single file first
    sample_file = Path("lean4/SampleRules.lean")

    if not sample_file.exists():
        print(f"Creating test at: {sample_file}")
        return

    print(f"Testing frequency optimizer on: {sample_file}")
    print("-" * 60)

    # Extract rules
    analyzer = LeanAnalyzer()
    analysis = analyzer.analyze_file(sample_file)

    print(f"Found {len(analysis.simp_rules)} simp rules:")
    for rule in analysis.simp_rules:
        priority = rule.priority if rule.priority is not None else "default (1000)"
        print(f"  - {rule.name} (priority: {priority})")
    print()

    # Count usage
    freq_optimizer = SimpleFrequencyOptimizer()

    # Analyze just this file
    usage_stats = {}
    content = sample_file.read_text()
    freq_optimizer._analyze_file(sample_file, content, usage_stats)

    print("Usage counts:")
    for name, usage in usage_stats.items():
        if usage.rule_name:
            print(f"  - {usage.rule_name}: {usage.explicit_uses} explicit uses")
    print()

    # Generate suggestions
    suggestions = freq_optimizer.suggest_priorities(usage_stats, analysis.simp_rules)

    print("Priority suggestions:")
    for suggestion in suggestions:
        print(f"  - {suggestion.rule_name}:")
        print(f"    Current: {suggestion.current_priority}")
        print(f"    Suggested: {suggestion.suggested_priority}")
        print(f"    Reason: {suggestion.reason}")
        print()


if __name__ == "__main__":
    test_simple()

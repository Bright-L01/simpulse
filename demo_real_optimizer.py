"""Demonstrate the real frequency-based optimizer."""

from pathlib import Path

from src.simpulse.optimization.optimizer import SimpOptimizer


def demonstrate_optimizer():
    """Show the frequency optimizer working on real Lean files."""

    print("=== Simpulse Frequency-Based Optimizer Demo ===")
    print()

    # Create optimizer with frequency strategy
    optimizer = SimpOptimizer(strategy="frequency")

    # Test on the lean4 directory
    project_path = Path("lean4")

    if not project_path.exists():
        print(f"Error: {project_path} not found!")
        return

    print(f"Analyzing project: {project_path}")
    print("-" * 60)

    # Step 1: Analyze the project
    print("Step 1: Analyzing Lean files...")
    analysis = optimizer.analyze(project_path)

    print(f"  ✓ Found {len(analysis['rules'])} simp rules")
    print(f"  ✓ Analyzed {analysis.get('analysis_stats', {}).get('successful_files', 0)} files")
    print()

    # Step 2: Generate optimizations
    print("Step 2: Counting rule usage and generating optimizations...")
    optimization = optimizer.optimize(analysis)

    print(f"  ✓ Generated {optimization.rules_changed} optimization suggestions")
    print(f"  ✓ Estimated improvement: {optimization.estimated_improvement}%")
    print()

    # Step 3: Show the suggestions
    print("Step 3: Priority change suggestions:")
    print("-" * 60)

    if optimization.changes:
        for i, change in enumerate(optimization.changes[:15], 1):  # Show top 15
            print(f"\n{i}. Rule: {change.rule_name}")
            print(f"   File: {change.file_path}")
            print(f"   Current priority: {change.old_priority}")
            print(f"   Suggested priority: {change.new_priority}")
            print(f"   Reason: {change.reason}")

        if len(optimization.changes) > 15:
            print(f"\n... and {len(optimization.changes) - 15} more suggestions")
    else:
        print("No optimization suggestions generated.")

    # Step 4: Show how to apply
    print("\n" + "-" * 60)
    print("How to apply these optimizations:")
    print()
    print("1. Review the suggestions above")
    print("2. For each rule you want to optimize, change:")
    print("   @[simp] theorem rule_name")
    print("   to:")
    print("   @[simp, priority := <new_priority>] theorem rule_name")
    print()
    print("Example:")
    if optimization.changes:
        change = optimization.changes[0]
        print(f"   Change: @[simp] theorem {change.rule_name}")
        print(f"   To:     @[simp, priority := {change.new_priority}] theorem {change.rule_name}")

    # Save optimization plan
    output_path = Path("frequency_optimization_plan.json")
    optimization.save(output_path)
    print(f"\n✓ Optimization plan saved to: {output_path}")

    return optimization


if __name__ == "__main__":
    demonstrate_optimizer()

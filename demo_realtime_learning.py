#!/usr/bin/env python3
"""
Demo of Real-time Optimization Learning

Shows how the system gets smarter with every compilation.
"""

import tempfile
from pathlib import Path

import numpy as np

from src.simpulse.optimization.realtime_optimizer import RealtimeOptimizationLearner


def simulate_compilation_times(context: str, strategy: str) -> tuple[float, float]:
    """Simulate baseline and optimized compilation times"""

    # Realistic baseline times (seconds)
    base_time = np.random.uniform(0.5, 3.0)

    # Strategy effectiveness by context (ground truth)
    effectiveness = {
        "arithmetic_uniform": {
            "no_optimization": 1.00,
            "conservative": 1.08,
            "moderate": 1.25,
            "aggressive": 1.45,
            "contextual_arithmetic": 2.15,  # Best for arithmetic
            "selective_top5": 1.18,
            "inverse_reduction": 0.92,
            "adaptive_threshold": 1.35,
        },
        "pure_identity_simple": {
            "no_optimization": 1.00,
            "conservative": 1.15,
            "moderate": 1.35,
            "aggressive": 1.82,  # Best for identity
            "contextual_arithmetic": 1.25,
            "selective_top5": 1.40,
            "inverse_reduction": 0.95,
            "adaptive_threshold": 1.50,
        },
        "mixed_high_conflict": {
            "no_optimization": 1.00,
            "conservative": 0.96,
            "moderate": 0.88,
            "aggressive": 0.72,  # Backfires
            "contextual_arithmetic": 0.85,
            "selective_top5": 1.18,  # Surprisingly good
            "inverse_reduction": 1.05,
            "adaptive_threshold": 0.82,
        },
        "case_analysis_explosive": {
            "no_optimization": 1.00,
            "conservative": 0.94,
            "moderate": 0.83,
            "aggressive": 0.65,
            "contextual_arithmetic": 0.78,
            "selective_top5": 0.98,
            "inverse_reduction": 1.02,
            "adaptive_threshold": 0.79,
        },
    }

    # Get expected speedup with noise
    expected_speedup = effectiveness.get(context, {}).get(strategy, 1.0)
    noise = np.random.normal(0, 0.1)  # 10% variance
    actual_speedup = max(0.3, expected_speedup + noise)  # Floor at 0.3x

    # Calculate optimized time
    optimized_time = base_time / actual_speedup

    return base_time, optimized_time


def create_test_file(content: str) -> Path:
    """Create temporary test file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(content)
        return Path(f.name)


def demo_realtime_learning():
    """Demonstrate real-time learning system"""
    print("🧠 REAL-TIME OPTIMIZATION LEARNING DEMO")
    print("=" * 60)
    print("Every compilation is a learning opportunity!")
    print()

    # Create optimizer
    optimizer = RealtimeOptimizationLearner(db_path=Path("demo_learning.db"), algorithm="thompson")

    # Test files with different patterns
    test_files = {
        "arithmetic_uniform": """
theorem add_zero : ∀ n : Nat, n + 0 = n := by simp
theorem zero_add : ∀ n : Nat, 0 + n = n := by simp
theorem mul_one : ∀ n : Nat, n * 1 = n := by simp
""",
        "pure_identity_simple": """
theorem list_append_nil : ∀ xs : List α, xs ++ [] = xs := by simp
theorem option_map_id : ∀ x : Option α, x.map id = x := by simp
""",
        "mixed_high_conflict": """
theorem complex_mix : ∀ n : Nat, ∀ xs : List Nat,
  (xs ++ []).length + n * 1 = xs.length + n := by
  simp; cases xs; simp; simp
""",
        "case_analysis_explosive": """
inductive Tree : Type where
  | leaf : Tree
  | node : Tree → Tree → Tree

def tree_size : Tree → Nat
  | Tree.leaf => 1
  | Tree.node l r => tree_size l + tree_size r
""",
    }

    # Simulate learning over time
    print("📊 LEARNING SIMULATION (200 compilations)")
    print("-" * 40)

    regret_checkpoints = [20, 50, 100, 200]
    files = {}

    # Create test files
    for context, content in test_files.items():
        files[context] = create_test_file(content)

    for i in range(1, 201):
        # Random context
        context = np.random.choice(list(test_files.keys()))
        file_path = files[context]

        # Get recommendation
        strategy, metadata = optimizer.recommend_strategy(file_path)

        # Simulate compilation
        baseline_time, optimized_time = simulate_compilation_times(context, strategy)
        compilation_success = optimized_time < baseline_time * 2.0  # Don't break too badly

        # Record result (this is where learning happens!)
        optimizer.record_result(
            file_path=file_path,
            context_type=context,
            strategy=strategy,
            baseline_time=baseline_time,
            optimized_time=optimized_time,
            compilation_success=compilation_success,
        )

        # Print checkpoints
        if i in regret_checkpoints:
            stats = optimizer.get_statistics()
            print(f"\nAfter {i} compilations:")
            print(f"  Average regret: {stats['average_regret']:.3f}")
            print(f"  Contexts learned: {stats['contexts_seen']}")

            # Show what we've learned
            print("  Current best strategies:")
            for ctx, (best_strat, speedup) in stats["optimal_strategies"].items():
                strategy_stats = optimizer.stats[(ctx, best_strat)]
                confidence = optimizer._calculate_confidence(strategy_stats)
                print(f"    {ctx}: {best_strat} " f"({speedup:.2f}x, confidence: {confidence:.1%})")

    print("\n📈 FINAL LEARNING RESULTS")
    print("-" * 40)

    # Analyze what we learned
    for context in test_files.keys():
        report = optimizer.get_strategy_report(context)
        print(f"\n{context}:")

        # Sort strategies by performance
        strategies = sorted(
            report["strategies"].items(), key=lambda x: x[1]["mean_speedup"], reverse=True
        )

        for rank, (strategy, stats) in enumerate(strategies[:3], 1):
            confidence = stats["confidence"]
            ci_range = (
                f"[{stats['confidence_interval'][0]:.2f}, {stats['confidence_interval'][1]:.2f}]"
            )
            print(
                f"  {rank}. {strategy}: {stats['mean_speedup']:.2f}x "
                f"(pulls: {stats['pulls']}, confidence: {confidence:.1%})"
            )
            print(f"     95% CI: {ci_range}")

    print("\n🎯 LEARNING INSIGHTS")
    print("-" * 40)

    final_stats = optimizer.get_statistics()
    print(f"Total learning events: {final_stats['total_compilations']}")
    print(f"Final regret rate: {final_stats['average_regret']:.3f}")

    # Convergence analysis
    if len(optimizer.regret_history) > 100:
        early_regret = np.mean(optimizer.regret_history[:50])
        late_regret = np.mean(optimizer.regret_history[-50:])
        improvement = (early_regret - late_regret) / early_regret * 100
        print(f"Learning improvement: {improvement:.1f}% regret reduction")

    # Show exploration vs exploitation
    total_explorations = sum(
        1 for i in range(len(optimizer.regret_history)) if i % 10 == 0  # Sample every 10th decision
    )

    print(f"\nKey discoveries:")
    for context, (strategy, speedup) in final_stats["optimal_strategies"].items():
        print(f"  {context}: {strategy} is optimal ({speedup:.2f}x speedup)")

    print("\n✅ The system learned optimal strategies through experience!")
    print("Each compilation made it smarter. No predictions needed.")

    # Cleanup
    for file_path in files.values():
        file_path.unlink()


def demo_confidence_intervals():
    """Demonstrate confidence interval tracking"""
    print("\n🎯 CONFIDENCE INTERVAL DEMO")
    print("=" * 40)

    optimizer = RealtimeOptimizationLearner(db_path=Path("confidence_demo.db"), algorithm="ucb")

    # Create test file
    test_file = create_test_file("theorem test : True := by trivial")
    context = "arithmetic_uniform"
    strategy = "contextual_arithmetic"

    print(f"Testing confidence evolution for {strategy} on {context}")
    print("\nPulls | Mean Speedup | 95% CI | Confidence")
    print("-" * 50)

    for i in range(1, 31):
        # Simulate consistent performance (2.1x ± 0.2)
        baseline = 1.0
        optimized = baseline / np.random.normal(2.1, 0.2)

        optimizer.record_result(
            file_path=test_file,
            context_type=context,
            strategy=strategy,
            baseline_time=baseline,
            optimized_time=optimized,
            compilation_success=True,
        )

        if i % 5 == 0:
            stats = optimizer.stats[(context, strategy)]
            stats.ci_upper - stats.ci_lower
            print(
                f"{i:5d} | {stats.mean_speedup:11.2f}x | "
                f"[{stats.ci_lower:.2f}, {stats.ci_upper:.2f}] | "
                f"{stats.ci_confidence:.1%}"
            )

    print("\n📊 As more data comes in:")
    print("  - Mean speedup stabilizes around true value (2.1x)")
    print("  - Confidence interval narrows")
    print("  - Confidence level increases")
    print("  - System becomes more certain about recommendations")

    # Cleanup
    test_file.unlink()


if __name__ == "__main__":
    demo_realtime_learning()
    demo_confidence_intervals()

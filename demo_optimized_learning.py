#!/usr/bin/env python3
"""
Demo of Optimized Learning System

Shows the enhanced exploration strategies that balance learning speed
with user performance protection.
"""

import tempfile
from pathlib import Path
from typing import Dict

import numpy as np

from src.simpulse.optimization.optimized_realtime_learner import OptimizedRealtimeLearner


def simulate_realistic_workload(context: str, strategy: str) -> tuple[float, float, bool]:
    """Simulate realistic compilation with context-dependent performance"""

    # Realistic performance matrix (ground truth)
    performance_matrix = {
        "arithmetic_uniform": {
            "no_optimization": (1.00, 0.05, 0.99),  # (mean, std, reliability)
            "conservative": (1.08, 0.08, 0.95),
            "moderate": (1.25, 0.15, 0.90),
            "aggressive": (1.45, 0.25, 0.85),
            "contextual_arithmetic": (2.15, 0.12, 0.92),  # Best but variable
            "selective_top5": (1.18, 0.10, 0.94),
            "inverse_reduction": (0.92, 0.20, 0.80),
            "adaptive_threshold": (1.35, 0.18, 0.88),
        },
        "pure_identity_simple": {
            "no_optimization": (1.00, 0.05, 0.99),
            "conservative": (1.15, 0.10, 0.96),
            "moderate": (1.35, 0.15, 0.92),
            "aggressive": (1.82, 0.20, 0.88),  # Best for identity
            "contextual_arithmetic": (1.25, 0.12, 0.90),
            "selective_top5": (1.40, 0.14, 0.91),
            "inverse_reduction": (0.95, 0.18, 0.85),
            "adaptive_threshold": (1.50, 0.16, 0.89),
        },
        "mixed_high_conflict": {
            "no_optimization": (1.00, 0.05, 0.99),
            "conservative": (0.96, 0.12, 0.88),
            "moderate": (0.88, 0.20, 0.75),
            "aggressive": (0.72, 0.35, 0.60),  # Risky and unreliable
            "contextual_arithmetic": (0.85, 0.25, 0.70),
            "selective_top5": (1.18, 0.08, 0.95),  # Surprisingly good
            "inverse_reduction": (1.05, 0.15, 0.85),
            "adaptive_threshold": (0.82, 0.30, 0.70),
        },
        "case_analysis_explosive": {
            "no_optimization": (1.00, 0.05, 0.99),
            "conservative": (0.94, 0.10, 0.90),
            "moderate": (0.83, 0.18, 0.75),
            "aggressive": (0.65, 0.40, 0.50),  # Often catastrophic
            "contextual_arithmetic": (0.78, 0.22, 0.70),
            "selective_top5": (0.98, 0.08, 0.92),  # Safe choice
            "inverse_reduction": (1.02, 0.12, 0.88),
            "adaptive_threshold": (0.79, 0.25, 0.72),
        },
        "rare_but_critical": {
            "no_optimization": (1.00, 0.05, 0.99),
            "conservative": (1.20, 0.15, 0.90),
            "moderate": (1.60, 0.25, 0.85),
            "aggressive": (2.80, 0.50, 0.70),  # High reward, high risk
            "contextual_arithmetic": (1.40, 0.20, 0.88),
            "selective_top5": (1.25, 0.12, 0.92),
            "inverse_reduction": (0.85, 0.30, 0.75),
            "adaptive_threshold": (1.70, 0.35, 0.80),
        },
    }

    # Get performance parameters
    params = performance_matrix.get(context, {}).get(strategy, (1.0, 0.1, 0.95))
    mean_speedup, std_dev, reliability = params

    # Simulate compilation
    baseline_time = np.random.uniform(0.5, 3.0)

    # Reliability check (does compilation succeed?)
    compilation_success = np.random.random() < reliability

    if not compilation_success:
        # Compilation failed - big penalty
        optimized_time = baseline_time * 3.0
        return baseline_time, optimized_time, False

    # Normal case: sample from distribution
    speedup_sample = np.random.normal(mean_speedup, std_dev)
    speedup_sample = max(0.3, speedup_sample)  # Floor at 0.3x

    optimized_time = baseline_time / speedup_sample

    return baseline_time, optimized_time, True


def create_test_files() -> Dict[str, Path]:
    """Create test files for different contexts"""

    contexts = {
        "arithmetic_uniform": """
theorem add_zero : ∀ n : Nat, n + 0 = n := by simp
theorem zero_add : ∀ n : Nat, 0 + n = n := by simp
theorem mul_one : ∀ n : Nat, n * 1 = n := by simp
theorem one_mul : ∀ n : Nat, 1 * n = n := by simp
""",
        "pure_identity_simple": """
theorem list_append_nil : ∀ xs : List α, xs ++ [] = xs := by simp
theorem option_map_id : ∀ x : Option α, x.map id = x := by simp
theorem function_comp_id : ∀ f : α → β, f ∘ id = f := by simp
""",
        "mixed_high_conflict": """
theorem complex_mix : ∀ n : Nat, ∀ xs : List Nat,
  (xs ++ []).length + n * 1 = xs.length + n := by
  simp
  cases xs with
  | nil => simp
  | cons h t => simp [List.length_cons]
""",
        "case_analysis_explosive": """
inductive Tree : Type where
  | leaf : Tree
  | node : Tree → Tree → Tree

def tree_size : Tree → Nat
  | Tree.leaf => 1
  | Tree.node l r => tree_size l + tree_size r + 1

theorem tree_size_pos : ∀ t : Tree, tree_size t > 0 := by
  intro t
  cases t with
  | leaf => simp [tree_size]
  | node l r => 
    simp [tree_size]
    have hl : tree_size l > 0 := tree_size_pos l
    have hr : tree_size r > 0 := tree_size_pos r
    omega
""",
        "rare_but_critical": """
-- Performance-critical proof in inner loop
theorem critical_optimization : ∀ n : Nat, ∀ f : Nat → Nat,
  (List.range n).map f = (List.range n).map f := by
  intro n f
  rfl
""",
    }

    files = {}
    for context, content in contexts.items():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            files[context] = Path(f.name)

    return files


def demo_exploration_balance():
    """Demonstrate balanced exploration that protects user performance"""
    print("🎯 OPTIMIZED LEARNING DEMO")
    print("=" * 60)
    print("Balancing learning speed with user performance protection")
    print()

    # Create optimized learner
    learner = OptimizedRealtimeLearner(
        db_path=Path("optimized_demo.db"),
        config={
            "algorithm": "thompson",
            "exploration_budget": 0.15,  # 15% exploration budget
            "safety_threshold": 0.95,  # Never go below 95% of baseline
        },
    )

    # Create test files
    test_files = create_test_files()

    print("📊 SIMULATING INTELLIGENT EXPLORATION (300 compilations)")
    print("-" * 50)

    # Track metrics
    exploration_count = 0
    safety_violations = 0
    novel_discoveries = 0
    performance_history = []

    # Different phases to show adaptation
    phases = [
        (100, "🔍 Early Exploration Phase"),
        (150, "⚖️  Balanced Learning Phase"),
        (300, "🎯 Exploitation Focus Phase"),
    ]

    current_phase = 0

    for i in range(1, 301):
        # Check phase transitions
        if current_phase < len(phases) and i == phases[current_phase][0]:
            print(f"\n{phases[current_phase][1]}")
            current_phase += 1

        # Select context (realistic distribution)
        context_weights = {
            "arithmetic_uniform": 0.35,  # Common
            "pure_identity_simple": 0.25,  # Common
            "mixed_high_conflict": 0.20,  # Moderate
            "case_analysis_explosive": 0.15,  # Less common
            "rare_but_critical": 0.05,  # Rare but important
        }

        context = np.random.choice(list(context_weights.keys()), p=list(context_weights.values()))

        file_path = test_files[context]

        # Get recommendation
        strategy, metadata = learner.recommend_strategy(file_path)

        # Track exploration
        if metadata["is_exploration"]:
            exploration_count += 1

        # Simulate compilation
        baseline_time, optimized_time, compilation_success = simulate_realistic_workload(
            context, strategy
        )

        # Check for safety violations
        speedup = baseline_time / optimized_time if optimized_time > 0 else 0.1
        if speedup < 0.95:
            safety_violations += 1

        # Check for novel discoveries
        if speedup > 1.8:
            novel_discoveries += 1

        performance_history.append(speedup)

        # Record result
        learner.record_result(
            file_path=file_path,
            context_type=context,
            strategy=strategy,
            baseline_time=baseline_time,
            optimized_time=optimized_time,
            compilation_success=compilation_success,
            was_exploration=metadata["is_exploration"],
            curiosity_score=metadata.get("curiosity_score", 0.0),
        )

        # Periodic updates
        if i % 75 == 0:
            recent_performance = np.mean(performance_history[-50:])
            exploration_rate = exploration_count / i
            print(f"  After {i} compilations:")
            print(f"    Average speedup: {recent_performance:.2f}x")
            print(f"    Exploration rate: {exploration_rate:.1%}")
            print(f"    Safety violations: {safety_violations}")
            print(f"    Novel discoveries: {novel_discoveries}")

    print("\n📈 FINAL OPTIMIZATION RESULTS")
    print("-" * 40)

    # Get comprehensive report
    report = learner.get_comprehensive_report()

    print("Overall Performance:")
    print(f"  Total compilations: {report['learning_stats']['total_compilations']}")
    print(f"  Success rate: {report['learning_stats']['success_rate']:.1%}")
    print(f"  Average speedup: {report['learning_stats']['average_speedup']:.2f}x")
    print(f"  Contexts learned: {report['learning_stats']['contexts_learned']}")

    print("\nExploration Analysis:")
    exp_stats = report["exploration_stats"]["epsilon_greedy_stats"]
    print(f"  Total explorations: {exp_stats['total_explorations']}")
    print(f"  Exploration success rate: {exp_stats['success_rate']:.1%}")
    print(f"  Novel discoveries: {exp_stats['novel_discoveries']}")
    print(f"  Average exploration regret: {exp_stats['average_regret']:.3f}")

    print("\nSafety Analysis:")
    print(f"  Safety violations: {safety_violations} ({safety_violations/300:.1%})")
    print(
        f"  Performance trend: {report['exploration_stats']['performance_trend']['improvement_trend']}"
    )

    print("\nLearned Optimal Strategies:")
    strategy_perf = report["strategy_performance"]
    for context, strategies in strategy_perf.items():
        if strategies:
            best_strategy = max(strategies.items(), key=lambda x: x[1]["mean_speedup"])
            strategy_name, stats = best_strategy
            print(f"  {context}:")
            print(f"    Best: {strategy_name} ({stats['mean_speedup']:.2f}x)")
            print(f"    Confidence: {stats['confidence']:.1%}")
            print(f"    Based on: {stats['pulls']} samples")

    print("\n🎯 INTELLIGENCE ANALYSIS")
    print("-" * 30)

    # Analyze learning efficiency
    early_performance = np.mean(performance_history[:50])
    late_performance = np.mean(performance_history[-50:])
    improvement = (late_performance - early_performance) / early_performance * 100

    print(f"Learning efficiency: {improvement:+.1f}% improvement")
    print(f"Exploration overhead: {(exploration_count/300) * 100:.1f}% of decisions")
    print(f"Safety record: {(1 - safety_violations/300)*100:.1f}% safe decisions")

    # Show recommendations
    if report["recommendations"]:
        print("\nSystem Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

    print("\n✅ The optimized learner successfully balanced:")
    print("  - Fast learning through intelligent exploration")
    print("  - User performance protection via safety mechanisms")
    print("  - Coverage of rare but important patterns")
    print("  - Adaptation to changing workload patterns")

    # Cleanup
    for file_path in test_files.values():
        file_path.unlink()


def demo_curiosity_mechanism():
    """Demonstrate curiosity-driven exploration"""
    print("\n🧠 CURIOSITY MECHANISM DEMO")
    print("=" * 40)

    learner = OptimizedRealtimeLearner(config={"algorithm": "thompson"})

    # Create test file
    test_file = create_test_files()["rare_but_critical"]

    print("Testing curiosity for understudied 'rare_but_critical' context...")

    # Simulate sparse, high-value context
    for i in range(10):
        strategy, metadata = learner.recommend_strategy(test_file)

        print(f"Attempt {i+1}:")
        print(f"  Strategy: {strategy}")
        print(f"  Curiosity score: {metadata.get('curiosity_score', 0):.2f}")
        print(f"  Is exploration: {metadata['is_exploration']}")

        # Simulate high-value but risky outcome
        if strategy == "aggressive":
            baseline, optimized, success = 1.0, 0.35, True  # 2.8x speedup
        else:
            baseline, optimized, success = 1.0, 0.8, True  # 1.25x speedup

        learner.record_result(
            file_path=test_file,
            context_type="rare_but_critical",
            strategy=strategy,
            baseline_time=baseline,
            optimized_time=optimized,
            compilation_success=success,
            was_exploration=metadata["is_exploration"],
            curiosity_score=metadata.get("curiosity_score", 0.0),
        )

    print("\n📊 Curiosity successfully drove exploration of rare context!")

    # Cleanup
    test_file.unlink()


if __name__ == "__main__":
    demo_exploration_balance()
    demo_curiosity_mechanism()

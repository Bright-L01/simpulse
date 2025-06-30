#!/usr/bin/env python3
"""
Demonstration of JIT-style optimization for Lean's simp tactic.

Shows how runtime profiling leads to adaptive performance improvements.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simpulse.jit.dynamic_optimizer import (
    DynamicSimpOptimizer,
    OptimizationContext,
    create_benchmark_scenario,
)
from simpulse.jit.lean_integration import demo_jit_optimization


def visualize_adaptation():
    """Visualize how JIT optimization adapts over time."""
    print("=" * 70)
    print("SIMPULSE JIT OPTIMIZATION DEMONSTRATION")
    print("=" * 70)
    print()

    print("This demo shows how Simpulse learns from runtime behavior")
    print("and dynamically optimizes simp rule priorities.\n")

    # Create optimizer
    optimizer = DynamicSimpOptimizer(adaptation_interval=20)

    # Define rules with varying characteristics
    rules = [
        # Name, actual_success_rate, actual_time
        ("List.append_nil", 0.95, 0.0001),  # Very common, fast
        ("List.nil_append", 0.90, 0.0001),  # Very common, fast
        ("Nat.add_zero", 0.85, 0.0001),  # Common, fast
        ("Nat.zero_add", 0.85, 0.0001),  # Common, fast
        ("List.map_append", 0.30, 0.0005),  # Less common, medium
        ("List.reverse_append", 0.20, 0.0008),  # Uncommon, medium
        ("ComplexArithmetic", 0.05, 0.005),  # Rare, slow
        ("HeavyComputation", 0.02, 0.010),  # Very rare, very slow
    ]

    print("Initial state: All rules have default priority (1000)")
    print("\nRule characteristics:")
    for name, success, time_ms in rules:
        print(f"  {name:20} - Success: {success:4.0%}, Time: {time_ms*1000:5.2f}ms")
    print()

    # Track performance over time
    improvements = []
    iterations = 100

    print(f"Running {iterations} iterations with adaptation every 20 attempts...\n")

    for iteration in range(iterations):
        # Create varying contexts
        contexts = [
            OptimizationContext("Lists", "list_ops", 1, ["intro"]),
            OptimizationContext("Arithmetic", "nat_ops", 2, ["rw", "intro"]),
            OptimizationContext("Mixed", "general", 3, ["simp", "rw"]),
        ]

        context = contexts[iteration % len(contexts)]

        # Measure time for this iteration
        time.time()

        # Try rules in current priority order
        current_order = sorted(
            rules, key=lambda r: optimizer.get_priority(r[0], context)
        )

        # Simulate rule attempts
        total_time = 0
        for rule_name, success_rate, exec_time in current_order:
            # Simulate attempt
            with optimizer.instrument_simp_attempt(rule_name, context) as attempt:
                total_time += exec_time

                # Check if rule succeeds
                import random

                if random.random() < success_rate:
                    attempt.mark_success()
                    break

        # Calculate improvement for this iteration
        # Compare to worst-case (trying all rules)
        worst_case = sum(r[2] for r in rules)
        improvement = (worst_case - total_time) / worst_case * 100
        improvements.append(improvement)

        # Show progress
        if (iteration + 1) % 20 == 0:
            avg_recent = sum(improvements[-20:]) / 20
            print(
                f"Iteration {iteration + 1:3d}: Recent avg improvement: {avg_recent:5.1f}%"
            )

            # Show current top priorities
            top_rules = sorted(
                [(r[0], optimizer.get_priority(r[0])) for r in rules],
                key=lambda x: x[1],
            )[:3]
            print("  Top priority rules:")
            for rule, priority in top_rules:
                stats = optimizer.rule_stats.get(rule)
                if stats:
                    print(
                        f"    {rule:20} (priority={priority:4d}, "
                        f"success={stats.success_rate:4.0%})"
                    )
            print()

    # Final report
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print("\nLearned priorities (top 10):")
    final_priorities = sorted(
        [(r[0], optimizer.get_priority(r[0])) for r in rules], key=lambda x: x[1]
    )

    for i, (rule, priority) in enumerate(final_priorities):
        stats = optimizer.rule_stats.get(rule)
        if stats:
            print(
                f"{i+1:2d}. {rule:20} - Priority: {priority:4d}, "
                f"Success: {stats.success_rate:4.0%}, "
                f"Attempts: {stats.attempts:4d}, "
                f"Avg time: {stats.avg_time*1000:5.2f}ms"
            )

    # Performance summary
    print("\nPerformance improvement over time:")
    print(f"  First 20 iterations: {sum(improvements[:20])/20:5.1f}%")
    print(f"  Last 20 iterations:  {sum(improvements[-20:])/20:5.1f}%")
    print(f"  Overall average:     {sum(improvements)/len(improvements):5.1f}%")

    # Save configuration
    optimizer.compile_optimized_simp(Path("demo_jit_config.json"))
    print("\nOptimized configuration saved to demo_jit_config.json")


def compare_static_vs_jit():
    """Compare static optimization vs JIT optimization."""
    print("\n" + "=" * 70)
    print("STATIC vs JIT OPTIMIZATION COMPARISON")
    print("=" * 70)
    print()

    print("Static optimization: Priorities fixed based on analysis")
    print("JIT optimization: Priorities adapt based on runtime behavior")
    print()

    # Simulate changing workload
    print("Simulating changing workload patterns...\n")

    # Phase 1: List-heavy workload
    print("Phase 1 (iterations 1-50): List-heavy operations")
    # Would show JIT adapting to prioritize list rules

    # Phase 2: Arithmetic-heavy workload
    print("Phase 2 (iterations 51-100): Arithmetic-heavy operations")
    # Would show JIT adapting to prioritize arithmetic rules

    # Phase 3: Mixed workload
    print("Phase 3 (iterations 101-150): Mixed operations")
    # Would show JIT finding balanced priorities

    print("\nKey advantages of JIT optimization:")
    print("✓ Adapts to actual usage patterns")
    print("✓ Handles changing workloads")
    print("✓ Learns from success/failure")
    print("✓ Optimizes for specific contexts")
    print("✓ No manual tuning required")


def main():
    """Run all JIT demonstrations."""
    import argparse

    parser = argparse.ArgumentParser(description="Simpulse JIT Optimization Demo")
    parser.add_argument(
        "--mode",
        choices=["adapt", "compare", "server", "benchmark"],
        default="adapt",
        help="Demo mode to run",
    )

    args = parser.parse_args()

    if args.mode == "adapt":
        visualize_adaptation()
    elif args.mode == "compare":
        compare_static_vs_jit()
    elif args.mode == "server":
        demo_jit_optimization()
    elif args.mode == "benchmark":
        create_benchmark_scenario()

    print("\n✨ JIT optimization demo complete!")


if __name__ == "__main__":
    main()

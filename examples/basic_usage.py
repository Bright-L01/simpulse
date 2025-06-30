#!/usr/bin/env python3
"""Basic usage example for Simpulse."""

import asyncio
from pathlib import Path

from simpulse.analysis.health_checker import HealthChecker
from simpulse.optimization.optimizer import SimpOptimizer


async def main():
    """Run basic Simpulse optimization example."""
    print("🚀 Simpulse Basic Usage Example\n")

    # Example: Check a Lean project
    project_path = Path.cwd()  # Use current directory as example

    # Step 1: Check project health
    print("Step 1: Checking project health...")
    checker = HealthChecker()
    health_result = checker.check_project(project_path)

    print(f"  Total simp rules: {health_result.total_rules}")
    print(f"  Default priority usage: {health_result.default_priority_percentage:.0f}%")
    print(f"  Optimization potential: {health_result.score}/100")
    print(f"  Estimated improvement: {health_result.estimated_improvement}%\n")

    # Step 2: Run optimization (if high potential)
    if health_result.score > 40:
        print("Step 2: Running optimization...")
        optimizer = SimpOptimizer()

        # Analyze the project
        analysis = optimizer.analyze(project_path)
        print(f"  Found {len(analysis.rules)} rules to analyze")

        # Generate optimization plan
        optimization = optimizer.optimize(analysis)
        print("  Generated optimization plan")
        print(f"  Rules to change: {optimization.rules_changed}")
        print(f"  Expected improvement: {optimization.estimated_improvement}%")

        # In a real scenario, you would apply the optimization:
        # optimizer.apply(optimization, project_path)

        print("\n✅ Optimization complete!")
    else:
        print("ℹ️  Project already well-optimized or has few simp rules.")

    print("\n📝 Next steps:")
    print("  1. Review the optimization plan")
    print("  2. Apply changes with: optimizer.apply(optimization, project_path)")
    print("  3. Run benchmarks to verify improvements")


if __name__ == "__main__":
    asyncio.run(main())

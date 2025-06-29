#!/usr/bin/env python3
"""
Basic Simpulse Usage Example

This script demonstrates the basic usage of Simpulse for optimizing
Lean 4 simp rules in a simple project.
"""

import asyncio
import logging
from pathlib import Path

from simpulse.config import ClaudeConfig, Config, OptimizationConfig, PathConfig
from simpulse.evolution.evolution_engine import EvolutionEngine


async def basic_optimization_example():
    """Run a basic optimization on a Lean project."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("🧬 Starting basic Simpulse optimization example")

    # Configure Simpulse
    config = Config(
        optimization=OptimizationConfig(
            population_size=20,
            generations=25,
            time_budget=1800,  # 30 minutes
            target_improvement=10.0,
            max_parallel_evaluations=2,
        ),
        claude=ClaudeConfig(backend="claude_code"),  # Use Claude Code CLI
        paths=PathConfig(
            output_dir=Path("./simpulse_output"),
            cache_dir=Path("./simpulse_cache"),
            log_dir=Path("./simpulse_logs"),
        ),
    )

    # Create output directories
    config.paths.output_dir.mkdir(exist_ok=True)
    config.paths.cache_dir.mkdir(exist_ok=True)
    config.paths.log_dir.mkdir(exist_ok=True)

    # Initialize evolution engine
    engine = EvolutionEngine(config)

    # Define modules to optimize
    # You can specify exact module names or use "auto" to detect all modules
    modules_to_optimize = [
        "MyProject.BasicTheorems",
        "MyProject.Algebra.Ring",
        "MyProject.Data.List",
    ]

    logger.info(f"Optimizing modules: {modules_to_optimize}")

    try:
        # Run optimization
        result = await engine.run_evolution(
            modules=modules_to_optimize,
            source_path=Path.cwd(),  # Current directory
            time_budget=config.optimization.time_budget,
        )

        # Display results
        logger.info("🎉 Optimization completed!")
        logger.info(f"📈 Improvement: {result.improvement_percent:.1f}%")
        logger.info(f"🧬 Generations: {result.total_generations}")
        logger.info(f"⏱️ Execution time: {result.execution_time:.1f}s")
        logger.info(f"✅ Success: {result.success}")

        if result.best_candidate:
            logger.info(f"🔧 Mutations applied: {len(result.best_candidate.mutations)}")

            # Show some example mutations
            logger.info("🔍 Example mutations:")
            for i, mutation in enumerate(result.best_candidate.mutations[:3]):
                logger.info(
                    f"  {i+1}. {mutation.rule_name}: "
                    f"{mutation.old_attribute} -> {mutation.new_attribute}"
                )
                if i >= 2:  # Show max 3 examples
                    break

        # Save results
        import json

        results_file = config.paths.output_dir / "basic_optimization_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "improvement_percent": result.improvement_percent,
                    "total_generations": result.total_generations,
                    "execution_time": result.execution_time,
                    "success": result.success,
                    "modules": result.modules,
                    "mutations_count": (
                        len(result.best_candidate.mutations)
                        if result.best_candidate
                        else 0
                    ),
                },
                f,
                indent=2,
            )

        logger.info(f"📁 Results saved to: {results_file}")

    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        raise


async def targeted_optimization_example():
    """Example of optimizing specific rules with custom configuration."""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("🎯 Starting targeted optimization example")

    # More focused configuration for specific optimization
    config = Config(
        optimization=OptimizationConfig(
            population_size=15,
            generations=20,
            time_budget=900,  # 15 minutes
            target_improvement=5.0,  # Lower target for focused optimization
            mutation_rate=0.3,
            crossover_rate=0.7,
            elite_size=3,
            max_parallel_evaluations=3,
        ),
        claude=ClaudeConfig(backend="claude_code", timeout_seconds=30, max_retries=2),
        paths=PathConfig(
            output_dir=Path("./targeted_output"),
            cache_dir=Path("./targeted_cache"),
            log_dir=Path("./targeted_logs"),
        ),
    )

    # Create directories
    for path in [config.paths.output_dir, config.paths.cache_dir, config.paths.log_dir]:
        path.mkdir(exist_ok=True)

    # Focus on specific modules that are known to be performance bottlenecks
    target_modules = [
        "MyProject.Algorithms.Sort",  # Heavy computation module
        "MyProject.Data.Tree",  # Complex data structure module
    ]

    engine = EvolutionEngine(config)

    try:
        result = await engine.run_evolution(
            modules=target_modules,
            source_path=Path.cwd(),
            time_budget=config.optimization.time_budget,
        )

        logger.info("🎯 Targeted optimization completed!")
        logger.info(f"📈 Performance gain: {result.improvement_percent:.2f}%")

        if result.improvement_percent > 3.0:
            logger.info("🚀 Significant improvement achieved!")
        elif result.improvement_percent > 0:
            logger.info("✨ Minor improvement - every bit helps!")
        else:
            logger.info("📊 No improvement found - rules already well-optimized")

    except Exception as e:
        logger.error(f"❌ Targeted optimization failed: {e}")
        raise


if __name__ == "__main__":
    print("🧬 Simpulse Basic Usage Examples")
    print("================================")
    print()
    print("Choose an example to run:")
    print("1. Basic optimization (recommended for first-time users)")
    print("2. Targeted optimization (for specific modules)")
    print()

    choice = input("Enter choice (1-2): ").strip()

    if choice == "1":
        print("\n🚀 Running basic optimization example...")
        asyncio.run(basic_optimization_example())
    elif choice == "2":
        print("\n🎯 Running targeted optimization example...")
        asyncio.run(targeted_optimization_example())
    else:
        print("❌ Invalid choice. Please run the script again and choose 1 or 2.")

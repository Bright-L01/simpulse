#!/usr/bin/env python3
"""
Full scale experiment runner - 1000 files × 10 strategies = 10,000 experiments

This is the paradigm shift: no predictions, just empirical ground truth.
"""

import argparse
from pathlib import Path

from experiment_runner import ExperimentRunner, collect_diverse_lean_files


def main():
    parser = argparse.ArgumentParser(description="Run full scale optimization experiments")
    parser.add_argument(
        "--lean-path",
        type=Path,
        default=Path.home()
        / ".elan/toolchains/leanprover--lean4-nightly---nightly-2024-02-01/src/lean",
        help="Path to Lean source files",
    )
    parser.add_argument("--file-count", type=int, default=1000, help="Number of files to test")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument(
        "--timeout", type=int, default=30, help="Compilation timeout per file (seconds)"
    )

    args = parser.parse_args()

    print("🧪 EMPIRICAL OPTIMIZATION EXPERIMENT")
    print("=" * 50)
    print("Paradigm: Stop predicting, start experimenting")
    print(f"Target: {args.file_count} files × 10 strategies = {args.file_count * 10:,} experiments")
    print(f"Workers: {args.workers}")
    print(f"Timeout: {args.timeout}s per compilation")
    print()

    # Collect diverse lean files
    print("📁 Collecting diverse Lean files...")

    if args.lean_path.exists():
        lean_files = collect_diverse_lean_files(args.lean_path, args.file_count)
        print(f"✓ Found {len(lean_files)} files from {args.lean_path}")
    else:
        # Fallback to project files
        lean_files = list(Path.cwd().rglob("*.lean"))
        if len(lean_files) > args.file_count:
            lean_files = lean_files[: args.file_count]
        print(f"✓ Using {len(lean_files)} local project files")

    if not lean_files:
        print("❌ No Lean files found!")
        return

    # Show sample files
    print(f"\nSample files to test:")
    for i, f in enumerate(lean_files[:5]):
        print(f"  {i+1}. {f.name}")
    if len(lean_files) > 5:
        print(f"  ... and {len(lean_files) - 5} more")

    print(f"\n🚀 Starting experiments...")

    # Run experiments
    runner = ExperimentRunner(output_dir=Path("empirical_results"), lean_executable="lean")

    print(f"\nOptimization strategies to test:")
    for i, strategy in enumerate(runner.STRATEGIES, 1):
        print(f"  {i:2}. {strategy.name}: {strategy.description}")

    print(
        f"\n⏱️  Estimated time: {len(lean_files) * len(runner.STRATEGIES) * 5 / args.workers / 60:.1f} minutes"
    )
    print("(assuming 5s average per experiment)")

    # Confirm before running
    response = input("\nProceed with experiments? [y/N]: ")
    if response.lower() not in ["y", "yes"]:
        print("Experiments cancelled.")
        return

    print("\n" + "=" * 50)
    print("🔬 RUNNING EXPERIMENTS")
    print("=" * 50)

    runner.run_experiments(lean_files, max_workers=args.workers)

    print("\n" + "=" * 50)
    print("✅ EXPERIMENTS COMPLETE")
    print("=" * 50)
    print(f"Results saved to: empirical_results/")
    print("\nKey outputs:")
    print("  📊 payoff_matrix_heatmap.png - Context × Strategy → Speedup")
    print("  📈 strategy_comparison.png - Performance distribution")
    print("  📋 experiment_summary.json - Best strategies by context")
    print("\nThis is your ground truth data!")


if __name__ == "__main__":
    main()

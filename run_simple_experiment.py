#!/usr/bin/env python3
"""
Simple experiment runner to test the paradigm shift approach
"""

from pathlib import Path

from experiment_runner import ExperimentRunner


def main():
    # Find local Lean files in the project
    lean_files = list(Path.cwd().rglob("*.lean"))[:5]

    if not lean_files:
        print("No Lean files found in current directory")
        return

    print(f"Testing with {len(lean_files)} files:")
    for f in lean_files:
        print(f"  - {f.name}")

    # Run simplified experiment
    runner = ExperimentRunner(output_dir=Path("simple_experiment"))

    # Run just a few strategies to test the concept
    runner.STRATEGIES = runner.STRATEGIES[:5]  # First 5 strategies only

    print(f"\nTesting {len(runner.STRATEGIES)} strategies:")
    for s in runner.STRATEGIES:
        print(f"  - {s.name}: {s.description}")

    runner.run_experiments(lean_files, max_workers=1)


if __name__ == "__main__":
    main()

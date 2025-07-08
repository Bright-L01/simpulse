#!/usr/bin/env python3
"""
Simple benchmark plotter - outputs raw data for visualization.
"""

import json
import sys
from pathlib import Path


def create_plot_data(history_file: Path):
    """Create plot-ready data from benchmark history."""

    # Load history
    with open(history_file) as f:
        history = json.load(f)

    if not history.get("benchmarks"):
        print("No benchmarks found")
        return

    # Group by file
    by_file = {}

    for benchmark in history["benchmarks"]:
        timestamp = benchmark["timestamp"]
        commit = benchmark["git"]["commit"][:8]

        for file_name, summary in benchmark["summary"].items():
            if "avg_wall_time" not in summary:
                continue

            if file_name not in by_file:
                by_file[file_name] = []

            by_file[file_name].append(
                {
                    "timestamp": timestamp,
                    "commit": commit,
                    "wall_time": summary["avg_wall_time"],
                    "memory_mb": summary["avg_memory_mb"],
                }
            )

    # Output as simple text format for plotting
    print("# Benchmark Performance Over Time")
    print("# Format: timestamp, commit, wall_time, memory_mb")
    print()

    for file_name, data in by_file.items():
        print(f"## {file_name}")
        for point in sorted(data, key=lambda x: x["timestamp"]):
            print(
                f"{point['timestamp']}, {point['commit']}, {point['wall_time']:.3f}, {point['memory_mb']:.1f}"
            )
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        history_file = Path(sys.argv[1])
    else:
        history_file = Path("benchmark_history.json")

    if not history_file.exists():
        print(f"History file not found: {history_file}")
        sys.exit(1)

    create_plot_data(history_file)

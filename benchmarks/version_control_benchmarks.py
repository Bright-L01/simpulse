#!/usr/bin/env python3
"""
Version control benchmark results for tracking performance over time.
"""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List


class BenchmarkVersionControl:
    """Track benchmark results over git commits."""

    def __init__(self, history_file: Path = Path("benchmark_history.json")):
        self.history_file = history_file
        self.history = self.load_history()

    def load_history(self) -> Dict:
        """Load benchmark history."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return {"benchmarks": []}

    def save_history(self):
        """Save benchmark history."""
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_git_info(self) -> Dict:
        """Get current git commit information."""
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip()

            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
            ).stdout.strip()

            message = subprocess.run(
                ["git", "log", "-1", "--pretty=%B"], capture_output=True, text=True
            ).stdout.strip()

            author = subprocess.run(
                ["git", "log", "-1", "--pretty=%an <%ae>"], capture_output=True, text=True
            ).stdout.strip()

            # Check for uncommitted changes
            status = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True
            ).stdout

            has_changes = len(status.strip()) > 0

            return {
                "commit": commit,
                "branch": branch,
                "message": message.split("\n")[0],  # First line only
                "author": author,
                "has_uncommitted_changes": has_changes,
            }
        except:
            return {
                "commit": "unknown",
                "branch": "unknown",
                "message": "Git not available",
                "author": "unknown",
                "has_uncommitted_changes": False,
            }

    def add_benchmark_result(self, benchmark_file: Path, tags: List[str] = None):
        """Add a benchmark result to history."""

        # Load benchmark data
        with open(benchmark_file) as f:
            benchmark_data = json.load(f)

        # Extract summary statistics
        summary = self.create_summary(benchmark_data)

        # Create history entry
        entry = {
            "id": hashlib.sha256(f"{benchmark_data['metadata']['timestamp']}".encode()).hexdigest()[
                :12
            ],
            "timestamp": benchmark_data["metadata"]["timestamp"],
            "file": str(benchmark_file),
            "git": self.get_git_info(),
            "system": benchmark_data["metadata"]["system"],
            "summary": summary,
            "tags": tags or [],
        }

        # Add to history
        self.history["benchmarks"].append(entry)
        self.save_history()

        return entry["id"]

    def create_summary(self, benchmark_data: Dict) -> Dict:
        """Create summary statistics from benchmark data."""

        summary = {}

        for file_name, runs in benchmark_data["benchmarks"].items():
            # Calculate average metrics
            wall_times = [r["timing"]["wall_time"] for r in runs if r["exit_code"] == 0]
            memory_peaks = [
                r["memory"]["peak_rss_bytes"] / (1024 * 1024) for r in runs if r["exit_code"] == 0
            ]

            if wall_times:
                summary[file_name] = {
                    "avg_wall_time": sum(wall_times) / len(wall_times),
                    "min_wall_time": min(wall_times),
                    "max_wall_time": max(wall_times),
                    "avg_memory_mb": sum(memory_peaks) / len(memory_peaks),
                    "successful_runs": len(wall_times),
                    "total_runs": len(runs),
                }
            else:
                summary[file_name] = {
                    "error": "No successful runs",
                    "total_runs": len(runs),
                }

        return summary

    def compare_with_baseline(self, benchmark_id: str, baseline_id: str) -> Dict:
        """Compare a benchmark with a baseline."""

        # Find benchmarks
        benchmark = None
        baseline = None

        for b in self.history["benchmarks"]:
            if b["id"] == benchmark_id:
                benchmark = b
            if b["id"] == baseline_id:
                baseline = b

        if not benchmark or not baseline:
            return {"error": "Benchmark not found"}

        # Compare summaries
        comparison = {
            "benchmark": {
                "id": benchmark["id"],
                "commit": benchmark["git"]["commit"][:8],
                "timestamp": benchmark["timestamp"],
            },
            "baseline": {
                "id": baseline["id"],
                "commit": baseline["git"]["commit"][:8],
                "timestamp": baseline["timestamp"],
            },
            "files": {},
        }

        # Compare each file
        for file_name in benchmark["summary"]:
            if file_name in baseline["summary"]:
                b_data = benchmark["summary"][file_name]
                base_data = baseline["summary"][file_name]

                if "avg_wall_time" in b_data and "avg_wall_time" in base_data:
                    speedup = base_data["avg_wall_time"] / b_data["avg_wall_time"]
                    comparison["files"][file_name] = {
                        "baseline_time": base_data["avg_wall_time"],
                        "benchmark_time": b_data["avg_wall_time"],
                        "speedup": speedup,
                        "improvement_percent": (speedup - 1) * 100,
                    }

        return comparison

    def list_benchmarks(self, limit: int = 10) -> List[Dict]:
        """List recent benchmarks."""

        benchmarks = self.history["benchmarks"][-limit:]

        return [
            {
                "id": b["id"],
                "timestamp": b["timestamp"],
                "commit": b["git"]["commit"][:8],
                "branch": b["git"]["branch"],
                "tags": b["tags"],
                "files": list(b["summary"].keys()),
            }
            for b in reversed(benchmarks)
        ]

    def export_for_plotting(self, output_file: Path):
        """Export data in format suitable for plotting."""

        # Organize data by file
        plot_data = {}

        for benchmark in self.history["benchmarks"]:
            timestamp = benchmark["timestamp"]

            for file_name, summary in benchmark["summary"].items():
                if "avg_wall_time" not in summary:
                    continue

                if file_name not in plot_data:
                    plot_data[file_name] = {
                        "timestamps": [],
                        "wall_times": [],
                        "memory_mb": [],
                        "commits": [],
                    }

                plot_data[file_name]["timestamps"].append(timestamp)
                plot_data[file_name]["wall_times"].append(summary["avg_wall_time"])
                plot_data[file_name]["memory_mb"].append(summary["avg_memory_mb"])
                plot_data[file_name]["commits"].append(benchmark["git"]["commit"][:8])

        with open(output_file, "w") as f:
            json.dump(plot_data, f, indent=2)


def main():
    """Main entry point."""

    import argparse

    parser = argparse.ArgumentParser(description="Version control benchmarks")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add benchmark result")
    add_parser.add_argument("file", help="Benchmark result file")
    add_parser.add_argument("--tags", nargs="+", help="Tags for this benchmark")

    # List command
    list_parser = subparsers.add_parser("list", help="List benchmarks")
    list_parser.add_argument("--limit", type=int, default=10, help="Number to show")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare benchmarks")
    compare_parser.add_argument("benchmark", help="Benchmark ID")
    compare_parser.add_argument("baseline", help="Baseline ID")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export for plotting")
    export_parser.add_argument("output", help="Output file")

    args = parser.parse_args()

    vc = BenchmarkVersionControl()

    if args.command == "add":
        benchmark_id = vc.add_benchmark_result(Path(args.file), args.tags)
        print(f"Added benchmark: {benchmark_id}")

    elif args.command == "list":
        benchmarks = vc.list_benchmarks(args.limit)
        print(f"{'ID':<12} {'Timestamp':<20} {'Commit':<8} {'Branch':<15} {'Files'}")
        print("-" * 80)
        for b in benchmarks:
            print(
                f"{b['id']:<12} {b['timestamp']:<20} {b['commit']:<8} {b['branch']:<15} {len(b['files'])} files"
            )
            if b["tags"]:
                print(f"  Tags: {', '.join(b['tags'])}")

    elif args.command == "compare":
        comparison = vc.compare_with_baseline(args.benchmark, args.baseline)
        print(json.dumps(comparison, indent=2))

    elif args.command == "export":
        vc.export_for_plotting(Path(args.output))
        print(f"Exported to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

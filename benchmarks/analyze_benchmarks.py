#!/usr/bin/env python3
"""
Analyze benchmark results and output raw statistics.
No interpretation - just data.
"""

import csv
import json
import statistics
from pathlib import Path
from typing import Dict


def load_benchmark_results(file_path: Path) -> Dict:
    """Load benchmark JSON results."""
    with open(file_path) as f:
        return json.load(f)


def extract_metrics(results: Dict) -> Dict:
    """Extract key metrics from benchmark results."""

    metrics = {"metadata": results["metadata"], "files": {}}

    for file_name, runs in results["benchmarks"].items():
        file_metrics = {
            "runs": len(runs),
            "wall_times": [],
            "perf_times": [],
            "process_times": [],
            "peak_memory_mb": [],
            "avg_cpu_percent": [],
            "exit_codes": [],
            "profiler_data": [],
        }

        for run in runs:
            file_metrics["wall_times"].append(run["timing"]["wall_time"])
            file_metrics["perf_times"].append(run["timing"]["perf_counter"])
            file_metrics["process_times"].append(run["timing"]["process_time"])
            file_metrics["peak_memory_mb"].append(run["memory"]["peak_rss_bytes"] / (1024 * 1024))
            file_metrics["avg_cpu_percent"].append(run["cpu"]["average_percent"])
            file_metrics["exit_codes"].append(run["exit_code"])

            if "profiler_data" in run:
                file_metrics["profiler_data"].append(run["profiler_data"])

        # Calculate statistics
        file_metrics["statistics"] = {
            "wall_time": {
                "mean": statistics.mean(file_metrics["wall_times"]),
                "stdev": (
                    statistics.stdev(file_metrics["wall_times"])
                    if len(file_metrics["wall_times"]) > 1
                    else 0
                ),
                "min": min(file_metrics["wall_times"]),
                "max": max(file_metrics["wall_times"]),
            },
            "memory_mb": {
                "mean": statistics.mean(file_metrics["peak_memory_mb"]),
                "stdev": (
                    statistics.stdev(file_metrics["peak_memory_mb"])
                    if len(file_metrics["peak_memory_mb"]) > 1
                    else 0
                ),
                "min": min(file_metrics["peak_memory_mb"]),
                "max": max(file_metrics["peak_memory_mb"]),
            },
            "cpu_percent": {
                "mean": statistics.mean(file_metrics["avg_cpu_percent"]),
                "stdev": (
                    statistics.stdev(file_metrics["avg_cpu_percent"])
                    if len(file_metrics["avg_cpu_percent"]) > 1
                    else 0
                ),
                "min": min(file_metrics["avg_cpu_percent"]),
                "max": max(file_metrics["avg_cpu_percent"]),
            },
        }

        metrics["files"][file_name] = file_metrics

    return metrics


def output_csv(metrics: Dict, output_file: Path):
    """Output metrics as CSV."""

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["File", "Metric", "Mean", "StdDev", "Min", "Max", "Unit"])

        # Data
        for file_name, file_metrics in metrics["files"].items():
            stats = file_metrics["statistics"]

            # Wall time
            writer.writerow(
                [
                    file_name,
                    "wall_time",
                    stats["wall_time"]["mean"],
                    stats["wall_time"]["stdev"],
                    stats["wall_time"]["min"],
                    stats["wall_time"]["max"],
                    "seconds",
                ]
            )

            # Memory
            writer.writerow(
                [
                    file_name,
                    "peak_memory",
                    stats["memory_mb"]["mean"],
                    stats["memory_mb"]["stdev"],
                    stats["memory_mb"]["min"],
                    stats["memory_mb"]["max"],
                    "MB",
                ]
            )

            # CPU
            writer.writerow(
                [
                    file_name,
                    "avg_cpu",
                    stats["cpu_percent"]["mean"],
                    stats["cpu_percent"]["stdev"],
                    stats["cpu_percent"]["min"],
                    stats["cpu_percent"]["max"],
                    "percent",
                ]
            )


def output_json_summary(metrics: Dict, output_file: Path):
    """Output summary as JSON."""

    summary = {
        "system": metrics["metadata"]["system"],
        "benchmark_date": metrics["metadata"]["timestamp"],
        "results": {},
    }

    for file_name, file_metrics in metrics["files"].items():
        summary["results"][file_name] = {
            "runs": file_metrics["runs"],
            "statistics": file_metrics["statistics"],
            "all_exit_codes": list(set(file_metrics["exit_codes"])),
        }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)


def compare_benchmarks(file1: Path, file2: Path) -> Dict:
    """Compare two benchmark files."""

    metrics1 = extract_metrics(load_benchmark_results(file1))
    metrics2 = extract_metrics(load_benchmark_results(file2))

    comparison = {"file1": str(file1), "file2": str(file2), "comparisons": {}}

    # Find common files
    common_files = set(metrics1["files"].keys()) & set(metrics2["files"].keys())

    for file_name in common_files:
        stats1 = metrics1["files"][file_name]["statistics"]
        stats2 = metrics2["files"][file_name]["statistics"]

        comparison["comparisons"][file_name] = {
            "wall_time": {
                "file1_mean": stats1["wall_time"]["mean"],
                "file2_mean": stats2["wall_time"]["mean"],
                "ratio": stats2["wall_time"]["mean"] / stats1["wall_time"]["mean"],
                "difference": stats2["wall_time"]["mean"] - stats1["wall_time"]["mean"],
            },
            "memory": {
                "file1_mean": stats1["memory_mb"]["mean"],
                "file2_mean": stats2["memory_mb"]["mean"],
                "ratio": stats2["memory_mb"]["mean"] / stats1["memory_mb"]["mean"],
                "difference": stats2["memory_mb"]["mean"] - stats1["memory_mb"]["mean"],
            },
        }

    return comparison


def main():
    """Main entry point."""

    import argparse

    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("file", help="Benchmark result file")
    parser.add_argument("--csv", help="Output CSV to file")
    parser.add_argument("--json", help="Output JSON summary to file")
    parser.add_argument("--compare", help="Compare with another benchmark file")

    args = parser.parse_args()

    # Load and analyze
    results = load_benchmark_results(Path(args.file))
    metrics = extract_metrics(results)

    # Output formats
    if args.csv:
        output_csv(metrics, Path(args.csv))
        print(f"CSV output saved to: {args.csv}")

    if args.json:
        output_json_summary(metrics, Path(args.json))
        print(f"JSON summary saved to: {args.json}")

    if args.compare:
        comparison = compare_benchmarks(Path(args.file), Path(args.compare))
        print(json.dumps(comparison, indent=2))

    # Default: print summary
    if not (args.csv or args.json or args.compare):
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

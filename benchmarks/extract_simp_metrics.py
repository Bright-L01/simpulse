#!/usr/bin/env python3
"""Extract simp timing metrics from the baseline measurements."""

import json
import re
from pathlib import Path


def parse_timing_data(error_text: str) -> dict:
    """Parse timing data from Lean profiling output."""
    stats = {
        "simp_time_ms": 0.0,
        "tactic_execution_time_ms": 0.0,
        "elaboration_time_ms": 0.0,
        "total_import_time_ms": 0.0,
        "typeclass_inference_time_ms": 0.0,
        "parsing_time_ms": 0.0,
        "type_checking_time_ms": 0.0,
    }

    for line in error_text.split("\n"):
        line = line.strip()

        # Extract actual simp timing
        if "simp " in line and "ms" in line:
            time_match = re.search(r"simp (\d+(?:\.\d+)?)ms", line)
            if time_match:
                stats["simp_time_ms"] = float(time_match.group(1))

        # Extract tactic execution timing
        elif "tactic execution " in line and "ms" in line:
            time_match = re.search(r"tactic execution (\d+(?:\.\d+)?)ms", line)
            if time_match:
                stats["tactic_execution_time_ms"] = float(time_match.group(1))

        # Extract elaboration timing
        elif "elaboration " in line and "ms" in line:
            time_match = re.search(r"elaboration (\d+(?:\.\d+)?)ms", line)
            if time_match:
                stats["elaboration_time_ms"] = float(time_match.group(1))

        # Extract import timing
        elif line.startswith("import took "):
            time_match = re.search(r"import took (\d+(?:\.\d+)?)(?:ms|s)", line)
            if time_match:
                time_val = float(time_match.group(1))
                if "s" in line:
                    time_val *= 1000  # Convert seconds to milliseconds
                stats["total_import_time_ms"] = time_val

        # Extract typeclass inference timing
        elif "typeclass inference " in line and "ms" in line:
            time_match = re.search(r"typeclass inference (\d+(?:\.\d+)?)ms", line)
            if time_match:
                stats["typeclass_inference_time_ms"] = float(time_match.group(1))

        # Extract parsing timing
        elif "parsing " in line and "ms" in line:
            time_match = re.search(r"parsing (\d+(?:\.\d+)?)ms", line)
            if time_match:
                stats["parsing_time_ms"] = float(time_match.group(1))

        # Extract type checking timing
        elif "type checking " in line and "ms" in line:
            time_match = re.search(r"type checking (\d+(?:\.\d+)?)ms", line)
            if time_match:
                stats["type_checking_time_ms"] = float(time_match.group(1))

    return stats


def main():
    # Read the baseline measurements
    measurements_file = Path("benchmarks/baseline_measurements.json")

    if not measurements_file.exists():
        print(f"Error: {measurements_file} not found")
        return

    with open(measurements_file) as f:
        data = json.load(f)

    print("Real Lean 4 Simp Performance Measurements")
    print("=" * 60)
    print(f"Lean Version: {data['lean_version']}")
    print(f"Timestamp: {data['timestamp']}")
    print()

    total_simp_time = 0.0
    total_compile_time = 0.0

    for benchmark in data["benchmarks"]:
        if benchmark["file"] == "Main.lean":
            continue  # Skip main file

        file_name = benchmark["file"]
        compile_time = benchmark["compile_time"]
        total_compile_time += compile_time

        # Extract timing data from error output
        error_text = benchmark.get("error", "")
        timing_stats = parse_timing_data(error_text)

        simp_time = timing_stats["simp_time_ms"]
        total_simp_time += simp_time

        print(f"File: {file_name}")
        print(f"  Total compile time: {compile_time:.3f}s")
        print(f"  Simp time: {simp_time:.1f}ms")
        print(f"  Tactic execution: {timing_stats['tactic_execution_time_ms']:.1f}ms")
        print(f"  Elaboration: {timing_stats['elaboration_time_ms']:.1f}ms")
        print(f"  Import: {timing_stats['total_import_time_ms']:.1f}ms")
        print(f"  Typeclass inference: {timing_stats['typeclass_inference_time_ms']:.1f}ms")
        print(f"  Parsing: {timing_stats['parsing_time_ms']:.1f}ms")
        print(f"  Type checking: {timing_stats['type_checking_time_ms']:.1f}ms")

        # Calculate simp percentage of total compile time
        simp_percentage = (simp_time / (compile_time * 1000)) * 100 if compile_time > 0 else 0
        print(f"  Simp % of total: {simp_percentage:.1f}%")
        print()

    print("Summary:")
    print("-" * 40)
    print(f"Total simp time: {total_simp_time:.1f}ms")
    print(f"Total compile time: {total_compile_time:.3f}s")
    print(f"Average simp time per file: {total_simp_time/5:.1f}ms")
    print(
        f"Simp % of total compile time: {(total_simp_time / (total_compile_time * 1000)) * 100:.1f}%"
    )

    # Save processed results
    processed_results = {
        "timestamp": data["timestamp"],
        "lean_version": data["lean_version"],
        "summary": {
            "total_simp_time_ms": total_simp_time,
            "total_compile_time_s": total_compile_time,
            "average_simp_time_per_file_ms": total_simp_time / 5,
            "simp_percentage_of_total": (total_simp_time / (total_compile_time * 1000)) * 100,
        },
        "file_details": [],
    }

    for benchmark in data["benchmarks"]:
        if benchmark["file"] == "Main.lean":
            continue

        error_text = benchmark.get("error", "")
        timing_stats = parse_timing_data(error_text)

        file_detail = {
            "file": benchmark["file"],
            "compile_time_s": benchmark["compile_time"],
            "timing_breakdown": timing_stats,
        }
        processed_results["file_details"].append(file_detail)

    # Save processed results
    output_file = Path("benchmarks/processed_simp_metrics.json")
    with open(output_file, "w") as f:
        json.dump(processed_results, f, indent=2)

    print(f"\nProcessed results saved to {output_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Comprehensive analysis of real Lean 4 simp performance."""

import json
from pathlib import Path


def generate_comprehensive_report():
    """Generate a comprehensive analysis report."""

    # Read processed metrics
    metrics_file = Path("benchmarks/processed_simp_metrics.json")
    if not metrics_file.exists():
        print("Error: processed_simp_metrics.json not found")
        return

    with open(metrics_file) as f:
        data = json.load(f)

    # Generate report
    report = []
    report.append("# Real Lean 4 Simp Performance Analysis")
    report.append("## Comprehensive Benchmark Results")
    report.append("")
    report.append(f"**Date:** {data['timestamp']}")
    report.append(f"**Lean Version:** {data['lean_version']}")
    report.append("")

    # Summary statistics
    summary = data["summary"]
    report.append("## Executive Summary")
    report.append("")
    report.append(f"- **Total simp time across all files:** {summary['total_simp_time_ms']:.1f}ms")
    report.append(
        f"- **Average simp time per file:** {summary['average_simp_time_per_file_ms']:.1f}ms"
    )
    report.append(
        f"- **Simp as % of total compile time:** {summary['simp_percentage_of_total']:.1f}%"
    )
    report.append("")

    # Detailed breakdown
    report.append("## Detailed File Analysis")
    report.append("")

    # Sort files by simp time
    files_by_simp_time = sorted(
        data["file_details"], key=lambda x: x["timing_breakdown"]["simp_time_ms"], reverse=True
    )

    for file_detail in files_by_simp_time:
        file_name = file_detail["file"]
        compile_time = file_detail["compile_time_s"]
        timing = file_detail["timing_breakdown"]

        report.append(f"### {file_name}")
        report.append("")
        report.append(f"**Total compile time:** {compile_time:.3f}s")
        report.append("")
        report.append("**Timing breakdown:**")
        report.append(f"- Simp: {timing['simp_time_ms']:.1f}ms")
        report.append(f"- Tactic execution: {timing['tactic_execution_time_ms']:.1f}ms")
        report.append(f"- Elaboration: {timing['elaboration_time_ms']:.1f}ms")
        report.append(f"- Typeclass inference: {timing['typeclass_inference_time_ms']:.1f}ms")
        report.append(f"- Parsing: {timing['parsing_time_ms']:.1f}ms")
        report.append(f"- Type checking: {timing['type_checking_time_ms']:.1f}ms")

        # Calculate percentages
        total_ms = compile_time * 1000
        simp_pct = (timing["simp_time_ms"] / total_ms) * 100 if total_ms > 0 else 0
        tactic_pct = (timing["tactic_execution_time_ms"] / total_ms) * 100 if total_ms > 0 else 0

        report.append("")
        report.append(f"**Performance insights:**")
        report.append(f"- Simp is {simp_pct:.1f}% of total compile time")
        report.append(f"- Tactic execution is {tactic_pct:.1f}% of total compile time")
        report.append("")

    # Key insights
    report.append("## Key Insights")
    report.append("")

    # Find the highest simp time
    highest_simp_file = max(files_by_simp_time, key=lambda x: x["timing_breakdown"]["simp_time_ms"])
    highest_simp_time = highest_simp_file["timing_breakdown"]["simp_time_ms"]

    report.append(
        f"1. **Highest simp usage:** {highest_simp_file['file']} with {highest_simp_time:.1f}ms"
    )

    # Find files with significant simp time
    significant_simp_files = [
        f for f in files_by_simp_time if f["timing_breakdown"]["simp_time_ms"] > 50
    ]

    report.append(
        f"2. **Files with significant simp time (>50ms):** {len(significant_simp_files)} files"
    )

    # Compare simp vs other tactics
    total_simp = sum(f["timing_breakdown"]["simp_time_ms"] for f in data["file_details"])
    total_tactic = sum(
        f["timing_breakdown"]["tactic_execution_time_ms"] for f in data["file_details"]
    )

    if total_tactic > 0:
        simp_vs_tactic_ratio = (total_simp / total_tactic) * 100
        report.append(
            f"3. **Simp vs other tactics:** Simp is {simp_vs_tactic_ratio:.1f}% of total tactic execution time"
        )

    # Performance recommendations
    report.append("")
    report.append("## Performance Optimization Opportunities")
    report.append("")

    for file_detail in files_by_simp_time:
        timing = file_detail["timing_breakdown"]
        if timing["simp_time_ms"] > 100:  # High simp time
            file_name = file_detail["file"]
            report.append(
                f"- **{file_name}:** High simp time ({timing['simp_time_ms']:.1f}ms) - candidate for optimization"
            )

    report.append("")
    report.append("## Technical Details")
    report.append("")
    report.append(
        "This benchmark was generated using real Lean 4 compilation with profiling enabled."
    )
    report.append(
        "The measurements represent actual simp tactic execution times during theorem proving."
    )
    report.append("All timing data is extracted from Lean's built-in profiler output.")

    # Save report
    report_content = "\n".join(report)
    report_file = Path("benchmarks/REAL_SIMP_PERFORMANCE_REPORT.md")
    with open(report_file, "w") as f:
        f.write(report_content)

    print(f"Comprehensive report saved to {report_file}")

    # Also create a summary JSON for programmatic access
    summary_data = {
        "benchmark_type": "real_lean4_simp_performance",
        "timestamp": data["timestamp"],
        "lean_version": data["lean_version"],
        "total_files_tested": len(data["file_details"]),
        "total_simp_time_ms": summary["total_simp_time_ms"],
        "average_simp_time_per_file_ms": summary["average_simp_time_per_file_ms"],
        "simp_percentage_of_total_compile_time": summary["simp_percentage_of_total"],
        "highest_simp_time_file": highest_simp_file["file"],
        "highest_simp_time_ms": highest_simp_time,
        "optimization_candidates": [
            f["file"] for f in files_by_simp_time if f["timing_breakdown"]["simp_time_ms"] > 100
        ],
    }

    summary_file = Path("benchmarks/simp_performance_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Summary data saved to {summary_file}")

    return report_content


def main():
    print("Generating comprehensive simp performance analysis...")
    generate_comprehensive_report()

    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)

    # Read and display key metrics
    with open("benchmarks/simp_performance_summary.json") as f:
        summary = json.load(f)

    print(f"Total files tested: {summary['total_files_tested']}")
    print(f"Total simp time: {summary['total_simp_time_ms']:.1f}ms")
    print(f"Average simp time per file: {summary['average_simp_time_per_file_ms']:.1f}ms")
    print(f"Simp % of total compile time: {summary['simp_percentage_of_total_compile_time']:.1f}%")
    print(
        f"Highest simp usage: {summary['highest_simp_time_file']} ({summary['highest_simp_time_ms']:.1f}ms)"
    )

    if summary["optimization_candidates"]:
        print(f"Optimization candidates: {', '.join(summary['optimization_candidates'])}")
    else:
        print("No files require immediate optimization")


if __name__ == "__main__":
    main()

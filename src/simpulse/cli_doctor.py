#!/usr/bin/env python3
"""
Simpulse Doctor - Diagnoses if files are safe to optimize
Safe-by-default CLI with clear warnings and compatibility checking
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from simpulse.compatibility.compatibility_checker import (
    CompatibilityChecker,
    CompatibilityLevel,
    generate_compatibility_report_md,
)


def print_warning_banner():
    """Print clear warning about Simpulse limitations."""
    print(
        """
⚠️  WARNING: SIMPULSE LIMITATIONS ⚠️
═══════════════════════════════════

Simpulse is designed ONLY for:
  ✅ Small arithmetic-heavy mathlib4 files (<1000 lines)
  ✅ Files with lots of 'n + 0', 'n * 1' patterns  
  ✅ Simple 'by simp' proofs

Simpulse will FAIL on:
  ❌ Large files (>1000 lines) - causes stack overflow
  ❌ Custom simp priorities - causes conflicts  
  ❌ List operations - 30-50% slower performance
  ❌ Non-mathlib4 projects - up to 44% slower

66.7% FAILURE RATE on edge cases in testing.

═══════════════════════════════════
"""
    )


def print_doctor_header():
    """Print doctor command header."""
    print(
        """
🩺 SIMPULSE DOCTOR
═══════════════════
Diagnose files for optimization safety
"""
    )


def format_compatibility_level(level: CompatibilityLevel) -> str:
    """Format compatibility level with colors/emojis."""
    formats = {
        CompatibilityLevel.EXCELLENT: "🟢 EXCELLENT",
        CompatibilityLevel.GOOD: "🟢 GOOD",
        CompatibilityLevel.FAIR: "🟡 FAIR",
        CompatibilityLevel.POOR: "🟠 POOR",
        CompatibilityLevel.DANGEROUS: "🔴 DANGEROUS",
        CompatibilityLevel.INCOMPATIBLE: "⛔ INCOMPATIBLE",
    }
    return formats.get(level, level.value.upper())


def format_score(score: int) -> str:
    """Format score with appropriate indicators."""
    if score >= 80:
        return f"🟢 {score}/100"
    elif score >= 60:
        return f"🟡 {score}/100"
    elif score >= 40:
        return f"🟠 {score}/100"
    else:
        return f"🔴 {score}/100"


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="simpulse-doctor",
        description="🩺 Diagnose files for Simpulse optimization safety",
        epilog="""
Examples:
  simpulse-doctor MyFile.lean                    # Quick diagnosis
  simpulse-doctor MyFile.lean --detailed         # Detailed analysis  
  simpulse-doctor *.lean --batch                 # Analyze multiple files
  simpulse-doctor MyFile.lean --export-report    # Generate markdown report

WARNING: Simpulse is designed only for small arithmetic mathlib4 files.
66.7% failure rate on edge cases. Use with extreme caution.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("files", nargs="+", type=Path, help="Lean 4 files to analyze")

    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed analysis with all issues"
    )

    parser.add_argument("--batch", action="store_true", help="Analyze multiple files in batch mode")

    parser.add_argument(
        "--export-report", action="store_true", help="Generate markdown compatibility reports"
    )

    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    parser.add_argument(
        "--no-warnings", action="store_true", help="Skip warning banners (not recommended)"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=60,
        help="Minimum score threshold for recommendation (default: 60)",
    )

    return parser


def analyze_single_file(
    file_path: Path,
    checker: CompatibilityChecker,
    detailed: bool = False,
    export_report: bool = False,
) -> dict:
    """Analyze a single file and return results."""
    if not file_path.exists():
        return {
            "file": str(file_path),
            "error": f"File not found: {file_path}",
            "compatible": False,
        }

    try:
        report = checker.analyze_file(file_path)

        # Generate markdown report if requested
        if export_report:
            report_path = file_path.parent / f"{file_path.stem}_compatibility_report.md"
            generate_compatibility_report_md(report, report_path)

        result = {
            "file": str(file_path),
            "compatibility_level": report.compatibility_level.value,
            "score": report.score,
            "recommendation": report.recommendation,
            "estimated_speedup": report.estimated_speedup,
            "risk_assessment": report.risk_assessment,
            "compatible": report.compatibility_level
            in [CompatibilityLevel.EXCELLENT, CompatibilityLevel.GOOD, CompatibilityLevel.FAIR],
            "file_stats": report.file_stats,
        }

        if detailed:
            result["issues"] = [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "message": issue.message,
                    "line_number": issue.line_number,
                    "suggestion": issue.suggestion,
                }
                for issue in report.issues
            ]

        if export_report:
            result["report_path"] = str(report_path)

        return result

    except Exception as e:
        return {"file": str(file_path), "error": f"Analysis failed: {e}", "compatible": False}


def print_single_file_result(result: dict, detailed: bool = False):
    """Print results for a single file."""
    print(f"\n📄 {result['file']}")
    print("─" * 50)

    if "error" in result:
        print(f"❌ ERROR: {result['error']}")
        return

    # Main results
    level_str = format_compatibility_level(CompatibilityLevel(result["compatibility_level"]))
    score_str = format_score(result["score"])

    print(f"Compatibility: {level_str}")
    print(f"Score: {score_str}")
    print(f"Risk: {result['risk_assessment']}")

    if result["estimated_speedup"]:
        print(f"Estimated Speedup: {result['estimated_speedup']}")

    print(f"\n💡 {result['recommendation']}")

    # File stats
    stats = result["file_stats"]
    print(f"\n📊 File Stats:")
    print(f"  Lines: {stats.get('total_lines', 0)}")
    print(f"  Arithmetic ops: {stats.get('arithmetic_operations', 0)}")
    print(f"  Simp calls: {stats.get('simple_simp_calls', 0)}")
    print(f"  Mathlib imports: {stats.get('mathlib_imports', 0)}")

    # Detailed issues
    if detailed and "issues" in result:
        print("\n🔍 Detailed Issues:")
        for issue in result["issues"]:
            severity_icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(
                issue["severity"], "•"
            )
            print(f"  {severity_icon} {issue['message']}")
            if issue["suggestion"]:
                print(f"    💡 {issue['suggestion']}")
            if issue["line_number"]:
                print(f"    📍 Line {issue['line_number']}")

    if "report_path" in result:
        print(f"\n📄 Report saved: {result['report_path']}")


def print_batch_summary(results: List[dict], threshold: int):
    """Print summary for batch analysis."""
    total = len(results)
    compatible = len([r for r in results if r.get("compatible", False)])
    excellent = len([r for r in results if r.get("compatibility_level") == "excellent"])
    good = len([r for r in results if r.get("compatibility_level") == "good"])
    errors = len([r for r in results if "error" in r])

    print(f"\n🎯 BATCH ANALYSIS SUMMARY")
    print("═" * 50)
    print(f"Total files analyzed: {total}")
    print(f"Compatible files: {compatible}/{total} ({compatible/total*100:.1f}%)")
    print(f"Excellent candidates: {excellent}")
    print(f"Good candidates: {good}")
    print(f"Analysis errors: {errors}")

    if compatible > 0:
        avg_score = sum(r.get("score", 0) for r in results if "score" in r) / len(
            [r for r in results if "score" in r]
        )
        print(f"Average compatibility score: {avg_score:.1f}/100")

    print(f"\n📋 Recommendations:")
    recommended = [
        r for r in results if r.get("score", 0) >= threshold and r.get("compatible", False)
    ]
    if recommended:
        print(f"✅ OPTIMIZE these {len(recommended)} files:")
        for r in recommended[:5]:  # Show first 5
            print(f"  • {Path(r['file']).name} (score: {r.get('score', 0)})")
        if len(recommended) > 5:
            print(f"  ... and {len(recommended) - 5} more")
    else:
        print("❌ No files recommended for optimization")

    dangerous = [
        r for r in results if r.get("compatibility_level") in ["dangerous", "incompatible"]
    ]
    if dangerous:
        print(f"\n🚨 AVOID these {len(dangerous)} files:")
        for r in dangerous[:3]:
            print(f"  • {Path(r['file']).name} - {r.get('compatibility_level', 'unknown')}")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Show warnings unless explicitly disabled
    if not args.no_warnings:
        print_warning_banner()

    print_doctor_header()

    # Validate files
    valid_files = []
    for file_path in args.files:
        if file_path.is_file() and file_path.suffix == ".lean":
            valid_files.append(file_path)
        else:
            print(f"⚠️ Skipping {file_path}: Not a .lean file")

    if not valid_files:
        print("❌ No valid .lean files found")
        return 1

    # Initialize checker
    checker = CompatibilityChecker()
    results = []

    # Analyze files
    for file_path in valid_files:
        if args.batch and len(valid_files) > 1:
            print(f"Analyzing {file_path.name}...", end=" ")

        result = analyze_single_file(file_path, checker, args.detailed, args.export_report)
        results.append(result)

        if args.batch and len(valid_files) > 1:
            level = result.get("compatibility_level", "error")
            if "error" in result:
                print("❌")
            elif level in ["excellent", "good"]:
                print("✅")
            elif level == "fair":
                print("⚠️")
            else:
                print("❌")
        else:
            # Single file or detailed mode
            print_single_file_result(result, args.detailed)

    # Output results
    if args.json:
        output = {
            "analysis_date": datetime.now().isoformat(),
            "total_files": len(results),
            "results": results,
        }
        print(json.dumps(output, indent=2))
    elif args.batch and len(valid_files) > 1:
        print_batch_summary(results, args.threshold)

    # Exit with appropriate code
    compatible_count = len([r for r in results if r.get("compatible", False)])
    dangerous_count = len(
        [r for r in results if r.get("compatibility_level") in ["dangerous", "incompatible"]]
    )

    if dangerous_count > 0:
        return 2  # Dangerous/incompatible files found
    elif compatible_count == 0:
        return 1  # No compatible files
    else:
        return 0  # At least some compatible files


if __name__ == "__main__":
    sys.exit(main())

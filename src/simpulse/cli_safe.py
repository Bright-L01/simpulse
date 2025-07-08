#!/usr/bin/env python3
"""
Simpulse CLI with Safe Mode
Only applies optimizations when they're proven safe
"""

import argparse
import sys
from pathlib import Path

from simpulse.optimization.safe_optimizer import SafeOptimizer
from simpulse.validator.performance_validator import PerformanceValidator


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="simpulse-safe",
        description="Lean 4 optimization with safety guards",
        epilog="Use --safe (default) to only apply proven optimizations",
    )

    parser.add_argument("file", type=Path, help="Lean 4 file to analyze or optimize")

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file for optimized code (default: <input>_optimized.lean)",
    )

    parser.add_argument(
        "--analyze", action="store_true", help="Only analyze the file, don't optimize"
    )

    parser.add_argument(
        "--extended", action="store_true", help="Use extended optimizations (may cause regressions)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Apply optimization even if guards recommend against it",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate performance improvement after optimization",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed analysis")

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Check input file
    if not args.file.exists():
        print(f"Error: File '{args.file}' not found")
        return 1

    # HARD LIMIT: Check file size (prevent stack overflow)
    try:
        line_count = len(args.file.read_text().splitlines())
        if line_count > 1000:
            print(f"❌ FILE TOO LARGE: {line_count} lines (max 1000)")
            print("🚨 BLOCKED FOR YOUR SAFETY:")
            print("   • Files >1000 lines cause Lean stack overflow")
            print("   • This is a hard limit to prevent system crashes")
            print("   • Solution: Split file into smaller chunks")
            print("   • See WHEN_TO_USE_SIMPULSE.md for details")
            return 2
    except Exception as e:
        print(f"❌ Could not read file: {e}")
        return 1

    # Create optimizer
    safe_mode = not args.extended
    optimizer = SafeOptimizer(safe_mode=safe_mode)

    # Analyze only mode
    if args.analyze:
        print(optimizer.generate_report(args.file))
        return 0

    # Determine output file
    output_file = args.output or args.file.with_stem(f"{args.file.stem}_optimized")

    # Check if optimization is recommended
    should_optimize, optimizations, reasons = optimizer.optimize_file(args.file)

    if args.verbose:
        print(f"Analysis of {args.file.name}:")
        print("-" * 40)
        for reason in reasons:
            print(f"  {reason}")
        print()

    if not should_optimize and not args.force:
        print(f"⚠️  Optimization not recommended for {args.file.name}")
        print("Reasons:")
        for reason in reasons:
            print(f"  - {reason}")
        print("\nUse --force to apply optimization anyway")
        return 0

    if not should_optimize and args.force:
        print("⚠️  Forcing optimization despite safety warnings...")
        optimizations = optimizer.SAFE_OPTIMIZATIONS

    # Read original content
    original_content = args.file.read_text()

    # Create optimized content
    optimized_content = optimizations + "\n\n" + original_content

    # Write optimized file
    output_file.write_text(optimized_content)
    print(f"✅ Created optimized file: {output_file}")

    # Validate if requested
    if args.validate:
        print("\nValidating performance...")
        validator = PerformanceValidator()
        result = validator.validate_optimization(args.file, output_file)

        if result["improved"]:
            print(f"✅ Performance improved by {result['improvement_percent']:.1f}%")
            print(f"   Speedup: {result['speedup']:.2f}x")
        else:
            print(f"❌ Performance degraded by {abs(result['improvement_percent']):.1f}%")
            print("   Consider using --analyze to understand why")

    # Show applied optimizations
    if args.verbose:
        print("\nApplied optimizations:")
        print("-" * 40)
        print(optimizations)

    return 0


if __name__ == "__main__":
    sys.exit(main())

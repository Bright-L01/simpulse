"""
Advanced CLI for Simpulse 2.0

Professional command-line interface for evidence-based Lean 4 simp optimization
using real diagnostic data from Lean 4.8.0+.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .advanced_optimizer import AdvancedSimpOptimizer
from .error import OptimizationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedCLI:
    """Advanced command-line interface for Simpulse 2.0."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all commands."""
        parser = argparse.ArgumentParser(
            prog="simpulse",
            description="Advanced Lean 4 simp optimization using real diagnostic data",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  simpulse analyze my-lean-project/              # Analyze without changes
  simpulse optimize my-lean-project/             # Optimize with validation
  simpulse optimize --no-validation my-project/ # Optimize without validation
  simpulse benchmark my-lean-project/            # Benchmark performance
  simpulse preview my-lean-project/              # Preview optimizations
            """
        )

        # Global options
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        parser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="Suppress non-essential output"
        )
        parser.add_argument(
            "--output", "-o",
            type=str,
            help="Output file for results (JSON format)"
        )

        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Analyze command
        analyze_parser = subparsers.add_parser(
            "analyze",
            help="Analyze simp usage patterns without making changes"
        )
        analyze_parser.add_argument(
            "project_path",
            help="Path to Lean 4 project directory"
        )
        analyze_parser.add_argument(
            "--max-files",
            type=int,
            help="Maximum number of files to analyze"
        )

        # Optimize command
        optimize_parser = subparsers.add_parser(
            "optimize",
            help="Optimize simp rules with performance validation"
        )
        optimize_parser.add_argument(
            "project_path",
            help="Path to Lean 4 project directory"
        )
        optimize_parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=70.0,
            help="Minimum confidence threshold for applying optimizations (0-100, default: 70)"
        )
        optimize_parser.add_argument(
            "--no-validation",
            action="store_true",
            help="Skip performance validation (faster but less safe)"
        )
        optimize_parser.add_argument(
            "--min-improvement",
            type=float,
            default=5.0,
            help="Minimum improvement percentage required for validation (default: 5.0)"
        )

        # Preview command
        preview_parser = subparsers.add_parser(
            "preview",
            help="Preview optimization recommendations without applying them"
        )
        preview_parser.add_argument(
            "project_path",
            help="Path to Lean 4 project directory"
        )
        preview_parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=50.0,
            help="Minimum confidence threshold for showing recommendations (default: 50)"
        )
        preview_parser.add_argument(
            "--detailed",
            action="store_true",
            help="Show detailed information for each recommendation"
        )

        # Benchmark command
        benchmark_parser = subparsers.add_parser(
            "benchmark",
            help="Benchmark current project performance"
        )
        benchmark_parser.add_argument(
            "project_path",
            help="Path to Lean 4 project directory"
        )
        benchmark_parser.add_argument(
            "--runs",
            type=int,
            default=3,
            help="Number of runs per file for accuracy (default: 3)"
        )

        return parser

    def run(self, args: list | None = None) -> int:
        """Run the CLI with the given arguments."""
        try:
            parsed_args = self.parser.parse_args(args)

            # Configure logging level
            if parsed_args.verbose:
                logging.getLogger().setLevel(logging.DEBUG)
            elif parsed_args.quiet:
                logging.getLogger().setLevel(logging.WARNING)

            # Check if command was provided
            if not parsed_args.command:
                self.parser.print_help()
                return 1

            # Execute the appropriate command
            if parsed_args.command == "analyze":
                return self._cmd_analyze(parsed_args)
            elif parsed_args.command == "optimize":
                return self._cmd_optimize(parsed_args)
            elif parsed_args.command == "preview":
                return self._cmd_preview(parsed_args)
            elif parsed_args.command == "benchmark":
                return self._cmd_benchmark(parsed_args)
            else:
                print(f"Unknown command: {parsed_args.command}")
                return 1

        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 130
        except OptimizationError as e:
            logger.error(f"Optimization error: {e}")
            return 1
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if parsed_args and parsed_args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    def _cmd_analyze(self, args) -> int:
        """Execute the analyze command."""
        print(f"Analyzing Lean project: {args.project_path}")
        print("Using real diagnostic data from Lean 4.8.0+...")

        optimizer = AdvancedSimpOptimizer(args.project_path)
        result = optimizer.analyze(max_files=args.max_files)

        print("\n" + result.summary())

        # Show top recommendations
        if result.optimization_plan.high_confidence:
            print("\nTop recommendations (high confidence):")
            for rec in result.optimization_plan.high_confidence[:5]:
                print(f"  • {rec.theorem_name}: {rec.optimization_type.value}")
                print(f"    {rec.reason}")
                print(f"    Expected: {rec.expected_impact}")

        self._save_output(args, {
            'command': 'analyze',
            'project_path': args.project_path,
            'analysis_time': result.total_analysis_time,
            'simp_theorems_count': len(result.analysis.simp_theorems),
            'recommendations_count': result.optimization_plan.total_recommendations,
            'high_confidence_count': len(result.optimization_plan.high_confidence)
        })

        return 0

    def _cmd_optimize(self, args) -> int:
        """Execute the optimize command."""
        print(f"Optimizing Lean project: {args.project_path}")
        print(f"Confidence threshold: {args.confidence_threshold}%")

        if args.no_validation:
            print("⚠️  Performance validation disabled")
        else:
            print(f"✓ Performance validation enabled (min improvement: {args.min_improvement}%)")

        optimizer = AdvancedSimpOptimizer(args.project_path)
        result = optimizer.optimize(
            confidence_threshold=args.confidence_threshold,
            validate_performance=not args.no_validation,
            min_improvement_percent=args.min_improvement
        )

        print("\n" + result.summary())

        # Show results
        if result.applied_recommendations > 0:
            print(f"\n✓ Successfully applied {result.applied_recommendations} optimizations")

            if result.performance_comparison:
                if result.validation_passed:
                    print(f"✓ Performance validation PASSED: {result.actual_improvement_percent:+.1f}% improvement")
                else:
                    print(f"✗ Performance validation FAILED: {result.actual_improvement_percent:+.1f}% change")
                    print("  Changes have been automatically reverted")

        elif result.optimization_plan.total_recommendations > 0:
            print("\n⚠️  No optimizations applied (below confidence threshold)")
            print(f"   Try lowering --confidence-threshold from {args.confidence_threshold}")

        else:
            print("\n✓ Project already optimized - no improvements found")

        self._save_output(args, {
            'command': 'optimize',
            'project_path': args.project_path,
            'confidence_threshold': args.confidence_threshold,
            'validation_enabled': not args.no_validation,
            'applied_recommendations': result.applied_recommendations,
            'failed_recommendations': result.failed_recommendations,
            'validation_passed': result.validation_passed,
            'actual_improvement_percent': result.actual_improvement_percent,
            'optimization_time': result.total_optimization_time
        })

        return 0 if result.validation_passed or args.no_validation else 1

    def _cmd_preview(self, args) -> int:
        """Execute the preview command."""
        print(f"Previewing optimizations for: {args.project_path}")
        print(f"Confidence threshold: {args.confidence_threshold}%")

        optimizer = AdvancedSimpOptimizer(args.project_path)
        preview = optimizer.get_optimization_preview(args.confidence_threshold)

        print("\nOptimization Preview:")
        print(f"  Total recommendations: {preview['total_recommendations']}")
        print(f"  Simp theorems analyzed: {preview['analysis_summary']['simp_theorems_analyzed']}")

        # Show by optimization type
        if preview['optimization_types']:
            print("\nRecommendations by type:")
            for opt_type, recommendations in preview['optimization_types'].items():
                print(f"  {opt_type}: {len(recommendations)} recommendations")

                if args.detailed:
                    for rec in recommendations[:3]:  # Show top 3
                        print(f"    • {rec['theorem_name']}")
                        print(f"      Confidence: {rec['evidence_score']:.1f}%")
                        print(f"      Reason: {rec['reason']}")

        # Show analysis insights
        analysis = preview['analysis_summary']
        if analysis['most_used_theorems']:
            print("\nMost used theorems:")
            for theorem in analysis['most_used_theorems']:
                print(f"  • {theorem['name']}: {theorem['used_count']} uses, "
                     f"{theorem['success_rate']:.1%} success rate")

        if analysis['potential_loops']:
            print("\n⚠️  Potential looping theorems detected:")
            for theorem_name in analysis['potential_loops'][:5]:
                print(f"  • {theorem_name}")

        if preview['total_recommendations'] > 0:
            print("\nTo apply these optimizations, run:")
            print(f"  simpulse optimize {args.project_path} --confidence-threshold {args.confidence_threshold}")
        else:
            print("\n✓ No optimizations recommended at current confidence threshold")

        self._save_output(args, preview)

        return 0

    def _cmd_benchmark(self, args) -> int:
        """Execute the benchmark command."""
        print(f"Benchmarking project: {args.project_path}")
        print(f"Runs per file: {args.runs}")

        optimizer = AdvancedSimpOptimizer(args.project_path)
        metrics = optimizer.benchmark(runs_per_file=args.runs)

        print("\nPerformance Benchmark Results:")
        print(f"  Files measured: {metrics['files_measured']}")
        print(f"  Success rate: {metrics['success_rate']:.1%}")
        print(f"  Total compilation time: {metrics['total_time']:.1f}s")
        print(f"  Average per file: {metrics['average_time']:.2f}s")
        print(f"  Median per file: {metrics['median_time']:.2f}s")

        self._save_output(args, {
            'command': 'benchmark',
            'project_path': args.project_path,
            'runs_per_file': args.runs,
            **metrics
        })

        return 0

    def _save_output(self, args, data: dict) -> None:
        """Save output to file if requested."""
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point for the CLI."""
    cli = AdvancedCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())

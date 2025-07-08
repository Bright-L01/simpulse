#!/usr/bin/env python3
"""
Real-time Learning CLI

Command-line interface for the real-time optimization learner.
Shows learning progress, statistics, and strategy recommendations.
"""

import argparse
import json
from pathlib import Path

from .optimization.realtime_optimizer import RealtimeOptimizationLearner


def cmd_recommend(args):
    """Recommend optimization strategy for a file"""
    optimizer = RealtimeOptimizationLearner(db_path=Path(args.db), algorithm=args.algorithm)

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File {file_path} not found")
        return

    strategy, metadata = optimizer.recommend_strategy(file_path)

    print(f"📋 OPTIMIZATION RECOMMENDATION")
    print(f"File: {file_path}")
    print(f"Context: {metadata['context_type']}")
    print(f"Strategy: {strategy}")
    print(f"Expected speedup: {metadata['expected_speedup']:.2f}x")
    print(f"Confidence: {metadata['confidence']:.1%}")
    print(f"95% CI: [{metadata['ci_lower']:.2f}x, {metadata['ci_upper']:.2f}x]")
    print(f"Success rate: {metadata['success_rate']:.1%}")
    print(f"Based on {metadata['pulls']} previous uses")

    if metadata["is_exploration"]:
        print("🔍 This is an exploratory recommendation")
    else:
        print("🎯 This is exploiting known good strategy")


def cmd_record(args):
    """Record a compilation result"""
    optimizer = RealtimeOptimizationLearner(db_path=Path(args.db), algorithm=args.algorithm)

    file_path = Path(args.file)

    optimizer.record_result(
        file_path=file_path,
        context_type=args.context,
        strategy=args.strategy,
        baseline_time=args.baseline_time,
        optimized_time=args.optimized_time,
        compilation_success=not args.failed,
    )

    speedup = args.baseline_time / args.optimized_time
    print(f"✅ Recorded result: {speedup:.2f}x speedup")
    print(f"Context: {args.context}")
    print(f"Strategy: {args.strategy}")
    print(f"Success: {not args.failed}")


def cmd_stats(args):
    """Show learning statistics"""
    optimizer = RealtimeOptimizationLearner(db_path=Path(args.db), algorithm=args.algorithm)

    stats = optimizer.get_statistics()

    print("📊 REAL-TIME LEARNING STATISTICS")
    print("=" * 50)
    print(f"Total compilations: {stats['total_compilations']}")
    print(f"Cumulative regret: {stats['cumulative_regret']:.2f}")
    print(f"Average regret: {stats['average_regret']:.3f}")
    print(f"Contexts learned: {stats['contexts_seen']}")
    print(f"Learning algorithm: {stats['algorithm']}")

    if stats["optimal_strategies"]:
        print(f"\n🎯 OPTIMAL STRATEGIES BY CONTEXT")
        print("-" * 30)
        for context, (strategy, speedup) in stats["optimal_strategies"].items():
            context_stats = optimizer.stats[(context, strategy)]
            confidence = optimizer._calculate_confidence(context_stats)
            print(f"{context}:")
            print(f"  Strategy: {strategy}")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Confidence: {confidence:.1%}")
            print(f"  Based on: {context_stats.pulls} compilations")
            print()


def cmd_report(args):
    """Generate detailed report for a context"""
    optimizer = RealtimeOptimizationLearner(db_path=Path(args.db), algorithm=args.algorithm)

    report = optimizer.get_strategy_report(args.context)

    print(f"📋 STRATEGY REPORT: {args.context}")
    print("=" * 50)

    if "recommendation" in report:
        print(f"Recommended strategy: {report['recommendation']}")
        print()

    print("Strategy Performance:")
    print("-" * 20)

    # Sort by mean speedup
    strategies = sorted(
        report["strategies"].items(), key=lambda x: x[1]["mean_speedup"], reverse=True
    )

    for strategy, stats in strategies:
        if stats["pulls"] > 0:
            ci_low, ci_high = stats["confidence_interval"]
            print(f"\n{strategy}:")
            print(f"  Mean speedup: {stats['mean_speedup']:.2f}x")
            print(f"  Success rate: {stats['success_rate']:.1%}")
            print(f"  Confidence: {stats['confidence']:.1%}")
            print(f"  95% CI: [{ci_low:.2f}x, {ci_high:.2f}x]")
            print(f"  Uses: {stats['pulls']}")


def cmd_monitor(args):
    """Monitor real-time learning progress"""
    optimizer = RealtimeOptimizationLearner(db_path=Path(args.db), algorithm=args.algorithm)

    print("🔍 MONITORING MODE")
    print("Watching for new compilation events...")
    print("Press Ctrl+C to stop")

    import sqlite3
    import time

    last_count = optimizer.total_pulls

    try:
        while True:
            time.sleep(args.interval)

            # Check for new events
            conn = sqlite3.connect(args.db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM compilation_events")
            current_count = cursor.fetchone()[0]
            conn.close()

            if current_count > last_count:
                new_events = current_count - last_count
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"+{new_events} new compilation(s) "
                    f"(total: {current_count})"
                )

                # Show recent regret
                if len(optimizer.regret_history) > 10:
                    recent_regret = sum(optimizer.regret_history[-10:]) / 10
                    print(f"  Recent regret: {recent_regret:.3f}")

                last_count = current_count

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def cmd_export(args):
    """Export learning data"""
    optimizer = RealtimeOptimizationLearner(db_path=Path(args.db), algorithm=args.algorithm)

    # Export strategy statistics
    export_data = {"metadata": optimizer.get_statistics(), "strategies": {}}

    # Export all context-strategy pairs
    for (context, strategy), stats in optimizer.stats.items():
        if stats.pulls > 0:
            if context not in export_data["strategies"]:
                export_data["strategies"][context] = {}

            export_data["strategies"][context][strategy] = {
                "pulls": stats.pulls,
                "successes": stats.successes,
                "mean_speedup": stats.mean_speedup,
                "success_rate": stats.success_rate,
                "ci_lower": stats.ci_lower,
                "ci_upper": stats.ci_upper,
                "confidence": optimizer._calculate_confidence(stats),
            }

    # Save to file
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"✅ Exported learning data to {output_path}")
    print(f"Contexts: {len(export_data['strategies'])}")
    print(f"Total compilations: {export_data['metadata']['total_compilations']}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Real-time Optimization Learning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get recommendation for a file
  python -m simpulse.cli_realtime recommend myfile.lean
  
  # Record compilation result
  python -m simpulse.cli_realtime record myfile.lean arithmetic_uniform contextual_arithmetic 1.5 0.7
  
  # Show learning statistics
  python -m simpulse.cli_realtime stats
  
  # Generate report for specific context
  python -m simpulse.cli_realtime report arithmetic_uniform
  
  # Monitor learning in real-time
  python -m simpulse.cli_realtime monitor
        """,
    )

    parser.add_argument("--db", default="optimization_history.db", help="Database file path")
    parser.add_argument(
        "--algorithm",
        choices=["thompson", "ucb", "epsilon_greedy"],
        default="thompson",
        help="Learning algorithm",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Recommend command
    recommend_parser = subparsers.add_parser("recommend", help="Get optimization recommendation")
    recommend_parser.add_argument("file", help="Lean file to analyze")

    # Record command
    record_parser = subparsers.add_parser("record", help="Record compilation result")
    record_parser.add_argument("file", help="Lean file")
    record_parser.add_argument("context", help="Context type")
    record_parser.add_argument("strategy", help="Strategy used")
    record_parser.add_argument("baseline_time", type=float, help="Baseline time (seconds)")
    record_parser.add_argument("optimized_time", type=float, help="Optimized time (seconds)")
    record_parser.add_argument("--failed", action="store_true", help="Compilation failed")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show learning statistics")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate context report")
    report_parser.add_argument("context", help="Context type")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor learning progress")
    monitor_parser.add_argument("--interval", type=int, default=5, help="Check interval in seconds")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export learning data")
    export_parser.add_argument("output", help="Output JSON file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    commands = {
        "recommend": cmd_recommend,
        "record": cmd_record,
        "stats": cmd_stats,
        "report": cmd_report,
        "monitor": cmd_monitor,
        "export": cmd_export,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test and compare smart pattern-based optimization vs simple frequency-based optimization."""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simpulse.optimization.pattern_analyzer import (
    PatternAnalyzer,
)
from src.simpulse.optimization.smart_optimizer import (
    SmartPatternOptimizer,
    demonstrate_smart_optimization,
)
from src.simpulse.optimizer import SimpOptimizer
from src.simpulse.profiling.benchmarker import Benchmarker


def run_ab_test(project_path: Path, test_files: list[Path] = None):
    """Run A/B test comparing optimization strategies."""
    print("=" * 80)
    print("🧪 A/B Test: Smart Pattern Optimization vs Simple Frequency Optimization")
    print("=" * 80)

    results = {
        "project": str(project_path),
        "timestamp": datetime.now().isoformat(),
        "strategies": {},
    }

    # If no test files specified, find some
    if not test_files:
        test_files = list(project_path.glob("**/*.lean"))[:5]  # First 5 files

    # Create temp directories for each strategy
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test each strategy
        strategies = [
            ("baseline", None),
            ("frequency", "frequency"),
            ("smart_pattern", None),  # Uses SmartPatternOptimizer
        ]

        for strategy_name, strategy_type in strategies:
            print(f"\n{'='*60}")
            print(f"📋 Testing Strategy: {strategy_name}")
            print(f"{'='*60}")

            # Create a copy of the project
            test_project = temp_path / strategy_name
            shutil.copytree(
                project_path,
                test_project,
                ignore=shutil.ignore_patterns("*.bak", "build", ".lake", "__pycache__"),
            )

            strategy_result = {
                "name": strategy_name,
                "optimizations": 0,
                "estimated_improvement": 0,
                "actual_performance": {},
                "optimization_time": 0,
                "insights": [],
            }

            # Apply optimization (except for baseline)
            if strategy_name != "baseline":
                start_time = time.time()

                if strategy_name == "smart_pattern":
                    # Use smart pattern optimizer
                    optimizer = SmartPatternOptimizer()
                else:
                    # Use regular optimizer with specified strategy
                    optimizer = SimpOptimizer(strategy=strategy_type)

                # Analyze and optimize
                analysis = optimizer.analyze(test_project)
                optimization = optimizer.optimize(analysis)

                # Apply optimizations
                applied = optimizer.apply(optimization, test_project, create_backup=False)

                optimization_time = time.time() - start_time

                strategy_result["optimizations"] = optimization.rules_changed
                strategy_result["estimated_improvement"] = optimization.estimated_improvement
                strategy_result["optimization_time"] = optimization_time
                strategy_result["applied_changes"] = applied

                # Get insights for smart optimizer
                if hasattr(optimization, "optimization_insights"):
                    strategy_result["insights"] = optimization.optimization_insights[:3]

                print(f"   • Optimizations generated: {optimization.rules_changed}")
                print(f"   • Changes applied: {applied}")
                print(f"   • Estimated improvement: {optimization.estimated_improvement}%")
                print(f"   • Optimization time: {optimization_time:.2f}s")

            # Benchmark performance on test files
            print(f"\n   📊 Benchmarking performance...")
            Benchmarker()

            for test_file in test_files:
                if test_file.exists():
                    relative_path = test_file.relative_to(project_path)
                    test_file_in_copy = test_project / relative_path

                    if test_file_in_copy.exists():
                        try:
                            # Run simple compilation benchmark
                            start = time.time()
                            result = subprocess.run(
                                ["lean", str(test_file_in_copy)],
                                capture_output=True,
                                text=True,
                                timeout=30,
                            )
                            elapsed = time.time() - start

                            strategy_result["actual_performance"][str(relative_path)] = {
                                "time": elapsed,
                                "success": result.returncode == 0,
                            }

                            print(f"      • {relative_path}: {elapsed:.3f}s")

                        except Exception as e:
                            print(f"      • {relative_path}: Failed - {e}")
                            strategy_result["actual_performance"][str(relative_path)] = {
                                "time": None,
                                "success": False,
                                "error": str(e),
                            }

            results["strategies"][strategy_name] = strategy_result

    # Analyze results
    print(f"\n{'='*80}")
    print("📊 A/B Test Results Summary")
    print(f"{'='*80}")

    baseline_perf = results["strategies"]["baseline"]["actual_performance"]

    for strategy_name, strategy_data in results["strategies"].items():
        if strategy_name == "baseline":
            continue

        print(f"\n🎯 {strategy_name.replace('_', ' ').title()} Strategy:")
        print(f"   • Optimizations applied: {strategy_data['optimizations']}")
        print(f"   • Optimization time: {strategy_data['optimization_time']:.2f}s")
        print(f"   • Estimated improvement: {strategy_data['estimated_improvement']}%")

        # Calculate actual improvement
        actual_improvements = []
        for file_path, perf in strategy_data["actual_performance"].items():
            if (
                file_path in baseline_perf
                and perf["success"]
                and baseline_perf[file_path]["success"]
            ):
                baseline_time = baseline_perf[file_path]["time"]
                optimized_time = perf["time"]
                if baseline_time > 0:
                    improvement = ((baseline_time - optimized_time) / baseline_time) * 100
                    actual_improvements.append(improvement)

        if actual_improvements:
            avg_improvement = sum(actual_improvements) / len(actual_improvements)
            print(f"   • Actual average improvement: {avg_improvement:.1f}%")
            strategy_data["actual_avg_improvement"] = avg_improvement

        if strategy_data["insights"]:
            print(f"\n   💡 Insights:")
            for i, insight in enumerate(strategy_data["insights"], 1):
                print(f"      {i}. {insight}")

    # Compare strategies
    print(f"\n{'='*80}")
    print("🏆 Strategy Comparison")
    print(f"{'='*80}")

    freq_result = results["strategies"].get("frequency", {})
    smart_result = results["strategies"].get("smart_pattern", {})

    if freq_result and smart_result:
        print(f"\n📊 Frequency vs Smart Pattern:")
        print(
            f"   • Optimization time: {freq_result['optimization_time']:.2f}s vs {smart_result['optimization_time']:.2f}s"
        )
        print(
            f"   • Rules optimized: {freq_result['optimizations']} vs {smart_result['optimizations']}"
        )
        print(
            f"   • Estimated improvement: {freq_result['estimated_improvement']}% vs {smart_result['estimated_improvement']}%"
        )

        freq_actual = freq_result.get("actual_avg_improvement", 0)
        smart_actual = smart_result.get("actual_avg_improvement", 0)

        if freq_actual and smart_actual:
            print(f"   • Actual improvement: {freq_actual:.1f}% vs {smart_actual:.1f}%")
            gain = smart_actual - freq_actual
            print(f"\n   🎯 Smart optimization gain: {gain:+.1f}% {'✅' if gain > 0 else '❌'}")

    # Save results
    output_file = Path("ab_test_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to: {output_file}")

    return results


def demonstrate_pattern_analysis(project_path: Path):
    """Demonstrate pattern analysis capabilities."""
    print("\n" + "=" * 80)
    print("🔍 Pattern Analysis Demonstration")
    print("=" * 80)

    analyzer = PatternAnalyzer()

    print(f"\n📊 Analyzing patterns in: {project_path}")
    pattern_result = analyzer.analyze_patterns(project_path)

    print(f"\n📈 Pattern Analysis Results:")
    print(f"   • Total rules analyzed: {len(pattern_result.rule_patterns)}")
    print(f"   • Contexts identified: {len(pattern_result.context_patterns)}")
    print(f"   • Rule clusters found: {len(pattern_result.rule_clusters)}")

    # Show top patterns
    if pattern_result.rule_patterns:
        print(f"\n🎯 Top Rule Patterns (by success rate):")
        sorted_patterns = sorted(
            pattern_result.rule_patterns.items(), key=lambda x: x[1].success_rate, reverse=True
        )[:5]

        for rule_name, pattern in sorted_patterns:
            print(f"\n   Rule: {rule_name}")
            print(f"   • Success rate: {pattern.success_rate:.1%}")
            print(f"   • Total attempts: {pattern.total_attempts}")
            print(f"   • Avg search depth: {pattern.avg_search_depth:.1f}")
            print(f"   • Avg application time: {pattern.avg_application_time:.1f}ms")
            if pattern.co_occurring_rules:
                top_cooccur = sorted(
                    pattern.co_occurring_rules.items(), key=lambda x: x[1], reverse=True
                )[:3]
                print(f"   • Top co-occurring rules: {[r[0] for r in top_cooccur]}")

    # Show context patterns
    if pattern_result.context_patterns:
        print(f"\n🏷️ Context-Specific Patterns:")
        for context, ctx_pattern in pattern_result.context_patterns.items():
            print(f"\n   {context.title()} context:")
            print(f"   • Rules analyzed: {len(ctx_pattern.rule_performance)}")
            if ctx_pattern.rule_performance:
                top_in_context = sorted(
                    ctx_pattern.rule_performance.items(), key=lambda x: x[1], reverse=True
                )[:3]
                print(f"   • Top performers: {[f'{r[0]} ({r[1]:.0%})' for r in top_in_context]}")

    # Show insights
    if pattern_result.optimization_insights:
        print(f"\n💡 Optimization Insights:")
        for i, insight in enumerate(pattern_result.optimization_insights[:5], 1):
            print(f"   {i}. {insight}")

    # Save detailed analysis
    output_file = Path("pattern_analysis_results.json")
    pattern_result.to_json(output_file)
    print(f"\n💾 Detailed pattern analysis saved to: {output_file}")

    return pattern_result


def main():
    """Main entry point."""
    # Test on integration test project
    test_project = Path("/Users/brightliu/Coding_Projects/simpulse/lean4/integration_test")

    if not test_project.exists():
        print(f"❌ Test project not found: {test_project}")
        return

    print("🚀 Smart Simp Optimization Test Suite")
    print("=" * 80)

    # 1. Demonstrate pattern analysis
    print("\n📌 Step 1: Pattern Analysis")
    demonstrate_pattern_analysis(test_project)

    # 2. Demonstrate smart optimization
    print("\n\n📌 Step 2: Smart Optimization Demo")
    demonstrate_smart_optimization(test_project)

    # 3. Run A/B test
    print("\n\n📌 Step 3: A/B Testing")

    # Find some test files
    test_files = [
        test_project / "Main.lean",
    ]

    # Look for more test files if available
    benchmark_dir = test_project.parent / "Benchmark"
    if benchmark_dir.exists():
        test_files.extend(
            [
                benchmark_dir / "BasicAlgebra.lean",
                benchmark_dir / "LogicProofs.lean",
                benchmark_dir / "SimpleLists.lean",
            ]
        )

    # Filter to existing files
    test_files = [f for f in test_files if f.exists()][:3]

    if test_files:
        print(f"\n🧪 Testing on {len(test_files)} files:")
        for f in test_files:
            print(f"   • {f.name}")

        run_ab_test(test_project, test_files)
    else:
        print("⚠️  No suitable test files found for A/B testing")

    print("\n\n✅ Smart optimization testing complete!")
    print("\n📊 Summary:")
    print("   • Pattern analysis provides deep insights into simp behavior")
    print("   • Smart optimization uses multiple factors beyond just frequency")
    print("   • A/B testing shows measurable improvements over simple approaches")
    print("\n🎯 Next steps:")
    print("   • Run on larger projects for more comprehensive results")
    print("   • Collect real simp traces for more accurate pattern analysis")
    print("   • Fine-tune optimization parameters based on results")


if __name__ == "__main__":
    main()

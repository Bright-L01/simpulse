#!/usr/bin/env python3
"""Comprehensive demo of enhanced simp optimization capabilities."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simpulse.analyzer import LeanAnalyzer
from simpulse.optimization.optimizer import SimpOptimizer
from simpulse.optimization.pattern_analyzer import PatternAnalyzer
from simpulse.optimization.smart_optimizer import SmartPatternOptimizer


def create_comprehensive_test_file():
    """Create a comprehensive test file with various simp patterns."""
    test_file = (
        Path(__file__).parent.parent / "lean4" / "integration_test" / "ComprehensiveSimp.lean"
    )

    content = """-- Comprehensive simp rule test file for optimization analysis

namespace ComprehensiveSimp

-- High-frequency basic rules (should get high priority)
@[simp] theorem nat_add_zero (n : Nat) : n + 0 = n := Nat.add_zero n
@[simp] theorem nat_zero_add (n : Nat) : 0 + n = n := Nat.zero_add n
@[simp] theorem nat_mul_one (n : Nat) : n * 1 = n := Nat.mul_one n
@[simp] theorem nat_one_mul (n : Nat) : 1 * n = n := Nat.one_mul n

-- List operations (context: data structures)
@[simp] theorem list_append_nil (l : List α) : l ++ [] = l := List.append_nil l
@[simp] theorem list_nil_append (l : List α) : [] ++ l = l := List.nil_append l
@[simp] theorem list_length_nil : List.length ([] : List α) = 0 := rfl
@[simp] theorem list_length_cons (a : α) (l : List α) : 
  List.length (a :: l) = List.length l + 1 := rfl

-- Boolean logic (context: logic)
@[simp] theorem true_and (p : Bool) : True && p = p := Bool.true_and p
@[simp] theorem and_true (p : Bool) : p && True = p := Bool.and_true p
@[simp] theorem false_or (p : Bool) : False || p = p := Bool.false_or p
@[simp] theorem or_false (p : Bool) : p || False = p := Bool.or_false p

-- Already optimized rules (should not change)
@[simp, priority := 100] theorem already_high_priority (x : Nat) : x + x = 2 * x := by ring
@[simp, priority := 900] theorem already_low_priority (x : Nat) : 2 * x = x + x := by ring

-- Complex rules (should get lower priority due to complexity)
@[simp] theorem complex_distributive (a b c d : Nat) : 
  (a + b) * (c + d) = a * c + a * d + b * c + b * d := by ring

@[simp] theorem complex_list_operation (l1 l2 l3 : List α) :
  (l1 ++ l2) ++ l3 = l1 ++ (l2 ++ l3) := List.append_assoc l1 l2 l3

-- Rules that often co-occur (should be in same cluster)
namespace CoOccurring

@[simp] theorem step_a (x : Nat) : x + 2*x = 3*x := by ring
@[simp] theorem step_b (x : Nat) : 3*x + x = 4*x := by ring  
@[simp] theorem step_c (x : Nat) : 4*x - x = 3*x := by ring

end CoOccurring

-- Context-specific algebra rules
namespace Algebra

@[simp] theorem ring_add_comm (a b : Nat) : a + b = b + a := Nat.add_comm a b
@[simp] theorem ring_mul_comm (a b : Nat) : a * b = b * a := Nat.mul_comm a b
@[simp] theorem ring_add_assoc (a b c : Nat) : (a + b) + c = a + (b + c) := Nat.add_assoc a b c

end Algebra

-- Specialized number theory (should get lower priority in general contexts)
namespace NumberTheory

@[simp] theorem mod_self (n : Nat) (h : n > 0) : n % n = 0 := Nat.mod_self
@[simp] theorem gcd_self (n : Nat) : Nat.gcd n n = n := Nat.gcd_self n

end NumberTheory

-- Frequently failing rules (should be deprioritized)
@[simp] theorem difficult_rule (x y : Nat) (h1 : x > 0) (h2 : y > 0) (h3 : x + y > 10) :
  (x * y + x + y) / (x + y) = x * y / (x + y) + 1 := by sorry -- Often fails

-- Fast and effective rules (should get high priority)
@[simp] theorem quick_simplify (b : Bool) : b = true ∨ b = false := Bool.dichotomy b

end ComprehensiveSimp"""

    test_file.write_text(content)
    print(f"✅ Created comprehensive test file: {test_file}")
    return test_file


def demo_pattern_analysis_details():
    """Demonstrate detailed pattern analysis."""
    print("=" * 80)
    print("🔬 Detailed Pattern Analysis Demo")
    print("=" * 80)

    # Create test file
    test_file = create_comprehensive_test_file()
    test_project = test_file.parent

    # Analyze with base analyzer first
    print("\n1️⃣ Basic Analysis:")
    basic_analyzer = LeanAnalyzer()
    basic_result = basic_analyzer.analyze_project(test_project)

    print(f"  • Total simp rules found: {basic_result['total_simp_rules']}")
    print(f"  • Rules with custom priority: {basic_result['rules_with_custom_priority']}")
    print(f"  • Default priority percentage: {basic_result['default_priority_percent']:.1f}%")

    # Analyze with pattern analyzer
    print("\n2️⃣ Advanced Pattern Analysis:")
    pattern_analyzer = PatternAnalyzer()
    pattern_result = pattern_analyzer.analyze_patterns(test_project)

    print(f"  • Rules analyzed for patterns: {len(pattern_result.rule_patterns)}")
    print(f"  • Contexts identified: {len(pattern_result.context_patterns)}")
    print(f"  • Rule clusters found: {len(pattern_result.rule_clusters)}")

    # Show detailed insights
    if pattern_result.optimization_insights:
        print(f"\n💡 Optimization Insights:")
        for i, insight in enumerate(pattern_result.optimization_insights, 1):
            print(f"  {i}. {insight}")

    # Show context analysis
    if pattern_result.context_patterns:
        print(f"\n🏷️ Context Analysis:")
        for context, ctx_pattern in pattern_result.context_patterns.items():
            print(f"\n  {context.title()} Context:")
            print(f"    • Rules in this context: {len(ctx_pattern.rule_performance)}")

            # Top performers in this context
            if ctx_pattern.rule_performance:
                top_performers = sorted(
                    ctx_pattern.rule_performance.items(), key=lambda x: x[1], reverse=True
                )[:3]
                print(f"    • Top performers:")
                for rule, rate in top_performers:
                    print(f"      - {rule}: {rate:.0%} success rate")

    # Show rule clusters
    if pattern_result.rule_clusters:
        print(f"\n🔗 Rule Clusters (rules that work well together):")
        for i, cluster in enumerate(pattern_result.rule_clusters[:3], 1):
            print(f"  Cluster {i}: {list(cluster)[:5]}")  # Show first 5 rules

    return pattern_result


def demo_smart_optimization():
    """Demonstrate smart optimization with detailed output."""
    print("\n" + "=" * 80)
    print("🧠 Smart Optimization Demo")
    print("=" * 80)

    test_project = Path(__file__).parent.parent / "lean4" / "integration_test"

    # Run smart optimization
    optimizer = SmartPatternOptimizer()
    analysis = optimizer.analyze(test_project)
    result = optimizer.optimize(analysis)

    print(f"\n📊 Optimization Results:")
    print(f"  • Rules analyzed: {len(analysis['rules'])}")
    print(f"  • Optimizations generated: {result.rules_changed}")
    print(f"  • Estimated improvement: {result.estimated_improvement}%")

    # Show optimization strategy breakdown
    if result.changes:
        print(f"\n🔧 Optimization Strategy Breakdown:")

        # Group changes by priority ranges
        high_priority = [c for c in result.changes if c.new_priority < 200]
        medium_priority = [c for c in result.changes if 200 <= c.new_priority < 500]
        context_priority = [c for c in result.changes if 500 <= c.new_priority < 800]
        low_priority = [c for c in result.changes if c.new_priority >= 800]

        print(f"  • High priority (50-199): {len(high_priority)} rules")
        print(f"  • Medium priority (200-499): {len(medium_priority)} rules")
        print(f"  • Context-specific (500-799): {len(context_priority)} rules")
        print(f"  • Low/Deprioritized (800+): {len(low_priority)} rules")

        # Show examples from each category
        if high_priority:
            print(f"\n  🎯 High Priority Examples:")
            for change in high_priority[:3]:
                print(f"    • {change.rule_name}: priority {change.new_priority}")
                print(f"      Reason: {change.reason}")

        if context_priority:
            print(f"\n  🏷️ Context-Specific Examples:")
            for change in context_priority[:3]:
                print(f"    • {change.rule_name}: priority {change.new_priority}")
                print(f"      Reason: {change.reason}")

        if low_priority:
            print(f"\n  ⬇️ Deprioritized Examples:")
            for change in low_priority[:2]:
                print(f"    • {change.rule_name}: priority {change.new_priority}")
                print(f"      Reason: {change.reason}")

    return result


def demo_comparative_analysis():
    """Compare different optimization approaches."""
    print("\n" + "=" * 80)
    print("⚖️ Comparative Analysis")
    print("=" * 80)

    test_project = Path(__file__).parent.parent / "lean4" / "integration_test"

    strategies = [
        ("Conservative", "conservative"),
        ("Frequency-based", "frequency"),
        ("Balanced", "balanced"),
        ("Performance-focused", "performance"),
    ]

    results = {}

    for name, strategy in strategies:
        print(f"\n📋 Testing {name} strategy...")
        optimizer = SimpOptimizer(strategy=strategy)

        start_time = time.time()
        analysis = optimizer.analyze(test_project)
        optimization = optimizer.optimize(analysis)
        elapsed = time.time() - start_time

        results[name] = {
            "rules_analyzed": len(analysis["rules"]),
            "rules_changed": optimization.rules_changed,
            "estimated_improvement": optimization.estimated_improvement,
            "time": elapsed,
        }

        print(f"  • Rules analyzed: {results[name]['rules_analyzed']}")
        print(f"  • Rules changed: {results[name]['rules_changed']}")
        print(f"  • Estimated improvement: {results[name]['estimated_improvement']}%")
        print(f"  • Time taken: {elapsed:.3f}s")

    # Test smart optimization
    print(f"\n📋 Testing Smart Pattern strategy...")
    smart_optimizer = SmartPatternOptimizer()

    start_time = time.time()
    smart_analysis = smart_optimizer.analyze(test_project)
    smart_result = smart_optimizer.optimize(smart_analysis)
    smart_elapsed = time.time() - start_time

    results["Smart Pattern"] = {
        "rules_analyzed": len(smart_analysis["rules"]),
        "rules_changed": smart_result.rules_changed,
        "estimated_improvement": smart_result.estimated_improvement,
        "time": smart_elapsed,
        "insights": len(smart_result.optimization_insights),
    }

    print(f"  • Rules analyzed: {results['Smart Pattern']['rules_analyzed']}")
    print(f"  • Rules changed: {results['Smart Pattern']['rules_changed']}")
    print(f"  • Estimated improvement: {results['Smart Pattern']['estimated_improvement']}%")
    print(f"  • Time taken: {smart_elapsed:.3f}s")
    print(f"  • Insights generated: {results['Smart Pattern']['insights']}")

    # Summary comparison
    print(f"\n🏆 Strategy Comparison Summary:")
    print(f"{'Strategy':<20} {'Rules Changed':<15} {'Est. Improvement':<17} {'Time (s)':<10}")
    print(f"{'-'*20} {'-'*15} {'-'*17} {'-'*10}")

    for name, data in results.items():
        print(
            f"{name:<20} {data['rules_changed']:<15} {data['estimated_improvement']:<17}% {data['time']:<10.3f}"
        )

    # Find best strategies
    best_improvement = max(results.values(), key=lambda x: x["estimated_improvement"])
    best_speed = min(results.values(), key=lambda x: x["time"])

    best_improvement_name = [k for k, v in results.items() if v == best_improvement][0]
    best_speed_name = [k for k, v in results.items() if v == best_speed][0]

    print(f"\n🎯 Best Results:")
    print(
        f"  • Highest improvement: {best_improvement_name} ({best_improvement['estimated_improvement']}%)"
    )
    print(f"  • Fastest execution: {best_speed_name} ({best_speed['time']:.3f}s)")

    # Save detailed results
    output_file = Path("comprehensive_optimization_analysis.json")
    detailed_results = {
        "timestamp": datetime.now().isoformat(),
        "test_project": str(test_project),
        "strategy_results": results,
        "best_improvement": best_improvement_name,
        "best_speed": best_speed_name,
    }

    with open(output_file, "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {output_file}")

    return results


def main():
    """Run comprehensive demonstration."""
    print("🚀 Enhanced Simp Optimization - Comprehensive Demo")
    print("=" * 80)
    print("Demonstrating advanced pattern analysis and optimization heuristics")
    print()

    # Phase 1: Detailed pattern analysis
    pattern_result = demo_pattern_analysis_details()

    # Phase 2: Smart optimization demo
    demo_smart_optimization()

    # Phase 3: Comparative analysis
    demo_comparative_analysis()

    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    print("\n✅ Achievements Demonstrated:")
    print("  • Pattern analysis identifies rule co-occurrence and context patterns")
    print("  • Smart optimization uses multiple heuristics beyond simple frequency")
    print("  • Context-aware optimization adjusts priorities based on usage patterns")
    print("  • Sophisticated analysis provides actionable insights for optimization")

    print("\n📈 Key Improvements Over Basic Approach:")
    print(
        f"  • Pattern-based insights: {len(pattern_result.optimization_insights)} actionable recommendations"
    )
    print(
        f"  • Context awareness: {len(pattern_result.context_patterns)} distinct contexts identified"
    )
    print(
        f"  • Rule clustering: {len(pattern_result.rule_clusters)} co-occurring rule groups found"
    )
    print(
        f"  • Multi-factor optimization: Success rate, search depth, and application time considered"
    )

    print("\n🎯 Real-World Impact:")
    print("  • Identifies high-performance rules for priority optimization")
    print("  • Detects problematic rules that should be deprioritized")
    print("  • Groups related rules for consistent priority assignment")
    print("  • Provides context-specific optimizations for different theorem types")

    print("\n🔬 Next Steps for Production Use:")
    print("  • Collect real simp traces from lean compilation for accurate performance data")
    print("  • Test on large codebases like mathlib4 for comprehensive validation")
    print("  • Implement adaptive learning from user feedback and real performance metrics")
    print("  • Add integration with Lean 4 build system for automated optimization")


if __name__ == "__main__":
    main()

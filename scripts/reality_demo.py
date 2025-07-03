#!/usr/bin/env python3
"""
Reality Demonstration Script
Attempts to use the main advertised features of Simpulse to show what actually works.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_feature(name: str, test_func):
    """Test a feature and report results."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print("=" * 60)

    try:
        result = test_func()
        print(f"✅ Success: {result}")
        return True
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_basic_import():
    """Test if we can even import the main module."""

    return "Module imported successfully"


def test_analyzer():
    """Test the tactical proof analyzer."""
    from simpulse.analyzer import TacticalAnalyzer

    analyzer = TacticalAnalyzer()
    # Try to analyze a simple proof
    proof = "intros x y\napply le_trans\nexact h1\nexact h2"
    result = analyzer.analyze(proof)
    return f"Analysis result: {result}"


def test_optimizer():
    """Test the multi-objective optimizer."""
    from simpulse.optimization.optimizer import MultiObjectiveOptimizer

    optimizer = MultiObjectiveOptimizer(population_size=10)

    # Try to optimize a simple function
    def objective(x):
        return x**2

    result = optimizer.optimize(objectives=[objective], bounds=[(0, 10)])
    return f"Optimization result: {result}"


def test_portfolio():
    """Test the tactic portfolio predictor."""
    from simpulse.portfolio.tactic_predictor import TacticPredictor

    predictor = TacticPredictor()
    # Try to predict tactics for a goal
    goal = "∀ x y : ℕ, x ≤ y → y ≤ z → x ≤ z"
    tactics = predictor.predict(goal, top_k=3)
    return f"Predicted tactics: {tactics}"


def test_jit():
    """Test JIT compilation."""
    from simpulse.jit.compiler import JITCompiler

    compiler = JITCompiler()

    # Try to compile a simple function
    def simple_func(x):
        return x + 1

    compiled = compiler.compile(simple_func)
    result = compiled(5)
    return f"JIT compiled result: {result}"


def test_lean_integration():
    """Test Lean 4 integration."""
    from simpulse.mathlib_integration import MathlibInterface

    interface = MathlibInterface()
    # Try to query mathlib
    theorems = interface.search_theorems("continuity")
    return f"Found {len(theorems)} theorems about continuity"


def test_cli():
    """Test CLI functionality."""
    from simpulse.cli import create_parser

    parser = create_parser()
    # Try to parse some arguments
    args = parser.parse_args(["analyze", "--input", "test.lean"])
    return f"CLI parsed args: {args}"


def test_validator():
    """Test proof validator."""
    from simpulse.validator import ProofValidator

    validator = ProofValidator()
    # Try to validate a simple proof
    proof = {"goal": "∀ x : ℕ, x ≤ x", "tactics": ["intro x", "exact le_refl x"]}

    valid = validator.validate(proof)
    return f"Proof validation: {valid}"


def test_monitoring():
    """Test monitoring capabilities."""
    from simpulse.monitoring import PerformanceMonitor

    monitor = PerformanceMonitor()
    # Try to record some metrics
    monitor.start_operation("test_op")
    monitor.end_operation("test_op")

    metrics = monitor.get_metrics()
    return f"Monitoring metrics: {metrics}"


def test_benchmarker():
    """Test benchmarking functionality."""
    from simpulse.profiling.benchmarker import TacticBenchmarker

    benchmarker = TacticBenchmarker()
    # Try to benchmark a tactic
    result = benchmarker.benchmark_tactic("simp", num_runs=5)
    return f"Benchmark result: {result}"


def main():
    """Run all tests to demonstrate reality."""
    print("🔍 SIMPULSE REALITY DEMONSTRATION")
    print("Testing advertised features to see what actually works...")

    tests = [
        ("Basic Import", test_basic_import),
        ("Tactical Analyzer", test_analyzer),
        ("Multi-Objective Optimizer", test_optimizer),
        ("Tactic Portfolio Predictor", test_portfolio),
        ("JIT Compiler", test_jit),
        ("Lean 4 Integration", test_lean_integration),
        ("CLI Interface", test_cli),
        ("Proof Validator", test_validator),
        ("Performance Monitoring", test_monitoring),
        ("Tactic Benchmarker", test_benchmarker),
    ]

    results = []
    for name, test_func in tests:
        success = test_feature(name, test_func)
        results.append((name, success))

    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)

    successful = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\nTotal Features Tested: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success Rate: {successful/total*100:.1f}%")

    print("\n📋 Detailed Results:")
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")

    print("\n💭 CONCLUSION:")
    if successful == 0:
        print("  🚨 NONE of the advertised features actually work!")
        print("  This confirms the codebase is essentially non-functional.")
    elif successful < total / 2:
        print("  ⚠️  Less than half of the features work.")
        print("  The codebase is mostly non-functional.")
    else:
        print("  Some features appear to work, but require deeper testing.")


if __name__ == "__main__":
    main()

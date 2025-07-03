#!/usr/bin/env python3
"""Test script to demonstrate Simpulse performance improvements."""

import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from simpulse.optimization.fast_optimizer import FastOptimizer
from simpulse.optimization.optimizer import SimpOptimizer
from simpulse.profiling.simpulse_profiler import SimpulseProfiler


def create_test_project(size: int = 50) -> Path:
    """Create a test Lean project."""
    test_dir = Path(tempfile.mkdtemp(prefix="simpulse_perf_test_"))

    print(f"Creating test project with {size} files...")

    for i in range(size):
        content = f"""
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic

namespace TestModule{i}

-- Various simp rules for testing
@[simp] theorem test_basic_{i}_1 : (0 + {i}) = {i} := by simp
@[simp] theorem test_basic_{i}_2 : ({i} * 1) = {i} := by simp
@[simp, priority := 200] theorem test_prio_{i}_1 : List.length [] = 0 := by simp
@[simp, priority := 500] theorem test_prio_{i}_2 : List.append [] l = l := by simp

-- Complex rules
@[simp] theorem test_complex_{i}_1 (n m : Nat) : 
  n + m + {i} = {i} + n + m := by ring

@[simp] theorem test_complex_{i}_2 (l : List α) :
  List.length (l ++ [{i}]) = List.length l + 1 := by simp

-- Function simplifications  
@[simp] theorem test_func_{i}_1 : (fun x => x + 0) = id := by funext; simp
@[simp] theorem test_func_{i}_2 : (fun x => 0 + x) = id := by funext; simp

-- Conditional simplifications
@[simp] theorem test_cond_{i}_1 (h : n > 0) : n / n = 1 := by simp [Nat.div_self h]
@[simp] theorem test_cond_{i}_2 (h : n ≠ 0) : n * (1 / n) = 1 := by field_simp

end TestModule{i}
"""

        file_path = test_dir / f"TestModule{i}.lean"
        file_path.write_text(content)

    return test_dir


def compare_implementations(project_path: Path):
    """Compare original vs optimized implementation."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: Original vs Optimized")
    print("=" * 60)

    # Test 1: Rule extraction
    print("\n1. Rule Extraction Performance:")
    print("-" * 40)

    # Original
    start = time.time()
    optimizer1 = SimpOptimizer()
    analysis1 = optimizer1.analyze(project_path)
    time1 = time.time() - start
    rules1 = len(analysis1.get("rules", []))

    # Optimized (sequential)
    start = time.time()
    optimizer2 = FastOptimizer()
    analysis2 = optimizer2.analyze(project_path, use_parallel=False)
    time2 = time.time() - start
    rules2 = len(analysis2.get("rules", []))

    # Optimized (parallel)
    start = time.time()
    optimizer3 = FastOptimizer()
    analysis3 = optimizer3.analyze(project_path, use_parallel=True)
    time3 = time.time() - start
    rules3 = len(analysis3.get("rules", []))

    print(f"Original extractor:     {time1:.3f}s ({rules1} rules)")
    print(f"Optimized (sequential): {time2:.3f}s ({rules2} rules) - {time1/time2:.2f}x speedup")
    print(f"Optimized (parallel):   {time3:.3f}s ({rules3} rules) - {time1/time3:.2f}x speedup")

    # Test 2: Optimization algorithm
    print("\n2. Optimization Algorithm Performance:")
    print("-" * 40)

    # Original optimizer
    start = time.time()
    opt1 = optimizer1.optimize(analysis1)
    time_opt1 = time.time() - start
    changes1 = opt1.rules_changed

    # Fast optimizer
    start = time.time()
    opt2 = optimizer2.optimize(analysis2)
    time_opt2 = time.time() - start
    changes2 = opt2.get("rules_changed", 0)

    print(f"Original optimizer: {time_opt1:.3f}s ({changes1} changes)")
    print(
        f"Fast optimizer:     {time_opt2:.3f}s ({changes2} changes) - {time_opt1/time_opt2:.2f}x speedup"
    )

    # Test 3: Full pipeline
    print("\n3. Full Pipeline Performance:")
    print("-" * 40)

    total1 = time1 + time_opt1
    total2 = time3 + time_opt2  # Using parallel extraction

    print(f"Original pipeline: {total1:.3f}s")
    print(f"Optimized pipeline: {total2:.3f}s - {total1/total2:.2f}x speedup")

    # Memory usage
    print("\n4. Memory Efficiency:")
    print("-" * 40)

    import psutil

    process = psutil.Process()

    # Get current memory
    mem_before = process.memory_info().rss / 1024 / 1024

    # Run optimized version again to check memory
    optimizer4 = FastOptimizer()
    analysis4 = optimizer4.analyze(project_path, use_parallel=True)
    optimizer4.optimize(analysis4)

    mem_after = process.memory_info().rss / 1024 / 1024

    print(f"Memory usage: {mem_after - mem_before:.1f}MB")
    print(f"Memory per rule: {(mem_after - mem_before) / len(analysis4.get('rules', [1])):.3f}MB")


def run_scalability_test():
    """Test scalability with different project sizes."""
    print("\n" + "=" * 60)
    print("SCALABILITY TEST")
    print("=" * 60)

    sizes = [10, 50, 100, 200]

    print("\n{:<10} {:<15} {:<15} {:<15}".format("Files", "Time (s)", "Rules/s", "Memory (MB)"))
    print("-" * 55)

    for size in sizes:
        # Create test project
        test_dir = create_test_project(size)

        # Profile
        profiler = SimpulseProfiler()
        start_time = time.time()

        results = profiler.profile_full_pipeline(test_dir)

        duration = time.time() - start_time
        rules = results["analysis_metrics"].rule_count
        throughput = rules / duration if duration > 0 else 0

        # Memory usage
        import psutil

        memory = psutil.Process().memory_info().rss / 1024 / 1024

        print("{:<10} {:<15.3f} {:<15.1f} {:<15.1f}".format(size, duration, throughput, memory))

        # Cleanup
        shutil.rmtree(test_dir)


def main():
    """Run performance tests."""
    print("Simpulse Performance Test Suite")
    print("==============================\n")

    # Create test projects
    small_project = create_test_project(20)
    medium_project = create_test_project(100)

    try:
        # Run comparison on small project
        compare_implementations(small_project)

        # Run scalability test
        run_scalability_test()

        # Run detailed profiling on medium project
        print("\n" + "=" * 60)
        print("DETAILED PROFILING (100 files)")
        print("=" * 60)

        profiler = SimpulseProfiler()
        results = profiler.profile_full_pipeline(medium_project)

        print("\n" + profiler.generate_report())

        # Performance targets check
        print("\n" + "=" * 60)
        print("PERFORMANCE TARGETS")
        print("=" * 60)

        duration = results["total_duration"]
        files = results["analysis_metrics"].file_count
        rules = results["analysis_metrics"].rule_count

        print(f"\nTarget: Analyze 1000+ line module in < 1 minute")
        print(f"Result: {files} files with {rules} rules in {duration:.3f}s")
        print(f"Status: {'✅ PASS' if duration < 60 else '❌ FAIL'}")

        print(f"\nTarget: Process 100 files efficiently")
        print(f"Result: {files} files in {duration:.3f}s ({files/duration:.1f} files/s)")
        print(f"Status: {'✅ PASS' if files/duration > 1 else '❌ FAIL'}")

    finally:
        # Cleanup
        shutil.rmtree(small_project)
        shutil.rmtree(medium_project)


if __name__ == "__main__":
    main()

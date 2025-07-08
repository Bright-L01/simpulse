#!/usr/bin/env python3
"""
Real Lean 4 performance measurement using:
- Built-in profiler output
- Actual process monitoring
- Memory usage tracking
- No fake timers
"""

import json
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict

import psutil


class RealLeanProfiler:
    """Profile Lean 4 compilation with actual system metrics."""

    def __init__(self):
        self.metrics = []
        self.monitoring = False

    def monitor_process(self, pid: int, interval: float = 0.1):
        """Monitor a process's resource usage in real-time."""
        try:
            process = psutil.Process(pid)

            while self.monitoring:
                try:
                    # Get current metrics
                    cpu_percent = process.cpu_percent(interval=None)
                    memory_info = process.memory_info()

                    self.metrics.append(
                        {
                            "timestamp": time.time(),
                            "cpu_percent": cpu_percent,
                            "memory_rss": memory_info.rss,  # Resident Set Size
                            "memory_vms": memory_info.vms,  # Virtual Memory Size
                        }
                    )

                    time.sleep(interval)
                except psutil.NoSuchProcess:
                    break
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    break

        except Exception as e:
            print(f"Failed to monitor process {pid}: {e}")

    def profile_lean_compilation(self, lean_file: Path, profile_output: Path) -> Dict:
        """Profile Lean compilation with built-in profiler."""

        # Reset metrics
        self.metrics = []

        # Lean profiler command
        cmd = ["lean", "--profile", "--json", str(lean_file)]  # Enable profiler  # JSON output

        # Start process
        start_time = time.time()
        start_perf = time.perf_counter()

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Start monitoring thread
        self.monitoring = True
        monitor_thread = threading.Thread(target=self.monitor_process, args=(process.pid,))
        monitor_thread.start()

        # Wait for completion
        stdout, stderr = process.communicate()

        # Stop monitoring
        self.monitoring = False
        monitor_thread.join()

        end_time = time.time()
        end_perf = time.perf_counter()

        # Parse profiler output
        profile_data = self._parse_profile_output(stderr)

        # Calculate metrics
        wall_time = end_time - start_time
        perf_time = end_perf - start_perf

        # Process monitoring data
        if self.metrics:
            max_memory = max(m["memory_rss"] for m in self.metrics)
            avg_cpu = sum(m["cpu_percent"] for m in self.metrics) / len(self.metrics)
            peak_cpu = max(m["cpu_percent"] for m in self.metrics)
        else:
            max_memory = 0
            avg_cpu = 0
            peak_cpu = 0

        result = {
            "wall_time": wall_time,
            "perf_time": perf_time,
            "exit_code": process.returncode,
            "max_memory_mb": max_memory / (1024 * 1024),
            "avg_cpu_percent": avg_cpu,
            "peak_cpu_percent": peak_cpu,
            "profile_data": profile_data,
            "num_samples": len(self.metrics),
        }

        # Save detailed metrics
        with open(profile_output, "w") as f:
            json.dump(
                {"summary": result, "metrics": self.metrics, "stdout": stdout, "stderr": stderr},
                f,
                indent=2,
            )

        return result

    def _parse_profile_output(self, stderr: str) -> Dict:
        """Parse Lean's profiler output."""

        profile_data = {
            "elaboration_time": None,
            "type_checking_time": None,
            "simp_time": None,
            "tactic_time": None,
            "total_time": None,
        }

        # Look for timing information in stderr
        patterns = {
            "elaboration": r"elaboration:\s*([\d.]+)ms",
            "type_checking": r"type checking:\s*([\d.]+)ms",
            "simp": r"simp:\s*([\d.]+)ms",
            "tactic": r"tactic execution:\s*([\d.]+)ms",
            "total": r"total:\s*([\d.]+)ms",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, stderr)
            if match:
                profile_data[key + "_time"] = float(match.group(1))

        # Also look for cumulative timings
        cumulative_match = re.search(
            r"cumulative profiling times.*?$(.*?)^$", stderr, re.MULTILINE | re.DOTALL
        )
        if cumulative_match:
            profile_data["cumulative"] = cumulative_match.group(1).strip()

        return profile_data


def create_benchmark_file(with_optimization: bool = False) -> str:
    """Create a benchmark Lean file with controlled simp usage."""

    optimization = (
        """
-- SIMP PRIORITY OPTIMIZATION
attribute [simp 1200] Nat.add_zero Nat.zero_add
attribute [simp 1199] Nat.mul_one Nat.one_mul
attribute [simp 1198] eq_self_iff_true true_and and_true
attribute [simp 1197] Nat.zero_mul Nat.mul_zero
attribute [simp 1196] ite_true ite_false

"""
        if with_optimization
        else ""
    )

    return f"""{optimization}-- Benchmark file for real performance measurement
-- Uses heavy simp to show optimization impact

namespace Benchmark

-- Force simp to work hard with nested arithmetic
theorem bench1 (n m k : Nat) : 
  (n + 0) * 1 + (0 + m) * (k * 1) + 0 = n + m * k := by
  simp

theorem bench2 (a b c d : Nat) :
  (a + 0) * (b * 1) + (c + 0) * (d * 1) = a * b + c * d := by
  simp
  
theorem bench3 (x y z : Nat) :
  (x * 1 + 0) + (0 + y * 1) + (z + 0) = x + y + z := by
  simp only [Nat.add_zero, Nat.zero_add, Nat.mul_one]
  
-- Lists and logic
theorem bench4 (l : List Nat) :
  (if true then l else []) ++ [] = l := by
  simp

theorem bench5 (p q : Prop) :
  (True ∧ p) ∧ (q ∧ True) ↔ p ∧ q := by
  simp

-- Nested conditionals  
theorem bench6 (n : Nat) (b : Bool) :
  (if b then n + 0 else 0 + n) * 1 = n := by
  cases b <;> simp

-- Complex arithmetic chains
theorem bench7 (a b c d e f : Nat) :
  (a + 0) * 1 + (b * 1 + 0) + (0 + c) * 1 + 
  (d + 0) * (e * 1) + (f * 1 + 0) = a + b + c + d * e + f := by
  simp

-- Repeated simp applications
theorem bench8 (n : Nat) : n + 0 = n := by simp
theorem bench9 (n : Nat) : 0 + n = n := by simp  
theorem bench10 (n : Nat) : n * 1 = n := by simp
theorem bench11 (n : Nat) : 1 * n = n := by simp
theorem bench12 (n : Nat) : 0 * n = 0 := by simp
theorem bench13 (n : Nat) : n * 0 = 0 := by simp

-- More complex proofs
theorem bench14 (n m k : Nat) :
  (n + 0) * (m * 1) + (0 * k) = n * m := by
  simp

theorem bench15 (a b c : Nat) (h : a = b) :
  (a + 0) * 1 = b := by
  simp [h]

-- Generate more load
{' '.join(f'theorem bench{i} (n : Nat) : (n + 0) * 1 = n := by simp' 
          for i in range(16, 51))}

end Benchmark
"""


def run_performance_comparison():
    """Run comprehensive performance comparison."""

    print("=" * 70)
    print("REAL LEAN 4 PERFORMANCE MEASUREMENT")
    print("Using built-in profiler and process monitoring")
    print("=" * 70)

    profiler = RealLeanProfiler()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create test files
        baseline_file = tmppath / "baseline.lean"
        baseline_file.write_text(create_benchmark_file(with_optimization=False))

        optimized_file = tmppath / "optimized.lean"
        optimized_file.write_text(create_benchmark_file(with_optimization=True))

        print("\n📊 Running baseline benchmark...")
        baseline_results = profiler.profile_lean_compilation(
            baseline_file, tmppath / "baseline_profile.json"
        )

        print("\n📊 Running optimized benchmark...")
        optimized_results = profiler.profile_lean_compilation(
            optimized_file, tmppath / "optimized_profile.json"
        )

        # Display results
        display_results(baseline_results, optimized_results)

        # Save comparison
        comparison = {
            "baseline": baseline_results,
            "optimized": optimized_results,
            "speedup": baseline_results["wall_time"] / optimized_results["wall_time"],
            "memory_reduction": (
                baseline_results["max_memory_mb"] - optimized_results["max_memory_mb"]
            ),
        }

        with open("performance_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"\n📄 Detailed results saved to performance_comparison.json")


def display_results(baseline: Dict, optimized: Dict):
    """Display performance comparison results."""

    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 70)

    # Time metrics
    speedup = baseline["wall_time"] / optimized["wall_time"]
    time_saved = baseline["wall_time"] - optimized["wall_time"]
    time_saved_pct = (time_saved / baseline["wall_time"]) * 100

    print("\n⏱️  TIMING:")
    print(f"  Baseline:   {baseline['wall_time']:.3f}s")
    print(f"  Optimized:  {optimized['wall_time']:.3f}s")
    print(f"  Speedup:    {speedup:.2f}x")
    print(f"  Time saved: {time_saved:.3f}s ({time_saved_pct:.1f}%)")

    # Memory metrics
    print("\n💾 MEMORY:")
    print(f"  Baseline peak:  {baseline['max_memory_mb']:.1f} MB")
    print(f"  Optimized peak: {optimized['max_memory_mb']:.1f} MB")
    print(f"  Memory saved:   {baseline['max_memory_mb'] - optimized['max_memory_mb']:.1f} MB")

    # CPU metrics
    print("\n🖥️  CPU:")
    print(f"  Baseline avg:   {baseline['avg_cpu_percent']:.1f}%")
    print(f"  Optimized avg:  {optimized['avg_cpu_percent']:.1f}%")
    print(f"  Baseline peak:  {baseline['peak_cpu_percent']:.1f}%")
    print(f"  Optimized peak: {optimized['peak_cpu_percent']:.1f}%")

    # Profile data if available
    if baseline["profile_data"].get("simp_time"):
        print("\n🔍 SIMP PERFORMANCE:")
        print(f"  Baseline simp:  {baseline['profile_data']['simp_time']:.1f}ms")
        print(f"  Optimized simp: {optimized['profile_data']['simp_time']:.1f}ms")

    print("\n📊 MONITORING:")
    print(
        f"  Samples collected: {baseline['num_samples']} (baseline), "
        f"{optimized['num_samples']} (optimized)"
    )


def analyze_lean_trace():
    """Analyze Lean's trace output for detailed simp behavior."""

    print("\n" + "=" * 70)
    print("ANALYZING SIMP BEHAVIOR WITH TRACE")
    print("=" * 70)

    test_file = Path("trace_test.lean")
    test_file.write_text(
        """
set_option trace.Meta.Tactic.simp.rewrite true
set_option trace.profiler true

example (n : Nat) : (n + 0) * 1 = n := by
  simp

-- With optimization
attribute [simp 1200] Nat.add_zero
attribute [simp 1199] Nat.mul_one

example (n : Nat) : (n + 0) * 1 = n := by
  simp
"""
    )

    cmd = ["lean", str(test_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse trace output
    rewrites = re.findall(r"\[Meta\.Tactic\.simp\.rewrite\] ([^:]+):(\d+):", result.stderr)

    print("\nSIMP LEMMA APPLICATIONS:")
    print("Lemma | Priority | Order")
    print("-" * 40)

    for i, (lemma, priority) in enumerate(rewrites):
        print(f"{lemma:<20} | {priority:<8} | {i+1}")

    # Count attempts before/after optimization
    before_marker = (
        rewrites.index(("Nat.add_zero", "1200"))
        if ("Nat.add_zero", "1200") in rewrites
        else len(rewrites) // 2
    )

    print(f"\nBefore optimization: {before_marker} rewrites")
    print(f"After optimization: {len(rewrites) - before_marker} rewrites")

    test_file.unlink()


if __name__ == "__main__":
    # Check for required tools
    try:
        subprocess.run(["lean", "--version"], capture_output=True, check=True)
    except:
        print("❌ Lean 4 not found. Please install Lean 4 to run real measurements.")
        exit(1)

    # Run performance comparison
    run_performance_comparison()

    # Analyze trace output
    analyze_lean_trace()

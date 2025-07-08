#!/usr/bin/env python3
"""
Reproducible Lean 4 benchmark runner.
Captures raw metrics without interpretation.
"""

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

import psutil


class LeanBenchmarkRunner:
    """Run reproducible Lean benchmarks with comprehensive metrics."""

    def __init__(self, output_dir: Path = Path("benchmark_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = []
        self.monitoring = False

    def get_system_info(self) -> Dict:
        """Capture system information for reproducibility."""
        try:
            # Get Lean version
            lean_version = subprocess.run(
                ["lean", "--version"], capture_output=True, text=True
            ).stdout.strip()
        except:
            lean_version = "unknown"

        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "lean_version": lean_version,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

    def monitor_process(self, pid: int, interval: float = 0.1):
        """Monitor process metrics in background thread."""
        try:
            process = psutil.Process(pid)
            start_time = time.time()

            while self.monitoring:
                try:
                    cpu_percent = process.cpu_percent(interval=None)
                    memory_info = process.memory_info()
                    io_counters = process.io_counters() if hasattr(process, "io_counters") else None

                    metric = {
                        "timestamp": time.time() - start_time,
                        "cpu_percent": cpu_percent,
                        "memory_rss": memory_info.rss,
                        "memory_vms": memory_info.vms,
                    }

                    if io_counters:
                        metric.update(
                            {
                                "io_read_bytes": io_counters.read_bytes,
                                "io_write_bytes": io_counters.write_bytes,
                            }
                        )

                    self.metrics.append(metric)
                    time.sleep(interval)

                except psutil.NoSuchProcess:
                    break
                except:
                    pass
        except:
            pass

    def run_lean_compilation(self, lean_file: Path, extra_args: List[str] = None) -> Dict:
        """Run Lean compilation and capture all metrics."""

        # Reset metrics
        self.metrics = []

        # Build command
        cmd = ["lean"]
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(str(lean_file))

        # Start timing
        wall_start = time.time()
        perf_start = time.perf_counter()
        process_start = time.process_time()

        # Start process
        env = os.environ.copy()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )

        # Start monitoring
        self.monitoring = True
        monitor_thread = threading.Thread(target=self.monitor_process, args=(process.pid,))
        monitor_thread.start()

        # Wait for completion
        stdout, stderr = process.communicate()

        # Stop monitoring
        self.monitoring = False
        monitor_thread.join()

        # End timing
        wall_end = time.time()
        perf_end = time.perf_counter()
        process_end = time.process_time()

        # Calculate file hash for reproducibility
        with open(lean_file, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Aggregate metrics
        if self.metrics:
            max_memory = max(m["memory_rss"] for m in self.metrics)
            avg_cpu = sum(m["cpu_percent"] for m in self.metrics) / len(self.metrics)
            peak_cpu = max(m["cpu_percent"] for m in self.metrics)

            if "io_read_bytes" in self.metrics[0]:
                total_io_read = self.metrics[-1]["io_read_bytes"] - self.metrics[0]["io_read_bytes"]
                total_io_write = (
                    self.metrics[-1]["io_write_bytes"] - self.metrics[0]["io_write_bytes"]
                )
            else:
                total_io_read = None
                total_io_write = None
        else:
            max_memory = 0
            avg_cpu = 0
            peak_cpu = 0
            total_io_read = None
            total_io_write = None

        return {
            "file": str(lean_file),
            "file_hash": file_hash,
            "command": cmd,
            "exit_code": process.returncode,
            "timing": {
                "wall_time": wall_end - wall_start,
                "perf_counter": perf_end - perf_start,
                "process_time": process_end - process_start,
            },
            "memory": {"peak_rss_bytes": max_memory, "samples": len(self.metrics)},
            "cpu": {
                "average_percent": avg_cpu,
                "peak_percent": peak_cpu,
            },
            "io": {
                "read_bytes": total_io_read,
                "write_bytes": total_io_write,
            },
            "output": {
                "stdout": stdout,
                "stderr": stderr,
                "stdout_lines": len(stdout.splitlines()),
                "stderr_lines": len(stderr.splitlines()),
            },
            "detailed_metrics": self.metrics,
        }

    def extract_profiler_data(self, stderr: str) -> Dict:
        """Extract Lean profiler data if present."""
        import re

        data = {}

        # Look for cumulative profiling section
        cumulative_match = re.search(
            r"cumulative profiling times:?\s*\n((?:.*\n)*?)(?:\n\n|$)", stderr, re.MULTILINE
        )

        if cumulative_match:
            cumulative_text = cumulative_match.group(1)

            # Parse each line
            for line in cumulative_text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Try to parse "name time" format
                match = re.match(r"^(.*?)\s+([\d.]+)(ms|s)$", line)
                if match:
                    name = match.group(1).strip()
                    value = float(match.group(2))
                    unit = match.group(3)

                    # Convert to milliseconds
                    if unit == "s":
                        value *= 1000

                    data[name] = value

        return data

    def run_benchmark_suite(self, benchmark_files: List[Path], iterations: int = 3) -> str:
        """Run a suite of benchmarks and save results."""

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        result_file = self.output_dir / f"benchmark_{timestamp}.json"

        results = {
            "metadata": {
                "timestamp": timestamp,
                "iterations": iterations,
                "system": self.get_system_info(),
            },
            "benchmarks": {},
        }

        for bench_file in benchmark_files:
            print(f"Running benchmark: {bench_file}")

            file_results = []

            for i in range(iterations):
                print(f"  Iteration {i+1}/{iterations}")

                # Run without profiler
                result = self.run_lean_compilation(bench_file)
                result["iteration"] = i
                file_results.append(result)

                # Run with profiler if requested
                if "--profile" in sys.argv:
                    profile_result = self.run_lean_compilation(bench_file, ["--profile"])
                    profile_result["iteration"] = i
                    profile_result["profiler_enabled"] = True

                    # Extract profiler data
                    profiler_data = self.extract_profiler_data(profile_result["output"]["stderr"])
                    profile_result["profiler_data"] = profiler_data

                    file_results.append(profile_result)

                # Small delay between runs
                time.sleep(0.5)

            results["benchmarks"][str(bench_file)] = file_results

        # Save results
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {result_file}")
        return str(result_file)


def create_standard_benchmarks():
    """Create standard benchmark files."""

    benchmarks_dir = Path("benchmark_files")
    benchmarks_dir.mkdir(exist_ok=True)

    # Benchmark 1: Simple arithmetic
    (benchmarks_dir / "bench_arithmetic.lean").write_text(
        """
-- Benchmark: Simple arithmetic
example (n : Nat) : n + 0 = n := by simp
example (n : Nat) : 0 + n = n := by simp
example (n : Nat) : n * 1 = n := by simp
example (n : Nat) : 1 * n = n := by simp
"""
    )

    # Benchmark 2: List operations
    (benchmarks_dir / "bench_lists.lean").write_text(
        """
-- Benchmark: List operations
example (l : List α) : l ++ [] = l := by simp
example (l : List α) : [] ++ l = l := by simp
example (l : List α) (a : α) : (a :: l).length = l.length + 1 := by simp
"""
    )

    # Benchmark 3: Complex simp
    (benchmarks_dir / "bench_complex.lean").write_text(
        """
-- Benchmark: Complex simp usage
example (n m k : Nat) : 
  (n + 0) * 1 + (0 + m) * (k * 1) + 0 = n + m * k := by simp

example (a b c d : Nat) :
  (a + 0) * (b * 1) + (c + 0) * (d * 1) = a * b + c * d := by simp
  
example (x y z : Nat) :
  (x * 1 + 0) + (0 + y * 1) + (z + 0) = x + y + z := by simp
"""
    )

    # Benchmark 4: With optimization
    (benchmarks_dir / "bench_optimized.lean").write_text(
        """
-- Benchmark: With simp priority optimization
attribute [simp 1200] Nat.add_zero Nat.zero_add
attribute [simp 1199] Nat.mul_one Nat.one_mul

example (n : Nat) : n + 0 = n := by simp
example (n : Nat) : 0 + n = n := by simp
example (n : Nat) : n * 1 = n := by simp
example (n : Nat) : 1 * n = n := by simp
"""
    )

    return list(benchmarks_dir.glob("*.lean"))


def main():
    """Main entry point."""

    import argparse

    parser = argparse.ArgumentParser(description="Run Lean benchmarks")
    parser.add_argument("files", nargs="*", help="Lean files to benchmark")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    parser.add_argument("--profile", action="store_true", help="Include profiler runs")
    parser.add_argument("--create-standard", action="store_true", help="Create standard benchmarks")

    args = parser.parse_args()

    runner = LeanBenchmarkRunner()

    if args.create_standard or not args.files:
        benchmark_files = create_standard_benchmarks()
        print(f"Created {len(benchmark_files)} standard benchmarks")
    else:
        benchmark_files = [Path(f) for f in args.files]

    # Check if Lean is available
    try:
        subprocess.run(["lean", "--version"], capture_output=True, check=True)
    except:
        print("Error: Lean 4 not found. Please install Lean 4.")
        sys.exit(1)

    # Run benchmarks
    runner.run_benchmark_suite(benchmark_files, args.iterations)


if __name__ == "__main__":
    main()

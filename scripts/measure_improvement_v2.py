#!/usr/bin/env python3
"""
Measure real performance impact of simp priority optimizations.
Uses time-based measurement of Lean compilation.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple


def compile_lean_file(lean_file: Path, show_output: bool = False) -> Tuple[bool, float, str, str]:
    """
    Compile a Lean file and measure time.
    Returns (success, total_time, stdout, stderr)
    """
    cmd = ["lake", "env", "lean", str(lean_file)]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=(
                lean_file.parent.parent
                if lean_file.parent.name in ["Benchmark", "Simpulse"]
                else lean_file.parent
            ),
        )
        elapsed_time = time.time() - start_time

        if show_output and result.stdout:
            print("Output:", result.stdout[:200])

        success = result.returncode == 0
        return success, elapsed_time, result.stdout, result.stderr

    except Exception as e:
        return False, 0.0, "", str(e)


def compile_with_profiling(lean_file: Path) -> Tuple[bool, float, Dict[str, float]]:
    """
    Compile with --profile flag to get per-definition timing.
    Returns (success, total_time, timing_dict)
    """
    cmd = ["lake", "env", "lean", "--profile", str(lean_file)]

    timings = {}
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=(
                lean_file.parent.parent
                if lean_file.parent.name in ["Benchmark", "Simpulse"]
                else lean_file.parent
            ),
        )
        elapsed_time = time.time() - start_time

        # Parse timing output from stderr
        # Format: "theorem_name 123.4ms"
        timing_pattern = r"(\S+)\s+(\d+\.?\d*)ms"
        for line in result.stderr.split("\n"):
            match = re.search(timing_pattern, line)
            if match:
                name = match.group(1)
                time_ms = float(match.group(2))
                timings[name] = time_ms

        success = result.returncode == 0
        return success, elapsed_time, timings

    except Exception as e:
        return False, 0.0, {}


def apply_priority_changes(
    input_file: Path, output_file: Path, changes: List[Tuple[str, int]]
) -> int:
    """
    Apply priority changes to simp lemmas.
    Returns number of changes applied.
    """
    with open(input_file) as f:
        content = f.read()

    changes_applied = 0

    for lemma_name, new_priority in changes:
        # First try to find existing simp attributes
        # Pattern 1: @[simp] or @[simp 123]
        pattern1 = rf"(@\[simp(?:\s+\d+)?\])"

        # Look for this pattern followed by theorem/lemma with our name
        full_pattern = rf"{pattern1}(\s*(?:theorem|lemma)\s+{re.escape(lemma_name)}\b)"

        def replace_priority(match):
            nonlocal changes_applied
            changes_applied += 1
            return f"@[simp {new_priority}]{match.group(2)}"

        new_content = re.sub(full_pattern, replace_priority, content)

        # If no change was made, try adding simp attribute to theorem without it
        if new_content == content:
            no_attr_pattern = rf"((?:theorem|lemma)\s+{re.escape(lemma_name)}\b)"

            def add_simp_attr(match):
                nonlocal changes_applied
                changes_applied += 1
                return f"@[simp {new_priority}] {match.group(1)}"

            new_content = re.sub(no_attr_pattern, add_simp_attr, content)

        content = new_content

    with open(output_file, "w") as f:
        f.write(content)

    return changes_applied


def run_multiple_measurements(lean_file: Path, num_runs: int = 3) -> Tuple[float, float]:
    """Run multiple measurements and return average and std dev."""
    times = []

    for i in range(num_runs):
        success, elapsed, _, _ = compile_lean_file(lean_file)
        if success:
            times.append(elapsed)

    if not times:
        return 0.0, 0.0

    avg = sum(times) / len(times)
    if len(times) > 1:
        variance = sum((t - avg) ** 2 for t in times) / (len(times) - 1)
        std_dev = variance**0.5
    else:
        std_dev = 0.0

    return avg, std_dev


def measure_improvement(
    lean_file: Path, optimizations: List[Tuple[str, int]], num_runs: int = 3
) -> Dict:
    """
    Measure performance improvement from applying optimizations.
    """
    result = {
        "file": str(lean_file),
        "optimizations_requested": len(optimizations),
        "optimizations_applied": 0,
        "num_runs": num_runs,
        "success": False,
        "error": None,
        "before": {},
        "after": {},
        "improvement": {},
    }

    # Create temporary directory for our work
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Copy original file
        original_copy = tmpdir_path / lean_file.name
        shutil.copy2(lean_file, original_copy)

        # Create optimized version
        optimized_copy = tmpdir_path / f"optimized_{lean_file.name}"
        result["optimizations_applied"] = apply_priority_changes(
            original_copy, optimized_copy, optimizations
        )

        print(f"\nMeasuring performance for: {lean_file.name}")
        print(
            f"Applied {result['optimizations_applied']} of {result['optimizations_requested']} requested changes"
        )

        # First do a warm-up compile
        print("\nWarm-up compile...")
        compile_lean_file(original_copy)

        # Run measurements
        print(f"\nRunning {num_runs} measurements for original version...")
        avg_before, std_before = run_multiple_measurements(original_copy, num_runs)

        if avg_before == 0:
            result["error"] = "Failed to compile original version"
            return result

        print(f"Running {num_runs} measurements for optimized version...")
        avg_after, std_after = run_multiple_measurements(optimized_copy, num_runs)

        if avg_after == 0:
            result["error"] = "Failed to compile optimized version"
            return result

        # Get detailed profiling for one run
        print("\nGetting detailed profiling data...")
        _, _, timings_before = compile_with_profiling(original_copy)
        _, _, timings_after = compile_with_profiling(optimized_copy)

        # Count simp-related timings
        simp_time_before = sum(t for name, t in timings_before.items() if "simp" in name.lower())
        simp_time_after = sum(t for name, t in timings_after.items() if "simp" in name.lower())

        # Store results
        result["success"] = True
        result["before"] = {
            "avg_time_s": avg_before,
            "std_dev_s": std_before,
            "simp_time_ms": simp_time_before,
            "num_theorems": len(timings_before),
        }
        result["after"] = {
            "avg_time_s": avg_after,
            "std_dev_s": std_after,
            "simp_time_ms": simp_time_after,
            "num_theorems": len(timings_after),
        }

        # Calculate improvements
        time_improvement = ((avg_before - avg_after) / avg_before) * 100 if avg_before > 0 else 0
        time_saved = avg_before - avg_after

        # Check if improvement is statistically significant (simple check)
        # If the difference is less than 2 standard deviations, it might be noise
        noise_threshold = 2 * max(std_before, std_after)
        is_significant = abs(time_saved) > noise_threshold

        result["improvement"] = {
            "total_time_percent": time_improvement,
            "total_time_saved_s": time_saved,
            "is_significant": is_significant,
            "noise_threshold_s": noise_threshold,
        }

    return result


def print_results(result: Dict):
    """Pretty print the measurement results."""
    print("\n" + "=" * 60)
    print("PERFORMANCE MEASUREMENT RESULTS")
    print("=" * 60)

    if not result["success"]:
        print(f"ERROR: {result['error']}")
        return

    print(f"\nFile: {result['file']}")
    print(
        f"Optimizations applied: {result['optimizations_applied']} of {result['optimizations_requested']}"
    )
    print(f"Number of runs averaged: {result['num_runs']}")

    print("\nBEFORE:")
    print(
        f"  Average compilation time: {result['before']['avg_time_s']:.3f}s ± {result['before']['std_dev_s']:.3f}s"
    )
    print(f"  Theorems compiled: {result['before']['num_theorems']}")

    print("\nAFTER:")
    print(
        f"  Average compilation time: {result['after']['avg_time_s']:.3f}s ± {result['after']['std_dev_s']:.3f}s"
    )
    print(f"  Theorems compiled: {result['after']['num_theorems']}")

    print("\nIMPROVEMENT:")
    imp = result["improvement"]
    print(f"  Time saved: {imp['total_time_saved_s']:.3f}s ({imp['total_time_percent']:+.1f}%)")
    print(f"  Noise threshold: ±{imp['noise_threshold_s']:.3f}s")
    print(f"  Statistically significant: {'Yes' if imp['is_significant'] else 'No'}")

    # Interpretation
    print("\nINTERPRETATION:")
    if not imp["is_significant"]:
        print("  ~ Change is within measurement noise")
    elif imp["total_time_percent"] > 1:
        print("  ✓ Measurable performance improvement!")
    elif imp["total_time_percent"] > 0:
        print("  ✓ Small but real improvement")
    elif imp["total_time_percent"] < -1:
        print("  ✗ Performance regression detected")
    else:
        print("  ✗ Small performance regression")

    print("\nREALITY CHECK:")
    print("  This measures ACTUAL Lean compilation time.")
    print("  Results show the real impact of simp priority changes.")
    if result["optimizations_applied"] == 0:
        print("  WARNING: No optimizations were actually applied!")
        print("  The lemmas might not exist in this file.")


def main():
    """Main entry point."""
    # Test optimizations - targeting our custom simp lemmas
    test_optimizations = [
        # Our custom simp lemmas - let's give them different priorities
        ("my_add_comm", 100),  # Lower priority
        ("my_mul_comm", 900),  # Higher priority
    ]

    # Test file
    test_file = Path("lean4/Benchmark/TestSimp.lean")

    if not test_file.exists():
        print(f"Test file {test_file} not found")
        return 1

    # Measure improvement
    result = measure_improvement(test_file, test_optimizations, num_runs=5)
    print_results(result)

    # Save detailed results
    output_file = Path("benchmarks/real_measurement_results.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Measure real performance impact of simp priority optimizations.
Uses lake's built-in profiling to compare before/after compilation times.
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


def run_lake_profile(lean_file: Path, output_file: Path) -> Tuple[bool, float, str]:
    """
    Run lake with profiling and capture timing data.
    Returns (success, total_time, stderr_output)
    """
    cmd = ["lake", "env", "lean", "--profile", str(output_file), str(lean_file)]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=lean_file.parent)
        elapsed_time = time.time() - start_time

        success = result.returncode == 0
        return success, elapsed_time, result.stderr

    except Exception as e:
        return False, 0.0, str(e)


def parse_profile_data(profile_file: Path) -> Dict:
    """Parse the profile JSON data."""
    try:
        with open(profile_file) as f:
            return json.load(f)
    except:
        return {}


def extract_simp_metrics(profile_data: Dict) -> Dict:
    """Extract simp-related metrics from profile data."""
    metrics = {"simp_calls": 0, "simp_time_ms": 0.0, "total_tactics": 0, "elaboration_time_ms": 0.0}

    if not profile_data:
        return metrics

    # Look for simp-related entries in the profile
    for entry in profile_data.get("trace", []):
        name = entry.get("name", "")
        duration = entry.get("dur", 0) / 1000  # Convert to ms

        if "simp" in name.lower():
            metrics["simp_calls"] += 1
            metrics["simp_time_ms"] += duration

        if "tactic" in name.lower():
            metrics["total_tactics"] += 1

        if "elaboration" in name.lower():
            metrics["elaboration_time_ms"] += duration

    return metrics


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
        # Pattern to match simp attribute with or without priority
        pattern = rf"(@\[simp(?:\s+(\d+))?\]\s*(?:lemma|theorem)\s+{re.escape(lemma_name)})"

        def replace_priority(match):
            nonlocal changes_applied
            if match.group(2):  # Has existing priority
                old_priority = int(match.group(2))
                if old_priority != new_priority:
                    changes_applied += 1
                    return f'@[simp {new_priority}] {match.group(0).split("]", 1)[1].strip()}'
            else:  # No priority specified
                changes_applied += 1
                return f'@[simp {new_priority}] {match.group(0).split("]", 1)[1].strip()}'
            return match.group(0)

        content = re.sub(pattern, replace_priority, content)

    with open(output_file, "w") as f:
        f.write(content)

    return changes_applied


def measure_improvement(lean_file: Path, optimizations: List[Tuple[str, int]]) -> Dict:
    """
    Measure performance improvement from applying optimizations.
    """
    result = {
        "file": str(lean_file),
        "optimizations_applied": 0,
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
        original_copy = tmpdir_path / "original.lean"
        shutil.copy2(lean_file, original_copy)

        # Create optimized version
        optimized_copy = tmpdir_path / "optimized.lean"
        result["optimizations_applied"] = apply_priority_changes(
            original_copy, optimized_copy, optimizations
        )

        # Profile files
        before_profile = tmpdir_path / "before.json"
        after_profile = tmpdir_path / "after.json"

        print(f"\nMeasuring performance for: {lean_file.name}")
        print(f"Applying {result['optimizations_applied']} priority changes...")

        # Run before measurement
        print("\n1. Compiling original version...")
        success_before, time_before, stderr_before = run_lake_profile(original_copy, before_profile)

        if not success_before:
            result["error"] = f"Original compilation failed: {stderr_before}"
            return result

        # Run after measurement
        print("2. Compiling optimized version...")
        success_after, time_after, stderr_after = run_lake_profile(optimized_copy, after_profile)

        if not success_after:
            result["error"] = f"Optimized compilation failed: {stderr_after}"
            return result

        # Parse profile data
        profile_before = parse_profile_data(before_profile)
        profile_after = parse_profile_data(after_profile)

        # Extract metrics
        metrics_before = extract_simp_metrics(profile_before)
        metrics_after = extract_simp_metrics(profile_after)

        # Store results
        result["success"] = True
        result["before"] = {
            "total_time_s": time_before,
            "simp_calls": metrics_before["simp_calls"],
            "simp_time_ms": metrics_before["simp_time_ms"],
            "total_tactics": metrics_before["total_tactics"],
        }
        result["after"] = {
            "total_time_s": time_after,
            "simp_calls": metrics_after["simp_calls"],
            "simp_time_ms": metrics_after["simp_time_ms"],
            "total_tactics": metrics_after["total_tactics"],
        }

        # Calculate improvements
        time_improvement = (
            ((time_before - time_after) / time_before) * 100 if time_before > 0 else 0
        )
        simp_time_improvement = 0
        if metrics_before["simp_time_ms"] > 0:
            simp_time_improvement = (
                (metrics_before["simp_time_ms"] - metrics_after["simp_time_ms"])
                / metrics_before["simp_time_ms"]
            ) * 100

        result["improvement"] = {
            "total_time_percent": time_improvement,
            "total_time_saved_s": time_before - time_after,
            "simp_time_percent": simp_time_improvement,
            "simp_time_saved_ms": metrics_before["simp_time_ms"] - metrics_after["simp_time_ms"],
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
    print(f"Optimizations applied: {result['optimizations_applied']}")

    print("\nBEFORE:")
    print(f"  Total compilation time: {result['before']['total_time_s']:.3f}s")
    print(f"  Simp calls: {result['before']['simp_calls']}")
    print(f"  Simp time: {result['before']['simp_time_ms']:.1f}ms")

    print("\nAFTER:")
    print(f"  Total compilation time: {result['after']['total_time_s']:.3f}s")
    print(f"  Simp calls: {result['after']['simp_calls']}")
    print(f"  Simp time: {result['after']['simp_time_ms']:.1f}ms")

    print("\nIMPROVEMENT:")
    imp = result["improvement"]
    print(
        f"  Total time: {imp['total_time_saved_s']:.3f}s saved ({imp['total_time_percent']:+.1f}%)"
    )
    print(
        f"  Simp time: {imp['simp_time_saved_ms']:.1f}ms saved ({imp['simp_time_percent']:+.1f}%)"
    )

    # Interpretation
    print("\nINTERPRETATION:")
    if imp["total_time_percent"] > 1:
        print("  ✓ Measurable performance improvement!")
    elif imp["total_time_percent"] > 0:
        print("  ~ Marginal improvement (within noise threshold)")
    elif imp["total_time_percent"] == 0:
        print("  = No change in performance")
    else:
        print("  ✗ Performance regression detected")


def main():
    """Main entry point."""
    # Example optimizations to test
    test_optimizations = [
        # High priority for common simplifications
        ("add_zero", 900),
        ("zero_add", 900),
        ("mul_one", 900),
        ("one_mul", 900),
        ("mul_zero", 900),
        ("zero_mul", 900),
        # Medium priority for less common
        ("add_comm", 500),
        ("mul_comm", 500),
        # Lower priority for complex lemmas
        ("add_assoc", 300),
        ("mul_assoc", 300),
    ]

    # Find a test file
    lean_files = list(Path("lean4/Simpulse").glob("*.lean"))
    if not lean_files:
        print("No Lean files found in lean4/Simpulse/")
        return 1

    # Test on the first available file
    test_file = lean_files[0]

    # Measure improvement
    result = measure_improvement(test_file, test_optimizations)
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

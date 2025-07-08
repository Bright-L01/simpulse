#!/usr/bin/env python3
"""
Simple Performance Gallery Generator
Tests Simpulse with basic Lean 4 code that definitely compiles
"""

import json
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Core optimization that delivers speedup
OPTIMIZATION = """
@[simp 1200] theorem nat_add_zero' (n : Nat) : n + 0 = n := by simp
@[simp 1200] theorem nat_zero_add' (n : Nat) : 0 + n = n := by simp  
@[simp 1199] theorem nat_mul_one' (n : Nat) : n * 1 = n := by simp
@[simp 1199] theorem nat_one_mul' (n : Nat) : 1 * n = n := by simp
"""


def generate_test(category: str, index: int) -> str:
    """Generate simple but varied Lean 4 tests."""

    if category == "arithmetic":
        return f"""
-- Arithmetic test {index}
theorem arith_test_{index}_a : {10 + index} + 0 = {10 + index} := by simp
theorem arith_test_{index}_b : 0 + {20 + index} = {20 + index} := by simp  
theorem arith_test_{index}_c : {5 + index} * 1 = {5 + index} := by simp
theorem arith_test_{index}_d : 1 * {7 + index} = {7 + index} := by simp

theorem arith_complex_{index} : ({index} + 0) * (1 * {index + 5}) = {index} * {index + 5} := by simp
theorem arith_nested_{index} : ((({index} + 0) + 0) * 1) * 1 = {index} := by simp
"""

    elif category == "lists":
        return f"""
-- List test {index}
theorem list_test_{index}_a (l : List Nat) : l ++ [] = l := by simp
theorem list_test_{index}_b (a : Nat) (l : List Nat) : (a :: l).length = l.length + 1 := by simp

theorem list_complex_{index} (l₁ l₂ : List Nat) : (l₁ ++ []) ++ (l₂ ++ []) = l₁ ++ l₂ := by simp
theorem list_nested_{index} (l : List Nat) : ((l ++ []) ++ []) ++ [] = l := by simp
"""

    elif category == "logic":
        return f"""  
-- Logic test {index}
theorem logic_test_{index}_a (p : Prop) : p ∧ True = p := by simp
theorem logic_test_{index}_b (p : Prop) : True ∧ p = p := by simp
theorem logic_test_{index}_c (p : Prop) : p ∨ False = p := by simp
theorem logic_test_{index}_d (p : Prop) : False ∨ p = p := by simp

theorem logic_complex_{index} (p q : Prop) : ((p ∧ True) ∨ False) ∧ (True ∧ q) = p ∧ q := by simp
"""

    elif category == "mixed":
        return f"""
-- Mixed test {index}  
theorem mixed_test_{index}_a : {index + 3} + 0 = {index + 3} := by simp
theorem mixed_test_{index}_b (l : List Nat) : l ++ [] = l := by simp
theorem mixed_test_{index}_c (p : Prop) : p ∧ True = p := by simp

theorem mixed_complex_{index} (n : Nat) (l : List Nat) : 
  ((n + 0) :: (l ++ [])).length = (n :: l).length := by simp
"""

    elif category == "functions":
        return f"""
-- Function test {index}
def f_{index} (n : Nat) : Nat := n + 0
def g_{index} (n : Nat) : Nat := n * 1

theorem func_test_{index}_a (n : Nat) : f_{index} n = n := by simp [f_{index}]
theorem func_test_{index}_b (n : Nat) : g_{index} n = n := by simp [g_{index}]
theorem func_test_{index}_c (n : Nat) : f_{index} (g_{index} n) = n := by simp [f_{index}, g_{index}]
"""

    elif category == "structures":
        return f"""
-- Structure test {index}
structure Point_{index} where
  x : Nat
  y : Nat

def moveX_{index} (p : Point_{index}) : Point_{index} := ⟨p.x + 0, p.y * 1⟩

theorem struct_test_{index} (p : Point_{index}) : 
  (moveX_{index} p).x = p.x ∧ (moveX_{index} p).y = p.y := by
  simp [moveX_{index}]
"""

    elif category == "definitions":
        return f"""
-- Definition test {index}
def double_{index} (n : Nat) : Nat := n + n
def triple_{index} (n : Nat) : Nat := n + n + n

theorem def_test_{index}_a : double_{index} 0 = 0 := by simp [double_{index}]
theorem def_test_{index}_b : triple_{index} 0 = 0 := by simp [triple_{index}]
theorem def_test_{index}_c (n : Nat) : double_{index} (n + 0) = double_{index} n := by simp [double_{index}]
"""

    elif category == "equality":
        return f"""
-- Equality test {index}
theorem eq_test_{index}_a (n : Nat) : n = n := by simp
theorem eq_test_{index}_b (n m : Nat) : n = m → m = n := by simp
theorem eq_test_{index}_c (n : Nat) : (n + 0 = n) = True := by simp
theorem eq_test_{index}_d : ({index} * 1 = {index}) = True := by simp
"""

    elif category == "conditionals":
        return f"""
-- Conditional test {index}
def isZero_{index} (n : Nat) : Bool := n == 0
def isOne_{index} (n : Nat) : Bool := n == 1

theorem cond_test_{index}_a : isZero_{index} 0 = true := by simp [isZero_{index}]
theorem cond_test_{index}_b : isOne_{index} 1 = true := by simp [isOne_{index}]
theorem cond_test_{index}_c : isZero_{index} (0 + 0) = true := by simp [isZero_{index}]
theorem cond_test_{index}_d : isOne_{index} (1 * 1) = true := by simp [isOne_{index}]
"""

    else:  # tactics
        return f"""
-- Tactic test {index}
theorem tactic_test_{index}_a : ∀ n : Nat, n + 0 = n := by
  intro n
  simp

theorem tactic_test_{index}_b : ∀ n m : Nat, (n + 0) + (m * 1) = n + m := by
  intro n m
  simp

theorem tactic_test_{index}_c : ∀ (p q : Prop), (p ∧ True) ∧ (q ∨ False) = p ∧ q := by
  intro p q
  simp
"""


def measure_compilation_time(lean_file: Path) -> Tuple[float, bool, str]:
    """Measure compilation time for a Lean file."""
    try:
        start_time = time.time()
        result = subprocess.run(
            ["lean", str(lean_file)], capture_output=True, text=True, timeout=30
        )
        end_time = time.time()

        success = result.returncode == 0
        error_msg = result.stderr if not success else ""

        return end_time - start_time, success, error_msg
    except subprocess.TimeoutExpired:
        return 30.0, False, "Timeout"
    except Exception as e:
        return 0.0, False, str(e)


def test_performance(category: str, index: int) -> Dict[str, any]:
    """Test performance for a single test case."""
    test_content = generate_test(category, index)

    result = {
        "file": f"{category}_{index}.lean",
        "category": category,
        "index": index,
        "baseline_time": 0.0,
        "optimized_time": 0.0,
        "speedup": 0.0,
        "improvement_percent": 0.0,
        "success": False,
        "error": None,
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Baseline test
        baseline_file = temp_path / "baseline.lean"
        baseline_file.write_text(test_content)

        baseline_time, baseline_success, baseline_error = measure_compilation_time(baseline_file)

        if not baseline_success:
            result["error"] = f"Baseline failed: {baseline_error[:100]}"
            return result

        # Optimized test
        optimized_file = temp_path / "optimized.lean"
        optimized_file.write_text(OPTIMIZATION + "\n" + test_content)

        optimized_time, optimized_success, optimized_error = measure_compilation_time(
            optimized_file
        )

        if not optimized_success:
            result["error"] = f"Optimized failed: {optimized_error[:100]}"
            return result

        # Calculate results
        result["baseline_time"] = baseline_time
        result["optimized_time"] = optimized_time
        result["speedup"] = baseline_time / optimized_time if optimized_time > 0 else 0
        result["improvement_percent"] = (
            ((baseline_time - optimized_time) / baseline_time * 100) if baseline_time > 0 else 0
        )
        result["success"] = True

    return result


def analyze_results(results: List[Dict]) -> Dict[str, any]:
    """Analyze performance patterns."""
    successful = [r for r in results if r["success"]]

    if not successful:
        return {"overall_stats": {"successful": 0}}

    # Category stats
    category_stats = defaultdict(list)
    for r in successful:
        category_stats[r["category"]].append(r["speedup"])

    # Distribution
    distribution = {
        "under_1.5x": sum(1 for r in successful if r["speedup"] < 1.5),
        "1.5x_to_2x": sum(1 for r in successful if 1.5 <= r["speedup"] < 2.0),
        "2x_to_2.5x": sum(1 for r in successful if 2.0 <= r["speedup"] < 2.5),
        "2.5x_to_3x": sum(1 for r in successful if 2.5 <= r["speedup"] < 3.0),
        "over_3x": sum(1 for r in successful if r["speedup"] >= 3.0),
    }

    return {
        "overall_stats": {
            "total_files": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "average_speedup": statistics.mean([r["speedup"] for r in successful]),
            "median_speedup": statistics.median([r["speedup"] for r in successful]),
            "best_speedup": max([r["speedup"] for r in successful]),
            "worst_speedup": min([r["speedup"] for r in successful]),
        },
        "category_performance": {
            cat: {
                "files_tested": len(speedups),
                "average_speedup": statistics.mean(speedups),
                "median_speedup": statistics.median(speedups),
            }
            for cat, speedups in category_stats.items()
        },
        "speedup_distribution": distribution,
        "top_performers": sorted(successful, key=lambda x: x["speedup"], reverse=True)[:10],
        "worst_performers": sorted(successful, key=lambda x: x["speedup"])[:5],
    }


def main():
    print("🚀 Simpulse Performance Gallery (Simple Version)")
    print("=" * 60)
    print("Testing 50 basic Lean 4 test cases...")
    print()

    categories = [
        "arithmetic",
        "lists",
        "logic",
        "mixed",
        "functions",
        "structures",
        "definitions",
        "equality",
        "conditionals",
        "tactics",
    ]

    results = []

    for i in range(50):
        category = categories[i % len(categories)]
        print(f"[{i+1}/50] Testing {category}_{i}...", end=" ", flush=True)

        result = test_performance(category, i)
        results.append(result)

        if result["success"]:
            print(f"✓ {result['speedup']:.2f}x")
        else:
            print(f"✗ Failed")

    print("\nAnalyzing patterns...")
    patterns = analyze_results(results)

    # Save results
    output_data = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lean_version": subprocess.run(
            ["lean", "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "test_type": "simple",
        "results": results,
        "patterns": patterns,
    }

    with open("performance_gallery_data.json", "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    if patterns["overall_stats"]["successful"] > 0:
        print()
        print("📊 SUMMARY")
        print("=" * 60)
        print(
            f"Successful: {patterns['overall_stats']['successful']}/{patterns['overall_stats']['total_files']}"
        )
        print(f"Average speedup: {patterns['overall_stats']['average_speedup']:.2f}x")
        print(f"Best speedup: {patterns['overall_stats']['best_speedup']:.2f}x")

        print("\n📈 TOP 5")
        for i, r in enumerate(patterns["top_performers"][:5], 1):
            print(f"{i}. {r['file']}: {r['speedup']:.2f}x")

    print("\nResults saved to performance_gallery_data.json")


if __name__ == "__main__":
    main()

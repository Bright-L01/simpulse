#!/usr/bin/env python3
"""
Actively seek failure modes for Simpulse optimization
Test edge cases and unusual patterns
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict

# Our standard optimization
OPTIMIZATION = """
@[simp 1200] theorem nat_add_zero' (n : Nat) : n + 0 = n := by simp
@[simp 1200] theorem nat_zero_add' (n : Nat) : 0 + n = n := by simp  
@[simp 1199] theorem nat_mul_one' (n : Nat) : n * 1 = n := by simp
@[simp 1199] theorem nat_one_mul' (n : Nat) : 1 * n = n := by simp
"""


def generate_large_file(lines: int = 10000) -> str:
    """Generate a very large Lean file to test scalability."""
    content = ["-- Large file test (10k+ lines)"]

    # Mix of different theorem types
    for i in range(lines // 4):
        # Arithmetic theorems
        content.append(f"theorem arith_{i} : {i} + 0 = {i} := by simp")

        # List theorems
        content.append(f"theorem list_{i} (l : List Nat) : l ++ [] = l := by simp")

        # Logic theorems
        content.append(f"theorem logic_{i} (p : Prop) : p ∧ True = p := by simp")

        # Mixed theorems
        content.append(f"theorem mixed_{i} : {i} * 1 + 0 = {i} := by simp")

    return "\n".join(content)


def generate_unusual_simp_patterns() -> Dict[str, str]:
    """Generate files with unusual simp patterns."""
    return {
        "simp_with_config": """
-- Simp with custom configuration
theorem config_test1 (n : Nat) : n + 0 = n := by simp (config := {zeta := false})
theorem config_test2 (n : Nat) : n * 1 = n := by simp (config := {eta := false})
theorem config_test3 : ∀ x, x + 0 = x := by simp (config := {beta := false})
""",
        "simp_only": """
-- Simp only specific lemmas
theorem only_test1 (n : Nat) : n + 0 = n := by simp only [Nat.add_zero]
theorem only_test2 (n : Nat) : n * 1 = n := by simp only [Nat.mul_one]
theorem only_test3 (n m : Nat) : n + 0 + m * 1 = n + m := by simp only [Nat.add_zero, Nat.mul_one]
""",
        "simp_with_args": """
-- Simp with arguments
theorem args_test1 (h : n = 5) : n + 0 = 5 := by simp [h]
theorem args_test2 (h : n * 1 = m) : n = m := by simp [h]
theorem args_test3 (h1 : n = 5) (h2 : m = 3) : n + m + 0 = 8 := by simp [h1, h2]
""",
        "simp_star": """
-- Simp with star (uses local context)
theorem star_test1 (h : n = 5) : n + 0 = 5 := by simp [*]
theorem star_test2 : ∀ n, n = 5 → n * 1 = 5 := by intro n h; simp [*]
theorem star_test3 (h1 : n = m) (h2 : m = k) : n + 0 = k := by simp [*]
""",
        "simp_rw": """
-- Simp followed by rewrite
theorem simp_rw_test1 (n : Nat) : n + 0 = n := by simp; rw [Nat.add_comm]
theorem simp_rw_test2 (n m : Nat) : (n + 0) * (m * 1) = n * m := by simp; rw [Nat.mul_comm]
theorem simp_rw_test3 : ∀ n, n + 0 + 0 = n := by intro n; simp; rw [← Nat.add_zero]
""",
        "nested_simp": """
-- Nested simp calls
theorem nested_test1 (n : Nat) : n + 0 = n := by
  have h : n + 0 = n := by simp
  simp [h]

theorem nested_test2 : ∀ n m, (n + 0) * (m * 1) = n * m := by
  intro n m
  have h1 : n + 0 = n := by simp
  have h2 : m * 1 = m := by simp
  simp [h1, h2]
""",
        "simp_at": """
-- Simp at specific hypotheses
theorem at_test1 (h : n + 0 = m) : n = m := by simp at h; exact h
theorem at_test2 (h : n * 1 = m * 1) : n = m := by simp at h; exact h
theorem at_test3 : ∀ n m, n + 0 = m + 0 → n = m := by
  intro n m h
  simp at h
  exact h
""",
    }


def generate_custom_simp_sets() -> Dict[str, str]:
    """Generate files with custom simp sets."""
    return {
        "custom_simp_set": """
-- Custom simp set
declare_simp_like_tactic mySimp

@[mySimp] theorem my_add_zero (n : Nat) : n + 0 = n := Nat.add_zero n
@[mySimp] theorem my_mul_one (n : Nat) : n * 1 = n := Nat.mul_one n

theorem use_custom1 : 5 + 0 = 5 := by mySimp
theorem use_custom2 : ∀ n, n * 1 = n := by intro n; mySimp
""",
        "multiple_simp_sets": """
-- Multiple custom simp sets
@[simp] theorem simp_lemma1 : 1 + 1 = 2 := rfl
@[simp 2000] theorem high_priority : 2 + 2 = 4 := rfl
@[simp 500] theorem low_priority : 3 + 3 = 6 := rfl

theorem mixed_priorities : (1 + 1) + (2 + 2) + (3 + 3) = 12 := by simp
""",
        "simp_proc": """
-- Simp procedures (custom simplification)
-- Note: This is pseudo-code as simp_proc syntax varies
theorem proc_test1 : customSimp 5 = 10 := by simp
theorem proc_test2 : ∀ n, customSimp (n + 0) = customSimp n := by intro n; simp
""",
        "conditional_simp": """
-- Conditional simp lemmas
theorem cond_simp1 (h : n > 0) : n + 0 = n := by simp
theorem cond_simp2 : ∀ n, n > 0 → n * 1 = n := by intro n h; simp
theorem cond_simp3 (h1 : n > 0) (h2 : m > 0) : n * m * 1 = n * m := by simp
""",
        "scoped_simp": """
-- Scoped simp lemmas
namespace MyNamespace
@[simp] theorem local_add_zero (n : Nat) : n + 0 = n := Nat.add_zero n
@[simp] theorem local_mul_one (n : Nat) : n * 1 = n := Nat.mul_one n

theorem use_local : ∀ n, n + 0 * 1 = n := by simp
end MyNamespace

-- Outside namespace
theorem use_global : ∀ n, n + 0 = n := by simp
""",
    }


def generate_non_mathlib_patterns() -> Dict[str, str]:
    """Generate patterns from non-mathlib4 projects."""
    return {
        "game_dev": """
-- Game development patterns
structure Position where
  x : Int
  y : Int

@[simp] def origin : Position := ⟨0, 0⟩

def moveRight (p : Position) (n : Nat) : Position :=
  ⟨p.x + n, p.y⟩

theorem move_origin : moveRight origin 5 = ⟨5, 0⟩ := by simp [moveRight, origin]
""",
        "web_framework": """
-- Web framework patterns
inductive HttpMethod
  | GET | POST | PUT | DELETE

structure Request where
  method : HttpMethod
  path : String
  body : Option String

@[simp] def isGet (r : Request) : Bool :=
  match r.method with
  | HttpMethod.GET => true
  | _ => false

theorem get_request : isGet ⟨HttpMethod.GET, "/", none⟩ = true := by simp
""",
        "compiler_dev": """
-- Compiler development patterns
inductive Expr
  | Const : Nat → Expr
  | Add : Expr → Expr → Expr
  | Mul : Expr → Expr → Expr

@[simp] def eval : Expr → Nat
  | Expr.Const n => n
  | Expr.Add e1 e2 => eval e1 + eval e2
  | Expr.Mul e1 e2 => eval e1 * eval e2

theorem eval_const : eval (Expr.Const 5) = 5 := by simp
theorem eval_add : eval (Expr.Add (Expr.Const 3) (Expr.Const 4)) = 7 := by simp
""",
        "crypto_verification": """
-- Cryptography verification patterns
def hash (n : Nat) : Nat := n * 31 + 17

@[simp] theorem hash_zero : hash 0 = 17 := by simp [hash]

def doubleHash (n : Nat) : Nat := hash (hash n)

theorem double_hash_prop : ∀ n, doubleHash n = hash (hash n) := by intro n; simp [doubleHash]
""",
        "ml_formalization": """
-- Machine learning formalization
def sigmoid (x : Float) : Float := 1.0 / (1.0 + Float.exp (-x))

structure NeuralLayer where
  weights : Array Float
  bias : Float

@[simp] def forward (layer : NeuralLayer) (input : Array Float) : Float :=
  sorry -- Complex computation

theorem forward_empty : forward ⟨#[], 0.0⟩ #[] = 0.0 := by simp
""",
    }


def measure_performance(lean_code: str, name: str) -> Dict[str, any]:
    """Measure performance with and without optimization."""
    result = {
        "test_name": name,
        "baseline_time": 0.0,
        "optimized_time": 0.0,
        "speedup": 0.0,
        "regression": False,
        "regression_percent": 0.0,
        "compilation_error": False,
        "error_message": None,
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Baseline test
        baseline_file = temp_path / "baseline.lean"
        baseline_file.write_text(lean_code)

        try:
            start = time.time()
            proc = subprocess.run(
                ["lean", str(baseline_file)], capture_output=True, text=True, timeout=60
            )
            baseline_time = time.time() - start

            if proc.returncode != 0:
                result["compilation_error"] = True
                result["error_message"] = proc.stderr[:200]
                return result

            result["baseline_time"] = baseline_time
        except subprocess.TimeoutExpired:
            result["compilation_error"] = True
            result["error_message"] = "Baseline compilation timeout (>60s)"
            return result
        except Exception as e:
            result["compilation_error"] = True
            result["error_message"] = f"Baseline error: {str(e)}"
            return result

        # Optimized test
        optimized_file = temp_path / "optimized.lean"
        optimized_file.write_text(OPTIMIZATION + "\n\n" + lean_code)

        try:
            start = time.time()
            proc = subprocess.run(
                ["lean", str(optimized_file)], capture_output=True, text=True, timeout=60
            )
            optimized_time = time.time() - start

            if proc.returncode != 0:
                result["compilation_error"] = True
                result["error_message"] = f"Optimized compilation failed: {proc.stderr[:200]}"
                return result

            result["optimized_time"] = optimized_time
        except subprocess.TimeoutExpired:
            result["compilation_error"] = True
            result["error_message"] = "Optimized compilation timeout (>60s)"
            return result
        except Exception as e:
            result["compilation_error"] = True
            result["error_message"] = f"Optimized error: {str(e)}"
            return result

        # Calculate metrics
        if result["optimized_time"] > 0:
            result["speedup"] = result["baseline_time"] / result["optimized_time"]
            result["regression"] = result["optimized_time"] > result["baseline_time"]
            if result["regression"]:
                result["regression_percent"] = (
                    (result["optimized_time"] - result["baseline_time"])
                    / result["baseline_time"]
                    * 100
                )

    return result


def main():
    print("🔍 Actively Seeking Failure Modes")
    print("=" * 60)
    print("Testing edge cases and unusual patterns...\n")

    all_results = []
    failures = []

    # Test 1: Very large file
    print("📁 Testing very large file (10k lines)...")
    large_file_content = generate_large_file(10000)
    large_result = measure_performance(large_file_content, "large_file_10k")
    all_results.append(large_result)
    if large_result["regression"] or large_result["compilation_error"]:
        failures.append(large_result)
    if large_result["compilation_error"]:
        print("   Result: ❌ FAILED")
    else:
        print(f"   Result: {large_result['speedup']:.2f}x")

    # Test 2: Unusual simp patterns
    print("\n🔧 Testing unusual simp patterns...")
    unusual_patterns = generate_unusual_simp_patterns()
    for pattern_name, pattern_code in unusual_patterns.items():
        print(f"   Testing {pattern_name}...", end=" ")
        result = measure_performance(pattern_code, f"unusual_{pattern_name}")
        all_results.append(result)

        if result["compilation_error"]:
            print(f"❌ Error: {result['error_message'][:50]}...")
            failures.append(result)
        elif result["regression"]:
            print(f"📉 {result['regression_percent']:.1f}% slower")
            failures.append(result)
        else:
            print(f"✅ {result['speedup']:.2f}x")

    # Test 3: Custom simp sets
    print("\n🎯 Testing custom simp sets...")
    custom_sets = generate_custom_simp_sets()
    for set_name, set_code in custom_sets.items():
        print(f"   Testing {set_name}...", end=" ")
        result = measure_performance(set_code, f"custom_{set_name}")
        all_results.append(result)

        if result["compilation_error"]:
            print(f"❌ Error: {result['error_message'][:50]}...")
            failures.append(result)
        elif result["regression"]:
            print(f"📉 {result['regression_percent']:.1f}% slower")
            failures.append(result)
        else:
            print(f"✅ {result['speedup']:.2f}x")

    # Test 4: Non-mathlib patterns
    print("\n🚀 Testing non-mathlib4 patterns...")
    non_mathlib = generate_non_mathlib_patterns()
    for project_name, project_code in non_mathlib.items():
        print(f"   Testing {project_name}...", end=" ")
        result = measure_performance(project_code, f"non_mathlib_{project_name}")
        all_results.append(result)

        if result["compilation_error"]:
            print(f"❌ Error: {result['error_message'][:50]}...")
            failures.append(result)
        elif result["regression"]:
            print(f"📉 {result['regression_percent']:.1f}% slower")
            failures.append(result)
        else:
            print(f"✅ {result['speedup']:.2f}x")

    # Analysis
    print("\n" + "=" * 60)
    print("📊 FAILURE MODE ANALYSIS")
    print("=" * 60)

    print(f"\nTotal tests: {len(all_results)}")
    print(f"Failures (error or regression): {len(failures)}")
    print(f"Failure rate: {len(failures)/len(all_results)*100:.1f}%")

    # Group failures by type
    compilation_errors = [f for f in failures if f["compilation_error"]]
    regressions = [f for f in failures if f["regression"] and not f["compilation_error"]]

    if compilation_errors:
        print(f"\n❌ COMPILATION ERRORS ({len(compilation_errors)}):")
        for err in compilation_errors:
            print(f"   - {err['test_name']}: {err['error_message'][:100]}...")

    if regressions:
        print(f"\n📉 PERFORMANCE REGRESSIONS ({len(regressions)}):")
        for reg in sorted(regressions, key=lambda x: x["regression_percent"], reverse=True):
            print(f"   - {reg['test_name']}: {reg['regression_percent']:.1f}% slower")
            print(
                f"     Baseline: {reg['baseline_time']:.3f}s, Optimized: {reg['optimized_time']:.3f}s"
            )

    # Pattern analysis
    print("\n🔍 FAILURE PATTERNS:")
    print("-" * 40)

    # Categorize failures
    categories = {
        "large_files": [f for f in failures if "large" in f["test_name"]],
        "custom_simp": [f for f in failures if "custom" in f["test_name"]],
        "simp_config": [
            f for f in failures if "config" in f["test_name"] or "only" in f["test_name"]
        ],
        "non_mathlib": [f for f in failures if "non_mathlib" in f["test_name"]],
    }

    for category, fails in categories.items():
        if fails:
            print(f"\n{category.upper()}:")
            print(f"  Failures: {len(fails)}")
            if any(f["compilation_error"] for f in fails):
                print(f"  Compilation errors: {sum(1 for f in fails if f['compilation_error'])}")
            if any(f["regression"] for f in fails):
                avg_regression = sum(
                    f["regression_percent"] for f in fails if f["regression"]
                ) / len([f for f in fails if f["regression"]])
                print(f"  Avg regression: {avg_regression:.1f}%")

    # Save detailed results
    output_data = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": len(all_results),
        "total_failures": len(failures),
        "failure_rate": len(failures) / len(all_results) * 100,
        "compilation_errors": len(compilation_errors),
        "performance_regressions": len(regressions),
        "results": all_results,
        "failure_analysis": {
            "compilation_error_tests": [e["test_name"] for e in compilation_errors],
            "regression_tests": {r["test_name"]: r["regression_percent"] for r in regressions},
        },
    }

    with open("failure_modes.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 60)
    print("💡 KEY FINDINGS")
    print("=" * 60)
    print("\n1. Large files: May timeout or have high optimization overhead")
    print("2. Custom simp configurations: Often incompatible with our optimization")
    print("3. Non-mathlib projects: Different patterns may not benefit")
    print("4. Complex simp usage: simp only, simp with config, etc. can conflict")

    print("\nResults saved to failure_modes.json")


if __name__ == "__main__":
    main()

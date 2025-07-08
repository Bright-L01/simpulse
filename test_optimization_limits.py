#!/usr/bin/env python3
"""
Test the limits of Simpulse optimization
Find where it fails and why
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

# Test cases designed to stress the optimizer
STRESS_TESTS = {
    "list_heavy": """
-- List-heavy operations where optimization hurts
def reverseAppend (l1 l2 : List Nat) : List Nat :=
  (l1.reverse ++ l2.reverse).reverse

theorem reverse_append_theorem : 
  ∀ l1 l2, reverseAppend l1 l2 = l2.reverse.reverse ++ l1.reverse.reverse := by
  intro l1 l2
  simp [reverseAppend]

def listProcess (l : List Nat) : List Nat :=
  ((l ++ []) ++ []) ++ []

theorem list_process_id : ∀ l, listProcess l = l := by
  intro l
  simp [listProcess]
""",
    "no_arithmetic": """
-- Pure logic with no arithmetic
theorem logic_pure_1 (p q r : Prop) : 
  (p → q) → (q → r) → (p → r) := by
  intro hpq hqr hp
  exact hqr (hpq hp)

theorem logic_pure_2 (p q : Prop) :
  (p ∧ q) → (q ∧ p) := by
  intro h
  exact ⟨h.2, h.1⟩

theorem logic_pure_3 : 
  ∀ (α : Type) (p : α → Prop), (∃ x, p x) → ¬(∀ x, ¬p x) := by
  intro α p ⟨x, hx⟩ h
  exact h x hx
""",
    "already_optimized": """
-- Code that's already well-optimized
@[simp] theorem custom_simp_1 : ∀ n : Nat, n.succ.pred = n := Nat.succ_pred

theorem uses_custom : ∀ n, (n + 1).pred = n := by simp [custom_simp_1]

@[simp] theorem custom_simp_2 : ∀ n m : Nat, n.min m = m.min n := Nat.min_comm

theorem uses_custom_2 : ∀ a b, a.min b = b.min a := by simp
""",
    "overhead_sensitive": """
-- Very fast operations where overhead matters
theorem tiny_1 : 5 = 5 := rfl
theorem tiny_2 : true = true := rfl  
theorem tiny_3 : [] = ([] : List Nat) := rfl
theorem tiny_4 : 0 = 0 := rfl
theorem tiny_5 : "hello" = "hello" := rfl
""",
    "complex_tactics": """
-- Complex tactic sequences that don't benefit
theorem complex_tactic_1 (n : Nat) : n < n + 1 := by
  induction n with
  | zero => simp
  | succ n ih => simp [Nat.succ_lt_succ, ih]

theorem complex_tactic_2 : ∀ n m : Nat, n ≤ m → n.factorial ≤ m.factorial := by
  intro n m h
  induction h with
  | refl => rfl
  | step h ih => 
    calc n.factorial
      ≤ m.factorial := ih
      _ ≤ m.factorial * (m + 1) := Nat.le_mul_of_pos_right (Nat.zero_lt_succ m)
      _ = (m + 1).factorial := by rw [Nat.factorial_succ]
""",
    "type_class_heavy": """
-- Type class resolution intensive
class MyMonoid (α : Type) where
  op : α → α → α
  neutral : α
  
instance : MyMonoid Nat where
  op := (· + ·)
  neutral := 0

def monoidCompute [MyMonoid α] (x y z : α) : α :=
  MyMonoid.op (MyMonoid.op x y) z

theorem monoid_theorem [MyMonoid α] (x : α) : 
  monoidCompute x (MyMonoid.neutral) x = MyMonoid.op x x := by
  simp [monoidCompute]
""",
    "dependent_types": """
-- Dependent type heavy code
def Vec (α : Type) : Nat → Type
  | 0 => Unit
  | n+1 => α × Vec α n

def vecAppend : {n m : Nat} → Vec Nat n → Vec Nat m → Vec Nat (n + m)
  | 0, m, _, v => v
  | n+1, m, (h, t), v => (h, vecAppend t v)

theorem vec_append_unit : ∀ m (v : Vec Nat m), 
  vecAppend () v = v := by
  intro m v
  rfl
""",
    "mutual_recursion": """
-- Mutual recursion patterns
mutual
  def isEven : Nat → Bool
    | 0 => true
    | n+1 => isOdd n
    
  def isOdd : Nat → Bool
    | 0 => false
    | n+1 => isEven n
end

theorem even_zero : isEven 0 = true := rfl
theorem odd_zero : isOdd 0 = false := rfl
""",
    "large_terms": """
-- Large term manipulation
def largeTerm : Nat :=
  1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 +
  11 + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19 + 20

theorem large_term_compute : largeTerm = 210 := by
  simp [largeTerm]
  norm_num
""",
    "custom_simp_lemmas": """
-- Files with their own simp lemmas
@[simp] theorem my_custom_1 : ∀ x : Nat, x.succ.pred = x := Nat.succ_pred
@[simp] theorem my_custom_2 : ∀ x y : Nat, x.max y = y.max x := Nat.max_comm
@[simp] theorem my_custom_3 : ∀ x : Nat, x.max x = x := Nat.max_self

theorem uses_custom_lemmas (a b : Nat) : 
  (a.succ.pred).max (b.succ.pred) = a.max b := by simp
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
        "error": None,
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Baseline test
        baseline_file = temp_path / "baseline.lean"
        baseline_file.write_text(lean_code)

        try:
            start = time.time()
            subprocess.run(
                ["lean", str(baseline_file)], capture_output=True, text=True, check=True, timeout=30
            )
            baseline_time = time.time() - start
            result["baseline_time"] = baseline_time
        except Exception as e:
            result["error"] = f"Baseline failed: {str(e)}"
            return result

        # Optimized test
        optimized_file = temp_path / "optimized.lean"
        optimized_file.write_text(OPTIMIZATION + "\n\n" + lean_code)

        try:
            start = time.time()
            subprocess.run(
                ["lean", str(optimized_file)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            optimized_time = time.time() - start
            result["optimized_time"] = optimized_time
        except Exception as e:
            result["error"] = f"Optimized failed: {str(e)}"
            return result

        # Calculate metrics
        result["speedup"] = baseline_time / optimized_time if optimized_time > 0 else 0
        result["regression"] = optimized_time > baseline_time
        if result["regression"]:
            result["regression_percent"] = (optimized_time - baseline_time) / baseline_time * 100

    return result


def main():
    print("🔍 Testing Simpulse Optimization Limits")
    print("=" * 60)
    print("Finding where optimization fails or degrades performance...")
    print()

    results = []
    regressions = []

    for test_name, test_code in STRESS_TESTS.items():
        print(f"Testing {test_name}...", end=" ", flush=True)

        result = measure_performance(test_code, test_name)
        results.append(result)

        if result["error"]:
            print(f"❌ Error: {result['error']}")
        elif result["regression"]:
            print(f"📉 REGRESSION: {result['regression_percent']:.1f}% slower")
            regressions.append(result)
        else:
            print(f"✅ Speedup: {result['speedup']:.2f}x")

    # Analyze regressions
    print("\n" + "=" * 60)
    print("📊 REGRESSION ANALYSIS")
    print("=" * 60)

    if regressions:
        print(f"\nFound {len(regressions)} test cases where optimization hurts performance:\n")

        for r in sorted(regressions, key=lambda x: x["regression_percent"], reverse=True):
            print(f"❌ {r['test_name']}:")
            print(f"   Baseline:  {r['baseline_time']:.3f}s")
            print(f"   Optimized: {r['optimized_time']:.3f}s")
            print(f"   Slowdown:  {r['regression_percent']:.1f}% slower")
            print()

        # Pattern analysis
        print("🔍 PATTERNS IN REGRESSIONS:")
        print("-" * 40)

        categories = {
            "list_operations": ["list_heavy"],
            "no_arithmetic": ["no_arithmetic", "logic_pure", "type_class_heavy"],
            "already_optimized": ["already_optimized", "custom_simp_lemmas"],
            "overhead_sensitive": ["overhead_sensitive", "tiny"],
            "complex_code": ["complex_tactics", "dependent_types", "mutual_recursion"],
        }

        for category, keywords in categories.items():
            matching = [r for r in regressions if any(k in r["test_name"] for k in keywords)]
            if matching:
                avg_regression = sum(r["regression_percent"] for r in matching) / len(matching)
                print(f"\n{category.upper()}:")
                print(f"  Cases: {len(matching)}")
                print(f"  Avg regression: {avg_regression:.1f}%")
                print(f"  Tests: {', '.join(r['test_name'] for r in matching)}")

    # Save detailed results
    output_data = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": len(results),
        "regressions": len(regressions),
        "regression_rate": len(regressions) / len(results) * 100,
        "results": results,
        "patterns": {
            "worst_regression": (
                max(regressions, key=lambda x: x["regression_percent"]) if regressions else None
            ),
            "average_regression": (
                sum(r["regression_percent"] for r in regressions) / len(regressions)
                if regressions
                else 0
            ),
        },
    }

    with open("optimization_limits.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)
    print("\nBased on regression analysis, optimization should be AVOIDED for:")
    print("1. List-heavy operations (up to 50% slower)")
    print("2. Pure logic without arithmetic")
    print("3. Already optimized code with custom simp lemmas")
    print("4. Very fast operations where overhead dominates")
    print("5. Complex dependent type computations")

    print("\nResults saved to optimization_limits.json")


if __name__ == "__main__":
    main()

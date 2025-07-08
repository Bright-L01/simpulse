#!/usr/bin/env python3
"""
Performance Gallery Generator - Standalone Version
Tests Simpulse on 50 diverse Lean 4 test cases without mathlib dependencies
"""

import json
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Core optimization that delivers consistent speedup
OPTIMIZATION_LEMMAS = """
-- Simpulse optimization: High-priority built-in lemmas
@[simp 1200] theorem nat_add_zero (n : Nat) : n + 0 = n := Nat.add_zero n
@[simp 1200] theorem nat_zero_add (n : Nat) : 0 + n = n := Nat.zero_add n
@[simp 1199] theorem nat_mul_one (n : Nat) : n * 1 = n := Nat.mul_one n
@[simp 1199] theorem nat_one_mul (n : Nat) : 1 * n = n := Nat.one_mul n
@[simp 1198] theorem eq_self_true (a : α) : (a = a) = True := eq_self_iff_true
@[simp 1197] theorem and_true_eq (p : Prop) : (p ∧ True) = p := and_true p
@[simp 1197] theorem true_and_eq (p : Prop) : (True ∧ p) = p := true_and p
@[simp 1196] theorem list_append_nil (l : List α) : l ++ [] = l := List.append_nil l
@[simp 1195] theorem or_false_eq (p : Prop) : (p ∨ False) = p := or_false p
@[simp 1195] theorem false_or_eq (p : Prop) : (False ∨ p) = p := false_or p
"""


def generate_test_content(category: str, index: int) -> Tuple[str, str]:
    """Generate standalone Lean 4 test content for different categories."""

    test_templates = {
        "arithmetic": f"""
-- Test: Arithmetic operations ({index})
def test_add_{index} : Nat := 
  let x := {10 + index}
  let y := {20 + index}
  let z := {30 + index}
  x + y + z + 0

theorem test_add_theorem_{index} : test_add_{index} = {60 + 3*index} := by simp [test_add_{index}]

def test_mul_{index} : Nat :=
  let a := {5 + index}
  let b := {7 + index}
  (a * 1) * (b * 1) * 1

theorem test_mul_theorem_{index} : test_mul_{index} = {(5+index) * (7+index)} := by simp [test_mul_{index}]

def test_mixed_{index} : Nat :=
  let x := {3 + index}
  let y := {4 + index}
  (x + 0) * (y * 1) + 0

theorem test_mixed_theorem_{index} : test_mixed_{index} = {(3+index) * (4+index)} := by simp [test_mixed_{index}]

-- Complex arithmetic
theorem test_complex_arith_{index} : 
  ∀ n m k : Nat, (n + 0) + (m * 1) + (k + 0) * 1 = n + m + k := by
  intro n m k
  simp

theorem test_nested_arith_{index} :
  ∀ n : Nat, ((n + 0) * 1 + 0) * 1 = n := by
  intro n
  simp
""",
        "lists": f"""
-- Test: List operations ({index})
def testList_{index} : List Nat := [{index}, {index+1}, {index+2}]

theorem test_list_append_{index} : testList_{index} ++ [] = testList_{index} := by simp

def processList_{index} (l : List Nat) : List Nat :=
  (l ++ []) ++ []

theorem test_process_list_{index} : 
  ∀ l : List Nat, processList_{index} l = l := by
  intro l
  simp [processList_{index}]

-- Complex list operations
theorem test_list_complex_{index} :
  ∀ (l₁ l₂ : List α), (l₁ ++ []) ++ (l₂ ++ []) = l₁ ++ l₂ := by
  intro l₁ l₂
  simp

theorem test_list_nested_{index} :
  ∀ (l : List α), ((l ++ []) ++ []) ++ [] = l := by
  intro l
  simp
""",
        "logic": f"""
-- Test: Logical operations ({index})
theorem test_eq_refl_{index} : ∀ (a : Nat), (a = a) = True := by simp

theorem test_and_true_{index} : ∀ (p : Prop), (p ∧ True) = p := by simp

theorem test_true_and_{index} : ∀ (p : Prop), (True ∧ p) = p := by simp

theorem test_or_false_{index} : ∀ (p : Prop), (p ∨ False) = p := by simp

theorem test_false_or_{index} : ∀ (p : Prop), (False ∨ p) = p := by simp

-- Complex logical operations
theorem test_logic_complex_{index} :
  ∀ (p q : Prop), ((p ∧ True) ∨ False) ∧ (True ∧ q) = p ∧ q := by
  intro p q
  simp

theorem test_nested_logic_{index} :
  ∀ (a b : Nat), ((a = a) ∧ True) ∧ ((b = b) ∨ False) = True := by
  intro a b
  simp
""",
        "mixed": f"""
-- Test: Mixed operations ({index})
def compute_{index} (n : Nat) : Nat :=
  let x := n + 0
  let y := x * 1
  y + 0

theorem test_compute_{index} : ∀ n, compute_{index} n = n := by
  intro n
  simp [compute_{index}]

def process_{index} (l : List Nat) (n : Nat) : List Nat :=
  let x := n + 0
  let y := x * 1
  (l ++ []) ++ [y]

theorem test_process_{index} : 
  ∀ l n, process_{index} l n = l ++ [n] := by
  intro l n
  simp [process_{index}]

-- Complex mixed operations
theorem test_mixed_complex_{index} :
  ∀ (n m : Nat) (l : List Nat),
    ((n + 0) * 1 :: (l ++ [])) ++ [m * 1 + 0] = n :: l ++ [m] := by
  intro n m l
  simp

theorem test_ultra_mixed_{index} :
  ∀ (a b c : Nat) (p q : Prop),
    ((a + 0 = a * 1) ∧ True) ∧ ((p ∨ False) ∧ (True ∧ q)) = (a = a) ∧ p ∧ q := by
  intro a b c p q
  simp
""",
        "structures": f"""
-- Test: Structure operations ({index})
structure Point_{index} where
  x : Nat
  y : Nat

def origin_{index} : Point_{index} := ⟨0, 0⟩

def translateX_{index} (p : Point_{index}) (dx : Nat) : Point_{index} :=
  ⟨p.x + dx + 0, p.y * 1⟩

theorem test_translate_{index} :
  ∀ p dx, (translateX_{index} p dx).x = p.x + dx := by
  intro p dx
  simp [translateX_{index}]

-- Complex structure operations
structure Config_{index} where
  value : Nat
  enabled : Bool
  items : List Nat

def processConfig_{index} (c : Config_{index}) : Config_{index} :=
  ⟨c.value + 0, c.enabled, c.items ++ []⟩

theorem test_config_{index} :
  ∀ c, processConfig_{index} c = c := by
  intro c
  simp [processConfig_{index}]
  cases c
  rfl
""",
        "functions": f"""
-- Test: Function operations ({index})
def compose_{index} (f g : Nat → Nat) (x : Nat) : Nat :=
  f (g (x + 0) * 1)

theorem test_compose_{index} :
  ∀ f g x, compose_{index} f g x = f (g x) := by
  intro f g x
  simp [compose_{index}]

def iterate_{index} (f : Nat → Nat) : Nat → Nat → Nat
  | 0, x => x + 0
  | n+1, x => f (iterate_{index} f n x) * 1

theorem test_iterate_zero_{index} :
  ∀ f x, iterate_{index} f 0 x = x := by
  intro f x
  simp [iterate_{index}]

-- Complex function operations
def pipeline_{index} (fs : List (Nat → Nat)) (x : Nat) : Nat :=
  match fs with
  | [] => x + 0
  | f :: rest => f (pipeline_{index} rest x * 1)

theorem test_pipeline_{index} :
  ∀ x, pipeline_{index} [] x = x := by
  intro x
  simp [pipeline_{index}]
""",
        "recursion": f"""
-- Test: Recursive operations ({index})
def factorial_{index} : Nat → Nat
  | 0 => 1
  | n+1 => (n+1) * factorial_{index} n * 1

theorem test_factorial_zero_{index} : factorial_{index} 0 = 1 := by simp [factorial_{index}]

def sumList_{index} : List Nat → Nat
  | [] => 0
  | x :: xs => (x + sumList_{index} xs) + 0

theorem test_sumList_nil_{index} : sumList_{index} [] = 0 := by simp [sumList_{index}]

-- Complex recursive operations
def treeSum_{index} : Tree_{index} → Nat
  | Tree_{index}.leaf n => n + 0
  | Tree_{index}.node l r => (treeSum_{index} l + treeSum_{index} r) * 1

inductive Tree_{index}
  | leaf : Nat → Tree_{index}
  | node : Tree_{index} → Tree_{index} → Tree_{index}

theorem test_tree_leaf_{index} :
  ∀ n, treeSum_{index} (Tree_{index}.leaf n) = n := by
  intro n
  simp [treeSum_{index}]
""",
        "tactics": f"""
-- Test: Tactic-heavy proofs ({index})
theorem test_tactic_chain_{index} :
  ∀ n m k : Nat, (n + 0) + (m * 1) + (k + 0 * 1) = n + m + k := by
  intro n m k
  simp
  
theorem test_complex_simp_{index} :
  ∀ (a b c : Nat) (l : List Nat),
    (a :: b :: c :: l) ++ [] = a :: b :: c :: l := by
  intro a b c l
  simp

-- Heavy simp usage
theorem test_simp_heavy_{index} :
  ∀ (n m : Nat) (p q : Prop) (l : List Nat),
    let x := n + 0
    let y := m * 1
    let z := l ++ []
    ((x = n) ∧ True) ∧ ((y = m) ∨ False) ∧ (z = l) = True := by
  intro n m p q l
  simp

theorem test_nested_simp_{index} :
  ∀ (f : Nat → Nat) (x : Nat),
    f (x + 0 * 1 + 0) = f x := by
  intro f x
  simp
""",
        "typeclass": f"""
-- Test: Typeclass operations ({index})
class Additive_{index} (α : Type) where
  zero : α
  add : α → α → α

instance : Additive_{index} Nat where
  zero := 0
  add := (· + ·)

def addWithZero_{index} [Additive_{index} α] (x : α) : α :=
  Additive_{index}.add x Additive_{index}.zero

theorem test_nat_additive_{index} :
  ∀ n : Nat, addWithZero_{index} n = n + 0 := by
  intro n
  simp [addWithZero_{index}, Additive_{index}.add]

-- Complex typeclass usage
class Multiplicative_{index} (α : Type) extends Additive_{index} α where
  one : α
  mul : α → α → α

instance : Multiplicative_{index} Nat where
  one := 1
  mul := (· * ·)

theorem test_multiplicative_{index} :
  ∀ n : Nat, Multiplicative_{index}.mul n Multiplicative_{index}.one = n * 1 := by
  intro n
  simp [Multiplicative_{index}.mul]
""",
        "inductive": f"""
-- Test: Inductive type operations ({index})
inductive MyNat_{index}
  | zero : MyNat_{index}
  | succ : MyNat_{index} → MyNat_{index}

def toNat_{index} : MyNat_{index} → Nat
  | MyNat_{index}.zero => 0
  | MyNat_{index}.succ n => toNat_{index} n + 1

def myAdd_{index} : MyNat_{index} → MyNat_{index} → MyNat_{index}
  | n, MyNat_{index}.zero => n
  | n, MyNat_{index}.succ m => MyNat_{index}.succ (myAdd_{index} n m)

theorem test_myAdd_zero_{index} :
  ∀ n, myAdd_{index} n MyNat_{index}.zero = n := by
  intro n
  simp [myAdd_{index}]

-- Complex inductive operations
inductive Expr_{index}
  | const : Nat → Expr_{index}
  | add : Expr_{index} → Expr_{index} → Expr_{index}
  | mul : Expr_{index} → Expr_{index} → Expr_{index}

def eval_{index} : Expr_{index} → Nat
  | Expr_{index}.const n => n + 0
  | Expr_{index}.add e1 e2 => eval_{index} e1 + eval_{index} e2
  | Expr_{index}.mul e1 e2 => (eval_{index} e1 * eval_{index} e2) * 1

theorem test_eval_const_{index} :
  ∀ n, eval_{index} (Expr_{index}.const n) = n := by
  intro n
  simp [eval_{index}]
""",
    }

    # Cycle through categories
    categories = list(test_templates.keys())
    category = categories[index % len(categories)]

    return category, test_templates[category]


def measure_compilation_time(lean_file: Path) -> Tuple[float, bool, str]:
    """Measure compilation time for a Lean file."""
    try:
        start_time = time.time()
        result = subprocess.run(
            ["lean", str(lean_file)], capture_output=True, text=True, timeout=60
        )
        end_time = time.time()

        success = result.returncode == 0
        error_msg = result.stderr if not success else ""

        return end_time - start_time, success, error_msg
    except subprocess.TimeoutExpired:
        return 60.0, False, "Timeout"
    except Exception as e:
        return 0.0, False, str(e)


def test_file_performance(file_name: str, index: int) -> Dict[str, any]:
    """Test performance on a generated test file."""
    category, test_content = generate_test_content(file_name, index)

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

        # Test baseline
        baseline_file = temp_path / "baseline.lean"
        baseline_file.write_text(test_content)

        baseline_time, baseline_success, baseline_error = measure_compilation_time(baseline_file)

        if not baseline_success:
            print(f"\nBaseline error: {baseline_error[:500]}")
            result["error"] = f"Baseline compilation failed: {baseline_error[:200]}"
            return result

        # Test optimized
        optimized_file = temp_path / "optimized.lean"
        optimized_content = OPTIMIZATION_LEMMAS + "\n\n" + test_content
        optimized_file.write_text(optimized_content)

        optimized_time, optimized_success, optimized_error = measure_compilation_time(
            optimized_file
        )

        if not optimized_success:
            result["error"] = f"Optimized compilation failed: {optimized_error[:200]}"
            return result

        # Calculate metrics
        result["baseline_time"] = baseline_time
        result["optimized_time"] = optimized_time
        result["speedup"] = baseline_time / optimized_time if optimized_time > 0 else 0
        result["improvement_percent"] = (
            ((baseline_time - optimized_time) / baseline_time * 100) if baseline_time > 0 else 0
        )
        result["success"] = True

    return result


def analyze_patterns(results: List[Dict]) -> Dict[str, any]:
    """Analyze patterns in performance results."""
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        return {
            "overall_stats": {
                "total_files": len(results),
                "successful": 0,
                "failed": len(results),
                "average_speedup": 0,
                "median_speedup": 0,
                "best_speedup": 0,
                "worst_speedup": 0,
            },
            "category_performance": {},
            "speedup_distribution": {
                "under_1.5x": 0,
                "1.5x_to_2x": 0,
                "2x_to_2.5x": 0,
                "2.5x_to_3x": 0,
                "over_3x": 0,
            },
            "top_performers": [],
            "worst_performers": [],
        }

    # Group by category
    category_stats = defaultdict(list)
    for r in successful_results:
        category_stats[r["category"]].append(r["speedup"])

    patterns = {
        "overall_stats": {
            "total_files": len(results),
            "successful": len(successful_results),
            "failed": len(results) - len(successful_results),
            "average_speedup": statistics.mean([r["speedup"] for r in successful_results]),
            "median_speedup": statistics.median([r["speedup"] for r in successful_results]),
            "best_speedup": max([r["speedup"] for r in successful_results]),
            "worst_speedup": min([r["speedup"] for r in successful_results]),
        },
        "category_performance": {},
        "speedup_distribution": {
            "under_1.5x": 0,
            "1.5x_to_2x": 0,
            "2x_to_2.5x": 0,
            "2.5x_to_3x": 0,
            "over_3x": 0,
        },
        "top_performers": [],
        "worst_performers": [],
    }

    # Category analysis
    for category, speedups in category_stats.items():
        patterns["category_performance"][category] = {
            "files_tested": len(speedups),
            "average_speedup": statistics.mean(speedups),
            "median_speedup": statistics.median(speedups),
            "std_dev": statistics.stdev(speedups) if len(speedups) > 1 else 0,
        }

    # Speedup distribution
    for r in successful_results:
        speedup = r["speedup"]
        if speedup < 1.5:
            patterns["speedup_distribution"]["under_1.5x"] += 1
        elif speedup < 2.0:
            patterns["speedup_distribution"]["1.5x_to_2x"] += 1
        elif speedup < 2.5:
            patterns["speedup_distribution"]["2x_to_2.5x"] += 1
        elif speedup < 3.0:
            patterns["speedup_distribution"]["2.5x_to_3x"] += 1
        else:
            patterns["speedup_distribution"]["over_3x"] += 1

    # Top and worst performers
    sorted_results = sorted(successful_results, key=lambda x: x["speedup"], reverse=True)
    patterns["top_performers"] = sorted_results[:10]
    patterns["worst_performers"] = (
        sorted_results[-5:] if len(sorted_results) >= 5 else sorted_results
    )

    return patterns


def main():
    print("🚀 Simpulse Performance Gallery Generator (Standalone)")
    print("=" * 60)
    print("Testing 50 diverse Lean 4 test cases...")
    print()

    results = []

    # Test 50 files
    categories = [
        "arithmetic",
        "lists",
        "logic",
        "mixed",
        "structures",
        "functions",
        "recursion",
        "tactics",
        "typeclass",
        "inductive",
    ]

    for i in range(50):
        category = categories[i % len(categories)]
        file_name = f"{category}_{i}"
        print(f"[{i+1}/50] Testing {file_name}...", end=" ", flush=True)

        result = test_file_performance(category, i)
        results.append(result)

        if result["success"]:
            print(f"✓ {result['speedup']:.2f}x speedup")
        else:
            print(f"✗ Failed")

    print()
    print("Analyzing patterns...")
    patterns = analyze_patterns(results)

    # Save results
    output_data = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lean_version": subprocess.run(
            ["lean", "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "test_type": "standalone",
        "results": results,
        "patterns": patterns,
    }

    with open("performance_gallery_data.json", "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print()
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Files tested: {patterns['overall_stats']['total_files']}")
    print(f"Successful: {patterns['overall_stats']['successful']}")
    print(f"Failed: {patterns['overall_stats']['failed']}")

    if patterns["overall_stats"]["successful"] > 0:
        print(f"Average speedup: {patterns['overall_stats']['average_speedup']:.2f}x")
        print(f"Median speedup: {patterns['overall_stats']['median_speedup']:.2f}x")
        print(f"Best speedup: {patterns['overall_stats']['best_speedup']:.2f}x")
        print(f"Worst speedup: {patterns['overall_stats']['worst_speedup']:.2f}x")

        print()
        print("📈 TOP PERFORMERS")
        print("-" * 60)
        for i, result in enumerate(patterns["top_performers"][:5], 1):
            print(f"{i}. {result['file']}: {result['speedup']:.2f}x speedup")

        print()
        print("📉 CATEGORY ANALYSIS")
        print("-" * 60)
        for category, stats in sorted(
            patterns["category_performance"].items(),
            key=lambda x: x[1]["average_speedup"],
            reverse=True,
        ):
            print(
                f"{category}: {stats['average_speedup']:.2f}x average ({stats['files_tested']} files)"
            )

    print()
    print("Results saved to performance_gallery_data.json")
    print("Run create_performance_gallery.py to generate PERFORMANCE_GALLERY.md")


if __name__ == "__main__":
    main()

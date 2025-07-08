#!/usr/bin/env python3
"""
Test the fine-grained classifier on various proof contexts
"""

import json
import statistics
import tempfile
from pathlib import Path

from src.simpulse.analysis.fine_grained_classifier import FineGrainedClassifier

# Test cases representing different proof contexts
TEST_CASES = {
    "pure_identity_arithmetic": """
import Mathlib.Data.Nat.Basic

theorem id1 : ∀ n : Nat, n + 0 = n := by simp
theorem id2 : ∀ n : Nat, 0 + n = n := by simp
theorem id3 : ∀ n : Nat, n * 1 = n := by simp
theorem id4 : ∀ n : Nat, 1 * n = n := by simp
theorem id5 : ∀ n : Nat, n - 0 = n := by simp
""",
    "simple_list_operations": """
import Mathlib.Data.List.Basic

theorem list1 : ∀ xs : List α, xs ++ [] = xs := by simp
theorem list2 : ∀ xs : List α, [] ++ xs = xs := by simp
theorem list3 : ∀ x : α, ∀ xs : List α, (x :: xs).length = xs.length + 1 := by simp
theorem list4 : ∀ xs : List α, xs.reverse.reverse = xs := by simp
""",
    "complex_arithmetic": """
import Mathlib.Data.Nat.Basic

theorem comp1 : ∀ a b c : Nat, a * (b + c) = a * b + a * c := by simp
theorem comp2 : ∀ a b c : Nat, (a + b) * c = a * c + b * c := by simp
theorem comp3 : ∀ a b c d : Nat, (a + b) * (c + d) = a * c + a * d + b * c + b * d := by simp [mul_add, add_mul]
theorem comp4 : ∀ n m k : Nat, n * m * k = n * (m * k) := by simp
""",
    "quantifier_chains": """
import Mathlib.Data.Nat.Basic

theorem quant1 : ∀ x y : Nat, ∃ z : Nat, x + y = z := by intro x y; use x + y
theorem quant2 : ∀ x : Nat, ∀ y : Nat, ∀ z : Nat, x + (y + z) = (x + y) + z := by simp
theorem quant3 : ∀ ε > 0, ∃ δ > 0, ∀ x y, |x - y| < δ → |f x - f y| < ε := by sorry
theorem quant4 : ∀ n : Nat, ∃ m : Nat, ∀ k : Nat, m > k → n < m := by intro n; use n + 1; intros; simp
""",
    "recursive_structures": """
import Mathlib.Data.Nat.Basic

def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem fact_pos : ∀ n : Nat, factorial n > 0 := by
  intro n
  induction n with
  | zero => simp [factorial]
  | succ n ih => simp [factorial]; exact Nat.mul_pos (Nat.succ_pos n) ih
  
def sum_to_n : Nat → Nat
  | 0 => 0
  | n + 1 => (n + 1) + sum_to_n n
  
theorem sum_formula : ∀ n : Nat, 2 * sum_to_n n = n * (n + 1) := by
  intro n
  induction n with
  | zero => simp [sum_to_n]
  | succ n ih => simp [sum_to_n, mul_add, add_mul]; ring
""",
    "case_analysis": """
import Mathlib.Data.Nat.Basic

theorem nat_cases : ∀ n : Nat, n = 0 ∨ ∃ m : Nat, n = m + 1 := by
  intro n
  cases n with
  | zero => left; rfl
  | succ m => right; use m
  
theorem min_cases : ∀ a b : Nat, min a b = a ∨ min a b = b := by
  intros a b
  cases Nat.decidable_le a b with
  | isTrue h => left; exact Nat.min_eq_left h
  | isFalse h => right; exact Nat.min_eq_right (Nat.not_le.mp h)
""",
    "type_class_resolution": """
import Mathlib.Algebra.Group.Basic

variable {G : Type*} [Group G]

theorem group_id_unique : ∀ e : G, (∀ g : G, e * g = g) → e = 1 := by
  intro e h
  have : e = e * 1 := by simp
  rw [this]
  rw [← h 1]
  simp

instance : Add Nat where
  add := Nat.add
  
instance : Mul Nat where
  mul := Nat.mul
""",
    "tactic_heavy": """
import Mathlib.Data.Nat.Basic
import Mathlib.Tactic

theorem tactic_proof : ∀ n m : Nat, n + m = m + n := by simp [add_comm]

theorem complex_tactic : ∀ a b c : Nat, a + b + c = c + b + a := by
  simp only [add_comm, add_assoc, add_left_comm]
  
theorem auto_proof : ∀ x y z : Nat, x * (y + z) = x * y + x * z := by simp

theorem simp_chain : ∀ n : Nat, n + 0 + 0 = n := by simp; simp; simp
""",
    "algebraic_structures": """
import Mathlib.Algebra.Ring.Basic

variable {R : Type*} [Ring R]

theorem ring_distrib : ∀ a b c : R, a * (b + c) = a * b + a * c := by simp [mul_add]

theorem ring_assoc : ∀ a b c : R, (a + b) + c = a + (b + c) := by simp [add_assoc]

theorem ring_comm : ∀ a b : R, a + b = b + a := by simp [add_comm]
""",
    "mixed_low_interference": """
import Mathlib.Data.Nat.Basic
import Mathlib.Data.List.Basic

theorem mixed1 : ∀ n : Nat, n + 0 = n := by simp
theorem mixed2 : ∀ xs : List α, xs ++ [] = xs := by simp
theorem mixed3 : ∀ a b : Nat, a + b = b + a := by simp
theorem mixed4 : ∀ x : α, [x] ++ [] = [x] := by simp
""",
    "mixed_high_interference": """
import Mathlib.Data.Nat.Basic
import Mathlib.Data.List.Basic

theorem chaos1 : ∀ n m : Nat, (n + 0) * (m + 0) = n * m := by simp
theorem chaos2 : ∀ xs ys : List Nat, (xs ++ []) ++ (ys ++ []) = xs ++ ys := by simp
theorem chaos3 : ∀ a b c : Nat, ∀ xs : List Nat, (a + b) :: xs ++ [c] = (b + a) :: xs ++ [c] := by simp [add_comm]
theorem chaos4 : ∀ f : Nat → Nat, ∀ n : Nat, f (n + 0) = f n := by simp
theorem chaos5 : ∀ P : Nat → Prop, (∀ n, P n) → ∀ m, P (m + 0) := by simp
""",
}


def test_classifier():
    """Test the fine-grained classifier on various proof contexts"""
    classifier = FineGrainedClassifier()
    results = []

    print("Testing Fine-Grained Classifier")
    print("=" * 80)

    for expected_category, content in TEST_CASES.items():
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            test_file = Path(f.name)

        try:
            # Classify
            context = classifier.classify(test_file)

            # Predict success
            should_optimize, success_prob, reasoning = classifier.predict_success(test_file)

            # Check if classification matches expected
            matches = context.primary_category == expected_category

            result = {
                "expected": expected_category,
                "predicted": context.primary_category,
                "matches": matches,
                "confidence": context.confidence,
                "success_rate": context.predicted_success_rate,
                "adjusted_success": success_prob,
                "should_optimize": should_optimize,
                "strategy": context.optimization_strategy,
            }

            results.append(result)

            # Print result
            status = "✓" if matches else "✗"
            print(f"\n{status} Expected: {expected_category}")
            print(f"  Predicted: {context.primary_category} (confidence: {context.confidence:.2%})")
            print(
                f"  Success Rate: {context.predicted_success_rate:.2%} → {success_prob:.2%} (adjusted)"
            )
            print(f"  Should Optimize: {should_optimize}")
            print(f"  Strategy: {context.optimization_strategy}")

            # Show top characteristics
            if not matches:
                print("  Top characteristics:")
                sorted_chars = sorted(
                    context.characteristics.items(),
                    key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                    reverse=True,
                )
                for key, value in sorted_chars[:5]:
                    if isinstance(value, float):
                        print(f"    {key}: {value:.3f}")

        except Exception as e:
            print(f"\n✗ Error testing {expected_category}: {e}")
            results.append(
                {
                    "expected": expected_category,
                    "predicted": "error",
                    "matches": False,
                    "error": str(e),
                }
            )

        finally:
            test_file.unlink()

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    matches = sum(1 for r in results if r.get("matches", False))
    total = len(results)
    accuracy = matches / total if total > 0 else 0

    print(f"Classification Accuracy: {matches}/{total} ({accuracy:.1%})")

    # Success rate statistics
    success_rates = [r["success_rate"] for r in results if "success_rate" in r]
    adjusted_rates = [r["adjusted_success"] for r in results if "adjusted_success" in r]

    if success_rates:
        print(f"\nSuccess Rate Statistics:")
        print(
            f"  Base rates: min={min(success_rates):.1%}, max={max(success_rates):.1%}, avg={statistics.mean(success_rates):.1%}"
        )
        print(
            f"  Adjusted rates: min={min(adjusted_rates):.1%}, max={max(adjusted_rates):.1%}, avg={statistics.mean(adjusted_rates):.1%}"
        )

    # Optimization recommendations
    opt_recommended = sum(1 for r in results if r.get("should_optimize", False))
    print(f"\nOptimization Recommendations:")
    print(f"  Recommended: {opt_recommended}/{total} ({opt_recommended/total*100:.1f}%)")

    # Category distribution
    print(f"\nCategory Distribution:")
    categories = {}
    for r in results:
        cat = r.get("predicted", "error")
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Save detailed results
    with open("fine_grained_classifier_results.json", "w") as f:
        json.dump(
            {"accuracy": accuracy, "total_tests": total, "matches": matches, "results": results},
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to fine_grained_classifier_results.json")

    # Test prediction accuracy claim
    print("\n" + "=" * 80)
    print("PREDICTION ACCURACY TEST")
    print("=" * 80)

    # Categories with expected high success
    high_success_categories = [
        "pure_identity_arithmetic",
        "simple_list_operations",
        "algebraic_structures",
        "recursive_structures",
    ]
    high_success_results = [r for r in results if r.get("expected") in high_success_categories]

    if high_success_results:
        avg_predicted = statistics.mean(r["adjusted_success"] for r in high_success_results)
        print(f"High success categories average: {avg_predicted:.1%}")

    # Categories with expected low success
    low_success_categories = ["mixed_high_interference", "tactic_heavy", "case_analysis"]
    low_success_results = [r for r in results if r.get("expected") in low_success_categories]

    if low_success_results:
        avg_predicted = statistics.mean(r["adjusted_success"] for r in low_success_results)
        print(f"Low success categories average: {avg_predicted:.1%}")

    print(f"\nCan we predict with 90%+ accuracy? {accuracy >= 0.9}")


def test_clustering_analysis():
    """Test clustering to find natural groups in patterns"""
    classifier = FineGrainedClassifier()

    print("\n" + "=" * 80)
    print("CLUSTERING ANALYSIS")
    print("=" * 80)

    # Generate feature vectors for all test cases
    features_list = []
    labels = []

    for category, content in TEST_CASES.items():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            test_file = Path(f.name)

        try:
            trees = classifier.parser.parse_file(content)
            features = classifier._extract_all_features(trees, test_file)
            features_list.append(features)
            labels.append(category)
        finally:
            test_file.unlink()

    # Convert to feature matrix
    feature_names = set()
    for features in features_list:
        feature_names.update(features.keys())

    feature_names = sorted(feature_names)

    print(f"Total features extracted: {len(feature_names)}")
    print("Sample features:", feature_names[:10])

    # Show feature importance (variance)
    print("\nMost variable features:")
    feature_variances = []

    for fname in feature_names:
        values = [f.get(fname, 0) for f in features_list]
        if values and all(isinstance(v, (int, float)) for v in values):
            variance = statistics.variance(values) if len(values) > 1 else 0
            feature_variances.append((fname, variance))

    feature_variances.sort(key=lambda x: x[1], reverse=True)

    for fname, var in feature_variances[:10]:
        print(f"  {fname}: variance={var:.4f}")


if __name__ == "__main__":
    test_classifier()
    test_clustering_analysis()

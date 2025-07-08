#!/usr/bin/env python3
"""
Test workload characterization and success prediction accuracy
"""

import json
import statistics
import tempfile
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.simpulse.analysis.workload_characterizer import (
    WorkloadCharacterizer,
    predict_optimization_success,
)

# Enhanced test cases with more realistic examples
TEST_WORKLOADS = {
    "highly_optimizable": {
        "content": """
-- Pure identity arithmetic - should have high success
theorem id1 : ∀ n : Nat, n + 0 = n := by rfl
theorem id2 : ∀ n : Nat, 0 + n = n := by rfl  
theorem id3 : ∀ n : Nat, n * 1 = n := by rfl
theorem id4 : ∀ n : Nat, 1 * n = n := by rfl
theorem id5 : ∀ n : Nat, n - 0 = n := by rfl
theorem id6 : ∀ n m : Nat, (n + 0) + (m * 1) = n + m := by rfl
""",
        "expected_success": 0.45,
        "expected_style": "direct",
    },
    "moderately_optimizable": {
        "content": """
-- List operations with some uniformity
def append_nil {α : Type} : ∀ xs : List α, xs ++ [] = xs
  | [] => rfl
  | x :: xs => by simp [append_nil xs]

def reverse_invol {α : Type} : ∀ xs : List α, xs.reverse.reverse = xs
  | [] => rfl
  | x :: xs => by simp [reverse_invol xs]
  
theorem list_len : ∀ (x : α) (xs : List α), (x :: xs).length = xs.length + 1 := by intro x xs; rfl
""",
        "expected_success": 0.35,
        "expected_style": "inductive",
    },
    "computationally_intensive": {
        "content": """
-- Complex arithmetic with multiple operations
def poly (x y z : Nat) : Nat := x * x + 2 * x * y + y * y + z

theorem poly_expand : ∀ x y z : Nat, poly x y z = (x + y) * (x + y) + z := by
  intro x y z
  unfold poly
  ring
  
theorem poly_symm : ∀ x y z : Nat, poly x y z = poly y x z := by
  intro x y z
  unfold poly
  ring
  
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n
  
theorem fact_mul : ∀ n : Nat, factorial (n + 1) = (n + 1) * factorial n := by
  intro n
  rfl
""",
        "expected_success": 0.25,
        "expected_style": "computational",
    },
    "case_analysis_heavy": {
        "content": """
-- Pattern matching and case analysis
def classify_nat : Nat → String
  | 0 => "zero"
  | 1 => "one"  
  | n + 2 => "many"
  
theorem classify_correct : ∀ n : Nat, n > 1 → classify_nat n = "many" := by
  intro n h
  cases n with
  | zero => contradiction
  | succ m =>
    cases m with
    | zero => contradiction
    | succ k => rfl
    
inductive Tree (α : Type) where
  | leaf : α → Tree α
  | node : Tree α → Tree α → Tree α
  
def tree_size {α : Type} : Tree α → Nat
  | Tree.leaf _ => 1
  | Tree.node l r => tree_size l + tree_size r
""",
        "expected_success": 0.18,
        "expected_style": "case_based",
    },
    "mixed_chaos": {
        "content": """
-- Mixed patterns with high interference
import Mathlib.Data.Nat.Basic
import Mathlib.Data.List.Basic

theorem mixed1 : ∀ n : Nat, n + 0 = n := by simp
theorem mixed2 : ∀ xs : List Nat, xs ++ [] = xs := by simp
theorem mixed3 : ∀ x y : Nat, x * (y + 0) = x * y := by simp
theorem mixed4 : ∀ xs : List Nat, ∀ n : Nat, (n :: xs).length = xs.length + 1 := by simp
theorem mixed5 : ∀ P : Nat → Prop, (∀ n, P n) → P 0 := by intro P h; exact h 0
theorem mixed6 : ∀ f : Nat → Nat, f (0 + 0) = f 0 := by simp
""",
        "expected_success": 0.10,
        "expected_style": "mixed",
    },
    "highly_abstract": {
        "content": """
-- Type-level and abstract proofs
universe u v

def id_func {α : Type u} (x : α) : α := x

theorem id_compose {α : Type u} : ∀ x : α, id_func (id_func x) = id_func x := by
  intro x
  rfl
  
class Monoid (M : Type u) where
  mul : M → M → M
  one : M
  mul_assoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a, mul one a = a
  mul_one : ∀ a, mul a one = a
  
instance : Monoid Nat where
  mul := Nat.mul
  one := 1
  mul_assoc := Nat.mul_assoc
  one_mul := Nat.one_mul
  mul_one := Nat.mul_one
""",
        "expected_success": 0.20,
        "expected_style": "algebraic",
    },
}


def test_workload_characterization():
    """Test the workload characterizer on various proof types"""
    characterizer = WorkloadCharacterizer()
    results = []

    print("Testing Workload Characterization")
    print("=" * 80)

    for name, test_case in TEST_WORKLOADS.items():
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(test_case["content"])
            test_file = Path(f.name)

        try:
            # Characterize workload
            profile = characterizer.characterize(test_file)

            # Make prediction
            should_opt, success_prob, _ = predict_optimization_success(test_file)

            result = {
                "name": name,
                "expected_success": test_case["expected_success"],
                "predicted_success": success_prob,
                "expected_style": test_case["expected_style"],
                "predicted_style": profile.proof_style,
                "structural_complexity": profile.structural_complexity,
                "pattern_uniformity": profile.pattern_uniformity,
                "proof_depth": profile.proof_depth,
                "computational_intensity": profile.computational_intensity,
                "confidence": profile.confidence,
            }

            results.append(result)

            # Print result
            print(f"\n{name}:")
            print(f"  Expected success: {test_case['expected_success']:.2%}")
            print(f"  Predicted success: {success_prob:.2%}")
            print(f"  Error: {abs(success_prob - test_case['expected_success']):.2%}")
            print(f"  Expected style: {test_case['expected_style']}")
            print(f"  Predicted style: {profile.proof_style}")
            print(
                f"  Dimensions: SC={profile.structural_complexity:.2f}, "
                f"PU={profile.pattern_uniformity:.2f}, "
                f"PD={profile.proof_depth:.2f}, "
                f"CI={profile.computational_intensity:.2f}"
            )
            print(f"  Confidence: {profile.confidence:.2%}")

        except Exception as e:
            print(f"\n✗ Error testing {name}: {e}")

        finally:
            test_file.unlink()

    # Analyze prediction accuracy
    print("\n" + "=" * 80)
    print("PREDICTION ACCURACY ANALYSIS")
    print("=" * 80)

    if results:
        # Success rate prediction accuracy
        expected = [r["expected_success"] for r in results]
        predicted = [r["predicted_success"] for r in results]

        # Calculate metrics
        mae = statistics.mean(abs(e - p) for e, p in zip(expected, predicted))
        rmse = math.sqrt(statistics.mean((e - p) ** 2 for e, p in zip(expected, predicted)))

        # Correlation
        correlation = np.corrcoef(expected, predicted)[0, 1]

        print(f"Mean Absolute Error: {mae:.3f}")
        print(f"Root Mean Square Error: {rmse:.3f}")
        print(f"Correlation: {correlation:.3f}")

        # Binary classification metrics (threshold at 0.25)
        threshold = 0.25
        expected_binary = [1 if e >= threshold else 0 for e in expected]
        predicted_binary = [1 if p >= threshold else 0 for p in predicted]

        accuracy = accuracy_score(expected_binary, predicted_binary)
        precision = precision_score(expected_binary, predicted_binary, zero_division=0)
        recall = recall_score(expected_binary, predicted_binary, zero_division=0)
        f1 = f1_score(expected_binary, predicted_binary, zero_division=0)

        print(f"\nBinary Classification (threshold={threshold}):")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  F1 Score: {f1:.2%}")

        # Style prediction accuracy
        style_matches = sum(1 for r in results if r["predicted_style"] == r["expected_style"])
        style_accuracy = style_matches / len(results)
        print(f"\nProof Style Accuracy: {style_matches}/{len(results)} ({style_accuracy:.2%})")

        # Visualization
        visualize_predictions(results)

        # Save results
        with open("workload_prediction_results.json", "w") as f:
            json.dump(
                {
                    "results": results,
                    "metrics": {
                        "mae": mae,
                        "rmse": rmse,
                        "correlation": correlation,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "style_accuracy": style_accuracy,
                    },
                },
                f,
                indent=2,
            )

        print(f"\nDetailed results saved to workload_prediction_results.json")

        # Final verdict
        print("\n" + "=" * 80)
        print("CAN WE PREDICT WITH 90%+ ACCURACY?")
        print("=" * 80)

        if accuracy >= 0.9:
            print(f"YES! Binary classification accuracy: {accuracy:.2%}")
        else:
            print(f"NOT YET. Binary classification accuracy: {accuracy:.2%}")
            print(
                f"However, correlation of {correlation:.3f} shows meaningful prediction capability"
            )


def visualize_predictions(results: List[Dict]):
    """Create visualization of prediction accuracy"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    names = [r["name"] for r in results]
    expected = [r["expected_success"] for r in results]
    predicted = [r["predicted_success"] for r in results]

    # 1. Expected vs Predicted
    ax1.scatter(expected, predicted, s=100, alpha=0.7)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)  # Perfect prediction line
    ax1.set_xlabel("Expected Success Rate")
    ax1.set_ylabel("Predicted Success Rate")
    ax1.set_title("Prediction Accuracy")
    ax1.grid(True, alpha=0.3)

    # Add labels
    for i, name in enumerate(names):
        ax1.annotate(name.split("_")[0], (expected[i], predicted[i]), fontsize=8, alpha=0.7)

    # 2. Error distribution
    errors = [abs(e - p) for e, p in zip(expected, predicted)]
    ax2.bar(range(len(names)), errors)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.split("_")[0] for n in names], rotation=45)
    ax2.set_ylabel("Absolute Error")
    ax2.set_title("Prediction Errors by Workload Type")
    ax2.axhline(y=0.1, color="r", linestyle="--", alpha=0.5, label="10% error")
    ax2.legend()

    # 3. Workload dimensions
    dimensions = [
        "structural_complexity",
        "pattern_uniformity",
        "proof_depth",
        "computational_intensity",
    ]
    dim_values = np.array([[r[d] for d in dimensions] for r in results])

    im = ax3.imshow(dim_values.T, aspect="auto", cmap="RdYlGn_r")
    ax3.set_yticks(range(len(dimensions)))
    ax3.set_yticklabels([d.replace("_", " ").title() for d in dimensions])
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([n.split("_")[0] for n in names], rotation=45)
    ax3.set_title("Workload Characterization Dimensions")

    # Add colorbar
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Success rate by proof style
    style_groups = {}
    for r in results:
        style = r["predicted_style"]
        if style not in style_groups:
            style_groups[style] = []
        style_groups[style].append(r["predicted_success"])

    styles = list(style_groups.keys())
    avg_success = [statistics.mean(style_groups[s]) for s in styles]

    ax4.bar(styles, avg_success)
    ax4.set_ylabel("Average Predicted Success Rate")
    ax4.set_title("Success Rate by Proof Style")
    ax4.axhline(y=0.25, color="r", linestyle="--", alpha=0.5, label="Optimization threshold")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("workload_prediction_analysis.png", dpi=300, bbox_inches="tight")
    print("\nVisualization saved to workload_prediction_analysis.png")


import math  # Add this import for RMSE calculation

if __name__ == "__main__":
    test_workload_characterization()

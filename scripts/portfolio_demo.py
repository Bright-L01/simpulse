#!/usr/bin/env python3
"""
Demo of portfolio approach for Lean tactics.

Shows how the feature extraction and prediction would work.
"""

from simpulse.portfolio import extract_features


def demo_feature_extraction():
    """Demo feature extraction on various Lean goals."""

    test_goals = [
        ("⊢ x + 0 = x", "simp"),
        ("⊢ (x + y)^2 = x^2 + 2*x*y + y^2", "ring"),
        ("⊢ x < x + 1", "linarith"),
        ("⊢ 42 * 17 = 714", "norm_num"),
        ("⊢ x / x = 1", "field_simp"),
        ("⊢ ∀ x y : ℕ, x + y = y + x", "simp"),
        ("⊢ ∀ a b c : ℝ, a * (b + c) = a * b + a * c", "ring"),
        ("⊢ ∀ x : ℤ, x < x + 1", "linarith"),
        ("⊢ ∀ A B : Set α, A ∪ B = B ∪ A", "simp"),
        ("⊢ ∀ p q : Prop, p ∧ q → p", "tauto"),
    ]

    print("Portfolio Approach Demo - Feature Extraction")
    print("=" * 70)
    print()

    for goal_text, expected_tactic in test_goals:
        print(f"Goal: {goal_text}")
        print(f"Expected tactic: {expected_tactic}")

        # Extract features
        features = extract_features(goal_text)

        print(f"Goal type: {features.goal_type}")
        print("Key features:")

        # Show important features
        feature_summary = []

        if features.has_arithmetic:
            feature_summary.append("arithmetic")
        if features.has_algebra:
            feature_summary.append("algebra")
        if features.has_linear:
            feature_summary.append("linear")
        if features.has_logic:
            feature_summary.append("logic")
        if features.has_sets:
            feature_summary.append("sets")

        if features.is_equation:
            feature_summary.append("equation")
        if features.is_inequality:
            feature_summary.append("inequality")

        # Type information
        types = []
        if features.involves_nat:
            types.append("ℕ")
        if features.involves_int:
            types.append("ℤ")
        if features.involves_real:
            types.append("ℝ")
        if features.involves_set:
            types.append("Set")

        print(f"  - Features: {', '.join(feature_summary)}")
        print(f"  - Types: {', '.join(types) if types else 'none detected'}")
        print(f"  - Complexity: {features.total_terms} terms, depth {features.depth}")
        print(
            f"  - Operators: {dict(list(features.operators.items())[:5]) if features.operators else 'none'}"
        )

        # Show how features map to tactics
        print("Feature-based prediction:")

        if (
            features.goal_type == "equation"
            and features.has_arithmetic
            and not features.has_algebra
        ):
            predicted = "simp"
        elif (
            features.goal_type in ["algebraic_equation", "equation"]
            and features.has_algebra
        ):
            predicted = "ring"
        elif (
            features.goal_type in ["linear_inequality", "inequality"]
            and features.has_linear
        ):
            predicted = "linarith"
        elif all(c.isdigit() or c in "+-*/<>=^ " for c in goal_text):
            predicted = "norm_num"
        elif "/" in features.operators or "div" in features.operators:
            predicted = "field_simp"
        elif features.has_logic:
            predicted = "tauto"
        elif features.has_sets:
            predicted = "simp"
        else:
            predicted = "simp"  # Default

        match = "✓" if predicted == expected_tactic else "✗"
        print(f"  - Predicted: {predicted} {match}")

        print("-" * 70)
        print()

    # Show feature vector example
    print("Example Feature Vector (first goal):")
    example_goal = test_goals[0][0]
    features = extract_features(example_goal)
    vector = features.to_vector()

    print(f"Goal: {example_goal}")
    print(f"Feature vector length: {len(vector)}")
    print(f"First 10 features: {vector[:10]}")
    print()

    # Explain the approach
    print("How the Portfolio Approach Works:")
    print("1. Extract structural features from Lean goals")
    print("2. Use ML model (Random Forest) to predict best tactic")
    print("3. Try predicted tactic with timeout")
    print("4. Fall back to alternatives if needed")
    print("5. Learn from successes/failures to improve predictions")
    print()

    print("Benefits:")
    print("- Reduces time spent on failed tactic attempts")
    print("- Learns patterns from successful proofs")
    print("- Adapts to specific proof styles in a codebase")
    print("- Provides interpretable predictions")


if __name__ == "__main__":
    demo_feature_extraction()

#!/usr/bin/env python3
"""
Demo of portfolio approach without requiring scikit-learn.

Shows the concept and design of the ML-based tactic selection.
"""

from simpulse.portfolio.feature_extractor import LeanGoalParser, extract_features


def demo_feature_extraction():
    """Demo feature extraction from Lean goals."""

    print("Portfolio Approach Demo")
    print("=" * 70)
    print("\nThis system uses ML to predict the best tactic for a Lean goal.")
    print(
        "Currently demonstrating feature extraction (ML training requires scikit-learn).\n"
    )

    # Example goals
    test_goals = [
        ("⊢ x + 0 = x", "simp"),
        ("⊢ (a + b)^2 = a^2 + 2*a*b + b^2", "ring"),
        ("⊢ x < x + 1", "linarith"),
        ("⊢ 42 * 17 = 714", "norm_num"),
        ("⊢ x / x = 1", "field_simp"),
        ("⊢ a + b = b + a", "abel"),
        ("⊢ ∀ x y : ℕ, x + y = y + x", "simp"),
        ("⊢ ∀ a b c : ℝ, a * (b + c) = a * b + a * c", "ring"),
    ]

    LeanGoalParser()

    print("Feature Extraction Examples:")
    print("-" * 70)

    for goal, expected_tactic in test_goals:
        print(f"\nGoal: {goal}")
        print(f"Expected tactic: {expected_tactic}")

        # Extract features
        features = extract_features(goal)

        print(f"Goal type: {features.goal_type}")
        print(f"Features:")
        print(f"  - Arithmetic: {features.has_arithmetic}")
        print(f"  - Linear: {features.has_linear}")
        print(f"  - Equation: {features.is_equation}")
        print(f"  - Inequality: {features.is_inequality}")
        print(f"  - Complexity: {features.total_terms} terms, depth {features.depth}")

        # Show feature vector (first 10 elements)
        vector = features.to_vector()
        print(f"  - Feature vector: [{', '.join(f'{x:.1f}' for x in vector[:10])}...]")

        # Simple heuristic prediction (not ML)
        predicted = predict_tactic_heuristic(features)
        print(f"Heuristic prediction: {predicted}")
        print(f"Correct: {'✓' if predicted == expected_tactic else '✗'}")


def predict_tactic_heuristic(features):
    """Simple heuristic tactic prediction (placeholder for ML model)."""

    # This mimics what the ML model would learn
    if features.is_inequality and features.has_linear:
        return "linarith"
    elif features.has_exponentiation and features.is_equation:
        return "ring"
    elif features.involves_real and features.has_division:
        return "field_simp"
    elif features.goal_type == "algebraic_equation":
        return "ring"
    elif features.num_constants > features.num_variables and features.has_arithmetic:
        return "norm_num"
    elif features.has_arithmetic and features.is_equation:
        if features.total_terms > 10:
            return "ring"
        else:
            return "simp"
    else:
        return "simp"  # Default


def show_architecture():
    """Show the portfolio architecture."""

    print("\n\nPortfolio Architecture:")
    print("=" * 70)

    print(
        """
1. Feature Extraction (LeanGoalParser)
   - Parses goal structure
   - Detects patterns (arithmetic, algebraic, etc.)
   - Computes complexity metrics
   - Outputs numerical feature vector

2. ML Model (RandomForestClassifier)
   - Trained on mathlib4 proofs
   - Input: feature vector
   - Output: ranked list of tactics
   - Interpretable feature importance

3. Lean Integration (TacticPortfolio)
   - Custom 'portfolio' tactic
   - Calls Python predictor via IPC
   - Tries predicted tactics in order
   - Falls back to exhaustive search

4. Continuous Learning
   - Records successful tactics
   - Updates training data
   - Periodic model retraining
   - Adapts to codebase patterns
"""
    )


def show_benefits():
    """Show benefits of the portfolio approach."""

    print("\nBenefits of Portfolio Approach:")
    print("=" * 70)

    benefits = [
        ("Speed", "Predicts best tactic without trial-and-error"),
        ("Adaptability", "Learns from your specific codebase"),
        ("Interpretability", "Shows why each tactic was chosen"),
        ("Robustness", "Falls back to traditional search if needed"),
        ("Extensibility", "Easy to add new tactics to portfolio"),
    ]

    for benefit, description in benefits:
        print(f"\n{benefit}:")
        print(f"  {description}")

    print("\n\nExample Usage in Lean:")
    print("-" * 50)
    print(
        """
-- Basic usage
example (x : Nat) : x + 0 = x := by
  portfolio  -- Automatically selects 'simp'

-- With configuration
example (a b : Real) : (a + b)^2 = a^2 + 2*a*b + b^2 := by
  portfolio trainedConfig  -- Uses ML model

-- Convenience macro
example (x : Int) : x < x + 1 := by
  ml_auto  -- Shorthand for portfolio
"""
    )


def main():
    """Run the demo."""

    # Feature extraction demo
    demo_feature_extraction()

    # Show architecture
    show_architecture()

    # Show benefits
    show_benefits()

    print("\n\nTo use the full ML system:")
    print("-" * 50)
    print("1. Install ML dependencies: pip install -e '.[ml]'")
    print("2. Train on mathlib4: python scripts/train_portfolio.py mathlib <path>")
    print(
        "3. Integrate with Lean: python scripts/train_portfolio.py integrate <project> <model>"
    )

    print("\nThe portfolio approach brings modern ML to theorem proving!")


if __name__ == "__main__":
    main()

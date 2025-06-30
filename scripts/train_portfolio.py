#!/usr/bin/env python3
"""
Script to train tactic portfolio from mathlib4 or custom data.
"""

import argparse
import json

from simpulse.portfolio import (
    TacticDataset,
    TacticPredictor,
    create_lean_integration,
    train_from_mathlib,
)


def train_from_json(json_path: str, output_model: str):
    """Train model from JSON training data."""
    dataset = TacticDataset()
    dataset.add_from_json(json_path)

    # Add synthetic examples if needed
    if len(dataset.examples) < 100:
        print("Adding synthetic examples...")
        synthetic = dataset.create_synthetic_examples()
        dataset.examples.extend(synthetic)

    # Train
    predictor = TacticPredictor()
    balanced = dataset.get_balanced_dataset()

    print(f"Training on {len(balanced)} examples...")
    metrics = predictor.train(balanced)

    print(f"\nTraining complete!")
    print(f"Metrics: {metrics}")

    # Save
    predictor.save_model(output_model)

    return predictor


def evaluate_model(model_path: str, test_file: str):
    """Evaluate model on test data."""
    predictor = TacticPredictor(model_path)

    with open(test_file) as f:
        test_data = json.load(f)

    correct = 0
    total = 0

    print(f"Evaluating on {len(test_data)} test examples...")

    for example in test_data:
        goal = example["goal"]
        true_tactic = example["tactic"]

        if true_tactic not in predictor.SUPPORTED_TACTICS:
            continue

        pred = predictor.predict(goal)

        if pred.tactic == true_tactic:
            correct += 1

        total += 1

        if total % 100 == 0:
            print(f"  Processed {total} examples...")

    accuracy = correct / total if total > 0 else 0

    print(f"\nEvaluation Results:")
    print(f"  Total examples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.2%}")

    # Analyze errors
    print("\nAnalyzing prediction distribution...")
    predictions = {}

    for example in test_data[:1000]:  # Sample for analysis
        goal = example["goal"]
        pred = predictor.predict(goal)

        predictions[pred.tactic] = predictions.get(pred.tactic, 0) + 1

    print("\nPrediction distribution:")
    for tactic, count in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tactic}: {count}")


def demo_predictions(model_path: str):
    """Demo the trained model on various goals."""
    predictor = TacticPredictor(model_path)

    demo_goals = [
        # Arithmetic
        ("⊢ x + 0 = x", "simp"),
        ("⊢ 0 + x = x", "simp"),
        ("⊢ x * 1 = x", "simp"),
        # Ring
        ("⊢ (x + y)^2 = x^2 + 2*x*y + y^2", "ring"),
        ("⊢ (a - b) * (a + b) = a^2 - b^2", "ring"),
        # Linear arithmetic
        ("⊢ x < x + 1", "linarith"),
        ("⊢ x ≤ y → y ≤ z → x ≤ z", "linarith"),
        # Numerical
        ("⊢ 42 * 17 = 714", "norm_num"),
        ("⊢ 2^10 = 1024", "norm_num"),
        # Field
        ("⊢ x / x = 1", "field_simp"),
        ("⊢ (a / b) * b = a", "field_simp"),
        # Complex
        ("⊢ ∀ x y : ℕ, x + y = y + x", "simp"),
        ("⊢ ∀ a b c : ℝ, a * (b + c) = a * b + a * c", "ring"),
    ]

    print("Model Predictions Demo")
    print("=" * 70)

    correct = 0
    for goal, expected in demo_goals:
        pred = predictor.predict(goal)
        is_correct = pred.tactic == expected

        if is_correct:
            correct += 1

        print(f"\nGoal: {goal}")
        print(f"Expected: {expected}")
        print(f"Predicted: {pred.tactic} (confidence: {pred.confidence:.2f})")
        print(f"Correct: {'✓' if is_correct else '✗'}")

        if pred.alternatives:
            print(
                f"Alternatives: {[f'{t}({c:.2f})' for t, c in pred.alternatives[:2]]}"
            )

        # Explain prediction
        explanation = predictor.explain_prediction(goal)
        if explanation["key_features"]:
            print("Key features:")
            for feat, info in list(explanation["key_features"].items())[:3]:
                print(
                    f"  {feat}: {info['value']:.2f} (importance: {info['importance']:.3f})"
                )

    print(f"\n{'='*70}")
    print(
        f"Demo accuracy: {correct}/{len(demo_goals)} ({correct/len(demo_goals)*100:.1f}%)"
    )


def main():
    parser = argparse.ArgumentParser(description="Train tactic portfolio model")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train from mathlib
    mathlib_parser = subparsers.add_parser("mathlib", help="Train from mathlib4")
    mathlib_parser.add_argument("path", help="Path to mathlib4")
    mathlib_parser.add_argument(
        "--output", default="tactic_portfolio_model.pkl", help="Output model file"
    )
    mathlib_parser.add_argument(
        "--samples", type=int, default=10000, help="Number of training samples"
    )

    # Train from JSON
    json_parser = subparsers.add_parser("json", help="Train from JSON data")
    json_parser.add_argument("path", help="Path to JSON training data")
    json_parser.add_argument(
        "--output", default="tactic_portfolio_model.pkl", help="Output model file"
    )

    # Evaluate model
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("model", help="Path to model file")
    eval_parser.add_argument("test_data", help="Path to test data JSON")

    # Demo predictions
    demo_parser = subparsers.add_parser("demo", help="Demo model predictions")
    demo_parser.add_argument("model", help="Path to model file")

    # Create Lean integration
    integrate_parser = subparsers.add_parser(
        "integrate", help="Create Lean integration"
    )
    integrate_parser.add_argument("project", help="Lean project path")
    integrate_parser.add_argument("model", help="Path to model file")

    args = parser.parse_args()

    if args.command == "mathlib":
        predictor = train_from_mathlib(args.path, args.output, args.samples)
        print(f"\nModel saved to {args.output}")

    elif args.command == "json":
        predictor = train_from_json(args.path, args.output)
        print(f"\nModel saved to {args.output}")

    elif args.command == "evaluate":
        evaluate_model(args.model, args.test_data)

    elif args.command == "demo":
        demo_predictions(args.model)

    elif args.command == "integrate":
        create_lean_integration(args.project, args.model)

    else:
        # Train on synthetic data if no command given
        print("Training on synthetic data...")
        dataset = TacticDataset()
        synthetic = dataset.create_synthetic_examples()
        dataset.examples.extend(synthetic)

        predictor = TacticPredictor()
        metrics = predictor.train(dataset.examples)

        print(f"\nMetrics: {metrics}")

        # Save and demo
        predictor.save_model("synthetic_tactic_model.pkl")
        demo_predictions("synthetic_tactic_model.pkl")


if __name__ == "__main__":
    main()

"""
ML model for tactic prediction.

Uses Random Forest for interpretability and trains on mathlib4 proof data.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from .feature_extractor import extract_features


@dataclass
class TacticPrediction:
    """Prediction result for tactic selection."""

    tactic: str
    confidence: float
    alternatives: List[Tuple[str, float]]  # (tactic, confidence) pairs
    features_used: Optional[Dict[str, float]] = None  # Feature importance


class TacticPredictor:
    """ML-based tactic predictor using Random Forest."""

    # Focus on common tactics for initial version
    SUPPORTED_TACTICS = [
        "simp",
        "ring",
        "linarith",
        "norm_num",
        "field_simp",
        "abel",
        "omega",
        "tauto",
        "aesop",
        "exact",
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[RandomForestClassifier] = None
        self.label_encoder = LabelEncoder()
        self.feature_names = self._get_feature_names()
        self.model_path = model_path

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize a new Random Forest model."""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            # Enable feature importance
            oob_score=True,
        )

        # Initialize label encoder with supported tactics
        self.label_encoder.fit(self.SUPPORTED_TACTICS)

    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        return [
            # Binary features
            "has_arithmetic",
            "has_algebra",
            "has_linear",
            "has_logic",
            "has_sets",
            "is_equation",
            "is_inequality",
            "has_addition",
            "has_multiplication",
            "has_subtraction",
            "has_division",
            "has_exponentiation",
            "has_modulo",
            "involves_nat",
            "involves_int",
            "involves_real",
            "involves_complex",
            "involves_list",
            "involves_set",
            # Numerical features
            "depth",
            "num_subgoals",
            "num_variables",
            "num_constants",
            "num_functions",
            "max_nesting",
            "total_terms",
            # Operator frequencies
            "op_add",
            "op_mul",
            "op_sub",
            "op_div",
            "op_pow",
            "op_eq",
            "op_le",
            "op_lt",
            "op_and",
            "op_or",
        ]

    def train(
        self, training_data: List[Tuple[str, str]], validation_split: float = 0.2
    ) -> Dict[str, float]:
        """Train the model on goal-tactic pairs."""
        if not training_data:
            raise ValueError("No training data provided")

        print(f"Training on {len(training_data)} examples...")

        # Extract features and labels
        X = []
        y = []
        failed = 0

        for goal_text, tactic in training_data:
            if tactic not in self.SUPPORTED_TACTICS:
                continue  # Skip unsupported tactics

            try:
                features = extract_features(goal_text)
                X.append(features.to_vector())
                y.append(tactic)
            except Exception:
                failed += 1
                continue

        if failed > 0:
            print(f"Failed to extract features for {failed} examples")

        if len(X) < 10:
            raise ValueError("Insufficient training data after filtering")

        # Encode labels
        y_encoded = self.label_encoder.transform(y)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y_encoded,
            test_size=validation_split,
            random_state=42,
            stratify=y_encoded,
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)

        # Cross-validation for better estimate
        cv_scores = cross_val_score(self.model, X, y_encoded, cv=5)

        metrics = {
            "train_accuracy": train_score,
            "val_accuracy": val_score,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "n_samples": len(X),
            "n_features": len(self.feature_names),
        }

        print(f"Training complete!")
        print(f"Train accuracy: {train_score:.3f}")
        print(f"Validation accuracy: {val_score:.3f}")
        print(f"Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        return metrics

    def predict(self, goal_text: str, top_k: int = 3) -> TacticPrediction:
        """Predict best tactic for a goal."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Extract features
        features = extract_features(goal_text)
        feature_vector = features.to_vector()

        # Get prediction probabilities
        probas = self.model.predict_proba([feature_vector])[0]

        # Get top predictions
        tactic_probs = []
        for idx, prob in enumerate(probas):
            tactic = self.label_encoder.inverse_transform([idx])[0]
            tactic_probs.append((tactic, prob))

        # Sort by confidence
        tactic_probs.sort(key=lambda x: x[1], reverse=True)

        # Get feature importance for this prediction
        feature_importance = None
        if hasattr(self.model, "feature_importances_"):
            importance_dict = {}
            for fname, importance in zip(
                self.feature_names, self.model.feature_importances_
            ):
                if importance > 0.01:  # Only include significant features
                    importance_dict[fname] = float(importance)
            feature_importance = importance_dict

        # Create prediction
        best_tactic, best_conf = tactic_probs[0]
        alternatives = tactic_probs[1 : top_k + 1]

        return TacticPrediction(
            tactic=best_tactic,
            confidence=best_conf,
            alternatives=alternatives,
            features_used=feature_importance,
        )

    def predict_batch(self, goal_texts: List[str]) -> List[TacticPrediction]:
        """Predict tactics for multiple goals efficiently."""
        # Extract all features
        feature_vectors = []
        valid_indices = []

        for i, goal_text in enumerate(goal_texts):
            try:
                features = extract_features(goal_text)
                feature_vectors.append(features.to_vector())
                valid_indices.append(i)
            except:
                continue

        if not feature_vectors:
            return []

        # Batch prediction
        probas_batch = self.model.predict_proba(feature_vectors)

        # Process results
        predictions = []
        for probas in probas_batch:
            tactic_probs = []
            for idx, prob in enumerate(probas):
                tactic = self.label_encoder.inverse_transform([idx])[0]
                tactic_probs.append((tactic, prob))

            tactic_probs.sort(key=lambda x: x[1], reverse=True)

            best_tactic, best_conf = tactic_probs[0]
            alternatives = tactic_probs[1:4]

            predictions.append(
                TacticPrediction(
                    tactic=best_tactic, confidence=best_conf, alternatives=alternatives
                )
            )

        return predictions

    def explain_prediction(self, goal_text: str) -> Dict[str, any]:
        """Explain why a particular tactic was chosen."""
        # Get prediction
        prediction = self.predict(goal_text)

        # Extract features
        features = extract_features(goal_text)
        feature_vector = features.to_vector()

        # Get decision path for Random Forest
        explanation = {
            "predicted_tactic": prediction.tactic,
            "confidence": prediction.confidence,
            "alternatives": prediction.alternatives,
            "goal_type": features.goal_type,
            "key_features": {},
            "feature_values": {},
        }

        # Identify key features
        if prediction.features_used:
            for fname, importance in sorted(
                prediction.features_used.items(), key=lambda x: x[1], reverse=True
            )[:5]:
                idx = self.feature_names.index(fname)
                explanation["key_features"][fname] = {
                    "importance": importance,
                    "value": feature_vector[idx],
                }

        # Add interpretable feature values
        explanation["feature_values"] = {
            "has_arithmetic": features.has_arithmetic,
            "has_linear": features.has_linear,
            "is_equation": features.is_equation,
            "complexity": features.total_terms,
        }

        return explanation

    def save_model(self, path: str):
        """Save trained model to disk."""
        model_data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "supported_tactics": self.SUPPORTED_TACTICS,
        }

        with open(path, "wb") as f:
            joblib.dump(model_data, f)

        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model from disk."""
        with open(path, "rb") as f:
            model_data = joblib.load(f)

        self.model = model_data["model"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_names = model_data.get("feature_names", self._get_feature_names())

        print(f"Model loaded from {path}")


class TacticDataset:
    """Dataset for training tactic predictor."""

    def __init__(self):
        self.examples: List[Tuple[str, str]] = []

    def add_example(self, goal: str, tactic: str):
        """Add a training example."""
        self.examples.append((goal, tactic))

    def add_from_json(self, json_path: str):
        """Load examples from JSON file."""
        with open(json_path) as f:
            data = json.load(f)

        for example in data:
            if "goal" in example and "tactic" in example:
                self.add_example(example["goal"], example["tactic"])

    def create_synthetic_examples(self) -> List[Tuple[str, str]]:
        """Create synthetic training examples for initial model."""
        examples = []

        # Simp examples - simple arithmetic and algebraic identities
        simp_goals = [
            "⊢ x + 0 = x",
            "⊢ 0 + x = x",
            "⊢ x * 1 = x",
            "⊢ 1 * x = x",
            "⊢ x - x = 0",
            "⊢ x * 0 = 0",
            "⊢ (x + y) + 0 = x + y",
            "⊢ List.length [] = 0",
            "⊢ List.append [] l = l",
        ]
        examples.extend((g, "simp") for g in simp_goals)

        # Ring examples - polynomial equations
        ring_goals = [
            "⊢ (x + y) * (x - y) = x^2 - y^2",
            "⊢ (a + b)^2 = a^2 + 2*a*b + b^2",
            "⊢ x * (y + z) = x * y + x * z",
            "⊢ (x + 1) * (x - 1) = x^2 - 1",
            "⊢ x^2 + 2*x + 1 = (x + 1)^2",
            "⊢ a * b + a * c = a * (b + c)",
        ]
        examples.extend((g, "ring") for g in ring_goals)

        # Linarith examples - linear inequalities
        linarith_goals = [
            "⊢ x < x + 1",
            "⊢ x ≤ y → y ≤ z → x ≤ z",
            "⊢ 2 * x < 3 * x → 0 < x",
            "⊢ x + y < x + z → y < z",
            "⊢ a ≤ b → c ≤ d → a + c ≤ b + d",
            "⊢ 0 < x → 0 < y → 0 < x + y",
        ]
        examples.extend((g, "linarith") for g in linarith_goals)

        # Norm_num examples - numerical computations
        norm_num_goals = [
            "⊢ 2 + 3 = 5",
            "⊢ 7 * 8 = 56",
            "⊢ 100 / 5 = 20",
            "⊢ 2^10 = 1024",
            "⊢ 15 % 4 = 3",
            "⊢ 17 < 23",
        ]
        examples.extend((g, "norm_num") for g in norm_num_goals)

        # Field_simp examples - field arithmetic
        field_simp_goals = [
            "⊢ x / x = 1",
            "⊢ (a / b) * b = a",
            "⊢ 1 / (1 / x) = x",
            "⊢ (x / y) / z = x / (y * z)",
            "⊢ a / b + c / d = (a * d + b * c) / (b * d)",
        ]
        examples.extend((g, "field_simp") for g in field_simp_goals)

        # Abel examples - abelian group equations
        abel_goals = [
            "⊢ a + b = b + a",
            "⊢ (a + b) + c = a + (b + c)",
            "⊢ a + (b - a) = b",
            "⊢ -(-a) = a",
            "⊢ a - b + b = a",
        ]
        examples.extend((g, "abel") for g in abel_goals)

        return examples

    def get_balanced_dataset(self) -> List[Tuple[str, str]]:
        """Get a balanced dataset with equal examples per tactic."""
        from collections import defaultdict

        # Group by tactic
        by_tactic = defaultdict(list)
        for goal, tactic in self.examples:
            by_tactic[tactic].append((goal, tactic))

        # Find minimum count
        min_count = min(len(examples) for examples in by_tactic.values())

        # Balance dataset
        balanced = []
        for tactic, examples in by_tactic.items():
            balanced.extend(examples[:min_count])

        return balanced


if __name__ == "__main__":
    # Example usage
    dataset = TacticDataset()

    # Create synthetic examples
    synthetic = dataset.create_synthetic_examples()
    dataset.examples.extend(synthetic)

    # Train model
    predictor = TacticPredictor()
    metrics = predictor.train(dataset.examples)

    print(f"\nTraining metrics: {metrics}")

    # Test predictions
    test_goals = [
        "⊢ x + 0 = x",
        "⊢ (a + b)^2 = a^2 + 2*a*b + b^2",
        "⊢ x < x + 1",
        "⊢ 42 * 17 = 714",
    ]

    print("\nTest predictions:")
    for goal in test_goals:
        pred = predictor.predict(goal)
        print(f"\nGoal: {goal}")
        print(f"Predicted: {pred.tactic} (confidence: {pred.confidence:.2f})")
        print(f"Alternatives: {pred.alternatives}")

    # Save model
    predictor.save_model("tactic_model.pkl")

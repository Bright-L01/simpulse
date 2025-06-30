"""
Interface between Python ML models and Lean 4.

Provides communication and training data extraction from mathlib4.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .tactic_predictor import TacticDataset, TacticPredictor


@dataclass
class ProofStep:
    """A single step in a Lean proof."""

    goal: str
    tactic: str
    success: bool
    file_path: str
    line_number: int


class LeanPortfolioInterface:
    """Interface for portfolio tactic integration."""

    def __init__(self, lean_project: str, model_path: Optional[str] = None):
        self.lean_project = Path(lean_project)
        self.model_path = model_path or "tactic_portfolio_model.pkl"
        self.predictor = TacticPredictor(self.model_path)

        # IPC configuration
        self.socket_path = "/tmp/lean_portfolio.sock"
        self.process = None

    def start_server(self):
        """Start Python prediction server for Lean."""
        # In production, this would be a proper IPC server
        # For now, we'll use a simple file-based communication

        pred_file = self.lean_project / ".portfolio_predictions"
        pred_file.touch()

        print(f"Portfolio interface ready at {self.lean_project}")

    def predict_tactic(self, goal_text: str) -> Dict[str, any]:
        """Predict best tactic for a goal."""
        try:
            prediction = self.predictor.predict(goal_text)

            result = {
                "tactic": prediction.tactic,
                "confidence": prediction.confidence,
                "alternatives": [
                    {"tactic": t, "confidence": c} for t, c in prediction.alternatives
                ],
            }

            # Write to prediction file for Lean to read
            pred_file = self.lean_project / ".portfolio_predictions"
            with open(pred_file, "w") as f:
                json.dump(result, f)

            return result

        except Exception as e:
            return {
                "tactic": "simp",  # Default fallback
                "confidence": 0.1,
                "alternatives": [],
                "error": str(e),
            }

    def update_from_result(self, goal: str, tactic: str, success: bool):
        """Update model based on actual tactic success."""
        # In online learning mode, we could update the model
        # For now, just log the result

        log_file = self.lean_project / ".portfolio_feedback.jsonl"
        with open(log_file, "a") as f:
            json.dump(
                {
                    "goal": goal,
                    "tactic": tactic,
                    "success": success,
                    "timestamp": Path.ctime(Path()),
                },
                f,
            )
            f.write("\n")

    def extract_training_data(self, lean_file: Path) -> List[ProofStep]:
        """Extract proof steps from a Lean file."""
        proof_steps = []

        try:
            content = lean_file.read_text()

            # Regular expressions for common proof patterns
            # This is simplified - real implementation would use Lean parser

            # Pattern for 'by' proofs
            by_proof_pattern = re.compile(r":=\s*by\s+(\w+)(?:\s|$)", re.MULTILINE)

            # Pattern for tactic mode proofs
            tactic_pattern = re.compile(
                r"^\s*(\w+)(?:\s+\[.*?\])?(?:\s+.*?)?$", re.MULTILINE
            )

            # Pattern for goals (simplified)
            goal_pattern = re.compile(r"⊢\s+(.+?)(?=\n|$)", re.MULTILINE)

            # Extract matches
            for match in by_proof_pattern.finditer(content):
                tactic = match.group(1)
                line_num = content[: match.start()].count("\n") + 1

                # Try to find associated goal (heuristic)
                before_text = content[: match.start()]
                goal_matches = list(goal_pattern.finditer(before_text))

                if goal_matches:
                    goal = goal_matches[-1].group(1)

                    proof_steps.append(
                        ProofStep(
                            goal=goal,
                            tactic=tactic,
                            success=True,  # Assume success if in file
                            file_path=str(lean_file),
                            line_number=line_num,
                        )
                    )

        except Exception as e:
            print(f"Error parsing {lean_file}: {e}")

        return proof_steps

    def analyze_tactic_usage(self, mathlib_path: Path) -> Dict[str, int]:
        """Analyze tactic usage frequency in mathlib4."""
        tactic_counts = {}

        # Common Lean tactics to look for
        common_tactics = [
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
            "rfl",
            "apply",
            "intro",
            "cases",
            "induction",
            "rw",
            "calc",
        ]

        # Create pattern for tactic detection
        tactic_pattern = "|".join(re.escape(t) for t in common_tactics)
        pattern = re.compile(f"\\b({tactic_pattern})\\b", re.IGNORECASE)

        lean_files = list(mathlib_path.rglob("*.lean"))

        for i, lean_file in enumerate(lean_files):
            if i % 100 == 0:
                print(f"Analyzing file {i}/{len(lean_files)}...")

            try:
                content = lean_file.read_text()

                # Count tactic occurrences
                for match in pattern.finditer(content):
                    tactic = match.group(1).lower()
                    tactic_counts[tactic] = tactic_counts.get(tactic, 0) + 1

            except Exception:
                continue

        # Sort by frequency
        sorted_tactics = sorted(tactic_counts.items(), key=lambda x: x[1], reverse=True)

        print("\nTactic usage in mathlib4:")
        for tactic, count in sorted_tactics[:20]:
            print(f"  {tactic}: {count:,} uses")

        return dict(sorted_tactics)


def train_from_mathlib(
    mathlib_path: str,
    output_model: str = "tactic_portfolio_model.pkl",
    sample_size: int = 10000,
) -> TacticPredictor:
    """Train tactic predictor from mathlib4 proofs."""
    mathlib = Path(mathlib_path)

    if not mathlib.exists():
        raise ValueError(f"Mathlib path not found: {mathlib_path}")

    print(f"Extracting training data from {mathlib_path}...")

    dataset = TacticDataset()
    interface = LeanPortfolioInterface(mathlib_path)

    # Get Lean files
    lean_files = list(mathlib.rglob("*.lean"))[
        : sample_size // 10
    ]  # Estimate ~10 examples per file

    print(f"Processing {len(lean_files)} Lean files...")

    # Extract proof steps
    all_steps = []
    for i, lean_file in enumerate(lean_files):
        if i % 50 == 0:
            print(f"Processing file {i}/{len(lean_files)}...")

        steps = interface.extract_training_data(lean_file)
        all_steps.extend(steps)

        if len(all_steps) >= sample_size:
            break

    print(f"Extracted {len(all_steps)} proof steps")

    # Filter to supported tactics
    supported_steps = [
        (step.goal, step.tactic)
        for step in all_steps
        if step.tactic in TacticPredictor.SUPPORTED_TACTICS
    ]

    print(f"Found {len(supported_steps)} steps with supported tactics")

    # Add to dataset
    for goal, tactic in supported_steps:
        dataset.add_example(goal, tactic)

    # Add synthetic examples for balance
    if len(dataset.examples) < 1000:
        print("Adding synthetic examples for better coverage...")
        synthetic = dataset.create_synthetic_examples()
        dataset.examples.extend(synthetic)

    # Balance dataset
    balanced = dataset.get_balanced_dataset()
    print(f"Training on {len(balanced)} balanced examples...")

    # Train model
    predictor = TacticPredictor()
    metrics = predictor.train(balanced)

    print(f"\nTraining complete!")
    print(f"Metrics: {metrics}")

    # Save model
    predictor.save_model(output_model)

    # Analyze tactic distribution
    print("\nAnalyzing tactic usage patterns...")
    interface.analyze_tactic_usage(mathlib)

    return predictor


def create_lean_integration(project_path: str, model_path: str):
    """Create Lean 4 integration files for portfolio tactic."""
    project = Path(project_path)

    # Create integration file
    integration_content = f"""/-
  Auto-generated Portfolio Integration
  
  This file connects Lean tactics to the Python ML model.
-/

import TacticPortfolio.Portfolio

open TacticPortfolio

-- Configure portfolio to use trained model
def trainedConfig : PortfolioConfig := {{
  modelPath := some "{model_path}"
  useML := true
  logAttempts := false
  maxAttempts := 5
}}

-- Macro for easy usage
macro "ml_auto" : tactic => `(tactic| portfolio)

-- Export training data from this file
@[export_training_data]
def collectTrainingData : IO Unit := do
  -- This would collect successful proof steps
  -- and export them for model retraining
  exportStats "training_data.json"
"""

    integration_file = project / "MLIntegration.lean"
    integration_file.write_text(integration_content)

    print(f"Created Lean integration at {integration_file}")

    # Create README
    readme_content = f"""# Tactic Portfolio Integration

This project uses ML-based tactic selection trained on mathlib4.

## Usage

In your Lean files:

```lean
import MLIntegration

-- Use ml_auto for automatic tactic selection
example (x : Nat) : x + 0 = x := by ml_auto

-- Or use portfolio with custom config
example (a b : Real) : a * (b + c) = a * b + a * c := by
  portfolio trainedConfig
```

## Model Information

- Model path: {model_path}
- Supported tactics: simp, ring, linarith, norm_num, field_simp, abel, omega, tauto, aesop, exact
- Training data: mathlib4

## Retraining

To update the model with new examples:

```bash
python -m simpulse.portfolio.train --data training_data.json --model {model_path}
```
"""

    readme_file = project / "PORTFOLIO_README.md"
    readme_file.write_text(readme_content)

    print(f"Created README at {readme_file}")


if __name__ == "__main__":
    # Example: Train from mathlib4
    import sys

    if len(sys.argv) > 1:
        mathlib_path = sys.argv[1]
        predictor = train_from_mathlib(mathlib_path)

        # Test the trained model
        test_goals = [
            "⊢ x + 0 = x",
            "⊢ (a + b)^2 = a^2 + 2*a*b + b^2",
            "⊢ x < x + 1",
        ]

        print("\nTesting trained model:")
        for goal in test_goals:
            pred = predictor.predict(goal)
            print(f"\nGoal: {goal}")
            print(f"Predicted: {pred.tactic} ({pred.confidence:.2f})")
    else:
        print("Usage: python lean_interface.py <mathlib4_path>")

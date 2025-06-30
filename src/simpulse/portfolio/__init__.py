"""
Portfolio approach for Lean tactics.

ML-based tactic selection for optimal proof automation.
"""

from .feature_extractor import GoalFeatures, LeanGoalParser, extract_features

# ML components are optional - only import if available
try:
    from .lean_interface import LeanPortfolioInterface, train_from_mathlib
    from .tactic_predictor import TacticDataset, TacticPrediction, TacticPredictor

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    TacticPredictor = None
    TacticPrediction = None
    TacticDataset = None
    LeanPortfolioInterface = None
    train_from_mathlib = None

__all__ = [
    "GoalFeatures",
    "extract_features",
    "LeanGoalParser",
    "ML_AVAILABLE",
]

if ML_AVAILABLE:
    __all__.extend(
        [
            "TacticPredictor",
            "TacticPrediction",
            "TacticDataset",
            "LeanPortfolioInterface",
            "train_from_mathlib",
        ]
    )

"""
SimpNG (Simp Next Generation) - Revolutionary ML-powered simplification.

This module implements a complete reimagining of Lean's simp tactic using
transformer-based embeddings and neural proof search.
"""

from .core import SimpNGConfig, SimpNGEngine
from .embeddings import GoalEmbedder, RuleEmbedder
from .learning import SelfLearningSystem
from .search import NeuralProofSearch

__all__ = [
    "SimpNGEngine",
    "SimpNGConfig",
    "RuleEmbedder",
    "GoalEmbedder",
    "NeuralProofSearch",
    "SelfLearningSystem",
]

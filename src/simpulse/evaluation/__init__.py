"""Evaluation module for Simpulse.

This module provides fitness evaluation and performance measurement
capabilities for simp rule optimization.
"""

from .fitness_evaluator import Candidate, FitnessEvaluator, FitnessScore

__all__ = [
    'FitnessEvaluator',
    'FitnessScore', 
    'Candidate'
]
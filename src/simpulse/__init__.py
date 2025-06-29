"""Simpulse - ML-powered simp rule optimization for Lean 4.

Simpulse automatically optimizes the performance of Lean 4's simp tactic
by intelligently reordering simplification rule priorities.
"""

__version__ = "1.0.0"
__author__ = "Bright Liu"
__email__ = "bright.liu@example.com"

from .analysis.health_checker import HealthChecker
from .optimization.optimizer import SimpOptimizer
from .profiling.benchmarker import Benchmarker

__all__ = ["SimpOptimizer", "HealthChecker", "Benchmarker"]


# Convenience function
def optimize_project(project_path, strategy="balanced"):
    """Convenience function to optimize a project.

    Args:
        project_path: Path to Lean 4 project
        strategy: Optimization strategy (conservative, balanced, aggressive)

    Returns:
        OptimizationResult with proposed changes
    """
    optimizer = SimpOptimizer(strategy=strategy)
    analysis = optimizer.analyze(project_path)
    return optimizer.optimize(analysis)

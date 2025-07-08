"""Simpulse - Simple optimizer for Lean 4 simp rules.

Direct, no-nonsense optimization without the complexity.
"""

__version__ = "2.0.0"
__author__ = "Bright Liu"
__email__ = "bright.liu@example.com"

# The only thing that matters
from .unified_optimizer import Change, Rule, UnifiedOptimizer

__all__ = [
    "UnifiedOptimizer",
    "Rule",
    "Change",
    "optimize_project",
]


def optimize_project(project_path, strategy="frequency", apply=False):
    """Simple convenience function to optimize a project.

    Args:
        project_path: Path to Lean 4 project
        strategy: One of "frequency", "balanced", "conservative"
        apply: Whether to apply changes immediately

    Returns:
        Dictionary with optimization results
    """
    optimizer = UnifiedOptimizer(strategy=strategy)
    return optimizer.optimize(project_path, apply=apply)

"""Simpulse - High-performance optimization tool for Lean 4 simp tactics.

Simpulse analyzes and optimizes simp rule priorities in Lean 4 projects,
delivering measurable performance improvements for theorem proving workflows.
"""

__version__ = "1.1.0"
__author__ = "Bright Liu"
__email__ = "bright.liu@example.com"

# Core modules
from pathlib import Path

from . import analyzer, optimizer, validator
from .analyzer import LeanAnalyzer, LeanFileAnalysis, SimpRule
from .optimizer import OptimizationSuggestion, PriorityOptimizer
from .validator import OptimizationValidator

# Legacy modules (for backward compatibility)
try:
    from .analysis.health_checker import HealthChecker
    from .optimization.optimizer import SimpOptimizer
    from .profiling.benchmarker import Benchmarker
    from .reporting.report_generator import PerformanceReporter
except ImportError:
    # Handle missing legacy modules gracefully
    HealthChecker = None
    SimpOptimizer = None
    Benchmarker = None
    PerformanceReporter = None

__all__ = [
    "LeanAnalyzer",
    "LeanFileAnalysis",
    "OptimizationSuggestion",
    "OptimizationValidator",
    "PerformanceReporter",
    "PriorityOptimizer",
    "SimpRule",
    "analyzer",
    "optimizer",
    "validator",
]

# Include legacy classes if available
if HealthChecker:
    __all__.extend(["Benchmarker", "HealthChecker", "SimpOptimizer"])


# Convenience function
def optimize_project(project_path: Path, strategy: str = "balanced"):
    """Convenience function to optimize a project.

    Args:
        project_path: Path to Lean 4 project
        strategy: Optimization strategy (conservative, balanced, aggressive)

    Returns:
        List of OptimizationSuggestion objects
    """
    analyzer_obj = LeanAnalyzer()
    analysis = analyzer_obj.analyze_project(project_path)

    optimizer_obj = PriorityOptimizer()
    return optimizer_obj.optimize_project(analysis)

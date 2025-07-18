"""Simpulse 2.0 - Advanced Lean 4 simp optimization using real diagnostic data.

Evidence-based optimization with performance validation, powered by Lean 4.8.0+ diagnostics.
"""

__version__ = "2.0.0"
__author__ = "Bright Liu"
__email__ = "bright.liu@example.com"

# Advanced optimization engine
from .advanced_optimizer import AdvancedOptimizationResult, AdvancedSimpOptimizer
from .diagnostic_parser import DiagnosticAnalysis, SimpTheoremUsage
from .lake_integration import HybridDiagnosticCollector, LakeIntegration
from .optimization_engine import OptimizationPlan, OptimizationRecommendation, OptimizationType
from .performance_measurement import PerformanceComparison, PerformanceReport

__all__ = [
    "AdvancedOptimizationResult",
    "AdvancedSimpOptimizer",
    "DiagnosticAnalysis",
    "HybridDiagnosticCollector",
    "LakeIntegration",
    "OptimizationPlan",
    "OptimizationRecommendation",
    "OptimizationType",
    "PerformanceComparison",
    "PerformanceReport",
    "SimpTheoremUsage",
    "optimize_project",
]


def optimize_project(project_path,
                    confidence_threshold=70.0,
                    validate_performance=True,
                    min_improvement_percent=5.0):
    """Advanced optimization function using real diagnostic data.

    Args:
        project_path: Path to Lean 4 project
        confidence_threshold: Minimum confidence for applying optimizations (0-100)
        validate_performance: Whether to validate improvements with actual measurements
        min_improvement_percent: Minimum improvement required for validation

    Returns:
        AdvancedOptimizationResult with comprehensive analysis and validation

    Raises:
        OptimizationError: If optimization fails
    """
    try:
        optimizer = AdvancedSimpOptimizer(project_path)
        return optimizer.optimize(
            confidence_threshold=confidence_threshold,
            validate_performance=validate_performance,
            min_improvement_percent=min_improvement_percent
        )
    except Exception as e:
        import logging

        from .error import handle_error

        error_msg = handle_error(e, debug=False)
        logging.error(f"Advanced optimization failed: {error_msg}")
        raise

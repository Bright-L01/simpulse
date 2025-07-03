"""
Simpulse Validator Module

Provides correctness validation for Lean optimizations.
"""

from .correctness import CorrectnessValidator, ValidationResult

# Import OptimizationValidator from parent module for backward compatibility
try:
    from ..validator import OptimizationValidator
except ImportError:
    OptimizationValidator = None

__all__ = ["CorrectnessValidator", "ValidationResult"]

# Add OptimizationValidator if available
if OptimizationValidator:
    __all__.append("OptimizationValidator")

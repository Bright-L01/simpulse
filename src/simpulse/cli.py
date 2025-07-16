"""
Advanced CLI for Simpulse 2.0

Professional command-line interface using evidence-based optimization
with real diagnostic data from Lean 4.8.0+.
"""

import sys
from .advanced_cli import main

# Re-export main function for backward compatibility
__all__ = ["main"]

if __name__ == "__main__":
    sys.exit(main())
"""
Simple configuration using environment variables.
No frameworks, no complexity. Just environment variables.
"""

import os
from pathlib import Path

# Core executables
LEAN_PATH = os.environ.get("LEAN_PATH", "lean")
LAKE_PATH = os.environ.get("LAKE_PATH", "lake")

# Directories
CACHE_DIR = Path(os.environ.get("SIMPULSE_CACHE", "./cache"))
LOG_DIR = Path(os.environ.get("SIMPULSE_LOG_DIR", "./logs"))

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Lean project exclusions
LEAN_EXCLUDE_DIRS = ["lake-packages", ".lake", "build", "_target", "__pycache__"]

# Timeouts (in seconds)
LEAN_TIMEOUT = int(os.environ.get("LEAN_TIMEOUT", "300"))
BUILD_TIMEOUT = int(os.environ.get("BUILD_TIMEOUT", "600"))

# Performance settings
MAX_PARALLEL_JOBS = int(os.environ.get("SIMPULSE_JOBS", "4"))
BENCHMARK_RUNS = int(os.environ.get("BENCHMARK_RUNS", "3"))

# Development settings
DEBUG = os.environ.get("SIMPULSE_DEBUG", "false").lower() == "true"


def ensure_dirs():
    """Ensure required directories exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_lean_command():
    """Get the lean command to use."""
    return LEAN_PATH


def get_lake_command():
    """Get the lake command to use."""
    return LAKE_PATH


def should_skip_file(file_path: Path) -> bool:
    """Check if a file should be skipped based on exclusion patterns."""
    file_str = str(file_path)
    return any(exclude in file_str for exclude in LEAN_EXCLUDE_DIRS)

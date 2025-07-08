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
try:
    LEAN_TIMEOUT = int(os.environ.get("LEAN_TIMEOUT", "300"))
except ValueError:
    LEAN_TIMEOUT = 300

try:
    BUILD_TIMEOUT = int(os.environ.get("BUILD_TIMEOUT", "600"))
except ValueError:
    BUILD_TIMEOUT = 600

# Performance settings
try:
    MAX_PARALLEL_JOBS = int(os.environ.get("SIMPULSE_JOBS", "4"))
except ValueError:
    MAX_PARALLEL_JOBS = 4

try:
    BENCHMARK_RUNS = int(os.environ.get("BENCHMARK_RUNS", "3"))
except ValueError:
    BENCHMARK_RUNS = 3

# Development settings
DEBUG = os.environ.get("SIMPULSE_DEBUG", "false").lower() == "true"

# Safety limits
try:
    MAX_FILE_SIZE = int(os.environ.get("SIMPULSE_MAX_FILE_SIZE", "1000000"))  # 1MB default
except ValueError:
    MAX_FILE_SIZE = 1_000_000

try:
    MAX_MEMORY_USAGE = int(os.environ.get("SIMPULSE_MAX_MEMORY", "1000000000"))  # 1GB default
except ValueError:
    MAX_MEMORY_USAGE = 1_000_000_000

try:
    OPTIMIZATION_TIMEOUT = int(os.environ.get("SIMPULSE_TIMEOUT", "30"))  # 30 seconds default
except ValueError:
    OPTIMIZATION_TIMEOUT = 30


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

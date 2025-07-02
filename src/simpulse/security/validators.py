"""
Security validators and sanitizers for Simpulse.

This module provides security functions for input validation,
path sanitization, and safe subprocess execution.
"""

import os
import re
import shlex
from pathlib import Path
from typing import Any


def is_safe_path(path: Path, base_dir: Path | None = None) -> bool:
    """Validate that a path is safe and doesn't escape the base directory.

    Args:
        path: Path to validate
        base_dir: Base directory to restrict to (default: current directory)

    Returns:
        True if path is safe, False otherwise
    """
    try:
        # Resolve to absolute path
        abs_path = path.resolve()

        # Set base directory
        if base_dir is None:
            base_dir = Path.cwd()
        abs_base = base_dir.resolve()

        # Check if path is within base directory
        try:
            abs_path.relative_to(abs_base)

            # Additional checks
            if abs_path.parts[0] == "/" and str(abs_base) != "/":
                return False  # Absolute path outside base

            # Check for suspicious patterns
            str_path = str(path)
            if any(
                pattern in str_path for pattern in ["..", "~/", "/etc/", "/usr/", "/bin/", "/sbin/"]
            ):
                return False

            return True

        except ValueError:
            # Path is not relative to base directory
            return False

    except Exception:
        # Any error in path resolution is treated as unsafe
        return False


def sanitize_shell_arg(arg: str) -> str:
    """Sanitize a shell argument to prevent injection.

    Args:
        arg: Argument to sanitize

    Returns:
        Safely quoted argument
    """
    # Use shlex.quote for proper shell escaping
    return shlex.quote(arg)


def validate_module_name(name: str) -> bool:
    """Validate a Lean module name.

    Args:
        name: Module name to validate

    Returns:
        True if valid module name
    """
    if not name:
        return False

    # Module names should only contain alphanumeric, dots, and underscores
    pattern = r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)*$"
    return bool(re.match(pattern, name))


def validate_file_path(path: str | Path) -> Path:
    """Validate and sanitize a file path.

    Args:
        path: Path to validate

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid or unsafe
    """
    if isinstance(path, str):
        path = Path(path)

    # Check if path is safe
    if not is_safe_path(path):
        raise ValueError(f"Invalid file path: {path}")

    # Check file extension for Lean files
    if path.suffix not in [".lean", ".json", ".toml", ".yml", ".yaml", ".md", ".txt"]:
        raise ValueError(f"Invalid file type: {path.suffix}")

    return path


def validate_command_args(args: list[str]) -> list[str]:
    """Validate command arguments for subprocess execution.

    Args:
        args: Command arguments to validate

    Returns:
        Validated arguments

    Raises:
        ValueError: If arguments contain dangerous patterns
    """
    validated = []

    # Dangerous patterns that should not appear in arguments
    dangerous_patterns = [
        r"[;&|`$]",  # Shell metacharacters
        r"\$\(",  # Command substitution
        r"`.*`",  # Backtick substitution
        r">\s*/",  # Redirect to system paths
        r"<\s*/",  # Read from system paths
    ]

    for arg in args:
        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if re.search(pattern, arg):
                raise ValueError(f"Dangerous pattern in argument: {arg}")

        # Additional validation for specific arguments
        if arg.startswith("-"):
            # Validate flags
            if arg in ["--help", "--version", "--stats", "--profile"]:
                validated.append(arg)
            elif arg.startswith("--timeout="):
                # Validate timeout value
                try:
                    timeout = int(arg.split("=")[1])
                    if 0 < timeout < 3600:  # Max 1 hour
                        validated.append(arg)
                except (IndexError, ValueError):
                    raise ValueError(f"Invalid timeout argument: {arg}")
            else:
                # Unknown flag - sanitize
                validated.append(sanitize_shell_arg(arg))
        else:
            # Regular argument - validate as path or module name
            if "/" in arg or "\\" in arg or arg.endswith(".lean"):
                # Looks like a path
                try:
                    validated_path = validate_file_path(arg)
                    validated.append(str(validated_path))
                except ValueError:
                    # Not a valid path - sanitize as string
                    validated.append(sanitize_shell_arg(arg))
            elif "." in arg and validate_module_name(arg):
                # Looks like a module name
                validated.append(arg)
            else:
                # Generic argument - sanitize
                validated.append(sanitize_shell_arg(arg))

    return validated


def validate_json_structure(data: Any) -> bool:
    """Validate JSON structure for dangerous patterns.

    Args:
        data: JSON data to validate

    Returns:
        True if structure is safe
    """
    dangerous_keys = [
        "__proto__",
        "constructor",
        "prototype",
        "$where",
        "$ne",
        "$gt",
        "$lt",
        "$gte",
        "$lte",
        "$in",
        "$nin",
        "$exists",
        "$regex",
    ]

    def check_dict(d: dict[str, Any]) -> bool:
        """Recursively check dictionary for dangerous keys."""
        for key, value in d.items():
            if key in dangerous_keys:
                return False
            if isinstance(value, dict):
                if not check_dict(value):
                    return False
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and not check_dict(item):
                        return False
        return True

    if isinstance(data, dict):
        return check_dict(data)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and not check_dict(item):
                return False

    return True


def sanitize_json_input(json_str: str) -> str:
    """Sanitize JSON string input.

    Args:
        json_str: JSON string to sanitize

    Returns:
        Sanitized JSON string
    """
    # Remove dangerous patterns
    dangerous_patterns = [
        (r"__proto__", "proto"),
        (r"\$where", "where"),
        (r"\$ne", "ne"),
        (r"constructor\s*:", "construct:"),
        (r"prototype\s*:", "proto:"),
    ]

    sanitized = json_str
    for pattern, replacement in dangerous_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    return sanitized


def validate_file_size(path: Path, max_size_mb: int = 50) -> None:
    """Validate file size is within limits.

    Args:
        path: Path to file
        max_size_mb: Maximum size in megabytes

    Raises:
        ValueError: If file is too large
    """
    if not path.exists():
        raise ValueError(f"File does not exist: {path}")

    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    if size_mb > max_size_mb:
        raise ValueError(f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)")


def validate_api_key(key: str) -> None:
    """Validate API key format.

    Args:
        key: API key to validate

    Raises:
        ValueError: If key is invalid
    """
    if not key or not key.strip():
        raise ValueError("Invalid API key: empty")

    # Check length
    if len(key) < 10 or len(key) > 500:
        raise ValueError("Invalid API key: incorrect length")

    # Check for invalid characters
    if re.search(r"[\s;|`$<>]", key):
        raise ValueError("Invalid API key: contains invalid characters")

    # Check format (basic pattern)
    if not re.match(r"^[A-Za-z0-9_\-\.]+$", key):
        raise ValueError("Invalid API key: invalid format")


def get_safe_env_vars() -> dict[str, str]:
    """Get environment variables with sensitive values masked.

    Returns:
        Dictionary of environment variables with secrets masked
    """
    sensitive_patterns = [
        "TOKEN",
        "KEY",
        "SECRET",
        "PASSWORD",
        "CREDENTIAL",
        "AUTH",
        "PRIVATE",
    ]

    safe_env = {}

    for key, value in os.environ.items():
        # Check if this is a sensitive variable
        is_sensitive = any(pattern in key.upper() for pattern in sensitive_patterns)

        if is_sensitive and value:
            # Mask the value
            if len(value) > 8:
                safe_env[key] = value[:4] + "*" * (len(value) - 8) + value[-4:]
            else:
                safe_env[key] = "*" * len(value)
        else:
            safe_env[key] = value

    return safe_env


def is_valid_filename(filename: str) -> bool:
    """Check if filename is valid and safe.

    Args:
        filename: Filename to validate

    Returns:
        True if filename is valid
    """
    if not filename:
        return False

    # Check for null bytes
    if "\x00" in filename:
        return False

    # Check for path separators (should just be filename)
    if "/" in filename or "\\" in filename:
        return False

    # Check for special names
    if filename in [".", "..", "~"]:
        return False

    # Check for control characters
    if any(ord(c) < 32 for c in filename):
        return False

    # Check length
    if len(filename) > 255:
        return False

    return True


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_calls: int = 10, window_seconds: int = 60):
        """Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed in window
            window_seconds: Time window in seconds
        """
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls: list[float] = []

    def check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded.

        Returns:
            True if rate limited, False if OK to proceed
        """
        import time

        now = time.time()

        # Remove old calls outside window
        self.calls = [t for t in self.calls if now - t < self.window_seconds]

        # Check if limit exceeded
        if len(self.calls) >= self.max_calls:
            return True

        # Add current call
        self.calls.append(now)
        return False

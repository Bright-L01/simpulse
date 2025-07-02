"""Lean 4 runner for executing and profiling Lean code.

This module provides an async interface for running Lean 4 commands with
profiling and diagnostic capabilities.
"""

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Import security validators
from ..security.validators import validate_command_args, validate_file_path

logger = logging.getLogger(__name__)


class LeanExecutionMode(Enum):
    """Execution modes for Lean runner."""

    NORMAL = "normal"
    PROFILE = "profile"
    DIAGNOSE = "diagnose"


@dataclass
class LeanResult:
    """Result from a Lean execution."""

    stdout: str
    stderr: str
    returncode: int
    elapsed_time: float
    output_files: dict[str, Path] = None

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.returncode == 0


class LeanRunner:
    """Async runner for Lean 4 commands with profiling support."""

    def __init__(
        self,
        lake_path: str = "lake",
        lean_path: str = "lean",
        working_dir: Path | None = None,
    ):
        """Initialize Lean runner.

        Args:
            lake_path: Path to lake executable
            lean_path: Path to lean executable
            working_dir: Working directory for Lean commands
        """
        self.lake_path = lake_path
        self.lean_path = lean_path
        self.working_dir = working_dir or Path.cwd()
        self._rate_limiter = None  # Initialize if needed

    async def run_lean(
        self,
        file_path: Path,
        trace_flags: list[str] | None = None,
        timeout: float = 300.0,
        mode: LeanExecutionMode = LeanExecutionMode.NORMAL,
        extra_args: list[str] | None = None,
    ) -> LeanResult:
        """Execute Lean on a file with configurable flags.

        Args:
            file_path: Path to Lean file to execute
            trace_flags: List of trace flags to enable (e.g., ["profiler.output"])
            timeout: Execution timeout in seconds
            mode: Execution mode (normal, profile, diagnose)
            extra_args: Additional arguments to pass to Lean

        Returns:
            LeanResult with execution details
        """
        start_time = asyncio.get_event_loop().time()

        # Build command
        cmd = [self.lake_path, "env", self.lean_path]

        # Add trace flags
        if trace_flags:
            for flag in trace_flags:
                if not flag.startswith("trace."):
                    flag = f"trace.{flag}"
                cmd.extend(["-Dtrace." + flag.replace("trace.", "") + "=true"])

        # Add extra arguments
        if extra_args:
            cmd.extend(extra_args)

        # Add file path
        cmd.append(str(file_path))

        logger.debug(f"Executing command: {' '.join(cmd)}")

        # Prepare output files if in profile mode
        output_files = {}
        temp_file_handle = None
        if mode == LeanExecutionMode.PROFILE:
            # Use NamedTemporaryFile for security
            temp_file_handle = tempfile.NamedTemporaryFile(
                suffix=".json", prefix="lean_profile_", delete=False, mode="w"
            )
            temp_file_handle.close()  # Close but keep the file
            profile_output = temp_file_handle.name
            output_files["profile"] = Path(profile_output)
            cmd.extend(["-o", profile_output])

        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )

            # Wait for completion with timeout
            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Lean execution timed out after {timeout} seconds")

            elapsed_time = asyncio.get_event_loop().time() - start_time

            return LeanResult(
                stdout=stdout_data.decode("utf-8", errors="replace"),
                stderr=stderr_data.decode("utf-8", errors="replace"),
                returncode=process.returncode,
                elapsed_time=elapsed_time,
                output_files=output_files,
            )

        except Exception as e:
            logger.error(f"Error executing Lean: {e}")
            raise

    async def profile_module(
        self, module_name: str, options: dict[str, Any] | None = None
    ) -> tuple[LeanResult, dict | None]:
        """Profile a Lean module and return structured results.

        Args:
            module_name: Name of the module to profile (e.g., "Mathlib.Data.List.Basic")
            options: Profiling options including:
                - trace_flags: List of trace flags
                - timeout: Execution timeout
                - parse_output: Whether to parse profiler output

        Returns:
            Tuple of (LeanResult, parsed_profile_data)
        """
        options = options or {}

        # Create temporary file with import statement
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(f"import {module_name}\n")
            temp_file = Path(f.name)

        try:
            # Enable profiler output by default
            trace_flags = options.get("trace_flags", ["profiler.output"])
            if "profiler.output" not in trace_flags:
                trace_flags.append("profiler.output")

            # Run with profiling
            result = await self.run_lean(
                file_path=temp_file,
                trace_flags=trace_flags,
                timeout=options.get("timeout", 300.0),
                mode=LeanExecutionMode.PROFILE,
            )

            # Parse profile output if requested
            parsed_data = None
            if (
                options.get("parse_output", True)
                and result.output_files
                and result.output_files.get("profile")
            ):
                profile_file = result.output_files["profile"]
                if profile_file.exists():
                    try:
                        with open(profile_file) as f:
                            parsed_data = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to parse profile output: {e}")

            return result, parsed_data

        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()

    def get_trace_command(
        self,
        file_path: Path,
        output_file: Path | None = None,
        trace_flags: list[str] | None = None,
    ) -> list[str]:
        """Generate command for running Lean with trace output.

        Args:
            file_path: Path to Lean file
            output_file: Optional path for trace output
            trace_flags: List of trace flags to enable

        Returns:
            Command as list of strings
        """
        cmd = [self.lake_path, "env", self.lean_path]

        # Add trace flags
        if trace_flags:
            for flag in trace_flags:
                if not flag.startswith("trace."):
                    flag = f"trace.{flag}"
                cmd.extend([f"-D{flag}=true"])

        # Add output file if specified
        if output_file:
            cmd.extend(["-o", str(output_file)])

        # Add input file
        cmd.append(str(file_path))

        return cmd

    async def get_simp_diagnostics(
        self, file_path: Path, timeout: float = 300.0
    ) -> tuple[LeanResult, dict | None]:
        """Run Lean with simp diagnostics enabled.

        Args:
            file_path: Path to Lean file
            timeout: Execution timeout

        Returns:
            Tuple of (LeanResult, parsed_diagnostics)
        """
        # Enable simp trace flags for diagnostics
        trace_flags = [
            "Meta.Tactic.simp",
            "Meta.Tactic.simp.discharge",
            "Meta.Tactic.simp.unify",
            "Meta.Tactic.simp.rewrite",
        ]

        result = await self.run_lean(
            file_path=file_path,
            trace_flags=trace_flags,
            timeout=timeout,
            mode=LeanExecutionMode.DIAGNOSE,
        )

        # Parse diagnostics from stderr
        diagnostics = self._parse_simp_diagnostics(result.stderr)

        return result, diagnostics

    def _parse_simp_diagnostics(self, stderr: str) -> dict | None:
        """Parse simp diagnostics from stderr output.

        Args:
            stderr: Stderr output containing trace messages

        Returns:
            Parsed diagnostics dictionary
        """
        diagnostics = {
            "rewrites": [],
            "discharges": [],
            "unifications": [],
            "failures": [],
        }

        for line in stderr.split("\n"):
            if "[Meta.Tactic.simp.rewrite]" in line:
                diagnostics["rewrites"].append(line)
            elif "[Meta.Tactic.simp.discharge]" in line:
                diagnostics["discharges"].append(line)
            elif "[Meta.Tactic.simp.unify]" in line:
                diagnostics["unifications"].append(line)
            elif "failed" in line.lower() and "simp" in line.lower():
                diagnostics["failures"].append(line)

        return diagnostics

    def _validate_file_path(self, path: Path) -> None:
        """Validate file path for security.

        Args:
            path: Path to validate

        Raises:
            ValueError: If path is invalid or unsafe
        """
        # Use the security validator
        validate_file_path(path)

        # Additional Lean-specific validation
        if not str(path).endswith(".lean"):
            raise ValueError(f"File must be a Lean file: {path}")

    def _validate_command_args(self, args: list[str]) -> list[str]:
        """Validate and sanitize command arguments.

        Args:
            args: Command arguments to validate

        Returns:
            Validated arguments
        """
        return validate_command_args(args)

    def _validate_file_size(self, path: Path, max_size_mb: int = 50) -> None:
        """Validate file size is within limits.

        Args:
            path: Path to file
            max_size_mb: Maximum size in megabytes

        Raises:
            ValueError: If file is too large
        """
        from ..security.validators import validate_file_size

        validate_file_size(path, max_size_mb)

    def _check_rate_limit(self) -> bool:
        """Check if rate limited.

        Returns:
            True if rate limited, False if OK to proceed
        """
        if self._rate_limiter is None:
            from ..security.validators import RateLimiter

            self._rate_limiter = RateLimiter(max_calls=100, window_seconds=60)

        return self._rate_limiter.check_rate_limit()

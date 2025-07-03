"""
Correctness Validator for Simpulse

This module ensures that optimizations preserve the correctness of Lean proofs.
It applies optimizations incrementally and validates compilation after each change.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..monitoring import monitor_operation

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of applying a single optimization"""

    rule: str
    location: str
    success: bool
    error_message: Optional[str] = None
    compilation_time: Optional[float] = None


@dataclass
class ValidationResult:
    """Complete validation result for a file"""

    file_path: str
    original_compilation_time: float
    optimized_compilation_time: Optional[float] = None
    total_optimizations: int = 0
    successful_optimizations: int = 0
    failed_optimizations: int = 0
    optimization_results: List[OptimizationResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of optimizations"""
        if self.total_optimizations == 0:
            return 0.0
        return self.successful_optimizations / self.total_optimizations

    @property
    def speedup(self) -> float:
        """Calculate the speedup from optimizations"""
        if self.optimized_compilation_time is None:
            return 1.0
        if self.optimized_compilation_time == 0:
            return float("inf")
        return self.original_compilation_time / self.optimized_compilation_time


class CorrectnessValidator:
    """Validates that optimizations preserve Lean proof correctness"""

    def __init__(self, lean_exe: str = "lake", timeout: int = 60):
        """
        Initialize the correctness validator

        Args:
            lean_exe: Path to the Lean executable
            timeout: Timeout for compilation in seconds
        """
        self.lean_exe = lean_exe
        self.timeout = timeout
        self.temp_dir = None

    def _setup_temp_workspace(self, lean_file: Path) -> Path:
        """Create a temporary workspace for testing"""
        self.temp_dir = tempfile.mkdtemp(prefix="simpulse_validation_")
        temp_path = Path(self.temp_dir)

        # Copy the Lean file and any associated project files
        project_root = self._find_project_root(lean_file)
        if project_root:
            # Copy entire project structure
            shutil.copytree(project_root, temp_path / "project", dirs_exist_ok=True)
            temp_file = temp_path / "project" / lean_file.relative_to(project_root)
        else:
            # Just copy the single file
            temp_file = temp_path / lean_file.name
            shutil.copy2(lean_file, temp_file)

        return temp_file

    def _find_project_root(self, lean_file: Path) -> Optional[Path]:
        """Find the Lean project root (contains lakefile.lean)"""
        current = lean_file.parent
        while current != current.parent:
            if (current / "lakefile.lean").exists():
                return current
            current = current.parent
        return None

    def _cleanup_temp_workspace(self):
        """Clean up temporary workspace"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def _run_lake_build(self, work_dir: Path) -> Tuple[bool, float, Optional[str]]:
        """
        Run lake build and check if it succeeds

        Returns:
            Tuple of (success, compilation_time, error_message)
        """
        start_time = datetime.now()
        try:
            result = subprocess.run(
                [self.lean_exe, "build"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            compilation_time = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                return True, compilation_time, None
            else:
                error_msg = result.stderr or result.stdout
                return False, compilation_time, error_msg

        except subprocess.TimeoutExpired:
            compilation_time = (datetime.now() - start_time).total_seconds()
            return False, compilation_time, "Compilation timeout"
        except Exception as e:
            compilation_time = (datetime.now() - start_time).total_seconds()
            return False, compilation_time, str(e)

    def _apply_optimization(self, file_path: Path, optimization: Dict[str, Any]) -> bool:
        """
        Apply a single optimization to the file

        Args:
            file_path: Path to the Lean file
            optimization: Optimization dictionary with 'location' and 'replacement'

        Returns:
            True if successfully applied, False otherwise
        """
        try:
            with open(file_path) as f:
                content = f.read()

            # Apply the optimization
            lines = content.split("\n")
            line_num = optimization.get("line", 0) - 1  # Convert to 0-based

            if 0 <= line_num < len(lines):
                original_line = lines[line_num]

                # Simple replacement based on the optimization
                if "original" in optimization and "replacement" in optimization:
                    lines[line_num] = original_line.replace(
                        optimization["original"], optimization["replacement"]
                    )
                elif "new_line" in optimization:
                    lines[line_num] = optimization["new_line"]
                else:
                    return False

                # Write back the modified content
                with open(file_path, "w") as f:
                    f.write("\n".join(lines))

                return True
            else:
                logger.warning(f"Invalid line number: {line_num + 1}")
                return False

        except Exception as e:
            logger.error(f"Error applying optimization: {e}")
            return False

    @monitor_operation
    def validate_file(
        self, lean_file: Path, optimizations: List[Dict[str, Any]]
    ) -> ValidationResult:
        """
        Validate optimizations on a single Lean file

        Args:
            lean_file: Path to the Lean file
            optimizations: List of optimization suggestions

        Returns:
            ValidationResult with detailed information
        """
        result = ValidationResult(file_path=str(lean_file))

        try:
            # Setup temporary workspace
            temp_file = self._setup_temp_workspace(lean_file)
            work_dir = temp_file.parent if self._find_project_root(lean_file) else temp_file.parent

            # Test original compilation
            logger.info(f"Testing original compilation for {lean_file}")
            success, orig_time, error = self._run_lake_build(work_dir)

            if not success:
                result.error = f"Original file failed to compile: {error}"
                logger.error(result.error)
                return result

            result.original_compilation_time = orig_time
            logger.info(f"Original compilation time: {orig_time:.2f}s")

            # Create backup of original file
            backup_content = temp_file.read_text()

            # Apply optimizations incrementally
            successful_optimizations = []

            for i, optimization in enumerate(optimizations):
                logger.info(f"Testing optimization {i+1}/{len(optimizations)}")
                result.total_optimizations += 1

                # Apply the optimization
                if self._apply_optimization(temp_file, optimization):
                    # Test compilation with this optimization
                    success, comp_time, error = self._run_lake_build(work_dir)

                    opt_result = OptimizationResult(
                        rule=optimization.get("rule", "unknown"),
                        location=optimization.get("location", "unknown"),
                        success=success,
                        error_message=error if not success else None,
                        compilation_time=comp_time,
                    )

                    result.optimization_results.append(opt_result)

                    if success:
                        result.successful_optimizations += 1
                        successful_optimizations.append(optimization)
                        logger.info(f"Optimization {i+1} succeeded")
                    else:
                        result.failed_optimizations += 1
                        # Rollback this optimization
                        temp_file.write_text(backup_content)
                        logger.warning(f"Optimization {i+1} failed, rolling back")

                        # Re-apply successful optimizations
                        for prev_opt in successful_optimizations:
                            self._apply_optimization(temp_file, prev_opt)
                else:
                    result.failed_optimizations += 1
                    logger.warning(f"Failed to apply optimization {i+1}")

            # Final compilation time with successful optimizations
            if result.successful_optimizations > 0:
                success, final_time, _ = self._run_lake_build(work_dir)
                if success:
                    result.optimized_compilation_time = final_time
                    logger.info(
                        f"Final compilation time: {final_time:.2f}s (speedup: {result.speedup:.2f}x)"
                    )

        except Exception as e:
            result.error = str(e)
            logger.error(f"Validation error: {e}")

        finally:
            self._cleanup_temp_workspace()

        return result

    @monitor_operation
    def validate_batch(
        self, files_and_optimizations: List[Tuple[Path, List[Dict[str, Any]]]]
    ) -> Dict[str, Any]:
        """
        Validate optimizations on multiple files

        Args:
            files_and_optimizations: List of (file_path, optimizations) tuples

        Returns:
            Summary report dictionary
        """
        results = []
        total_files = len(files_and_optimizations)
        successful_files = 0
        total_optimizations = 0
        successful_optimizations = 0

        for i, (file_path, optimizations) in enumerate(files_and_optimizations):
            logger.info(f"Validating file {i+1}/{total_files}: {file_path}")

            result = self.validate_file(file_path, optimizations)
            results.append(result)

            if result.successful_optimizations > 0:
                successful_files += 1

            total_optimizations += result.total_optimizations
            successful_optimizations += result.successful_optimizations

        # Generate summary report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_files_tested": total_files,
            "files_successfully_optimized": successful_files,
            "files_preserved_correctness": sum(1 for r in results if r.error is None),
            "total_optimizations_attempted": total_optimizations,
            "successful_optimizations": successful_optimizations,
            "overall_success_rate": (
                successful_optimizations / total_optimizations if total_optimizations > 0 else 0.0
            ),
            "average_speedup": (
                sum(r.speedup for r in results if r.optimized_compilation_time) / len(results)
                if results
                else 1.0
            ),
            "file_results": [
                {
                    "file": r.file_path,
                    "success_rate": r.success_rate,
                    "speedup": r.speedup,
                    "successful_optimizations": r.successful_optimizations,
                    "failed_optimizations": r.failed_optimizations,
                    "error": r.error,
                }
                for r in results
            ],
            "failed_optimizations": [
                {
                    "file": r.file_path,
                    "optimization": opt.rule,
                    "location": opt.location,
                    "error": opt.error_message,
                }
                for r in results
                for opt in r.optimization_results
                if not opt.success
            ],
        }

        return report

    def generate_safety_report(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Generate a report categorizing optimizations by safety

        Args:
            validation_results: List of validation results

        Returns:
            Safety report dictionary
        """
        safe_rules = {}
        unsafe_rules = {}

        for result in validation_results:
            for opt_result in result.optimization_results:
                rule = opt_result.rule

                if opt_result.success:
                    safe_rules[rule] = safe_rules.get(rule, 0) + 1
                else:
                    unsafe_rules[rule] = unsafe_rules.get(rule, 0) + 1

        # Calculate safety scores
        safety_scores = {}
        for rule in set(safe_rules.keys()) | set(unsafe_rules.keys()):
            safe_count = safe_rules.get(rule, 0)
            unsafe_count = unsafe_rules.get(rule, 0)
            total = safe_count + unsafe_count

            safety_scores[rule] = {
                "safe_applications": safe_count,
                "unsafe_applications": unsafe_count,
                "total_applications": total,
                "safety_rate": safe_count / total if total > 0 else 0.0,
                "recommendation": (
                    "SAFE"
                    if unsafe_count == 0
                    else ("UNSAFE" if safe_count == 0 else "CONDITIONAL")
                ),
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "total_rules_tested": len(safety_scores),
            "safe_rules": [
                rule for rule, score in safety_scores.items() if score["recommendation"] == "SAFE"
            ],
            "unsafe_rules": [
                rule for rule, score in safety_scores.items() if score["recommendation"] == "UNSAFE"
            ],
            "conditional_rules": [
                rule
                for rule, score in safety_scores.items()
                if score["recommendation"] == "CONDITIONAL"
            ],
            "rule_safety_scores": safety_scores,
        }


def create_validator(lean_exe: str = "lake", timeout: int = 60) -> CorrectnessValidator:
    """Factory function to create a CorrectnessValidator instance"""
    return CorrectnessValidator(lean_exe=lean_exe, timeout=timeout)

"""
Advanced Lean 4 Simp Optimizer

Replaces the crude pattern-matching approach with sophisticated diagnostic analysis
using real performance data from Lean 4.8.0+ diagnostics infrastructure.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .diagnostic_parser import DiagnosticAnalysis
from .error import OptimizationError
from .lake_integration import HybridDiagnosticCollector
from .optimization_engine import OptimizationEngine, OptimizationPlan
from .performance_measurement import (
    OptimizationValidator,
    PerformanceComparison,
    PerformanceMeasurer,
)

logger = logging.getLogger(__name__)


@dataclass
class AdvancedOptimizationResult:
    """Results from advanced optimization with real performance validation."""

    project_path: str
    analysis: DiagnosticAnalysis
    optimization_plan: OptimizationPlan
    performance_comparison: PerformanceComparison | None = None
    applied_recommendations: int = 0
    failed_recommendations: int = 0
    total_analysis_time: float = 0.0
    total_optimization_time: float = 0.0
    validation_passed: bool = False

    @property
    def success_rate(self) -> float:
        """Calculate success rate of applied recommendations."""
        total = self.applied_recommendations + self.failed_recommendations
        if total == 0:
            return 0.0
        return self.applied_recommendations / total

    @property
    def actual_improvement_percent(self) -> float:
        """Get actual measured performance improvement."""
        if self.performance_comparison is None:
            return 0.0
        return self.performance_comparison.time_improvement_percent

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Advanced Simp Optimization Results:",
            f"  Project: {self.project_path}",
            f"  Simp theorems analyzed: {len(self.analysis.simp_theorems)}",
            f"  Recommendations generated: {self.optimization_plan.total_recommendations}",
            f"    High confidence: {len(self.optimization_plan.high_confidence)}",
            f"    Medium confidence: {len(self.optimization_plan.medium_confidence)}",
            f"    Low confidence: {len(self.optimization_plan.low_confidence)}",
        ]

        if self.applied_recommendations > 0 or self.failed_recommendations > 0:
            lines.extend(
                [
                    f"  Applied recommendations: {self.applied_recommendations}",
                    f"  Failed recommendations: {self.failed_recommendations}",
                    f"  Success rate: {self.success_rate:.1%}",
                ]
            )

        if self.performance_comparison:
            lines.extend(
                [
                    f"  Performance validation: {'✓ PASSED' if self.validation_passed else '✗ FAILED'}",
                    f"  Actual improvement: {self.actual_improvement_percent:+.1f}%",
                ]
            )

        lines.extend(
            [
                f"  Analysis time: {self.total_analysis_time:.1f}s",
                f"  Optimization time: {self.total_optimization_time:.1f}s",
            ]
        )

        return "\n".join(lines)


class AdvancedSimpOptimizer:
    """
    Advanced simp optimizer using real Lean 4.8.0+ diagnostic data.

    Replaces theoretical estimates with evidence-based optimization
    and performance validation.
    """

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.collector = HybridDiagnosticCollector(self.project_path)
        self.engine = OptimizationEngine(self.project_path)
        self.validator = OptimizationValidator(self.project_path)

        # Validate project path
        if not self.project_path.exists():
            raise OptimizationError(f"Project path does not exist: {self.project_path}")
        if not self.project_path.is_dir():
            raise OptimizationError(f"Project path is not a directory: {self.project_path}")

        # Check for Lean files
        lean_files = list(self.project_path.glob("**/*.lean"))
        if not lean_files:
            raise OptimizationError(f"No Lean files found in {self.project_path}")

        logger.info(f"Initialized advanced optimizer for {len(lean_files)} Lean files")

    def analyze(
        self, files: list[Path] | None = None, max_files: int | None = None
    ) -> AdvancedOptimizationResult:
        """
        Analyze project using real diagnostic data.

        Args:
            files: Specific files to analyze (None for all)
            max_files: Maximum number of files to analyze

        Returns:
            Complete analysis results with optimization recommendations
        """
        start_time = time.time()

        logger.info("Starting advanced diagnostic analysis...")

        try:
            # Step 1: Collect comprehensive analysis using Lake integration + fallback
            logger.info("Collecting comprehensive analysis (Lake + pattern-based)...")
            analysis = self.collector.collect_comprehensive_analysis()

            if not analysis.simp_theorems:
                logger.warning("No simp theorem usage data collected")
                return AdvancedOptimizationResult(
                    project_path=str(self.project_path),
                    analysis=analysis,
                    optimization_plan=OptimizationPlan(),
                    total_analysis_time=time.time() - start_time,
                )

            logger.info(f"Collected data for {len(analysis.simp_theorems)} simp theorems")

            # Step 2: Generate evidence-based optimization recommendations
            logger.info("Generating optimization recommendations...")
            optimization_plan = self.engine.analyze_and_recommend(analysis)

            analysis_time = time.time() - start_time

            result = AdvancedOptimizationResult(
                project_path=str(self.project_path),
                analysis=analysis,
                optimization_plan=optimization_plan,
                total_analysis_time=analysis_time,
            )

            logger.info(f"Analysis complete in {analysis_time:.1f}s")
            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise OptimizationError(f"Advanced analysis failed: {e}") from e

    def optimize(
        self,
        confidence_threshold: float = 70.0,
        validate_performance: bool = True,
        min_improvement_percent: float = 5.0,
    ) -> AdvancedOptimizationResult:
        """
        Perform complete optimization with performance validation.

        Args:
            confidence_threshold: Minimum confidence for applying recommendations
            validate_performance: Whether to measure actual performance improvement
            min_improvement_percent: Minimum improvement required for validation

        Returns:
            Complete optimization results with performance validation
        """
        # Step 1: Analyze project
        result = self.analyze()

        if result.optimization_plan.total_recommendations == 0:
            logger.info("No optimization recommendations generated")
            return result

        # Step 2: Apply optimizations with performance validation
        if validate_performance:
            result = self._optimize_with_validation(
                result, confidence_threshold, min_improvement_percent
            )
        else:
            result = self._optimize_without_validation(result, confidence_threshold)

        return result

    def _optimize_with_validation(
        self,
        result: AdvancedOptimizationResult,
        confidence_threshold: float,
        min_improvement_percent: float,
    ) -> AdvancedOptimizationResult:
        """Apply optimizations with performance validation."""
        start_time = time.time()

        logger.info("Applying optimizations with performance validation...")

        # Get files that will be modified
        files_to_optimize = list(
            set(
                [
                    rec.file_path
                    for rec in result.optimization_plan.recommendations
                    if rec.evidence_score >= confidence_threshold
                ]
            )
        )

        if not files_to_optimize:
            logger.info("No files to optimize with current confidence threshold")
            result.total_optimization_time = time.time() - start_time
            return result

        logger.info(f"Optimizing {len(files_to_optimize)} files...")

        def apply_optimizations():
            """Apply optimization function for validator."""
            applied, failed = self.engine.apply_plan(result.optimization_plan, confidence_threshold)
            result.applied_recommendations = applied
            result.failed_recommendations = failed

        try:
            # Use validator to apply optimizations and measure performance
            validation_passed, comparison = self.validator.create_backup_and_validate(
                files_to_optimize, apply_optimizations, min_improvement_percent
            )

            result.validation_passed = validation_passed
            result.performance_comparison = comparison
            result.total_optimization_time = time.time() - start_time

            if validation_passed:
                logger.info(f"✓ Optimization validated: {comparison.summary()}")
            else:
                logger.warning(f"✗ Optimization not validated: {comparison.summary()}")
                logger.info("Changes have been reverted")

            return result

        except Exception as e:
            logger.error(f"Optimization with validation failed: {e}")
            result.total_optimization_time = time.time() - start_time
            return result

    def _optimize_without_validation(
        self, result: AdvancedOptimizationResult, confidence_threshold: float
    ) -> AdvancedOptimizationResult:
        """Apply optimizations without performance validation."""
        start_time = time.time()

        logger.info("Applying optimizations without validation...")

        applied, failed = self.engine.apply_plan(result.optimization_plan, confidence_threshold)

        result.applied_recommendations = applied
        result.failed_recommendations = failed
        result.total_optimization_time = time.time() - start_time

        logger.info(f"Applied {applied} recommendations, {failed} failed")
        return result

    def benchmark(
        self, files: list[Path] | None = None, runs_per_file: int = 3
    ) -> dict[str, float]:
        """
        Benchmark current project performance.

        Args:
            files: Specific files to benchmark (None for all)
            runs_per_file: Number of runs per file for accuracy

        Returns:
            Performance metrics dictionary
        """
        logger.info("Benchmarking project performance...")

        measurer = PerformanceMeasurer(self.project_path)
        report = measurer.measure_project(files, runs_per_file)

        return {
            "total_time": report.total_time,
            "average_time": report.average_time,
            "median_time": report.median_time,
            "success_rate": report.success_rate,
            "files_measured": len(report.measurements),
        }

    def get_optimization_preview(self, confidence_threshold: float = 50.0) -> dict:
        """
        Get preview of optimizations without applying them.

        Args:
            confidence_threshold: Minimum confidence for including recommendations

        Returns:
            Preview information dictionary
        """
        result = self.analyze()

        recommendations = [
            rec
            for rec in result.optimization_plan.recommendations
            if rec.evidence_score >= confidence_threshold
        ]

        # Group by optimization type
        by_type = {}
        for rec in recommendations:
            opt_type = rec.optimization_type.value
            if opt_type not in by_type:
                by_type[opt_type] = []
            by_type[opt_type].append(
                {
                    "theorem_name": rec.theorem_name,
                    "file_path": str(rec.file_path),
                    "current_priority": rec.current_priority,
                    "recommended_priority": rec.recommended_priority,
                    "evidence_score": rec.evidence_score,
                    "reason": rec.reason,
                    "expected_impact": rec.expected_impact,
                }
            )

        return {
            "total_recommendations": len(recommendations),
            "confidence_threshold": confidence_threshold,
            "optimization_types": by_type,
            "analysis_summary": {
                "simp_theorems_analyzed": len(result.analysis.simp_theorems),
                "most_used_theorems": [
                    {"name": t.name, "used_count": t.used_count, "success_rate": t.success_rate}
                    for t in result.analysis.get_most_used_theorems(5)
                ],
                "least_efficient_theorems": [
                    {"name": t.name, "tried_count": t.tried_count, "success_rate": t.success_rate}
                    for t in result.analysis.get_least_efficient_theorems(5)
                ],
                "potential_loops": result.analysis.looping_theorems,
            },
        }


# Example usage
if __name__ == "__main__":
    import tempfile

    # Test with a temporary project
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)

        # Create a test file
        test_file = project_path / "test.lean"
        test_file.write_text("""
@[simp]
theorem test_theorem : 1 + 1 = 2 := by simp

theorem another_test : [1, 2].length = 2 := by simp [test_theorem]
""")

        try:
            optimizer = AdvancedSimpOptimizer(str(project_path))
            preview = optimizer.get_optimization_preview()

            print("Optimization preview:")
            print(f"  Total recommendations: {preview['total_recommendations']}")
            print(
                f"  Simp theorems analyzed: {preview['analysis_summary']['simp_theorems_analyzed']}"
            )

            if preview["total_recommendations"] > 0:
                print("✓ Advanced optimizer working correctly")
            else:
                print("i No optimizations recommended (expected for simple test)")

        except Exception as e:
            print(f"✗ Advanced optimizer test failed: {e}")

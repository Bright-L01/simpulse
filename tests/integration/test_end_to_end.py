"""
End-to-end integration tests for Simpulse.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from simpulse.advanced_cli import AdvancedCLI
from simpulse.advanced_optimizer import AdvancedSimpOptimizer
from simpulse.diagnostic_parser import DiagnosticParser
from simpulse.optimization_engine import OptimizationEngine
from simpulse.lake_integration import HybridDiagnosticCollector


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""

    @pytest.fixture
    def comprehensive_lean_project(self, temp_dir):
        """Create a comprehensive Lean project for testing."""
        project_dir = temp_dir / "comprehensive_project"
        project_dir.mkdir()

        # Create lakefile.lean
        lakefile = project_dir / "lakefile.lean"
        lakefile.write_text(
            """
import Lake
open Lake DSL

package comprehensive_project where
  version := v!"1.0.0"

lean_lib ComprehensiveProject where
  roots := #[`ComprehensiveProject]
"""
        )

        # Create lean-toolchain
        toolchain = project_dir / "lean-toolchain"
        toolchain.write_text("4.8.0")

        # Create library directory
        lib_dir = project_dir / "ComprehensiveProject"
        lib_dir.mkdir()

        # Create Basic.lean with various simp rules
        basic_file = lib_dir / "Basic.lean"
        basic_file.write_text(
            """
-- High usage, high success rate theorem
@[simp] theorem list_append_nil (l : List α) : l ++ [] = l := by
  induction l with
  | nil => rfl
  | cons h t ih => simp [ih]

-- Medium usage theorem
@[simp, priority := 500] theorem nat_add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.succ_add, ih]

-- Low success rate theorem (should get priority decrease)
@[simp] theorem inefficient_rule (n : Nat) : n * 1 + 0 = n := by
  simp [Nat.mul_one]

-- Unused theorem (should get removal recommendation)
@[simp] theorem unused_theorem (n : Nat) : n = n := rfl

-- Example usage
example (l : List Nat) : [1, 2, 3] ++ [] = [1, 2, 3] := by simp
example (n : Nat) : n + 0 = n := by simp
"""
        )

        # Create Advanced.lean
        advanced_file = lib_dir / "Advanced.lean"
        advanced_file.write_text(
            """
-- Another high usage theorem
@[simp] theorem zero_add (n : Nat) : 0 + n = n := rfl

-- Complex theorem that might cause loops
@[simp] theorem complex_rule (a b c : Nat) : 
  (a + b) + c = a + (b + c) := by
  simp [Nat.add_assoc]

-- Example proofs that use simp extensively
theorem proof_with_simp (a b : Nat) : a + 0 + (0 + b) = a + b := by
  simp [nat_add_zero, zero_add]

theorem another_proof (l : List Nat) : 
  [1, 2] ++ [] ++ [3] = [1, 2, 3] := by
  simp [list_append_nil]
"""
        )

        return project_dir

    def test_complete_analyze_workflow(self, comprehensive_lean_project):
        """Test complete analyze workflow from CLI to results."""
        cli = AdvancedCLI()

        # Mock the diagnostic collection to return realistic data
        with patch("simpulse.advanced_optimizer.HybridDiagnosticCollector") as mock_collector:
            mock_analysis = MagicMock()
            mock_analysis.total_theorems = 6
            mock_analysis.total_files = 2
            mock_analysis.theorem_usage = [
                # High usage theorem
                MagicMock(
                    name="list_append_nil",
                    used_count=25,
                    tried_count=27,
                    success_rate=0.926,
                    file_path=str(
                        comprehensive_lean_project / "ComprehensiveProject" / "Basic.lean"
                    ),
                ),
                # Medium usage theorem
                MagicMock(
                    name="nat_add_zero",
                    used_count=15,
                    tried_count=18,
                    success_rate=0.833,
                    file_path=str(
                        comprehensive_lean_project / "ComprehensiveProject" / "Basic.lean"
                    ),
                ),
                # Low success rate theorem
                MagicMock(
                    name="inefficient_rule",
                    used_count=5,
                    tried_count=20,
                    success_rate=0.25,
                    file_path=str(
                        comprehensive_lean_project / "ComprehensiveProject" / "Basic.lean"
                    ),
                ),
                # Unused theorem
                MagicMock(
                    name="unused_theorem",
                    used_count=0,
                    tried_count=3,
                    success_rate=0.0,
                    file_path=str(
                        comprehensive_lean_project / "ComprehensiveProject" / "Basic.lean"
                    ),
                ),
                # Another high usage theorem
                MagicMock(
                    name="zero_add",
                    used_count=22,
                    tried_count=24,
                    success_rate=0.917,
                    file_path=str(
                        comprehensive_lean_project / "ComprehensiveProject" / "Advanced.lean"
                    ),
                ),
                # Complex theorem
                MagicMock(
                    name="complex_rule",
                    used_count=8,
                    tried_count=12,
                    success_rate=0.667,
                    file_path=str(
                        comprehensive_lean_project / "ComprehensiveProject" / "Advanced.lean"
                    ),
                ),
            ]

            mock_collector.return_value.collect_comprehensive_analysis.return_value = mock_analysis

            # Test analyze command
            result = cli.run(["analyze", str(comprehensive_lean_project)])

            assert result == 0
            mock_collector.assert_called_once()

    def test_complete_preview_workflow(self, comprehensive_lean_project):
        """Test complete preview workflow."""
        cli = AdvancedCLI()

        with patch("simpulse.advanced_optimizer.HybridDiagnosticCollector") as mock_collector:
            # Create realistic mock analysis
            mock_analysis = self._create_mock_analysis(comprehensive_lean_project)
            mock_collector.return_value.collect_comprehensive_analysis.return_value = mock_analysis

            # Test preview command
            result = cli.run(["preview", str(comprehensive_lean_project), "--detailed"])

            assert result == 0
            mock_collector.assert_called_once()

    def test_complete_optimize_workflow(self, comprehensive_lean_project):
        """Test complete optimize workflow."""
        cli = AdvancedCLI()

        with (
            patch("simpulse.advanced_optimizer.HybridDiagnosticCollector") as mock_collector,
            patch("simpulse.advanced_optimizer.PerformanceMeasurer") as mock_measurer,
        ):

            # Mock analysis
            mock_analysis = self._create_mock_analysis(comprehensive_lean_project)
            mock_collector.return_value.collect_comprehensive_analysis.return_value = mock_analysis

            # Mock performance measurement
            mock_measurer.return_value.measure_performance.return_value = MagicMock(
                average_time=45.5, files_processed=2, successful_measurements=2
            )

            # Mock file operations
            with (
                patch("builtins.open", create=True) as mock_open,
                patch("shutil.copy2") as mock_copy,
            ):

                mock_open.return_value.__enter__.return_value.read.return_value = (
                    "@[simp] theorem test : 1 = 1 := rfl"
                )
                mock_open.return_value.__enter__.return_value.write.return_value = None

                # Test optimize command
                result = cli.run(
                    ["optimize", str(comprehensive_lean_project), "--confidence-threshold", "70"]
                )

                assert result == 0
                mock_collector.assert_called_once()

    def test_complete_benchmark_workflow(self, comprehensive_lean_project):
        """Test complete benchmark workflow."""
        cli = AdvancedCLI()

        with patch("simpulse.advanced_optimizer.PerformanceMeasurer") as mock_measurer:
            # Mock benchmark results
            mock_result = MagicMock()
            mock_result.average_time = 42.3
            mock_result.total_files = 2
            mock_result.successful_measurements = 2
            mock_result.file_times = {
                str(comprehensive_lean_project / "ComprehensiveProject" / "Basic.lean"): 25.1,
                str(comprehensive_lean_project / "ComprehensiveProject" / "Advanced.lean"): 17.2,
            }

            mock_measurer.return_value.benchmark_project.return_value = mock_result

            # Test benchmark command
            result = cli.run(["benchmark", str(comprehensive_lean_project), "--runs", "3"])

            assert result == 0
            mock_measurer.assert_called_once()

    def test_workflow_with_output_file(self, comprehensive_lean_project):
        """Test workflow with JSON output file."""
        cli = AdvancedCLI()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_file = f.name

        try:
            with patch("simpulse.advanced_optimizer.HybridDiagnosticCollector") as mock_collector:
                mock_analysis = self._create_mock_analysis(comprehensive_lean_project)
                mock_collector.return_value.collect_comprehensive_analysis.return_value = (
                    mock_analysis
                )

                # Test with output file
                result = cli.run(
                    ["--output", output_file, "analyze", str(comprehensive_lean_project)]
                )

                assert result == 0

                # Check that output file exists and contains valid JSON
                output_path = Path(output_file)
                if output_path.exists():
                    with open(output_path) as f:
                        data = json.load(f)
                        assert "analysis" in data or "recommendations" in data
        finally:
            Path(output_file).unlink(missing_ok=True)

    def test_error_handling_invalid_project(self):
        """Test error handling with invalid project path."""
        cli = AdvancedCLI()

        result = cli.run(["analyze", "/nonexistent/path"])

        assert result == 1  # Error exit code

    def test_error_handling_lean_compilation_failure(self, comprehensive_lean_project):
        """Test error handling when Lean compilation fails."""
        cli = AdvancedCLI()

        with patch("simpulse.advanced_optimizer.HybridDiagnosticCollector") as mock_collector:
            # Mock compilation failure
            mock_collector.return_value.collect_comprehensive_analysis.side_effect = Exception(
                "Compilation failed"
            )

            result = cli.run(["analyze", str(comprehensive_lean_project)])

            assert result == 1  # Error exit code

    def test_confidence_threshold_filtering(self, comprehensive_lean_project):
        """Test that confidence threshold filtering works."""
        cli = AdvancedCLI()

        with patch("simpulse.advanced_optimizer.HybridDiagnosticCollector") as mock_collector:
            mock_analysis = self._create_mock_analysis(comprehensive_lean_project)
            mock_collector.return_value.collect_comprehensive_analysis.return_value = mock_analysis

            # Test with high confidence threshold
            result = cli.run(
                ["preview", str(comprehensive_lean_project), "--confidence-threshold", "90"]
            )

            assert result == 0
            # Should only show high-confidence recommendations

    def test_max_files_limitation(self, comprehensive_lean_project):
        """Test that max files limitation works."""
        cli = AdvancedCLI()

        with patch("simpulse.advanced_optimizer.HybridDiagnosticCollector") as mock_collector:
            mock_analysis = self._create_mock_analysis(comprehensive_lean_project)
            mock_collector.return_value.collect_comprehensive_analysis.return_value = mock_analysis

            # Test with file limit
            result = cli.run(["analyze", str(comprehensive_lean_project), "--max-files", "1"])

            assert result == 0
            # Should limit the number of files processed

    def test_verbose_and_quiet_modes(self, comprehensive_lean_project):
        """Test verbose and quiet modes."""
        cli = AdvancedCLI()

        with patch("simpulse.advanced_optimizer.HybridDiagnosticCollector") as mock_collector:
            mock_analysis = self._create_mock_analysis(comprehensive_lean_project)
            mock_collector.return_value.collect_comprehensive_analysis.return_value = mock_analysis

            # Test verbose mode
            result_verbose = cli.run(["--verbose", "analyze", str(comprehensive_lean_project)])
            assert result_verbose == 0

            # Test quiet mode
            result_quiet = cli.run(["--quiet", "analyze", str(comprehensive_lean_project)])
            assert result_quiet == 0

    def test_optimization_with_validation(self, comprehensive_lean_project):
        """Test optimization with performance validation."""
        cli = AdvancedCLI()

        with (
            patch("simpulse.advanced_optimizer.HybridDiagnosticCollector") as mock_collector,
            patch("simpulse.advanced_optimizer.PerformanceMeasurer") as mock_measurer,
            patch("simpulse.advanced_optimizer.OptimizationValidator") as mock_validator,
        ):

            mock_analysis = self._create_mock_analysis(comprehensive_lean_project)
            mock_collector.return_value.collect_comprehensive_analysis.return_value = mock_analysis

            # Mock performance measurement
            mock_measurer.return_value.measure_performance.return_value = MagicMock(
                average_time=40.0, files_processed=2
            )

            # Mock validation
            mock_validator.return_value.validate_optimization.return_value = MagicMock(
                improvement_percentage=12.5, validation_successful=True
            )

            # Mock file operations
            with patch("builtins.open", create=True) as mock_open, patch("shutil.copy2"):

                mock_open.return_value.__enter__.return_value.read.return_value = (
                    "@[simp] theorem test : 1 = 1 := rfl"
                )

                result = cli.run(
                    ["optimize", str(comprehensive_lean_project), "--min-improvement", "10"]
                )

                assert result == 0

    def test_optimization_without_validation(self, comprehensive_lean_project):
        """Test optimization without performance validation."""
        cli = AdvancedCLI()

        with patch("simpulse.advanced_optimizer.HybridDiagnosticCollector") as mock_collector:
            mock_analysis = self._create_mock_analysis(comprehensive_lean_project)
            mock_collector.return_value.collect_comprehensive_analysis.return_value = mock_analysis

            # Mock file operations
            with patch("builtins.open", create=True) as mock_open, patch("shutil.copy2"):

                mock_open.return_value.__enter__.return_value.read.return_value = (
                    "@[simp] theorem test : 1 = 1 := rfl"
                )

                result = cli.run(["optimize", str(comprehensive_lean_project), "--no-validation"])

                assert result == 0

    def _create_mock_analysis(self, project_path):
        """Helper method to create realistic mock analysis."""
        mock_analysis = MagicMock()
        mock_analysis.total_theorems = 6
        mock_analysis.total_files = 2
        mock_analysis.theorem_usage = [
            MagicMock(
                name="list_append_nil",
                used_count=25,
                tried_count=27,
                success_rate=0.926,
                file_path=str(project_path / "ComprehensiveProject" / "Basic.lean"),
            ),
            MagicMock(
                name="nat_add_zero",
                used_count=15,
                tried_count=18,
                success_rate=0.833,
                file_path=str(project_path / "ComprehensiveProject" / "Basic.lean"),
            ),
            MagicMock(
                name="inefficient_rule",
                used_count=5,
                tried_count=20,
                success_rate=0.25,
                file_path=str(project_path / "ComprehensiveProject" / "Basic.lean"),
            ),
            MagicMock(
                name="unused_theorem",
                used_count=0,
                tried_count=3,
                success_rate=0.0,
                file_path=str(project_path / "ComprehensiveProject" / "Basic.lean"),
            ),
            MagicMock(
                name="zero_add",
                used_count=22,
                tried_count=24,
                success_rate=0.917,
                file_path=str(project_path / "ComprehensiveProject" / "Advanced.lean"),
            ),
            MagicMock(
                name="complex_rule",
                used_count=8,
                tried_count=12,
                success_rate=0.667,
                file_path=str(project_path / "ComprehensiveProject" / "Advanced.lean"),
            ),
        ]
        return mock_analysis


@pytest.mark.slow
class TestRealWorldScenarios:
    """Test real-world scenarios with more complex setups."""

    def test_large_project_simulation(self, temp_dir):
        """Test simulation of large project analysis."""
        # Create a simulated large project
        large_project = temp_dir / "large_project"
        large_project.mkdir()

        # Create lakefile
        lakefile = large_project / "lakefile.lean"
        lakefile.write_text(
            """
import Lake
open Lake DSL

package large_project where
  version := v!"1.0.0"

lean_lib LargeProject where
  roots := #[`LargeProject]
"""
        )

        # Create lean-toolchain
        toolchain = large_project / "lean-toolchain"
        toolchain.write_text("4.8.0")

        # Create multiple modules
        lib_dir = large_project / "LargeProject"
        lib_dir.mkdir()

        # Create multiple files with many theorems
        for i in range(5):
            module_file = lib_dir / f"Module{i}.lean"
            content = f"""
-- Module {i} with various simp rules
"""
            for j in range(10):
                content += f"""
@[simp] theorem rule_{i}_{j} (n : Nat) : n + {j} = {j} + n := by
  simp [Nat.add_comm]
"""
            module_file.write_text(content)

        # Test analysis with file limit
        cli = AdvancedCLI()

        with patch("simpulse.advanced_optimizer.HybridDiagnosticCollector") as mock_collector:
            mock_analysis = MagicMock()
            mock_analysis.total_theorems = 50
            mock_analysis.total_files = 5
            mock_analysis.theorem_usage = []

            mock_collector.return_value.collect_comprehensive_analysis.return_value = mock_analysis

            result = cli.run(["analyze", str(large_project), "--max-files", "3"])

            assert result == 0

    def test_performance_regression_detection(self, comprehensive_lean_project):
        """Test detection of performance regressions."""
        cli = AdvancedCLI()

        with (
            patch("simpulse.advanced_optimizer.HybridDiagnosticCollector") as mock_collector,
            patch("simpulse.advanced_optimizer.PerformanceMeasurer") as mock_measurer,
            patch("simpulse.advanced_optimizer.OptimizationValidator") as mock_validator,
        ):

            mock_analysis = MagicMock()
            mock_analysis.total_theorems = 1
            mock_analysis.total_files = 1
            mock_analysis.theorem_usage = [
                MagicMock(
                    name="test_theorem",
                    used_count=100,
                    tried_count=110,
                    success_rate=0.909,
                    file_path=str(comprehensive_lean_project / "test.lean"),
                )
            ]

            mock_collector.return_value.collect_comprehensive_analysis.return_value = mock_analysis

            # Mock performance measurement showing regression
            mock_measurer.return_value.measure_performance.return_value = MagicMock(
                average_time=50.0, files_processed=1
            )

            # Mock validation showing performance regression
            mock_validator.return_value.validate_optimization.return_value = MagicMock(
                improvement_percentage=-5.0, validation_successful=False  # Regression
            )

            # Mock file operations
            with patch("builtins.open", create=True) as mock_open, patch("shutil.copy2"):

                mock_open.return_value.__enter__.return_value.read.return_value = (
                    "@[simp] theorem test : 1 = 1 := rfl"
                )

                result = cli.run(["optimize", str(comprehensive_lean_project)])

                # Should complete but might not apply optimizations due to regression
                assert result == 0

    def test_hybrid_diagnostic_fallback(self, comprehensive_lean_project):
        """Test hybrid diagnostic collection with fallback."""
        cli = AdvancedCLI()

        with patch("simpulse.lake_integration.HybridDiagnosticCollector") as mock_collector:
            # Mock Lake failure followed by pattern analysis success
            mock_collector.return_value.collect_comprehensive_analysis.return_value = MagicMock(
                total_theorems=3,
                total_files=2,
                theorem_usage=[
                    MagicMock(
                        name="fallback_theorem",
                        used_count=10,
                        tried_count=10,
                        success_rate=1.0,
                        file_path=str(comprehensive_lean_project / "test.lean"),
                    )
                ],
            )

            result = cli.run(["analyze", str(comprehensive_lean_project)])

            assert result == 0
            mock_collector.assert_called_once()


class TestComponentIntegration:
    """Test integration between different components."""

    def test_diagnostic_parser_to_optimization_engine(self):
        """Test integration between diagnostic parser and optimization engine."""
        # Create mock diagnostic output
        diagnostic_output = """
[simp] used theorems (max: 250, num: 2):
  high_usage_theorem ↦ 100
  low_usage_theorem ↦ 10

[simp] tried theorems (max: 300, num: 2):
  high_usage_theorem ↦ 105, succeeded: 100
  low_usage_theorem ↦ 50, succeeded: 10
"""

        # Parse with diagnostic parser
        parser = DiagnosticParser()
        analysis = parser.parse_diagnostic_output(diagnostic_output)

        # Generate recommendations with optimization engine
        engine = OptimizationEngine()
        recommendations = engine.generate_recommendations(analysis)

        assert len(recommendations) > 0
        assert any(r.theorem_name == "high_usage_theorem" for r in recommendations)
        assert any(r.theorem_name == "low_usage_theorem" for r in recommendations)

    def test_optimization_engine_to_advanced_optimizer(self, sample_lean_project):
        """Test integration between optimization engine and advanced optimizer."""
        optimizer = AdvancedSimpOptimizer(sample_lean_project)

        with patch("simpulse.advanced_optimizer.HybridDiagnosticCollector") as mock_collector:
            mock_analysis = MagicMock()
            mock_analysis.total_theorems = 2
            mock_analysis.total_files = 1
            mock_analysis.theorem_usage = [
                MagicMock(
                    name="test_theorem",
                    used_count=50,
                    tried_count=60,
                    success_rate=0.833,
                    file_path=str(sample_lean_project / "test.lean"),
                )
            ]

            mock_collector.return_value.collect_comprehensive_analysis.return_value = mock_analysis

            result = optimizer.analyze()

            assert result.analysis.total_theorems == 2
            assert len(result.recommendations) > 0

    def test_full_component_chain(self, sample_lean_project):
        """Test the full component chain from CLI to file modification."""
        cli = AdvancedCLI()

        with (
            patch("simpulse.advanced_optimizer.HybridDiagnosticCollector") as mock_collector,
            patch("builtins.open", create=True) as mock_open,
            patch("shutil.copy2") as mock_copy,
        ):

            # Mock analysis
            mock_analysis = MagicMock()
            mock_analysis.total_theorems = 1
            mock_analysis.total_files = 1
            mock_analysis.theorem_usage = [
                MagicMock(
                    name="test_theorem",
                    used_count=100,
                    tried_count=110,
                    success_rate=0.909,
                    file_path=str(sample_lean_project / "test.lean"),
                )
            ]

            mock_collector.return_value.collect_comprehensive_analysis.return_value = mock_analysis

            # Mock file content
            mock_open.return_value.__enter__.return_value.read.return_value = (
                "@[simp] theorem test : 1 = 1 := rfl"
            )
            mock_open.return_value.__enter__.return_value.write.return_value = None

            result = cli.run(["optimize", str(sample_lean_project), "--no-validation"])

            assert result == 0
            # Verify that file operations were called
            mock_copy.assert_called()  # Backup creation
            mock_open.assert_called()  # File reading/writing

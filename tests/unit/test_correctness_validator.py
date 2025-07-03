"""Unit tests for the correctness validator."""

from pathlib import Path
from unittest.mock import Mock, patch


from simpulse.validator.correctness import (
    CorrectnessValidator,
    OptimizationResult,
    ValidationResult,
    create_validator,
)


class TestCorrectnessValidator:
    """Test the correctness validator functionality."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = CorrectnessValidator()
        assert validator.lean_exe == "lake"
        assert validator.timeout == 60
        assert validator.temp_dir is None

        # Custom initialization
        validator = CorrectnessValidator(lean_exe="lean", timeout=120)
        assert validator.lean_exe == "lean"
        assert validator.timeout == 120

    def test_optimization_result(self):
        """Test OptimizationResult dataclass."""
        result = OptimizationResult(
            rule="test_rule", location="line 10", success=True, compilation_time=1.5
        )
        assert result.rule == "test_rule"
        assert result.location == "line 10"
        assert result.success is True
        assert result.error_message is None
        assert result.compilation_time == 1.5

        # With error
        result = OptimizationResult(
            rule="bad_rule", location="line 20", success=False, error_message="Compilation failed"
        )
        assert result.success is False
        assert result.error_message == "Compilation failed"

    def test_validation_result(self):
        """Test ValidationResult dataclass."""
        result = ValidationResult(file_path="/test/file.lean", original_compilation_time=2.0)
        assert result.file_path == "/test/file.lean"
        assert result.original_compilation_time == 2.0
        assert result.optimized_compilation_time is None
        assert result.total_optimizations == 0
        assert result.successful_optimizations == 0
        assert result.failed_optimizations == 0
        assert result.optimization_results == []
        assert result.error is None

        # Test properties
        assert result.success_rate == 0.0
        assert result.speedup == 1.0

        # Add some results
        result.total_optimizations = 10
        result.successful_optimizations = 7
        result.failed_optimizations = 3
        result.optimized_compilation_time = 1.5

        assert result.success_rate == 0.7
        assert result.speedup == 2.0 / 1.5

    @patch("subprocess.run")
    def test_run_lake_build_success(self, mock_run):
        """Test successful lake build."""
        validator = CorrectnessValidator()

        # Mock successful compilation
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = "Build completed"
        mock_run.return_value = mock_result

        success, time, error = validator._run_lake_build(Path("/test"))

        assert success is True
        assert time > 0
        assert error is None
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_lake_build_failure(self, mock_run):
        """Test failed lake build."""
        validator = CorrectnessValidator()

        # Mock failed compilation
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Error: type mismatch"
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        success, time, error = validator._run_lake_build(Path("/test"))

        assert success is False
        assert time > 0
        assert error == "Error: type mismatch"

    def test_apply_optimization(self, tmp_path):
        """Test applying optimization to a file."""
        validator = CorrectnessValidator()

        # Create test file
        test_file = tmp_path / "test.lean"
        test_file.write_text(
            """@[simp] theorem test_rule : true = true := rfl
theorem other_rule : false = false := rfl"""
        )

        # Test line replacement
        optimization = {
            "line": 1,
            "original": "@[simp] theorem test_rule",
            "replacement": "@[simp, priority := 500] theorem test_rule",
        }

        assert validator._apply_optimization(test_file, optimization) is True

        content = test_file.read_text()
        assert "@[simp, priority := 500] theorem test_rule" in content
        assert "theorem other_rule" in content  # Unchanged

    def test_find_project_root(self, tmp_path):
        """Test finding Lean project root."""
        validator = CorrectnessValidator()

        # Create project structure
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "lakefile.lean").touch()

        src_dir = project_dir / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.lean"
        test_file.touch()

        # Should find project root
        root = validator._find_project_root(test_file)
        assert root == project_dir

        # No project root
        standalone_file = tmp_path / "standalone.lean"
        standalone_file.touch()
        root = validator._find_project_root(standalone_file)
        assert root is None

    @patch.object(CorrectnessValidator, "_run_lake_build")
    def test_validate_file_success(self, mock_build, tmp_path):
        """Test successful file validation."""
        validator = CorrectnessValidator()

        # Create test file
        test_file = tmp_path / "test.lean"
        test_file.write_text(
            """@[simp] theorem rule1 : true = true := rfl
@[simp] theorem rule2 : false = false := rfl"""
        )

        # Mock successful builds
        mock_build.side_effect = [
            (True, 1.0, None),  # Original compilation
            (True, 0.9, None),  # After first optimization
            (False, 1.1, "Error"),  # After second optimization (fails)
            (True, 0.9, None),  # Final compilation
        ]

        optimizations = [
            {
                "rule": "rule1",
                "location": "line 1",
                "line": 1,
                "original": "@[simp] theorem rule1",
                "replacement": "@[simp, priority := 500] theorem rule1",
            },
            {
                "rule": "rule2",
                "location": "line 2",
                "line": 2,
                "original": "@[simp] theorem rule2",
                "replacement": "@[simp, priority := INVALID] theorem rule2",  # Will fail
            },
        ]

        result = validator.validate_file(test_file, optimizations)

        assert result.file_path == str(test_file)
        assert result.original_compilation_time == 1.0
        assert result.optimized_compilation_time == 0.9
        assert result.total_optimizations == 2
        assert result.successful_optimizations == 1
        assert result.failed_optimizations == 1
        assert len(result.optimization_results) == 2
        assert result.optimization_results[0].success is True
        assert result.optimization_results[1].success is False

    def test_validate_batch(self, tmp_path):
        """Test batch validation."""
        validator = CorrectnessValidator()

        # Create test files
        files_and_optimizations = []
        for i in range(2):
            test_file = tmp_path / f"test{i}.lean"
            test_file.write_text(f"@[simp] theorem rule{i} : true = true := rfl")

            optimizations = [
                {
                    "rule": f"rule{i}",
                    "location": "line 1",
                    "line": 1,
                    "original": f"@[simp] theorem rule{i}",
                    "replacement": f"@[simp, priority := 500] theorem rule{i}",
                }
            ]

            files_and_optimizations.append((test_file, optimizations))

        # Mock validate_file
        with patch.object(validator, "validate_file") as mock_validate:
            mock_validate.side_effect = [
                ValidationResult(
                    file_path=str(files_and_optimizations[0][0]),
                    original_compilation_time=1.0,
                    optimized_compilation_time=0.8,
                    total_optimizations=1,
                    successful_optimizations=1,
                ),
                ValidationResult(
                    file_path=str(files_and_optimizations[1][0]),
                    original_compilation_time=1.2,
                    optimized_compilation_time=1.0,
                    total_optimizations=1,
                    successful_optimizations=1,
                ),
            ]

            report = validator.validate_batch(files_and_optimizations)

        assert report["total_files_tested"] == 2
        assert report["files_successfully_optimized"] == 2
        assert report["total_optimizations_attempted"] == 2
        assert report["successful_optimizations"] == 2
        assert report["overall_success_rate"] == 1.0
        assert len(report["file_results"]) == 2

    def test_generate_safety_report(self):
        """Test safety report generation."""
        validator = CorrectnessValidator()

        # Create mock validation results
        results = [
            ValidationResult(
                file_path="test1.lean",
                original_compilation_time=1.0,
                optimized_compilation_time=0.8,
            ),
            ValidationResult(
                file_path="test2.lean",
                original_compilation_time=1.2,
                optimized_compilation_time=1.0,
            ),
        ]

        # Add optimization results
        results[0].optimization_results = [
            OptimizationResult("safe_rule", "line 1", True),
            OptimizationResult("safe_rule", "line 2", True),
            OptimizationResult("unsafe_rule", "line 3", False, "Error"),
        ]

        results[1].optimization_results = [
            OptimizationResult("safe_rule", "line 1", True),
            OptimizationResult("conditional_rule", "line 2", True),
            OptimizationResult("conditional_rule", "line 3", False, "Error"),
        ]

        report = validator.generate_safety_report(results)

        assert report["total_rules_tested"] == 3
        assert "safe_rule" in report["safe_rules"]
        assert "unsafe_rule" in report["unsafe_rules"]
        assert "conditional_rule" in report["conditional_rules"]

        # Check safety scores
        assert report["rule_safety_scores"]["safe_rule"]["safety_rate"] == 1.0
        assert report["rule_safety_scores"]["unsafe_rule"]["safety_rate"] == 0.0
        assert report["rule_safety_scores"]["conditional_rule"]["safety_rate"] == 0.5

    def test_create_validator(self):
        """Test validator factory function."""
        validator = create_validator()
        assert isinstance(validator, CorrectnessValidator)
        assert validator.lean_exe == "lake"
        assert validator.timeout == 60

        validator = create_validator(lean_exe="lean", timeout=120)
        assert validator.lean_exe == "lean"
        assert validator.timeout == 120


class TestIntegrationWithOptimizer:
    """Test integration between optimizer and validator."""

    @patch("simpulse.validator.correctness.CorrectnessValidator.validate_file")
    def test_optimizer_with_validation(self, mock_validate):
        """Test optimizer with validation enabled."""
        from simpulse.optimizer import OptimizationSuggestion, PriorityOptimizer

        # Create optimizer with validation
        optimizer = PriorityOptimizer(validate_correctness=True)
        assert optimizer.validator is not None

        # Mock validation result
        mock_validate.return_value = ValidationResult(
            file_path="/test/file.lean",
            original_compilation_time=1.0,
            optimized_compilation_time=0.8,
            total_optimizations=1,
            successful_optimizations=1,
        )

        # Create test suggestion
        suggestion = OptimizationSuggestion(
            rule_name="test_rule",
            file_path="/test/file.lean",
            current_priority=None,
            suggested_priority=500,
            reason="Test",
            expected_speedup=0.2,
            confidence="high",
        )

        # Test apply with validation
        result = optimizer.apply_optimizations_with_validation(
            Path("/test/file.lean"), [suggestion]
        )

        assert result is not None
        assert result.successful_optimizations == 1
        mock_validate.assert_called_once()

    def test_optimizer_without_validation(self):
        """Test optimizer without validation."""
        from simpulse.optimizer import PriorityOptimizer

        optimizer = PriorityOptimizer(validate_correctness=False)
        assert optimizer.validator is None

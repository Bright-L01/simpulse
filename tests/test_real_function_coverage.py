"""
Test suite to achieve 100% coverage of REAL functions.

This test file focuses on testing actual implemented functionality,
ensuring comprehensive coverage of all non-stub code paths.

Test Categories:
- REAL_COVERAGE: Tests for real implemented functions
- EDGE_CASES: Edge case testing for real functions  
- ERROR_PATHS: Error handling in real functions
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simpulse.errors import ErrorHandler, ErrorCategory, ErrorSeverity, ErrorContext
from simpulse.errors import handle_file_error, handle_lean_error, handle_optimization_error
from simpulse.analyzer import SimpRuleAnalyzer
from simpulse.optimizer import SimpleFrequencyOptimizer
from simpulse.portfolio.feature_extractor import GoalFeatureExtractor
from simpulse.profiling.trace_parser import TraceParser
from simpulse.evolution.models import SimplificationRule, Mutation, OptimizationCandidate
from simpulse.jit.runtime_adapter import RuntimeAdapter
from simpulse.analysis.health_checker import HealthChecker


class TestErrorHandlerRealFunctions:
    """REAL_COVERAGE: Complete coverage of error handling real functions."""
    
    def setup_method(self):
        """Setup for each test."""
        self.error_handler = ErrorHandler()
        
    def test_error_handler_initialization_with_logger(self):
        """REAL_COVERAGE: Test initialization with custom logger."""
        import logging
        custom_logger = logging.getLogger("test_logger")
        handler = ErrorHandler(custom_logger)
        assert handler.logger == custom_logger
        
    def test_error_handler_initialization_default_logger(self):
        """REAL_COVERAGE: Test initialization with default logger."""
        handler = ErrorHandler()
        assert handler.logger is not None
        assert handler.errors == []
        assert handler.recovery_attempts == 0
        assert handler.max_recovery_attempts == 3
        
    def test_handle_error_complete_flow(self):
        """REAL_COVERAGE: Test complete error handling flow."""
        context = ErrorContext(
            operation="test_operation",
            file_path=Path("/test/file.lean"),
            rule_name="test_rule",
            strategy="test_strategy",
            additional_info={"key": "value"}
        )
        
        error = self.error_handler.handle_error(
            category=ErrorCategory.FILE_ACCESS,
            severity=ErrorSeverity.MEDIUM,
            message="Test error message",
            context=context,
            exception=FileNotFoundError("Test file not found")
        )
        
        assert error.category == ErrorCategory.FILE_ACCESS
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.message == "Test error message"
        assert error.context == context
        assert isinstance(error.original_exception, FileNotFoundError)
        assert len(error.recovery_suggestions) > 0
        assert error.timestamp > 0
        
        # Verify error was added to handler
        assert len(self.error_handler.errors) == 1
        assert self.error_handler.errors[0] == error
        
    def test_generate_recovery_suggestions_all_categories(self):
        """REAL_COVERAGE: Test recovery suggestions for all error categories."""
        test_cases = [
            (ErrorCategory.FILE_ACCESS, FileNotFoundError("test"), ["file path exists", "permissions"]),
            (ErrorCategory.LEAN_EXECUTION, Exception("lean failed"), ["Lean 4 is installed", "lake build"]),
            (ErrorCategory.OPTIMIZATION, Exception("opt failed"), ["conservative optimization", "reduce the number"]),
            (ErrorCategory.VALIDATION, Exception("validation failed"), ["compile correctly", "backup files"]),
            (ErrorCategory.CONFIGURATION, Exception("config error"), ["configuration file syntax", "required configuration"]),
            (ErrorCategory.DEPENDENCY, Exception("dep error"), ["missing dependencies", "Python version"]),
        ]
        
        for category, exception, expected_terms in test_cases:
            suggestions = self.error_handler._generate_recovery_suggestions(category, exception)
            assert len(suggestions) > 0
            
            # Check at least one expected term appears in suggestions
            all_suggestions_text = " ".join(suggestions).lower()
            found = any(term.lower() in all_suggestions_text for term in expected_terms)
            assert found, f"Expected one of {expected_terms} in suggestions for {category}: {suggestions}"
            
    def test_generate_recovery_suggestions_exception_specific(self):
        """REAL_COVERAGE: Test exception-specific recovery suggestions."""
        test_exceptions = [
            (FileNotFoundError("missing file"), "File not found"),
            (PermissionError("access denied"), "Permission denied"),
            (TimeoutError("timed out"), "timed out"),
        ]
        
        for exception, expected_text in test_exceptions:
            suggestions = self.error_handler._generate_recovery_suggestions(
                ErrorCategory.FILE_ACCESS, exception
            )
            
            found = any(expected_text.lower() in s.lower() for s in suggestions)
            assert found, f"Expected '{expected_text}' in suggestions for {exception}: {suggestions}"
            
    def test_get_user_friendly_summary_empty(self):
        """REAL_COVERAGE: Test summary with no errors."""
        summary = self.error_handler.get_user_friendly_summary()
        
        assert summary["status"] == "success"
        assert summary["errors"] == []
        
    def test_get_user_friendly_summary_with_errors(self):
        """REAL_COVERAGE: Test summary with various error types."""
        # Add different types of errors
        context = ErrorContext(operation="test")
        
        # High severity error
        self.error_handler.handle_error(
            ErrorCategory.FILE_ACCESS, ErrorSeverity.HIGH, "High error", context
        )
        
        # Medium severity error  
        self.error_handler.handle_error(
            ErrorCategory.OPTIMIZATION, ErrorSeverity.MEDIUM, "Medium error", context
        )
        
        # Low severity error
        self.error_handler.handle_error(
            ErrorCategory.VALIDATION, ErrorSeverity.LOW, "Low error", context
        )
        
        summary = self.error_handler.get_user_friendly_summary()
        
        assert summary["status"] == "error"  # High severity present
        assert summary["total_errors"] == 3
        assert summary["by_severity"]["high"] == 1
        assert summary["by_severity"]["medium"] == 1
        assert summary["by_severity"]["low"] == 1
        assert summary["by_category"]["file_access"] == 1
        assert summary["by_category"]["optimization"] == 1
        assert summary["by_category"]["validation"] == 1
        assert len(summary["recent_errors"]) == 3
        assert len(summary["recovery_suggestions"]) > 0
        
    def test_get_user_friendly_summary_warning_status(self):
        """REAL_COVERAGE: Test summary with only medium/low errors."""
        context = ErrorContext(operation="test")
        
        self.error_handler.handle_error(
            ErrorCategory.OPTIMIZATION, ErrorSeverity.MEDIUM, "Medium error", context
        )
        
        summary = self.error_handler.get_user_friendly_summary()
        assert summary["status"] == "warning"
        
    def test_has_fatal_errors(self):
        """REAL_COVERAGE: Test fatal error detection."""
        assert not self.error_handler.has_fatal_errors()
        
        context = ErrorContext(operation="test")
        self.error_handler.handle_error(
            ErrorCategory.OPTIMIZATION, ErrorSeverity.FATAL, "Fatal error", context
        )
        
        assert self.error_handler.has_fatal_errors()
        
    def test_has_high_severity_errors(self):
        """REAL_COVERAGE: Test high severity error detection."""
        assert not self.error_handler.has_high_severity_errors()
        
        context = ErrorContext(operation="test")
        self.error_handler.handle_error(
            ErrorCategory.OPTIMIZATION, ErrorSeverity.HIGH, "High error", context
        )
        
        assert self.error_handler.has_high_severity_errors()
        
    def test_clear_errors(self):
        """REAL_COVERAGE: Test error clearing."""
        context = ErrorContext(operation="test")
        self.error_handler.handle_error(
            ErrorCategory.OPTIMIZATION, ErrorSeverity.MEDIUM, "Test error", context
        )
        
        assert len(self.error_handler.errors) == 1
        
        self.error_handler.clear_errors()
        
        assert len(self.error_handler.errors) == 0
        assert self.error_handler.recovery_attempts == 0


class TestConvenienceErrorFunctions:
    """REAL_COVERAGE: Test convenience error handling functions."""
    
    def setup_method(self):
        """Setup for each test."""
        self.error_handler = ErrorHandler()
        
    def test_handle_file_error_filenotfound(self):
        """REAL_COVERAGE: Test file error handling for FileNotFoundError."""
        test_path = Path("/nonexistent/file.lean")
        exception = FileNotFoundError("File not found")
        
        error = handle_file_error(self.error_handler, "read_file", test_path, exception)
        
        assert error.category == ErrorCategory.FILE_ACCESS
        assert error.severity == ErrorSeverity.MEDIUM  # FileNotFoundError -> MEDIUM
        assert error.context.operation == "read_file"
        assert error.context.file_path == test_path
        assert "File operation failed" in error.message
        
    def test_handle_file_error_permission(self):
        """REAL_COVERAGE: Test file error handling for PermissionError."""
        test_path = Path("/root/file.lean")
        exception = PermissionError("Permission denied")
        
        error = handle_file_error(self.error_handler, "write_file", test_path, exception)
        
        assert error.category == ErrorCategory.FILE_ACCESS
        assert error.severity == ErrorSeverity.HIGH  # PermissionError -> HIGH
        assert error.context.operation == "write_file"
        assert error.context.file_path == test_path
        
    def test_handle_lean_error(self):
        """REAL_COVERAGE: Test Lean execution error handling."""
        test_path = Path("/project/file.lean")
        exception = Exception("Lean compilation failed")
        
        error = handle_lean_error(self.error_handler, "compile_lean", exception, test_path)
        
        assert error.category == ErrorCategory.LEAN_EXECUTION
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.operation == "compile_lean"
        assert error.context.file_path == test_path
        assert "Lean execution failed" in error.message
        
    def test_handle_lean_error_no_file_path(self):
        """REAL_COVERAGE: Test Lean error handling without file path."""
        exception = Exception("Lean setup failed")
        
        error = handle_lean_error(self.error_handler, "setup_lean", exception)
        
        assert error.category == ErrorCategory.LEAN_EXECUTION
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.operation == "setup_lean"
        assert error.context.file_path is None
        
    def test_handle_optimization_error_full_context(self):
        """REAL_COVERAGE: Test optimization error with full context."""
        exception = Exception("Optimization failed")
        
        error = handle_optimization_error(
            self.error_handler, 
            "optimize_rules", 
            exception,
            strategy="aggressive",
            rule_name="test_rule"
        )
        
        assert error.category == ErrorCategory.OPTIMIZATION
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context.operation == "optimize_rules"
        assert error.context.strategy == "aggressive"
        assert error.context.rule_name == "test_rule"
        assert "Optimization failed" in error.message
        
    def test_handle_optimization_error_minimal_context(self):
        """REAL_COVERAGE: Test optimization error with minimal context."""
        exception = Exception("Simple optimization error")
        
        error = handle_optimization_error(self.error_handler, "simple_opt", exception)
        
        assert error.category == ErrorCategory.OPTIMIZATION
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context.operation == "simple_opt"
        assert error.context.strategy is None
        assert error.context.rule_name is None


class TestAnalyzerRealFunctions:
    """REAL_COVERAGE: Test analyzer real functions."""
    
    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.analyzer = SimpRuleAnalyzer()
        
    def teardown_method(self):
        """Cleanup after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_analyzer_initialization(self):
        """REAL_COVERAGE: Test analyzer initialization."""
        assert self.analyzer is not None
        
    def test_analyze_project_empty_directory(self):
        """REAL_COVERAGE: Test analyzing empty directory."""
        result = self.analyzer.analyze_project(self.temp_dir)
        
        assert result is not None
        assert hasattr(result, 'total_files')
        assert result.total_files == 0
        
    def test_analyze_project_with_lean_files(self):
        """REAL_COVERAGE: Test analyzing directory with Lean files."""
        # Create test Lean files
        test_file1 = self.temp_dir / "test1.lean"
        test_file2 = self.temp_dir / "test2.lean"
        
        test_file1.write_text("""
@[simp] theorem test1 : 1 + 1 = 2 := rfl
@[simp] theorem test2 (n : Nat) : n + 0 = n := Nat.add_zero n
        """)
        
        test_file2.write_text("""
@[simp] theorem test3 : 0 + 0 = 0 := rfl
        """)
        
        result = self.analyzer.analyze_project(self.temp_dir)
        
        assert result.total_files == 2
        assert len(result.rules) > 0  # Should find some simp rules
        
    def test_get_statistics(self):
        """REAL_COVERAGE: Test getting statistics."""
        stats = self.analyzer.get_statistics()
        
        assert isinstance(stats, dict)
        assert "files_processed" in stats
        assert "rules_extracted" in stats
        assert "cache_hits" in stats
        

class TestFeatureExtractorRealFunctions:
    """REAL_COVERAGE: Test feature extractor real functions."""
    
    def setup_method(self):
        """Setup for each test."""
        self.extractor = GoalFeatureExtractor()
        
    def test_feature_extractor_initialization(self):
        """REAL_COVERAGE: Test feature extractor initialization."""
        assert self.extractor is not None
        
    def test_extract_basic_features(self):
        """REAL_COVERAGE: Test basic feature extraction."""
        goal = "∀ n : ℕ, n + 0 = n"
        features = self.extractor.extract_basic_features(goal)
        
        assert isinstance(features, dict)
        assert "length" in features
        assert "symbols" in features
        assert features["length"] == len(goal)
        
    def test_extract_syntactic_features(self):
        """REAL_COVERAGE: Test syntactic feature extraction."""
        goal = "∀ n : ℕ, n + 0 = n"
        features = self.extractor.extract_syntactic_features(goal)
        
        assert isinstance(features, dict)
        assert "forall_count" in features
        assert "arrow_count" in features
        assert features["forall_count"] > 0  # Should detect ∀
        
    def test_extract_features_combined(self):
        """REAL_COVERAGE: Test combined feature extraction."""
        goal = "∀ n : ℕ, n + 0 = n"
        features = self.extractor.extract_features(goal)
        
        assert isinstance(features, dict)
        # Should include both basic and syntactic features
        assert "length" in features
        assert "forall_count" in features
        

class TestTraceParserRealFunctions:
    """REAL_COVERAGE: Test trace parser real functions."""
    
    def setup_method(self):
        """Setup for each test."""
        self.parser = TraceParser()
        
    def test_trace_parser_initialization(self):
        """REAL_COVERAGE: Test trace parser initialization."""
        assert self.parser is not None
        
    def test_parse_simp_trace_empty(self):
        """REAL_COVERAGE: Test parsing empty trace."""
        result = self.parser.parse_simp_trace("")
        
        assert isinstance(result, dict)
        assert "rules_applied" in result
        assert len(result["rules_applied"]) == 0
        
    def test_parse_simp_trace_with_content(self):
        """REAL_COVERAGE: Test parsing trace with simp rules."""
        trace_content = """
simp [add_zero, zero_add] at h
simp only [mul_one] at *
        """
        
        result = self.parser.parse_simp_trace(trace_content)
        
        assert isinstance(result, dict)
        assert "rules_applied" in result
        # Should extract rule names from simp calls
        
    def test_extract_rule_applications(self):
        """REAL_COVERAGE: Test rule application extraction."""
        trace_line = "simp [add_zero, zero_add, mul_one]"
        rules = self.parser.extract_rule_applications(trace_line)
        
        assert isinstance(rules, list)
        assert "add_zero" in rules
        assert "zero_add" in rules
        assert "mul_one" in rules


class TestEvolutionModelsRealFunctions:
    """REAL_COVERAGE: Test evolution models real functions."""
    
    def test_simplification_rule_creation(self):
        """REAL_COVERAGE: Test SimplificationRule creation and methods."""
        rule = SimplificationRule(
            name="test_rule",
            lhs="x + 0",
            rhs="x",
            conditions=["x : ℕ"],
            priority=100
        )
        
        assert rule.name == "test_rule"
        assert rule.lhs == "x + 0"
        assert rule.rhs == "x"
        assert rule.conditions == ["x : ℕ"]
        assert rule.priority == 100
        
        # Test string representation
        rule_str = str(rule)
        assert "test_rule" in rule_str
        
        # Test equality
        rule2 = SimplificationRule(
            name="test_rule",
            lhs="x + 0", 
            rhs="x",
            conditions=["x : ℕ"],
            priority=100
        )
        assert rule == rule2
        
        # Test hash
        assert hash(rule) == hash(rule2)
        
    def test_mutation_creation(self):
        """REAL_COVERAGE: Test Mutation creation and methods."""
        rule = SimplificationRule("test", "x", "y", [], 100)
        
        mutation = Mutation(
            type="priority_change",
            target_rule=rule,
            old_value=100,
            new_value=50,
            impact_score=0.8
        )
        
        assert mutation.type == "priority_change"
        assert mutation.target_rule == rule
        assert mutation.old_value == 100
        assert mutation.new_value == 50
        assert mutation.impact_score == 0.8
        
        # Test string representation
        mutation_str = str(mutation)
        assert "priority_change" in mutation_str
        
    def test_optimization_candidate_creation(self):
        """REAL_COVERAGE: Test OptimizationCandidate creation and methods."""
        rule1 = SimplificationRule("rule1", "x", "y", [], 100)
        rule2 = SimplificationRule("rule2", "a", "b", [], 200)
        
        mutation1 = Mutation("priority_change", rule1, 100, 50, 0.8)
        mutation2 = Mutation("reorder", rule2, 0, 1, 0.6)
        
        candidate = OptimizationCandidate(
            mutations=[mutation1, mutation2],
            expected_improvement=0.15,
            confidence=0.85
        )
        
        assert len(candidate.mutations) == 2
        assert candidate.mutations[0] == mutation1
        assert candidate.mutations[1] == mutation2
        assert candidate.expected_improvement == 0.15
        assert candidate.confidence == 0.85
        
        # Test string representation
        candidate_str = str(candidate)
        assert "2 mutations" in candidate_str
        
        # Test fitness calculation
        fitness = candidate.calculate_fitness()
        assert isinstance(fitness, float)
        assert fitness > 0
        

class TestRuntimeAdapterRealFunctions:
    """REAL_COVERAGE: Test runtime adapter real functions."""
    
    def setup_method(self):
        """Setup for each test."""
        self.adapter = RuntimeAdapter()
        
    def test_runtime_adapter_initialization(self):
        """REAL_COVERAGE: Test runtime adapter initialization."""
        assert self.adapter is not None
        
    def test_is_lean_available_mock(self):
        """REAL_COVERAGE: Test Lean availability check with mocking."""
        with patch('shutil.which') as mock_which:
            mock_which.return_value = "/usr/bin/lean"
            assert self.adapter.is_lean_available() == True
            
            mock_which.return_value = None
            assert self.adapter.is_lean_available() == False
            
    def test_get_lean_version_mock(self):
        """REAL_COVERAGE: Test Lean version detection with mocking."""
        with patch('subprocess.run') as mock_run:
            # Mock successful version output
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Lean (version 4.2.0, commit 123abc, Release)"
            mock_run.return_value = mock_result
            
            version = self.adapter.get_lean_version()
            assert version == "4.2.0"
            
            # Mock failed version check
            mock_result.returncode = 1
            version = self.adapter.get_lean_version()
            assert version is None


class TestHealthCheckerRealFunctions:
    """REAL_COVERAGE: Test health checker real functions."""
    
    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checker = HealthChecker()
        
    def teardown_method(self):
        """Cleanup after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_health_checker_initialization(self):
        """REAL_COVERAGE: Test health checker initialization."""
        assert self.checker is not None
        
    def test_check_project_health_empty_directory(self):
        """REAL_COVERAGE: Test health check on empty directory."""
        result = self.checker.check_project_health(self.temp_dir)
        
        assert isinstance(result, dict)
        assert "overall_health" in result
        assert "issues" in result
        
    def test_check_project_health_with_files(self):
        """REAL_COVERAGE: Test health check with Lean files."""
        # Create test files
        test_file = self.temp_dir / "test.lean"
        test_file.write_text("theorem test : 1 = 1 := rfl")
        
        lakefile = self.temp_dir / "lakefile.lean"
        lakefile.write_text("require mathlib from git \"https://github.com/leanprover-community/mathlib4\"")
        
        result = self.checker.check_project_health(self.temp_dir)
        
        assert isinstance(result, dict)
        assert "overall_health" in result
        assert "file_count" in result
        assert result["file_count"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
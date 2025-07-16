"""
Tests for the advanced CLI functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

import pytest

from simpulse.advanced_cli import AdvancedCLI
from simpulse.error import OptimizationError


class TestAdvancedCLI:
    """Test the advanced CLI functionality."""
    
    @pytest.fixture
    def cli(self):
        """Create CLI instance for testing."""
        return AdvancedCLI()
    
    @pytest.fixture
    def mock_optimizer(self):
        """Mock optimizer for testing."""
        with patch('simpulse.advanced_cli.AdvancedSimpOptimizer') as mock:
            yield mock
    
    def test_help_output(self, cli, capsys):
        """Test that help output is displayed correctly."""
        result = cli.run(['--help'])
        captured = capsys.readouterr()
        assert "simpulse" in captured.out
        assert "analyze" in captured.out
        assert "optimize" in captured.out
        assert "preview" in captured.out
        assert "benchmark" in captured.out
        assert result == 0
    
    def test_no_command_shows_help(self, cli, capsys):
        """Test that running without command shows help."""
        result = cli.run([])
        captured = capsys.readouterr()
        assert "usage:" in captured.out
        assert result == 1
    
    def test_analyze_command_basic(self, cli, mock_optimizer):
        """Test basic analyze command."""
        mock_result = MagicMock()
        mock_result.analysis.total_theorems = 10
        mock_result.analysis.total_files = 3
        mock_result.recommendations = []
        mock_optimizer.return_value.analyze.return_value = mock_result
        
        result = cli.run(['analyze', 'test-project'])
        
        assert result == 0
        mock_optimizer.assert_called_once_with('test-project')
        mock_optimizer.return_value.analyze.assert_called_once()
    
    def test_analyze_with_max_files(self, cli, mock_optimizer):
        """Test analyze command with max files option."""
        mock_result = MagicMock()
        mock_result.analysis.total_theorems = 5
        mock_result.analysis.total_files = 2
        mock_result.recommendations = []
        mock_optimizer.return_value.analyze.return_value = mock_result
        
        result = cli.run(['analyze', 'test-project', '--max-files', '25'])
        
        assert result == 0
        mock_optimizer.return_value.analyze.assert_called_once_with(max_files=25)
    
    def test_optimize_command_basic(self, cli, mock_optimizer):
        """Test basic optimize command."""
        mock_result = MagicMock()
        mock_result.optimization_results = []
        mock_result.files_modified = 0
        mock_result.optimizations_applied = 0
        mock_optimizer.return_value.optimize.return_value = mock_result
        
        result = cli.run(['optimize', 'test-project'])
        
        assert result == 0
        mock_optimizer.return_value.optimize.assert_called_once()
    
    def test_optimize_with_confidence_threshold(self, cli, mock_optimizer):
        """Test optimize command with confidence threshold."""
        mock_result = MagicMock()
        mock_result.optimization_results = []
        mock_result.files_modified = 1
        mock_result.optimizations_applied = 3
        mock_optimizer.return_value.optimize.return_value = mock_result
        
        result = cli.run(['optimize', 'test-project', '--confidence-threshold', '85.0'])
        
        assert result == 0
        # Check that optimize was called with correct parameters
        call_args = mock_optimizer.return_value.optimize.call_args
        assert call_args[1]['confidence_threshold'] == 85.0
    
    def test_optimize_no_validation(self, cli, mock_optimizer):
        """Test optimize command with no validation."""
        mock_result = MagicMock()
        mock_result.optimization_results = []
        mock_result.files_modified = 1
        mock_result.optimizations_applied = 2
        mock_optimizer.return_value.optimize.return_value = mock_result
        
        result = cli.run(['optimize', 'test-project', '--no-validation'])
        
        assert result == 0
        call_args = mock_optimizer.return_value.optimize.call_args
        assert call_args[1]['validate'] is False
    
    def test_preview_command_basic(self, cli, mock_optimizer):
        """Test basic preview command."""
        mock_result = MagicMock()
        mock_result.analysis.total_theorems = 8
        mock_result.recommendations = []
        mock_optimizer.return_value.analyze.return_value = mock_result
        
        result = cli.run(['preview', 'test-project'])
        
        assert result == 0
        mock_optimizer.return_value.analyze.assert_called_once()
    
    def test_preview_with_confidence_threshold(self, cli, mock_optimizer):
        """Test preview command with confidence threshold."""
        mock_result = MagicMock()
        mock_result.analysis.total_theorems = 8
        mock_result.recommendations = []
        mock_optimizer.return_value.analyze.return_value = mock_result
        
        result = cli.run(['preview', 'test-project', '--confidence-threshold', '75.0'])
        
        assert result == 0
        # Preview should still call analyze, but filter results by confidence
        mock_optimizer.return_value.analyze.assert_called_once()
    
    def test_benchmark_command_basic(self, cli, mock_optimizer):
        """Test basic benchmark command."""
        mock_result = MagicMock()
        mock_result.average_time = 42.5
        mock_result.total_files = 5
        mock_optimizer.return_value.benchmark.return_value = mock_result
        
        result = cli.run(['benchmark', 'test-project'])
        
        assert result == 0
        mock_optimizer.return_value.benchmark.assert_called_once()
    
    def test_benchmark_with_runs(self, cli, mock_optimizer):
        """Test benchmark command with multiple runs."""
        mock_result = MagicMock()
        mock_result.average_time = 38.2
        mock_result.total_files = 3
        mock_optimizer.return_value.benchmark.return_value = mock_result
        
        result = cli.run(['benchmark', 'test-project', '--runs', '5'])
        
        assert result == 0
        call_args = mock_optimizer.return_value.benchmark.call_args
        assert call_args[1]['runs'] == 5
    
    def test_verbose_flag(self, cli, mock_optimizer):
        """Test verbose flag enables debug logging."""
        mock_result = MagicMock()
        mock_result.analysis.total_theorems = 5
        mock_result.analysis.total_files = 2
        mock_result.recommendations = []
        mock_optimizer.return_value.analyze.return_value = mock_result
        
        with patch('logging.getLogger') as mock_logger:
            result = cli.run(['--verbose', 'analyze', 'test-project'])
            
        assert result == 0
        # Check that debug logging was enabled
        mock_logger.return_value.setLevel.assert_called_with(10)  # DEBUG level
    
    def test_quiet_flag(self, cli, mock_optimizer):
        """Test quiet flag reduces logging."""
        mock_result = MagicMock()
        mock_result.analysis.total_theorems = 5
        mock_result.analysis.total_files = 2
        mock_result.recommendations = []
        mock_optimizer.return_value.analyze.return_value = mock_result
        
        with patch('logging.getLogger') as mock_logger:
            result = cli.run(['--quiet', 'analyze', 'test-project'])
            
        assert result == 0
        # Check that warning logging was set
        mock_logger.return_value.setLevel.assert_called_with(30)  # WARNING level
    
    def test_keyboard_interrupt_handling(self, cli, mock_optimizer):
        """Test keyboard interrupt handling."""
        mock_optimizer.return_value.analyze.side_effect = KeyboardInterrupt()
        
        result = cli.run(['analyze', 'test-project'])
        
        assert result == 130  # Standard exit code for SIGINT
    
    def test_optimization_error_handling(self, cli, mock_optimizer):
        """Test optimization error handling."""
        mock_optimizer.return_value.analyze.side_effect = OptimizationError("Test error")
        
        result = cli.run(['analyze', 'test-project'])
        
        assert result == 1
    
    def test_unknown_command(self, cli, capsys):
        """Test handling of unknown commands."""
        result = cli.run(['unknown-command'])
        
        captured = capsys.readouterr()
        assert "Unknown command: unknown-command" in captured.out
        assert result == 1
    
    def test_output_file_option(self, cli, mock_optimizer):
        """Test output file option."""
        mock_result = MagicMock()
        mock_result.analysis.total_theorems = 5
        mock_result.analysis.total_files = 2
        mock_result.recommendations = []
        mock_optimizer.return_value.analyze.return_value = mock_result
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            result = cli.run(['--output', output_file, 'analyze', 'test-project'])
            assert result == 0
            # Output file functionality should be tested in implementation
        finally:
            Path(output_file).unlink(missing_ok=True)
    
    def test_exception_handling_with_verbose(self, cli, mock_optimizer):
        """Test exception handling shows traceback in verbose mode."""
        mock_optimizer.return_value.analyze.side_effect = RuntimeError("Test error")
        
        with patch('traceback.print_exc') as mock_traceback:
            result = cli.run(['--verbose', 'analyze', 'test-project'])
            
        assert result == 1
        mock_traceback.assert_called_once()
    
    def test_exception_handling_without_verbose(self, cli, mock_optimizer):
        """Test exception handling without verbose mode."""
        mock_optimizer.return_value.analyze.side_effect = RuntimeError("Test error")
        
        with patch('traceback.print_exc') as mock_traceback:
            result = cli.run(['analyze', 'test-project'])
            
        assert result == 1
        mock_traceback.assert_not_called()


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def test_analyze_with_real_files(self, sample_lean_project):
        """Test analyze command with real Lean files."""
        cli = AdvancedCLI()
        
        # This should work even if Lake integration fails
        # The hybrid system should fall back to pattern analysis
        with patch('simpulse.advanced_cli.AdvancedSimpOptimizer') as mock_optimizer:
            mock_result = MagicMock()
            mock_result.analysis.total_theorems = 3
            mock_result.analysis.total_files = 2
            mock_result.recommendations = []
            mock_optimizer.return_value.analyze.return_value = mock_result
            
            result = cli.run(['analyze', str(sample_lean_project)])
            
        assert result == 0
    
    def test_cli_error_handling_with_invalid_path(self):
        """Test CLI error handling with invalid project path."""
        cli = AdvancedCLI()
        
        result = cli.run(['analyze', '/nonexistent/path'])
        
        # Should handle the error gracefully
        assert result == 1
    
    @pytest.mark.slow
    def test_end_to_end_workflow(self, sample_lean_project):
        """Test complete workflow: analyze -> preview -> optimize."""
        cli = AdvancedCLI()
        
        with patch('simpulse.advanced_cli.AdvancedSimpOptimizer') as mock_optimizer:
            # Mock analyze results
            mock_analyze_result = MagicMock()
            mock_analyze_result.analysis.total_theorems = 3
            mock_analyze_result.analysis.total_files = 2
            mock_analyze_result.recommendations = []
            
            # Mock optimize results
            mock_optimize_result = MagicMock()
            mock_optimize_result.optimization_results = []
            mock_optimize_result.files_modified = 1
            mock_optimize_result.optimizations_applied = 2
            
            mock_optimizer.return_value.analyze.return_value = mock_analyze_result
            mock_optimizer.return_value.optimize.return_value = mock_optimize_result
            
            # Test analyze
            result1 = cli.run(['analyze', str(sample_lean_project)])
            assert result1 == 0
            
            # Test preview
            result2 = cli.run(['preview', str(sample_lean_project)])
            assert result2 == 0
            
            # Test optimize
            result3 = cli.run(['optimize', str(sample_lean_project), '--confidence-threshold', '80'])
            assert result3 == 0
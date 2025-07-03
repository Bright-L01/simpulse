"""Tests for CLI module."""

from click.testing import CliRunner

from simpulse.cli import cli


class TestCLI:
    """Test CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_version_command(self):
        """Test version command."""
        result = self.runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "Simpulse v1.1.0" in result.output

    def test_check_command_help(self):
        """Test check command help."""
        result = self.runner.invoke(cli, ["check", "--help"])
        assert result.exit_code == 0
        assert "Check if your Lean 4 project" in result.output

    def test_optimize_command_help(self):
        """Test optimize command help."""
        result = self.runner.invoke(cli, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "Generate optimized simp rule priorities" in result.output

    def test_benchmark_command_help(self):
        """Test benchmark command help."""
        result = self.runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "Run performance benchmarks" in result.output

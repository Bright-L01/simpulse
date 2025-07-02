"""Tests for Simpulse profiling module."""

import shutil
import tempfile
from pathlib import Path

import pytest

from simpulse.profiling import LeanResult, LeanRunner, ProfileReport, TraceParser

# Check if Lean/Lake is available
LEAN_AVAILABLE = shutil.which("lake") is not None
requires_lean = pytest.mark.skipif(not LEAN_AVAILABLE, reason="Lean/Lake not installed")


@pytest.fixture
def lean_runner():
    """Create a LeanRunner instance."""
    return LeanRunner()


@pytest.fixture
def trace_parser():
    """Create a TraceParser instance."""
    return TraceParser()


@pytest.fixture
def sample_trace_output():
    """Sample trace output for testing."""
    return """
[typeclass_instances] 25.3 ms
[Meta.synthInstance] 12.5 ms
[Meta.Tactic.simp.rewrite] List.append_nil: x ++ [] ==> x (success)
[Meta.Tactic.simp.rewrite] List.map_id: List.map id x ==> x (success)
[elaboration] 150.2 ms
[Meta.Tactic.simp.rewrite] Nat.add_zero: n + 0 ==> n (failed)
[type_checking] 75.0 ms
"""


class TestLeanRunner:
    """Tests for LeanRunner class."""

    def test_initialization(self, lean_runner):
        """Test LeanRunner initialization."""
        assert lean_runner.lake_path == "lake"
        assert lean_runner.lean_path == "lean"
        assert lean_runner.working_dir == Path.cwd()

    def test_get_trace_command(self, lean_runner):
        """Test trace command generation."""
        cmd = lean_runner.get_trace_command(
            file_path=Path("test.lean"),
            trace_flags=["profiler.output", "Meta.Tactic.simp"],
        )

        assert cmd[0] == "lake"
        assert cmd[1] == "env"
        assert cmd[2] == "lean"
        assert "-Dtrace.profiler.output=true" in cmd
        assert "-Dtrace.Meta.Tactic.simp=true" in cmd
        assert str(Path("test.lean")) in cmd

    @pytest.mark.asyncio
    @requires_lean
    async def test_run_lean_with_nonexistent_file(self, lean_runner):
        """Test running Lean with a non-existent file."""
        with tempfile.NamedTemporaryFile(suffix=".lean", delete=True) as f:
            # File is deleted immediately
            pass

        result = await lean_runner.run_lean(file_path=Path(f.name), timeout=5.0)

        assert isinstance(result, LeanResult)
        assert not result.success
        assert result.returncode != 0


class TestTraceParser:
    """Tests for TraceParser class."""

    def test_parse_text_format(self, trace_parser, sample_trace_output):
        """Test parsing text-formatted trace output."""
        report = trace_parser.parse_content(sample_trace_output)

        assert isinstance(report, ProfileReport)
        assert len(report.entries) == 4  # 4 profile entries
        assert len(report.simp_rewrites) == 3  # 3 simp rewrites

        # Check profile entries
        entry_names = {e.name for e in report.entries}
        assert "typeclass_instances" in entry_names
        assert "elaboration" in entry_names

        # Check times
        elaboration = next(e for e in report.entries if e.name == "elaboration")
        assert elaboration.elapsed_ms == 150.2

    def test_parse_simp_trace(self, trace_parser, sample_trace_output):
        """Test parsing simp-specific traces."""
        stats = trace_parser.parse_simp_trace(sample_trace_output)

        assert stats["total_rewrites"] == 3
        assert stats["successful_rewrites"] == 2
        assert stats["failed_rewrites"] == 1
        assert len(stats["unique_theorems"]) == 3
        assert "List.append_nil" in stats["unique_theorems"]

    def test_generate_summary(self, trace_parser, sample_trace_output):
        """Test summary generation."""
        report = trace_parser.parse_content(sample_trace_output)
        summary = trace_parser.generate_summary(report)

        assert "Profile Report" in summary
        assert "Total Time:" in summary
        assert "Top 10 by elapsed time:" in summary
        assert "Simp Rewrites: 3" in summary

    def test_parse_json_format(self, trace_parser):
        """Test parsing JSON-formatted output."""
        json_content = """{
            "profiler": {
                "elaboration": {
                    "elapsed_ms": 100.5,
                    "count": 1,
                    "children": {
                        "type_checking": {
                            "elapsed_ms": 50.2,
                            "count": 2
                        }
                    }
                }
            },
            "metadata": {
                "version": "4.0.0"
            }
        }"""

        report = trace_parser.parse_content(json_content)

        assert len(report.entries) == 1
        assert report.entries[0].name == "elaboration"
        assert report.entries[0].elapsed_ms == 100.5
        assert len(report.entries[0].children) == 1
        assert report.metadata["version"] == "4.0.0"


@pytest.mark.asyncio
@requires_lean
async def test_integration(lean_runner, trace_parser):
    """Integration test with a simple Lean file."""
    # Create a minimal Lean file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(
            """
-- Test file for Simpulse
def hello : String := "world"

example : 1 + 1 = 2 := by simp
"""
        )
        temp_file = Path(f.name)

    try:
        # This will likely fail if Lean is not installed, but we test the flow
        result = await lean_runner.run_lean(
            file_path=temp_file, trace_flags=["Meta.Tactic.simp"], timeout=10.0
        )

        # Even if Lean fails, we should get a result
        assert isinstance(result, LeanResult)
        assert isinstance(result.elapsed_time, float)
        assert result.elapsed_time > 0

    finally:
        temp_file.unlink()

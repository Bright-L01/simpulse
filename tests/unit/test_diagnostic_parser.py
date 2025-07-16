"""
Tests for the diagnostic parser functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from simpulse.diagnostic_parser import (
    DiagnosticParser,
    SimpTheoremUsage,
    DiagnosticAnalysis,
    DiagnosticError
)


class TestSimpTheoremUsage:
    """Test the SimpTheoremUsage dataclass."""
    
    def test_creation_with_all_fields(self):
        """Test creating SimpTheoremUsage with all fields."""
        usage = SimpTheoremUsage(
            name="test_theorem",
            used_count=10,
            tried_count=15,
            success_rate=0.667,
            file_path="test.lean"
        )
        
        assert usage.name == "test_theorem"
        assert usage.used_count == 10
        assert usage.tried_count == 15
        assert usage.success_rate == 0.667
        assert usage.file_path == "test.lean"
    
    def test_creation_with_defaults(self):
        """Test creating SimpTheoremUsage with default values."""
        usage = SimpTheoremUsage(name="test_theorem")
        
        assert usage.name == "test_theorem"
        assert usage.used_count == 0
        assert usage.tried_count == 0
        assert usage.success_rate == 0.0
        assert usage.file_path == ""
    
    def test_success_rate_calculation(self):
        """Test automatic success rate calculation."""
        usage = SimpTheoremUsage(
            name="test_theorem",
            used_count=8,
            tried_count=10
        )
        
        # Success rate should be calculated automatically
        assert usage.success_rate == 0.8
    
    def test_zero_division_in_success_rate(self):
        """Test success rate calculation with zero tried count."""
        usage = SimpTheoremUsage(
            name="test_theorem",
            used_count=0,
            tried_count=0
        )
        
        assert usage.success_rate == 0.0


class TestDiagnosticAnalysis:
    """Test the DiagnosticAnalysis dataclass."""
    
    def test_creation_with_data(self):
        """Test creating DiagnosticAnalysis with data."""
        theorem_usage = [
            SimpTheoremUsage(name="theorem1", used_count=5, tried_count=10),
            SimpTheoremUsage(name="theorem2", used_count=3, tried_count=5)
        ]
        
        analysis = DiagnosticAnalysis(
            total_theorems=2,
            total_files=1,
            theorem_usage=theorem_usage
        )
        
        assert analysis.total_theorems == 2
        assert analysis.total_files == 1
        assert len(analysis.theorem_usage) == 2
        assert analysis.theorem_usage[0].name == "theorem1"
    
    def test_creation_with_defaults(self):
        """Test creating DiagnosticAnalysis with default values."""
        analysis = DiagnosticAnalysis()
        
        assert analysis.total_theorems == 0
        assert analysis.total_files == 0
        assert analysis.theorem_usage == []
    
    def test_get_most_used_theorems(self):
        """Test getting most used theorems."""
        theorem_usage = [
            SimpTheoremUsage(name="theorem1", used_count=10, tried_count=12),
            SimpTheoremUsage(name="theorem2", used_count=20, tried_count=25),
            SimpTheoremUsage(name="theorem3", used_count=5, tried_count=8)
        ]
        
        analysis = DiagnosticAnalysis(theorem_usage=theorem_usage)
        most_used = analysis.get_most_used_theorems(limit=2)
        
        assert len(most_used) == 2
        assert most_used[0].name == "theorem2"
        assert most_used[1].name == "theorem1"
    
    def test_get_high_success_rate_theorems(self):
        """Test getting theorems with high success rate."""
        theorem_usage = [
            SimpTheoremUsage(name="theorem1", used_count=10, tried_count=10),  # 100%
            SimpTheoremUsage(name="theorem2", used_count=8, tried_count=10),   # 80%
            SimpTheoremUsage(name="theorem3", used_count=2, tried_count=10)    # 20%
        ]
        
        analysis = DiagnosticAnalysis(theorem_usage=theorem_usage)
        high_success = analysis.get_high_success_rate_theorems(min_rate=0.75)
        
        assert len(high_success) == 2
        assert high_success[0].name == "theorem1"
        assert high_success[1].name == "theorem2"


class TestDiagnosticParser:
    """Test the DiagnosticParser class."""
    
    @pytest.fixture
    def parser(self):
        """Create a DiagnosticParser instance."""
        return DiagnosticParser()
    
    @pytest.fixture
    def sample_diagnostic_output(self):
        """Sample diagnostic output for testing."""
        return """
[simp] used theorems (max: 250, num: 3):
  list_append_nil ↦ 150
  zero_add ↦ 100
  add_zero ↦ 50

[simp] tried theorems (max: 300, num: 3):
  list_append_nil ↦ 152, succeeded: 150
  zero_add ↦ 120, succeeded: 100
  add_zero ↦ 80, succeeded: 50

[kernel] other diagnostic info...
"""
    
    def test_parse_diagnostic_output_basic(self, parser, sample_diagnostic_output):
        """Test basic parsing of diagnostic output."""
        analysis = parser.parse_diagnostic_output(sample_diagnostic_output)
        
        assert analysis.total_theorems == 3
        assert len(analysis.theorem_usage) == 3
        
        # Check specific theorem data
        list_append_nil = next(
            (t for t in analysis.theorem_usage if t.name == "list_append_nil"),
            None
        )
        assert list_append_nil is not None
        assert list_append_nil.used_count == 150
        assert list_append_nil.tried_count == 152
        assert list_append_nil.success_rate == pytest.approx(0.987, rel=1e-3)
    
    def test_parse_diagnostic_output_with_file_path(self, parser):
        """Test parsing diagnostic output with file path."""
        diagnostic_output = """
[simp] used theorems (max: 250, num: 1):
  theorem1 ↦ 25

[simp] tried theorems (max: 300, num: 1):
  theorem1 ↦ 30, succeeded: 25
"""
        
        analysis = parser.parse_diagnostic_output(diagnostic_output, "test.lean")
        
        assert len(analysis.theorem_usage) == 1
        assert analysis.theorem_usage[0].file_path == "test.lean"
    
    def test_parse_empty_diagnostic_output(self, parser):
        """Test parsing empty diagnostic output."""
        analysis = parser.parse_diagnostic_output("")
        
        assert analysis.total_theorems == 0
        assert analysis.total_files == 0
        assert len(analysis.theorem_usage) == 0
    
    def test_parse_diagnostic_output_no_simp_section(self, parser):
        """Test parsing diagnostic output without simp section."""
        diagnostic_output = """
[kernel] some kernel info
[reduction] some reduction info
"""
        
        analysis = parser.parse_diagnostic_output(diagnostic_output)
        
        assert analysis.total_theorems == 0
        assert len(analysis.theorem_usage) == 0
    
    def test_parse_diagnostic_output_malformed(self, parser):
        """Test parsing malformed diagnostic output."""
        diagnostic_output = """
[simp] used theorems (max: 250, num: 2):
  malformed_line_without_arrow
  valid_theorem ↦ 50

[simp] tried theorems (max: 300, num: 1):
  valid_theorem ↦ 60, succeeded: 50
"""
        
        analysis = parser.parse_diagnostic_output(diagnostic_output)
        
        # Should only parse the valid theorem
        assert len(analysis.theorem_usage) == 1
        assert analysis.theorem_usage[0].name == "valid_theorem"
    
    def test_parse_file_with_diagnostics(self, parser, temp_dir):
        """Test parsing file that contains diagnostic output."""
        lean_file = temp_dir / "test.lean"
        lean_file.write_text("""
theorem test : 1 = 1 := rfl
""")
        
        # Mock subprocess to return diagnostic output
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = """
[simp] used theorems (max: 250, num: 1):
  test_theorem ↦ 10

[simp] tried theorems (max: 300, num: 1):
  test_theorem ↦ 12, succeeded: 10
"""
            mock_run.return_value.returncode = 0
            
            analysis = parser.parse_file_with_diagnostics(lean_file)
            
        assert len(analysis.theorem_usage) == 1
        assert analysis.theorem_usage[0].name == "test_theorem"
        assert analysis.theorem_usage[0].file_path == str(lean_file)
    
    def test_parse_file_with_diagnostics_lean_error(self, parser, temp_dir):
        """Test parsing file when Lean compilation fails."""
        lean_file = temp_dir / "test.lean"
        lean_file.write_text("invalid lean code")
        
        # Mock subprocess to return error
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = "compilation error"
            mock_run.return_value.returncode = 1
            
            with pytest.raises(DiagnosticError):
                parser.parse_file_with_diagnostics(lean_file)
    
    def test_parse_file_with_diagnostics_timeout(self, parser, temp_dir):
        """Test parsing file with timeout."""
        lean_file = temp_dir / "test.lean"
        lean_file.write_text("theorem test : 1 = 1 := rfl")
        
        # Mock subprocess to raise timeout
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(['lean'], 60)
            
            with pytest.raises(DiagnosticError):
                parser.parse_file_with_diagnostics(lean_file, timeout=60)
    
    def test_parse_project_diagnostics(self, parser, sample_lean_project):
        """Test parsing diagnostics for entire project."""
        # Mock subprocess to return diagnostic output
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = """
[simp] used theorems (max: 250, num: 2):
  basic_rule_1 ↦ 5
  advanced_rule_1 ↦ 10

[simp] tried theorems (max: 300, num: 2):
  basic_rule_1 ↦ 6, succeeded: 5
  advanced_rule_1 ↦ 12, succeeded: 10
"""
            mock_run.return_value.returncode = 0
            
            analysis = parser.parse_project_diagnostics(sample_lean_project)
            
        assert analysis.total_theorems == 2
        assert len(analysis.theorem_usage) == 2
        
        # Check that file paths are set correctly
        for theorem in analysis.theorem_usage:
            assert theorem.file_path != ""
    
    def test_parse_project_diagnostics_max_files(self, parser, sample_lean_project):
        """Test parsing project diagnostics with file limit."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = ""
            mock_run.return_value.returncode = 0
            
            analysis = parser.parse_project_diagnostics(sample_lean_project, max_files=1)
            
        # Should limit the number of files processed
        assert mock_run.call_count <= 1
    
    def test_parse_project_diagnostics_nonexistent_path(self, parser):
        """Test parsing diagnostics for nonexistent project."""
        nonexistent_path = Path("/nonexistent/path")
        
        with pytest.raises(DiagnosticError):
            parser.parse_project_diagnostics(nonexistent_path)
    
    def test_merge_analyses(self, parser):
        """Test merging multiple diagnostic analyses."""
        analysis1 = DiagnosticAnalysis(
            total_theorems=2,
            total_files=1,
            theorem_usage=[
                SimpTheoremUsage(name="theorem1", used_count=5, tried_count=10),
                SimpTheoremUsage(name="theorem2", used_count=3, tried_count=5)
            ]
        )
        
        analysis2 = DiagnosticAnalysis(
            total_theorems=1,
            total_files=1,
            theorem_usage=[
                SimpTheoremUsage(name="theorem1", used_count=10, tried_count=15),
                SimpTheoremUsage(name="theorem3", used_count=7, tried_count=10)
            ]
        )
        
        merged = parser.merge_analyses([analysis1, analysis2])
        
        assert merged.total_theorems == 3  # unique theorems
        assert merged.total_files == 2
        assert len(merged.theorem_usage) == 3
        
        # Check that theorem1 counts were merged
        theorem1 = next(t for t in merged.theorem_usage if t.name == "theorem1")
        assert theorem1.used_count == 15  # 5 + 10
        assert theorem1.tried_count == 25  # 10 + 15
    
    def test_is_new_section(self, parser):
        """Test section detection in diagnostic output."""
        assert parser._is_new_section("[simp] used theorems")
        assert parser._is_new_section("[kernel] some info")
        assert parser._is_new_section("[reduction] data")
        assert not parser._is_new_section("  theorem_name ↦ 100")
        assert not parser._is_new_section("regular text")
    
    def test_parse_used_theorems_section(self, parser):
        """Test parsing used theorems section."""
        lines = [
            "[simp] used theorems (max: 250, num: 2):",
            "  theorem1 ↦ 100",
            "  theorem2 ↦ 50"
        ]
        
        used_theorems = parser._parse_used_theorems_section(lines)
        
        assert len(used_theorems) == 2
        assert used_theorems["theorem1"] == 100
        assert used_theorems["theorem2"] == 50
    
    def test_parse_tried_theorems_section(self, parser):
        """Test parsing tried theorems section."""
        lines = [
            "[simp] tried theorems (max: 300, num: 2):",
            "  theorem1 ↦ 120, succeeded: 100",
            "  theorem2 ↦ 80, succeeded: 50"
        ]
        
        tried_theorems = parser._parse_tried_theorems_section(lines)
        
        assert len(tried_theorems) == 2
        assert tried_theorems["theorem1"] == (120, 100)
        assert tried_theorems["theorem2"] == (80, 50)
    
    def test_combine_usage_data(self, parser):
        """Test combining used and tried theorem data."""
        used_theorems = {"theorem1": 100, "theorem2": 50}
        tried_theorems = {"theorem1": (120, 100), "theorem2": (80, 50)}
        
        combined = parser._combine_usage_data(used_theorems, tried_theorems)
        
        assert len(combined) == 2
        
        theorem1 = combined[0]
        assert theorem1.name == "theorem1"
        assert theorem1.used_count == 100
        assert theorem1.tried_count == 120
        assert theorem1.success_rate == pytest.approx(0.833, rel=1e-3)


class TestDiagnosticError:
    """Test the DiagnosticError exception."""
    
    def test_diagnostic_error_creation(self):
        """Test creating DiagnosticError."""
        error = DiagnosticError("Test error message")
        assert str(error) == "Test error message"
    
    def test_diagnostic_error_inheritance(self):
        """Test that DiagnosticError inherits from Exception."""
        error = DiagnosticError("Test error")
        assert isinstance(error, Exception)


# Import subprocess for timeout test
import subprocess
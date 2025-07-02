"""Unit tests for the analyzer module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from simpulse.analyzer import LeanAnalyzer, SimpRule


class TestLeanAnalyzer:
    """Test cases for LeanAnalyzer class."""

    def test_extract_simp_rules_basic(self):
        """Test basic simp rule extraction."""
        analyzer = LeanAnalyzer()
        content = "@[simp] theorem test_rule : true = true := rfl"

        rules = analyzer.extract_simp_rules(content)

        assert len(rules) == 1
        assert rules[0].name == "test_rule"
        assert rules[0].priority is None
        assert rules[0].line_number == 1

    def test_extract_simp_rules_with_priority(self):
        """Test simp rule extraction with custom priority."""
        analyzer = LeanAnalyzer()
        content = "@[simp, priority := 100] theorem high_prio : true = true := rfl"

        rules = analyzer.extract_simp_rules(content)

        assert len(rules) == 1
        assert rules[0].name == "high_prio"
        assert rules[0].priority == 100

    def test_extract_simp_rules_multiple(self, lean_content_with_priorities):
        """Test extraction of multiple simp rules with various priorities."""
        analyzer = LeanAnalyzer()

        rules = analyzer.extract_simp_rules(lean_content_with_priorities)

        assert len(rules) == 5

        # Check specific rules
        rule_names = [rule.name for rule in rules]
        assert "default_prio" in rule_names
        assert "high_prio" in rule_names
        assert "low_prio" in rule_names

        # Check priorities
        high_prio_rule = next(rule for rule in rules if rule.name == "high_prio")
        assert high_prio_rule.priority == 100

        default_rule = next(rule for rule in rules if rule.name == "default_prio")
        assert default_rule.priority is None

    def test_extract_simp_rules_no_rules(self):
        """Test extraction when no simp rules are present."""
        analyzer = LeanAnalyzer()
        content = """
        -- Just some regular theorems
        theorem regular_theorem : 1 + 1 = 2 := rfl
        theorem another_theorem (n : Nat) : n = n := rfl
        """

        rules = analyzer.extract_simp_rules(content)

        assert len(rules) == 0

    def test_extract_simp_rules_complex_patterns(self):
        """Test extraction with complex theorem patterns."""
        analyzer = LeanAnalyzer()
        content = """
        @[simp]
        theorem multiline_rule (n m : Nat) :
          n + m = m + n := by
          rw [Nat.add_comm]

        @[simp, priority := 200] theorem with_params {α : Type} (l : List α) :
          l ++ [] = l := List.append_nil l
        """

        rules = analyzer.extract_simp_rules(content)

        assert len(rules) == 2
        assert rules[0].name == "multiline_rule"
        assert rules[1].name == "with_params"
        assert rules[1].priority == 200

    def test_analyze_file_success(self, sample_lean_file):
        """Test successful file analysis."""
        analyzer = LeanAnalyzer()

        result = analyzer.analyze_file(sample_lean_file)

        assert result is not None
        assert "simp_rules" in result
        assert "file_path" in result
        assert len(result["simp_rules"]) > 0

    def test_analyze_file_not_found(self):
        """Test file analysis with non-existent file."""
        analyzer = LeanAnalyzer()
        non_existent_file = Path("nonexistent.lean")

        with pytest.raises(FileNotFoundError):
            analyzer.analyze_file(non_existent_file)

    def test_analyze_project_success(self, sample_lean_project):
        """Test successful project analysis."""
        analyzer = LeanAnalyzer()

        result = analyzer.analyze_project(sample_lean_project)

        assert result is not None
        assert "total_files" in result
        assert "total_simp_rules" in result
        assert "rules_with_custom_priority" in result
        assert result["total_files"] > 0
        assert result["total_simp_rules"] > 0

    def test_analyze_project_empty_directory(self, temp_dir):
        """Test project analysis with empty directory."""
        analyzer = LeanAnalyzer()

        result = analyzer.analyze_project(temp_dir)

        assert result["total_files"] == 0
        assert result["total_simp_rules"] == 0

    def test_calculate_statistics(self):
        """Test statistics calculation from rules."""
        analyzer = LeanAnalyzer()

        rules = [
            SimpRule("rule1", Path("test.lean"), 1, None, "pattern1", 10),
            SimpRule("rule2", Path("test.lean"), 2, 100, "pattern2", 20),
            SimpRule("rule3", Path("test.lean"), 3, None, "pattern3", 5),
            SimpRule("rule4", Path("test.lean"), 4, 200, "pattern4", 15),
        ]

        stats = analyzer._calculate_statistics(rules)

        assert stats["total_simp_rules"] == 4
        assert stats["rules_with_custom_priority"] == 2
        assert stats["default_priority_percent"] == 50.0
        assert stats["total_usage_frequency"] == 50

    def test_get_optimization_opportunities(self):
        """Test identification of optimization opportunities."""
        analyzer = LeanAnalyzer()

        rules = [
            SimpRule("high_freq_default", Path("test.lean"), 1, None, "p1", 100),  # Opportunity
            SimpRule("low_freq_default", Path("test.lean"), 2, None, "p2", 5),  # Maybe
            SimpRule("high_freq_custom", Path("test.lean"), 3, 100, "p3", 150),  # Already optimized
            SimpRule("medium_freq_default", Path("test.lean"), 4, None, "p4", 50),  # Opportunity
        ]

        opportunities = analyzer._get_optimization_opportunities(rules)

        # Should identify rules with high frequency but default priority
        assert len(opportunities) >= 2
        opportunity_names = [rule.name for rule in opportunities]
        assert "high_freq_default" in opportunity_names
        assert "medium_freq_default" in opportunity_names

    @patch("simpulse.analyzer.subprocess.run")
    def test_validate_lean_syntax_success(self, mock_run):
        """Test successful Lean syntax validation."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        analyzer = LeanAnalyzer()
        result = analyzer._validate_lean_syntax(Path("test.lean"))

        assert result is True
        mock_run.assert_called_once()

    @patch("simpulse.analyzer.subprocess.run")
    def test_validate_lean_syntax_failure(self, mock_run):
        """Test failed Lean syntax validation."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="syntax error")

        analyzer = LeanAnalyzer()
        result = analyzer._validate_lean_syntax(Path("test.lean"))

        assert result is False

    def test_filter_lean_files(self, temp_dir):
        """Test filtering of Lean files from directory."""
        # Create test files
        (temp_dir / "test1.lean").touch()
        (temp_dir / "test2.lean").touch()
        (temp_dir / "README.md").touch()
        (temp_dir / "build" / "temp.olean").mkdir(parents=True)

        analyzer = LeanAnalyzer()
        lean_files = analyzer._get_lean_files(temp_dir)

        lean_file_names = [f.name for f in lean_files]
        assert "test1.lean" in lean_file_names
        assert "test2.lean" in lean_file_names
        assert "README.md" not in lean_file_names
        assert len(lean_files) == 2


class TestSimpRule:
    """Test cases for SimpRule data class."""

    def test_simp_rule_creation(self):
        """Test SimpRule creation with all parameters."""
        rule = SimpRule(
            name="test_rule",
            file_path=Path("test.lean"),
            line_number=42,
            priority=100,
            pattern="test_pattern",
            frequency=10,
        )

        assert rule.name == "test_rule"
        assert rule.file_path == Path("test.lean")
        assert rule.line_number == 42
        assert rule.priority == 100
        assert rule.pattern == "test_pattern"
        assert rule.frequency == 10

    def test_simp_rule_defaults(self):
        """Test SimpRule creation with default values."""
        rule = SimpRule(name="test_rule", file_path=Path("test.lean"), line_number=1)

        assert rule.priority is None
        assert rule.pattern is None
        assert rule.frequency == 0

    def test_simp_rule_equality(self):
        """Test SimpRule equality comparison."""
        rule1 = SimpRule("test", Path("test.lean"), 1, 100, "pattern", 5)
        rule2 = SimpRule("test", Path("test.lean"), 1, 100, "pattern", 5)
        rule3 = SimpRule("other", Path("test.lean"), 1, 100, "pattern", 5)

        assert rule1 == rule2
        assert rule1 != rule3

    def test_simp_rule_string_representation(self):
        """Test SimpRule string representation."""
        rule = SimpRule("test_rule", Path("test.lean"), 42, 100)

        str_repr = str(rule)
        assert "test_rule" in str_repr
        assert "test.lean" in str_repr
        assert "42" in str_repr

    def test_simp_rule_hash(self):
        """Test SimpRule is hashable (for use in sets/dicts)."""
        rule1 = SimpRule("test", Path("test.lean"), 1)
        rule2 = SimpRule("test", Path("test.lean"), 1)

        # Should be able to use in set
        rule_set = {rule1, rule2}
        assert len(rule_set) == 1  # Should deduplicate


# Integration tests would go in tests/integration/
# Performance tests would go in tests/performance/

"""Unit tests for the optimizer module."""

from pathlib import Path
from unittest.mock import patch

from simpulse.analyzer import SimpRule
from simpulse.optimizer import OptimizationSuggestion, PriorityOptimizer


class TestPriorityOptimizer:
    """Test cases for PriorityOptimizer class."""

    def test_calculate_priority_high_frequency(self):
        """Test priority calculation for high frequency rules."""
        optimizer = PriorityOptimizer()

        rule = SimpRule(
            name="high_freq_rule",
            file_path=Path("test.lean"),
            line_number=1,
            priority=None,
            frequency=150,  # High frequency
        )

        priority = optimizer.calculate_priority(rule)

        # High frequency rules should get lower priority numbers (higher priority)
        assert priority < 500
        assert priority >= 100

    def test_calculate_priority_low_frequency(self):
        """Test priority calculation for low frequency rules."""
        optimizer = PriorityOptimizer()

        rule = SimpRule(
            name="low_freq_rule",
            file_path=Path("test.lean"),
            line_number=1,
            priority=None,
            frequency=5,  # Low frequency
        )

        priority = optimizer.calculate_priority(rule)

        # Low frequency rules should get higher priority numbers (lower priority)
        assert priority > 1000
        assert priority <= 2000

    def test_calculate_priority_medium_frequency(self):
        """Test priority calculation for medium frequency rules."""
        optimizer = PriorityOptimizer()

        rule = SimpRule(
            name="medium_freq_rule",
            file_path=Path("test.lean"),
            line_number=1,
            priority=None,
            frequency=50,  # Medium frequency
        )

        priority = optimizer.calculate_priority(rule)

        # Medium frequency rules should get medium priority
        assert 500 <= priority <= 1000

    def test_calculate_priority_with_existing_priority(self):
        """Test that rules with existing priorities are handled correctly."""
        optimizer = PriorityOptimizer()

        rule = SimpRule(
            name="existing_prio_rule",
            file_path=Path("test.lean"),
            line_number=1,
            priority=200,  # Already has priority
            frequency=100,
        )

        # Should still calculate new priority based on frequency
        priority = optimizer.calculate_priority(rule)
        assert isinstance(priority, int)
        assert 100 <= priority <= 2000

    def test_optimize_project_basic(self, analysis_result_sample):
        """Test basic project optimization."""
        optimizer = PriorityOptimizer()

        # Add some rules to the analysis result
        analysis_result_sample["rules_by_frequency"] = [
            SimpRule("rule1", Path("test.lean"), 1, None, "p1", 100),
            SimpRule("rule2", Path("test.lean"), 2, None, "p2", 50),
            SimpRule("rule3", Path("test.lean"), 3, 100, "p3", 25),  # Already has priority
        ]

        suggestions = optimizer.optimize_project(analysis_result_sample)

        assert len(suggestions) > 0
        assert all(isinstance(s, OptimizationSuggestion) for s in suggestions)

    def test_optimize_project_no_opportunities(self):
        """Test optimization when no opportunities exist."""
        optimizer = PriorityOptimizer()

        analysis_result = {
            "total_files": 1,
            "total_simp_rules": 2,
            "rules_with_custom_priority": 2,
            "rules_by_frequency": [
                SimpRule("rule1", Path("test.lean"), 1, 100, "p1", 50),
                SimpRule("rule2", Path("test.lean"), 2, 200, "p2", 30),
            ],
        }

        suggestions = optimizer.optimize_project(analysis_result)

        # Should still generate suggestions, but might be lower confidence
        assert isinstance(suggestions, list)

    def test_generate_optimization_suggestions(self):
        """Test generation of optimization suggestions."""
        optimizer = PriorityOptimizer()

        rules = [
            SimpRule("high_freq", Path("test.lean"), 1, None, "p1", 100),
            SimpRule("medium_freq", Path("test.lean"), 2, None, "p2", 50),
            SimpRule("low_freq", Path("test.lean"), 3, None, "p3", 10),
        ]

        suggestions = optimizer._generate_suggestions(rules)

        assert len(suggestions) == 3

        # Check suggestion structure
        high_freq_suggestion = next(s for s in suggestions if s.rule_name == "high_freq")
        assert high_freq_suggestion.current_priority is None
        assert high_freq_suggestion.suggested_priority < 500  # Should be high priority
        assert high_freq_suggestion.expected_speedup > 0
        assert high_freq_suggestion.confidence in ["high", "medium", "low"]

    def test_estimate_speedup_high_frequency(self):
        """Test speedup estimation for high frequency rules."""
        optimizer = PriorityOptimizer()

        rule = SimpRule("test", Path("test.lean"), 1, None, "p", 200)

        speedup = optimizer._estimate_speedup(rule, 100)

        assert speedup > 0
        assert speedup < 1.0  # Should be a percentage

    def test_estimate_speedup_low_frequency(self):
        """Test speedup estimation for low frequency rules."""
        optimizer = PriorityOptimizer()

        rule = SimpRule("test", Path("test.lean"), 1, None, "p", 5)

        speedup = optimizer._estimate_speedup(rule, 1500)

        assert speedup >= 0
        assert speedup < 0.1  # Should be small for low frequency

    def test_determine_confidence_high(self):
        """Test confidence determination for high confidence cases."""
        optimizer = PriorityOptimizer()

        rule = SimpRule("test", Path("test.lean"), 1, None, "p", 150)

        confidence = optimizer._determine_confidence(rule, 100, 0.25)

        assert confidence == "high"

    def test_determine_confidence_medium(self):
        """Test confidence determination for medium confidence cases."""
        optimizer = PriorityOptimizer()

        rule = SimpRule("test", Path("test.lean"), 1, None, "p", 75)

        confidence = optimizer._determine_confidence(rule, 300, 0.15)

        assert confidence == "medium"

    def test_determine_confidence_low(self):
        """Test confidence determination for low confidence cases."""
        optimizer = PriorityOptimizer()

        rule = SimpRule("test", Path("test.lean"), 1, None, "p", 20)

        confidence = optimizer._determine_confidence(rule, 800, 0.05)

        assert confidence == "low"

    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.exists", return_value=True)
    def test_apply_optimization_to_file(self, mock_exists, mock_read_text, mock_write_text):
        """Test applying optimization to a single file."""
        optimizer = PriorityOptimizer()

        # Mock file content
        original_content = """
@[simp] theorem test_rule : true = true := rfl
@[simp] theorem another_rule : false = false := rfl
"""
        mock_read_text.return_value = original_content

        suggestion = OptimizationSuggestion(
            rule_name="test_rule",
            file_path="test.lean",
            current_priority=None,
            suggested_priority=100,
            reason="High frequency",
            expected_speedup=0.2,
            confidence="high",
        )

        result = optimizer._apply_optimization_to_file(Path("test.lean"), [suggestion])

        assert result is True
        mock_read_text.assert_called_once()
        mock_write_text.assert_called_once()

    @patch("shutil.copy2")
    def test_create_backup(self, mock_copy):
        """Test backup file creation."""
        optimizer = PriorityOptimizer()

        file_path = Path("test.lean")
        backup_path = optimizer._create_backup(file_path)

        assert backup_path.suffix == ".bak"
        assert "test.lean" in str(backup_path)
        mock_copy.assert_called_once()

    def test_generate_optimization_script(self, temp_dir, optimization_suggestions_sample):
        """Test generation of optimization script."""
        optimizer = PriorityOptimizer()

        script_path = temp_dir / "optimize.py"
        optimizations = {"test.lean": optimization_suggestions_sample}

        optimizer.generate_optimization_script(optimizations, script_path)

        assert script_path.exists()
        script_content = script_path.read_text()
        assert "import" in script_content
        assert "test.lean" in script_content
        assert "list_append_nil" in script_content

    def test_priority_ranges(self):
        """Test that generated priorities are within valid ranges."""
        optimizer = PriorityOptimizer()

        # Test various frequency levels
        frequencies = [1, 10, 50, 100, 200, 500, 1000]

        for freq in frequencies:
            rule = SimpRule("test", Path("test.lean"), 1, None, "p", freq)
            priority = optimizer.calculate_priority(rule)

            # Priority should be in valid range
            assert 100 <= priority <= 2000
            assert isinstance(priority, int)

    def test_optimization_consistency(self):
        """Test that optimization results are consistent."""
        optimizer = PriorityOptimizer()

        rule = SimpRule("test", Path("test.lean"), 1, None, "p", 100)

        # Multiple calls should return same result
        priority1 = optimizer.calculate_priority(rule)
        priority2 = optimizer.calculate_priority(rule)

        assert priority1 == priority2


class TestOptimizationSuggestion:
    """Test cases for OptimizationSuggestion data class."""

    def test_optimization_suggestion_creation(self):
        """Test OptimizationSuggestion creation."""
        suggestion = OptimizationSuggestion(
            rule_name="test_rule",
            file_path="test.lean",
            current_priority=None,
            suggested_priority=100,
            reason="High frequency rule",
            expected_speedup=0.25,
            confidence="high",
        )

        assert suggestion.rule_name == "test_rule"
        assert suggestion.file_path == "test.lean"
        assert suggestion.current_priority is None
        assert suggestion.suggested_priority == 100
        assert suggestion.reason == "High frequency rule"
        assert suggestion.expected_speedup == 0.25
        assert suggestion.confidence == "high"

    def test_optimization_suggestion_string_representation(self):
        """Test OptimizationSuggestion string representation."""
        suggestion = OptimizationSuggestion(
            rule_name="test_rule",
            file_path="test.lean",
            current_priority=1000,
            suggested_priority=100,
            reason="High frequency",
            expected_speedup=0.15,
            confidence="medium",
        )

        str_repr = str(suggestion)
        assert "test_rule" in str_repr
        assert "100" in str_repr

    def test_optimization_suggestion_comparison(self):
        """Test OptimizationSuggestion comparison by speedup."""
        suggestion1 = OptimizationSuggestion(
            "rule1", "test.lean", None, 100, "reason", 0.25, "high"
        )
        suggestion2 = OptimizationSuggestion(
            "rule2", "test.lean", None, 200, "reason", 0.15, "medium"
        )

        # Should be sortable by expected speedup
        suggestions = [suggestion2, suggestion1]
        sorted_suggestions = sorted(suggestions, key=lambda s: s.expected_speedup, reverse=True)

        assert sorted_suggestions[0] == suggestion1  # Higher speedup first
        assert sorted_suggestions[1] == suggestion2

    def test_optimization_suggestion_equality(self):
        """Test OptimizationSuggestion equality."""
        suggestion1 = OptimizationSuggestion("test", "test.lean", None, 100, "reason", 0.2, "high")
        suggestion2 = OptimizationSuggestion("test", "test.lean", None, 100, "reason", 0.2, "high")
        suggestion3 = OptimizationSuggestion("other", "test.lean", None, 100, "reason", 0.2, "high")

        assert suggestion1 == suggestion2
        assert suggestion1 != suggestion3

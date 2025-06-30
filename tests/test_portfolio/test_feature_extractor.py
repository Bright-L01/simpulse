"""
Tests for portfolio feature extractor.
"""

import pytest

from simpulse.portfolio.feature_extractor import (
    FeatureCache,
    GoalFeatures,
    LeanGoalParser,
    extract_features,
)


class TestLeanGoalParser:
    """Test Lean goal parser."""

    @pytest.fixture
    def parser(self):
        return LeanGoalParser()

    def test_tokenize(self, parser):
        """Test tokenization of Lean expressions."""
        text = "⊢ ∀ x y : ℕ, x + y = y + x"
        tokens = parser._tokenize(text)

        assert "⊢" in tokens
        assert "∀" in tokens
        assert "+" in tokens
        assert "=" in tokens
        assert "x" in tokens
        assert "y" in tokens

    def test_detect_arithmetic(self, parser):
        """Test arithmetic pattern detection."""
        goal = "⊢ x + y * z = z * y + x"
        features = parser.parse_goal(goal)

        assert features.has_arithmetic
        assert features.has_addition
        assert features.has_multiplication
        assert features.is_equation

    def test_detect_inequality(self, parser):
        """Test inequality detection."""
        goal = "⊢ x < y → y ≤ z → x < z"
        features = parser.parse_goal(goal)

        assert features.is_inequality
        assert not features.is_equation
        assert features.has_logic  # Due to →

    def test_detect_types(self, parser):
        """Test type detection."""
        goals = [
            ("⊢ ∀ x : ℕ, x + 0 = x", "nat"),
            ("⊢ ∀ x : ℤ, x < x + 1", "int"),
            ("⊢ ∀ x : ℝ, x * 1 = x", "real"),
            ("⊢ ∀ l : List α, l ++ [] = l", "list"),
        ]

        for goal, expected_type in goals:
            features = parser.parse_goal(goal)
            assert getattr(features, f"involves_{expected_type}")

    def test_complexity_metrics(self, parser):
        """Test complexity metric calculation."""
        simple_goal = "⊢ x = x"
        complex_goal = "⊢ ∀ x y z : ℝ, (x + y) * (z + (a * b)) = x * z + x * (a * b) + y * z + y * (a * b)"

        simple_features = parser.parse_goal(simple_goal)
        complex_features = parser.parse_goal(complex_goal)

        assert simple_features.total_terms < complex_features.total_terms
        assert simple_features.num_variables < complex_features.num_variables
        assert simple_features.max_nesting < complex_features.max_nesting

    def test_goal_classification(self, parser):
        """Test goal type classification."""
        test_cases = [
            ("⊢ x + 0 = x", "equation"),
            ("⊢ x < y + 1", "linear_inequality"),
            ("⊢ x^2 + 2*x + 1 = (x + 1)^2", "algebraic_equation"),
            ("⊢ p ∧ q → p", "logical"),
            ("⊢ x ∈ A ∪ B", "set_theory"),
        ]

        for goal, expected_type in test_cases:
            features = parser.parse_goal(goal)
            assert expected_type in features.goal_type


class TestGoalFeatures:
    """Test GoalFeatures class."""

    def test_to_vector(self):
        """Test feature vector conversion."""
        features = GoalFeatures(
            goal_type="equation",
            depth=2,
            num_subgoals=1,
            operators={"add": 2, "mul": 1, "eq": 1},
            has_arithmetic=True,
            has_algebra=False,
            has_linear=True,
            has_logic=False,
            has_sets=False,
            num_variables=3,
            num_constants=1,
            num_functions=0,
            max_nesting=2,
            total_terms=10,
            is_equation=True,
            is_inequality=False,
            has_addition=True,
            has_multiplication=True,
            has_subtraction=False,
            has_division=False,
            has_exponentiation=False,
            has_modulo=False,
            involves_nat=True,
            involves_int=False,
            involves_real=False,
            involves_complex=False,
            involves_list=False,
            involves_set=False,
        )

        vector = features.to_vector()

        # Check vector properties
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)
        assert len(vector) > 20  # Should have many features

        # Check specific values
        assert vector[0] == 1.0  # has_arithmetic
        assert vector[5] == 1.0  # is_equation
        assert vector[6] == 0.0  # is_inequality

    def test_vector_normalization(self):
        """Test that numerical features are normalized."""
        features = GoalFeatures(
            goal_type="equation",
            depth=50,  # Very deep
            num_subgoals=10,  # Many subgoals
            operators={},
            has_arithmetic=False,
            has_algebra=False,
            has_linear=False,
            has_logic=False,
            has_sets=False,
            num_variables=100,  # Many variables
            num_constants=50,
            num_functions=30,
            max_nesting=20,
            total_terms=200,
            is_equation=True,
            is_inequality=False,
            has_addition=False,
            has_multiplication=False,
            has_subtraction=False,
            has_division=False,
            has_exponentiation=False,
            has_modulo=False,
            involves_nat=False,
            involves_int=False,
            involves_real=False,
            involves_complex=False,
            involves_list=False,
            involves_set=False,
        )

        vector = features.to_vector()

        # Check that normalized values are capped at 1.0
        # These are the numerical features that should be normalized
        assert all(0 <= x <= 1.0 for x in vector)


class TestFeatureCache:
    """Test feature caching."""

    def test_cache_operations(self):
        """Test cache get/put operations."""
        cache = FeatureCache(max_size=2)

        goal1 = "⊢ x + 0 = x"
        goal2 = "⊢ y * 1 = y"
        goal3 = "⊢ z < z + 1"

        # Extract and cache features
        features1 = extract_features(goal1, cache)
        extract_features(goal2, cache)

        # Test cache hits
        cached_features1 = extract_features(goal1, cache)
        assert cached_features1 == features1
        assert cache.hits == 1

        # Test cache eviction
        extract_features(goal3, cache)

        # goal1 should be evicted (FIFO)
        assert cache.get(goal1) is None
        assert cache.get(goal2) is not None
        assert cache.get(goal3) is not None

    def test_hit_rate(self):
        """Test cache hit rate calculation."""
        cache = FeatureCache()

        goals = ["⊢ x = x", "⊢ y = y", "⊢ x = x"]  # x=x appears twice

        for goal in goals:
            extract_features(goal, cache)

        assert cache.hits == 1  # Second x=x was a hit
        assert cache.misses == 2  # First two were misses
        assert cache.hit_rate == 1 / 3


class TestIntegration:
    """Integration tests for feature extraction."""

    def test_extract_features_comprehensive(self):
        """Test feature extraction on various goal types."""
        test_goals = [
            # Simple arithmetic
            "⊢ x + 0 = x",
            # Polynomial
            "⊢ (x + y)^2 = x^2 + 2*x*y + y^2",
            # Inequality
            "⊢ ∀ x y : ℝ, x < y → x + 1 < y + 1",
            # Logic
            "⊢ ∀ p q : Prop, p ∧ q → p ∨ q",
            # Set theory
            "⊢ ∀ A B : Set α, A ⊆ B → A ∩ B = A",
        ]

        for goal in test_goals:
            features = extract_features(goal)

            # Basic sanity checks
            assert features.goal_type is not None
            assert features.total_terms > 0

            # Check vector conversion
            vector = features.to_vector()
            assert len(vector) > 0
            assert all(isinstance(x, (int, float)) for x in vector)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

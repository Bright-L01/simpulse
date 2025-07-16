"""
Tests for the optimization engine functionality.
"""

import pytest
from unittest.mock import patch, MagicMock

from simpulse.optimization_engine import (
    OptimizationEngine,
    OptimizationRecommendation,
    OptimizationType,
    OptimizationError
)
from simpulse.diagnostic_parser import DiagnosticAnalysis, SimpTheoremUsage


class TestOptimizationType:
    """Test the OptimizationType enum."""
    
    def test_optimization_types_exist(self):
        """Test that all optimization types are defined."""
        assert OptimizationType.PRIORITY_INCREASE is not None
        assert OptimizationType.PRIORITY_DECREASE is not None
        assert OptimizationType.LOOP_DETECTION is not None
        assert OptimizationType.INEFFICIENT_REMOVAL is not None
    
    def test_optimization_type_values(self):
        """Test optimization type string values."""
        assert OptimizationType.PRIORITY_INCREASE.value == "priority_increase"
        assert OptimizationType.PRIORITY_DECREASE.value == "priority_decrease"
        assert OptimizationType.LOOP_DETECTION.value == "loop_detection"
        assert OptimizationType.INEFFICIENT_REMOVAL.value == "inefficient_removal"


class TestOptimizationRecommendation:
    """Test the OptimizationRecommendation dataclass."""
    
    def test_creation_with_all_fields(self):
        """Test creating OptimizationRecommendation with all fields."""
        recommendation = OptimizationRecommendation(
            theorem_name="test_theorem",
            file_path="test.lean",
            optimization_type=OptimizationType.PRIORITY_INCREASE,
            current_priority=1000,
            suggested_priority=100,
            reason="High usage frequency",
            confidence=85.5,
            usage_count=150,
            success_rate=0.92
        )
        
        assert recommendation.theorem_name == "test_theorem"
        assert recommendation.file_path == "test.lean"
        assert recommendation.optimization_type == OptimizationType.PRIORITY_INCREASE
        assert recommendation.current_priority == 1000
        assert recommendation.suggested_priority == 100
        assert recommendation.reason == "High usage frequency"
        assert recommendation.confidence == 85.5
        assert recommendation.usage_count == 150
        assert recommendation.success_rate == 0.92
    
    def test_creation_with_defaults(self):
        """Test creating OptimizationRecommendation with default values."""
        recommendation = OptimizationRecommendation(
            theorem_name="test_theorem",
            file_path="test.lean",
            optimization_type=OptimizationType.PRIORITY_INCREASE,
            reason="Test reason"
        )
        
        assert recommendation.current_priority == 1000  # default
        assert recommendation.suggested_priority == 1000  # default
        assert recommendation.confidence == 0.0  # default
        assert recommendation.usage_count == 0  # default
        assert recommendation.success_rate == 0.0  # default
    
    def test_is_high_confidence(self):
        """Test high confidence detection."""
        high_confidence = OptimizationRecommendation(
            theorem_name="test",
            file_path="test.lean",
            optimization_type=OptimizationType.PRIORITY_INCREASE,
            confidence=85.0,
            reason="Test"
        )
        
        low_confidence = OptimizationRecommendation(
            theorem_name="test",
            file_path="test.lean",
            optimization_type=OptimizationType.PRIORITY_INCREASE,
            confidence=45.0,
            reason="Test"
        )
        
        assert high_confidence.is_high_confidence()
        assert not low_confidence.is_high_confidence()
    
    def test_is_medium_confidence(self):
        """Test medium confidence detection."""
        medium_confidence = OptimizationRecommendation(
            theorem_name="test",
            file_path="test.lean",
            optimization_type=OptimizationType.PRIORITY_INCREASE,
            confidence=65.0,
            reason="Test"
        )
        
        high_confidence = OptimizationRecommendation(
            theorem_name="test",
            file_path="test.lean",
            optimization_type=OptimizationType.PRIORITY_INCREASE,
            confidence=85.0,
            reason="Test"
        )
        
        assert medium_confidence.is_medium_confidence()
        assert not high_confidence.is_medium_confidence()


class TestOptimizationEngine:
    """Test the OptimizationEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create an OptimizationEngine instance."""
        return OptimizationEngine()
    
    @pytest.fixture
    def sample_diagnostic_analysis(self):
        """Create sample diagnostic analysis for testing."""
        return DiagnosticAnalysis(
            total_theorems=5,
            total_files=2,
            theorem_usage=[
                SimpTheoremUsage(
                    name="high_usage_theorem",
                    used_count=150,
                    tried_count=160,
                    success_rate=0.9375,
                    file_path="test1.lean"
                ),
                SimpTheoremUsage(
                    name="low_success_theorem",
                    used_count=50,
                    tried_count=200,
                    success_rate=0.25,
                    file_path="test1.lean"
                ),
                SimpTheoremUsage(
                    name="medium_usage_theorem",
                    used_count=75,
                    tried_count=80,
                    success_rate=0.9375,
                    file_path="test2.lean"
                ),
                SimpTheoremUsage(
                    name="unused_theorem",
                    used_count=0,
                    tried_count=10,
                    success_rate=0.0,
                    file_path="test2.lean"
                ),
                SimpTheoremUsage(
                    name="excessive_usage_theorem",
                    used_count=500,
                    tried_count=505,
                    success_rate=0.99,
                    file_path="test1.lean"
                )
            ]
        )
    
    def test_generate_recommendations_basic(self, engine, sample_diagnostic_analysis):
        """Test basic recommendation generation."""
        recommendations = engine.generate_recommendations(sample_diagnostic_analysis)
        
        assert len(recommendations) > 0
        assert all(isinstance(r, OptimizationRecommendation) for r in recommendations)
        
        # Check that different types of recommendations are generated
        rec_types = {r.optimization_type for r in recommendations}
        assert OptimizationType.PRIORITY_INCREASE in rec_types
        assert OptimizationType.PRIORITY_DECREASE in rec_types
    
    def test_generate_priority_increase_recommendations(self, engine, sample_diagnostic_analysis):
        """Test priority increase recommendation generation."""
        recommendations = engine.generate_recommendations(sample_diagnostic_analysis)
        
        priority_increase_recs = [
            r for r in recommendations
            if r.optimization_type == OptimizationType.PRIORITY_INCREASE
        ]
        
        assert len(priority_increase_recs) > 0
        
        # Check that high usage theorems get priority increase recommendations
        high_usage_rec = next(
            (r for r in priority_increase_recs if r.theorem_name == "high_usage_theorem"),
            None
        )
        assert high_usage_rec is not None
        assert high_usage_rec.suggested_priority < high_usage_rec.current_priority
        assert high_usage_rec.confidence > 0
    
    def test_generate_priority_decrease_recommendations(self, engine, sample_diagnostic_analysis):
        """Test priority decrease recommendation generation."""
        recommendations = engine.generate_recommendations(sample_diagnostic_analysis)
        
        priority_decrease_recs = [
            r for r in recommendations
            if r.optimization_type == OptimizationType.PRIORITY_DECREASE
        ]
        
        assert len(priority_decrease_recs) > 0
        
        # Check that low success rate theorems get priority decrease recommendations
        low_success_rec = next(
            (r for r in priority_decrease_recs if r.theorem_name == "low_success_theorem"),
            None
        )
        assert low_success_rec is not None
        assert low_success_rec.suggested_priority > low_success_rec.current_priority
        assert low_success_rec.confidence > 0
    
    def test_generate_loop_detection_recommendations(self, engine, sample_diagnostic_analysis):
        """Test loop detection recommendation generation."""
        recommendations = engine.generate_recommendations(sample_diagnostic_analysis)
        
        loop_detection_recs = [
            r for r in recommendations
            if r.optimization_type == OptimizationType.LOOP_DETECTION
        ]
        
        # Check if excessive usage is detected
        excessive_usage_rec = next(
            (r for r in loop_detection_recs if r.theorem_name == "excessive_usage_theorem"),
            None
        )
        
        if excessive_usage_rec:
            assert excessive_usage_rec.confidence > 0
            assert "excessive" in excessive_usage_rec.reason.lower()
    
    def test_generate_inefficient_removal_recommendations(self, engine, sample_diagnostic_analysis):
        """Test inefficient removal recommendation generation."""
        recommendations = engine.generate_recommendations(sample_diagnostic_analysis)
        
        inefficient_removal_recs = [
            r for r in recommendations
            if r.optimization_type == OptimizationType.INEFFICIENT_REMOVAL
        ]
        
        # Check that completely unused theorems get removal recommendations
        unused_rec = next(
            (r for r in inefficient_removal_recs if r.theorem_name == "unused_theorem"),
            None
        )
        
        if unused_rec:
            assert unused_rec.confidence > 0
            assert "unused" in unused_rec.reason.lower() or "never used" in unused_rec.reason.lower()
    
    def test_calculate_confidence_priority_increase(self, engine):
        """Test confidence calculation for priority increase."""
        theorem_usage = SimpTheoremUsage(
            name="test_theorem",
            used_count=100,
            tried_count=110,
            success_rate=0.909
        )
        
        confidence = engine._calculate_confidence(
            theorem_usage,
            OptimizationType.PRIORITY_INCREASE
        )
        
        assert 0 <= confidence <= 100
        assert confidence > 50  # Should be high confidence for good success rate and usage
    
    def test_calculate_confidence_priority_decrease(self, engine):
        """Test confidence calculation for priority decrease."""
        theorem_usage = SimpTheoremUsage(
            name="test_theorem",
            used_count=20,
            tried_count=100,
            success_rate=0.2
        )
        
        confidence = engine._calculate_confidence(
            theorem_usage,
            OptimizationType.PRIORITY_DECREASE
        )
        
        assert 0 <= confidence <= 100
        assert confidence > 50  # Should be high confidence for low success rate
    
    def test_calculate_confidence_loop_detection(self, engine):
        """Test confidence calculation for loop detection."""
        theorem_usage = SimpTheoremUsage(
            name="test_theorem",
            used_count=1000,
            tried_count=1010,
            success_rate=0.99
        )
        
        confidence = engine._calculate_confidence(
            theorem_usage,
            OptimizationType.LOOP_DETECTION
        )
        
        assert 0 <= confidence <= 100
        # High usage should indicate potential loop
        assert confidence > 0
    
    def test_calculate_confidence_inefficient_removal(self, engine):
        """Test confidence calculation for inefficient removal."""
        theorem_usage = SimpTheoremUsage(
            name="test_theorem",
            used_count=0,
            tried_count=50,
            success_rate=0.0
        )
        
        confidence = engine._calculate_confidence(
            theorem_usage,
            OptimizationType.INEFFICIENT_REMOVAL
        )
        
        assert 0 <= confidence <= 100
        assert confidence > 50  # Should be high confidence for completely unused theorem
    
    def test_suggest_priority_increase(self, engine):
        """Test priority increase suggestion."""
        theorem_usage = SimpTheoremUsage(
            name="test_theorem",
            used_count=100,
            tried_count=110,
            success_rate=0.909
        )
        
        new_priority = engine._suggest_priority(
            theorem_usage,
            OptimizationType.PRIORITY_INCREASE
        )
        
        assert new_priority < 1000  # Should be higher priority (lower number)
        assert new_priority > 0  # Should be positive
    
    def test_suggest_priority_decrease(self, engine):
        """Test priority decrease suggestion."""
        theorem_usage = SimpTheoremUsage(
            name="test_theorem",
            used_count=20,
            tried_count=100,
            success_rate=0.2
        )
        
        new_priority = engine._suggest_priority(
            theorem_usage,
            OptimizationType.PRIORITY_DECREASE
        )
        
        assert new_priority > 1000  # Should be lower priority (higher number)
    
    def test_generate_reason_priority_increase(self, engine):
        """Test reason generation for priority increase."""
        theorem_usage = SimpTheoremUsage(
            name="test_theorem",
            used_count=100,
            tried_count=110,
            success_rate=0.909
        )
        
        reason = engine._generate_reason(
            theorem_usage,
            OptimizationType.PRIORITY_INCREASE
        )
        
        assert "used" in reason.lower()
        assert "success" in reason.lower()
        assert "100" in reason
    
    def test_generate_reason_priority_decrease(self, engine):
        """Test reason generation for priority decrease."""
        theorem_usage = SimpTheoremUsage(
            name="test_theorem",
            used_count=20,
            tried_count=100,
            success_rate=0.2
        )
        
        reason = engine._generate_reason(
            theorem_usage,
            OptimizationType.PRIORITY_DECREASE
        )
        
        assert "low" in reason.lower()
        assert "success" in reason.lower()
        assert "20.0%" in reason
    
    def test_filter_recommendations_by_confidence(self, engine, sample_diagnostic_analysis):
        """Test filtering recommendations by confidence threshold."""
        all_recommendations = engine.generate_recommendations(sample_diagnostic_analysis)
        
        high_confidence_recs = engine.filter_recommendations_by_confidence(
            all_recommendations,
            confidence_threshold=80.0
        )
        
        assert len(high_confidence_recs) <= len(all_recommendations)
        assert all(r.confidence >= 80.0 for r in high_confidence_recs)
    
    def test_filter_recommendations_by_type(self, engine, sample_diagnostic_analysis):
        """Test filtering recommendations by type."""
        all_recommendations = engine.generate_recommendations(sample_diagnostic_analysis)
        
        priority_increase_recs = engine.filter_recommendations_by_type(
            all_recommendations,
            OptimizationType.PRIORITY_INCREASE
        )
        
        assert all(
            r.optimization_type == OptimizationType.PRIORITY_INCREASE
            for r in priority_increase_recs
        )
    
    def test_sort_recommendations_by_confidence(self, engine, sample_diagnostic_analysis):
        """Test sorting recommendations by confidence."""
        recommendations = engine.generate_recommendations(sample_diagnostic_analysis)
        
        sorted_recs = engine.sort_recommendations_by_confidence(recommendations)
        
        # Check that recommendations are sorted by confidence (descending)
        for i in range(len(sorted_recs) - 1):
            assert sorted_recs[i].confidence >= sorted_recs[i + 1].confidence
    
    def test_sort_recommendations_by_usage(self, engine, sample_diagnostic_analysis):
        """Test sorting recommendations by usage count."""
        recommendations = engine.generate_recommendations(sample_diagnostic_analysis)
        
        sorted_recs = engine.sort_recommendations_by_usage(recommendations)
        
        # Check that recommendations are sorted by usage count (descending)
        for i in range(len(sorted_recs) - 1):
            assert sorted_recs[i].usage_count >= sorted_recs[i + 1].usage_count
    
    def test_empty_diagnostic_analysis(self, engine):
        """Test recommendation generation with empty analysis."""
        empty_analysis = DiagnosticAnalysis()
        
        recommendations = engine.generate_recommendations(empty_analysis)
        
        assert len(recommendations) == 0
    
    def test_single_theorem_analysis(self, engine):
        """Test recommendation generation with single theorem."""
        single_theorem_analysis = DiagnosticAnalysis(
            total_theorems=1,
            total_files=1,
            theorem_usage=[
                SimpTheoremUsage(
                    name="single_theorem",
                    used_count=50,
                    tried_count=60,
                    success_rate=0.833,
                    file_path="test.lean"
                )
            ]
        )
        
        recommendations = engine.generate_recommendations(single_theorem_analysis)
        
        assert len(recommendations) >= 1
        assert all(r.theorem_name == "single_theorem" for r in recommendations)
    
    def test_recommendation_validation(self, engine):
        """Test that generated recommendations are valid."""
        theorem_usage = SimpTheoremUsage(
            name="test_theorem",
            used_count=100,
            tried_count=110,
            success_rate=0.909,
            file_path="test.lean"
        )
        
        analysis = DiagnosticAnalysis(
            total_theorems=1,
            total_files=1,
            theorem_usage=[theorem_usage]
        )
        
        recommendations = engine.generate_recommendations(analysis)
        
        for rec in recommendations:
            # Basic validation
            assert rec.theorem_name
            assert rec.file_path
            assert rec.reason
            assert 0 <= rec.confidence <= 100
            assert rec.suggested_priority > 0
            assert rec.usage_count >= 0
            assert 0 <= rec.success_rate <= 1


class TestOptimizationError:
    """Test the OptimizationError exception."""
    
    def test_optimization_error_creation(self):
        """Test creating OptimizationError."""
        error = OptimizationError("Test error message")
        assert str(error) == "Test error message"
    
    def test_optimization_error_inheritance(self):
        """Test that OptimizationError inherits from Exception."""
        error = OptimizationError("Test error")
        assert isinstance(error, Exception)
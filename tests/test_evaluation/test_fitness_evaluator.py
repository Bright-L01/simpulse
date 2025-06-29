"""
Tests for fitness evaluation functionality.
"""

from unittest.mock import Mock, patch

import pytest

from simpulse.evaluation.fitness_evaluator import FitnessEvaluator, FitnessScore
from simpulse.evolution.models_v2 import Candidate


class TestFitnessEvaluator:
    """Test suite for FitnessEvaluator."""

    @pytest.fixture
    def evaluator(self, mock_config):
        """Create a FitnessEvaluator instance for testing."""
        return FitnessEvaluator(mock_config)

    @pytest.mark.asyncio
    async def test_evaluate_candidate_success(self, evaluator):
        """Test successful candidate evaluation."""
        candidate = Candidate(mutations=[])

        baseline_profiles = {
            "TestModule": {
                "simp_time": 10.0,
                "total_time": 50.0,
                "memory_peak": 256.0,
                "iterations": 100,
                "depth": 5,
            }
        }

        # Mock profiler response
        with patch.object(evaluator.profiler, "profile_with_mutations") as mock_profile:
            mock_result = Mock()
            mock_result.success = True
            mock_result.module_profiles = {
                "TestModule": {
                    "simp_time": 8.0,  # 20% improvement
                    "total_time": 42.0,  # 16% improvement
                    "memory_peak": 240.0,  # 6.25% improvement
                    "iterations": 85,  # 15% improvement
                    "depth": 4,  # 20% improvement
                }
            }
            mock_profile.return_value = mock_result

            # Evaluate candidate
            fitness = await evaluator.evaluate_candidate(candidate, baseline_profiles)

            # Verify fitness score
            assert isinstance(fitness, FitnessScore)
            assert fitness.time_score > 0.5
            assert fitness.memory_score > 0.5
            assert fitness.iterations_score > 0.5
            assert fitness.depth_score > 0.5
            assert 0 < fitness.composite_score < 1

    @pytest.mark.asyncio
    async def test_evaluate_candidate_failure(self, evaluator):
        """Test candidate evaluation with profiling failure."""
        candidate = Candidate(mutations=[])
        baseline_profiles = {"TestModule": {"simp_time": 10.0}}

        # Mock profiler failure
        with patch.object(evaluator.profiler, "profile_with_mutations") as mock_profile:
            mock_result = Mock()
            mock_result.success = False
            mock_result.error = "Profiling failed"
            mock_profile.return_value = mock_result

            # Evaluate candidate
            fitness = await evaluator.evaluate_candidate(candidate, baseline_profiles)

            # Should return zero fitness
            assert isinstance(fitness, FitnessScore)
            assert fitness.composite_score == 0.0

    @pytest.mark.asyncio
    async def test_batch_evaluation(self, evaluator):
        """Test batch candidate evaluation."""
        candidates = [Candidate(mutations=[]) for _ in range(5)]
        baseline_profiles = {"TestModule": {"simp_time": 10.0}}

        # Mock profiler
        with patch.object(evaluator.profiler, "profile_with_mutations") as mock_profile:
            mock_result = Mock()
            mock_result.success = True
            mock_result.module_profiles = {
                "TestModule": {
                    "simp_time": 9.0,
                    "total_time": 45.0,
                    "memory_peak": 250.0,
                    "iterations": 90,
                    "depth": 5,
                }
            }
            mock_profile.return_value = mock_result

            # Evaluate batch
            results = await evaluator.evaluate_batch(candidates, baseline_profiles)

            # Verify results
            assert len(results) == 5
            assert all(isinstance(r, FitnessScore) for r in results)
            assert all(r.composite_score > 0 for r in results)

    def test_calculate_time_score(self, evaluator):
        """Test time score calculation."""
        # Test improvement
        score = evaluator._calculate_time_score(10.0, 8.0)
        assert score > 0.5  # Improvement should give score > 0.5

        # Test no change
        score = evaluator._calculate_time_score(10.0, 10.0)
        assert score == 0.5

        # Test degradation
        score = evaluator._calculate_time_score(10.0, 12.0)
        assert score < 0.5  # Degradation should give score < 0.5

    def test_calculate_memory_score(self, evaluator):
        """Test memory score calculation."""
        # Test improvement (less memory)
        score = evaluator._calculate_memory_score(256.0, 200.0)
        assert score > 0.5

        # Test no change
        score = evaluator._calculate_memory_score(256.0, 256.0)
        assert score == 0.5

        # Test degradation (more memory)
        score = evaluator._calculate_memory_score(256.0, 300.0)
        assert score < 0.5

    def test_calculate_iterations_score(self, evaluator):
        """Test iterations score calculation."""
        # Test improvement (fewer iterations)
        score = evaluator._calculate_iterations_score(100, 80)
        assert score > 0.5

        # Test no change
        score = evaluator._calculate_iterations_score(100, 100)
        assert score == 0.5

        # Test degradation (more iterations)
        score = evaluator._calculate_iterations_score(100, 120)
        assert score < 0.5

    def test_calculate_depth_score(self, evaluator):
        """Test depth score calculation."""
        # Test improvement (less depth)
        score = evaluator._calculate_depth_score(10, 8)
        assert score > 0.5

        # Test no change
        score = evaluator._calculate_depth_score(10, 10)
        assert score == 0.5

        # Test degradation (more depth)
        score = evaluator._calculate_depth_score(10, 12)
        assert score < 0.5

    def test_normalize_score(self, evaluator):
        """Test score normalization."""
        # Test various improvements
        assert evaluator._normalize_score(0.2) > 0.5  # 20% improvement
        assert evaluator._normalize_score(0.0) == 0.5  # No change
        assert evaluator._normalize_score(-0.2) < 0.5  # 20% degradation

        # Test bounds
        assert 0 <= evaluator._normalize_score(1.0) <= 1
        assert 0 <= evaluator._normalize_score(-1.0) <= 1

    @pytest.mark.asyncio
    async def test_parallel_evaluation(self, evaluator):
        """Test parallel candidate evaluation."""
        candidates = [Candidate(mutations=[]) for _ in range(10)]
        baseline_profiles = {"TestModule": {"simp_time": 10.0}}

        with patch.object(evaluator, "evaluate_candidate") as mock_eval:
            # Different scores for each candidate
            mock_eval.side_effect = [
                FitnessScore(composite_score=0.5 + i * 0.05) for i in range(10)
            ]

            # Evaluate in parallel
            results = await evaluator.evaluate_batch(
                candidates, baseline_profiles, max_parallel=4
            )

            # Verify results
            assert len(results) == 10
            assert results[0].composite_score == 0.5
            assert results[9].composite_score == 0.95

    @pytest.mark.asyncio
    async def test_caching(self, evaluator, temp_dir):
        """Test fitness evaluation caching."""
        evaluator.cache_dir = temp_dir
        evaluator.use_cache = True

        candidate = Candidate(mutations=[])
        baseline_profiles = {"TestModule": {"simp_time": 10.0}}

        # First evaluation
        with patch.object(evaluator.profiler, "profile_with_mutations") as mock_profile:
            mock_result = Mock()
            mock_result.success = True
            mock_result.module_profiles = {"TestModule": {"simp_time": 8.0}}
            mock_profile.return_value = mock_result

            fitness1 = await evaluator.evaluate_candidate(candidate, baseline_profiles)

            # Should call profiler
            assert mock_profile.call_count == 1

        # Second evaluation of same candidate
        with patch.object(evaluator.profiler, "profile_with_mutations") as mock_profile:
            fitness2 = await evaluator.evaluate_candidate(candidate, baseline_profiles)

            # Should use cache, not call profiler
            assert mock_profile.call_count == 0
            assert fitness2.composite_score == fitness1.composite_score

    def test_multi_objective_scoring(self, evaluator):
        """Test multi-objective fitness scoring."""
        # Create candidate with specific improvements
        baseline = {
            "simp_time": 10.0,
            "total_time": 50.0,
            "memory_peak": 256.0,
            "iterations": 100,
            "depth": 5,
        }

        optimized = {
            "simp_time": 5.0,  # 50% improvement
            "total_time": 55.0,  # 10% degradation
            "memory_peak": 128.0,  # 50% improvement
            "iterations": 150,  # 50% degradation
            "depth": 3,  # 40% improvement
        }

        fitness = evaluator._calculate_multi_objective_fitness(baseline, optimized)

        # Should balance improvements and degradations
        assert isinstance(fitness, FitnessScore)
        assert fitness.time_score > 0.5  # Good simp time improvement
        assert fitness.memory_score > 0.5  # Good memory improvement
        assert fitness.iterations_score < 0.5  # Poor iterations
        assert fitness.depth_score > 0.5  # Good depth improvement


class TestFitnessScore:
    """Test suite for FitnessScore dataclass."""

    def test_fitness_score_creation(self):
        """Test creating FitnessScore instances."""
        score = FitnessScore(
            time_score=0.8,
            memory_score=0.7,
            iterations_score=0.9,
            depth_score=0.85,
            composite_score=0.8125,
        )

        assert score.time_score == 0.8
        assert score.memory_score == 0.7
        assert score.iterations_score == 0.9
        assert score.depth_score == 0.85
        assert score.composite_score == 0.8125

    def test_fitness_score_defaults(self):
        """Test FitnessScore default values."""
        score = FitnessScore()

        assert score.time_score == 0.0
        assert score.memory_score == 0.0
        assert score.iterations_score == 0.0
        assert score.depth_score == 0.0
        assert score.composite_score == 0.0

    def test_fitness_score_comparison(self):
        """Test comparing fitness scores."""
        score1 = FitnessScore(composite_score=0.8)
        score2 = FitnessScore(composite_score=0.6)

        # Should be comparable by composite score
        assert score1.composite_score > score2.composite_score

    def test_fitness_score_validation(self):
        """Test fitness score validation."""
        # All scores should be between 0 and 1
        score = FitnessScore(
            time_score=0.5,
            memory_score=0.5,
            iterations_score=0.5,
            depth_score=0.5,
            composite_score=0.5,
        )

        for value in [
            score.time_score,
            score.memory_score,
            score.iterations_score,
            score.depth_score,
            score.composite_score,
        ]:
            assert 0 <= value <= 1


@pytest.mark.integration
class TestFitnessEvaluatorIntegration:
    """Integration tests for fitness evaluator."""

    @pytest.mark.requires_lean
    @pytest.mark.asyncio
    async def test_real_evaluation(self, mock_config, mock_lean_project):
        """Test with real Lean evaluation."""
        FitnessEvaluator(mock_config)

        # Would test with real Lean modules if available
        # This is a placeholder for integration testing

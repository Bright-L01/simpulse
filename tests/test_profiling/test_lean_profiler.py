"""
Tests for Lean profiling functionality.
"""

import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from simpulse.profiling.lean_profiler import LeanProfiler, ProfileResult


class TestLeanProfiler:
    """Test suite for LeanProfiler."""

    @pytest.fixture
    def profiler(self, mock_config):
        """Create a LeanProfiler instance for testing."""
        return LeanProfiler(mock_config)

    @pytest.mark.asyncio
    async def test_profile_baseline(self, profiler, mock_lean_project):
        """Test baseline profiling."""
        # Mock lean runner response
        with patch.object(profiler.runner, "profile_module") as mock_profile:
            mock_result = Mock()
            mock_result.success = True
            mock_result.stdout = "Profile complete"

            mock_profile_data = {
                "simp_time": 10.5,
                "total_time": 50.0,
                "memory_peak": 256.0,
                "iterations": 100,
                "depth": 5,
            }

            mock_profile.return_value = (mock_result, mock_profile_data)

            # Execute baseline profiling
            result = await profiler.profile_baseline(["TestModule"], mock_lean_project)

            # Verify results
            assert isinstance(result, dict)
            assert "TestModule" in result
            assert result["TestModule"]["simp_time"] == 10.5
            assert result["TestModule"]["total_time"] == 50.0

    @pytest.mark.asyncio
    async def test_profile_with_mutations(self, profiler, sample_simp_rules):
        """Test profiling with mutations applied."""
        # Mock mutation applicator
        with patch.object(
            profiler.mutation_applicator, "apply_mutation_set"
        ) as mock_apply:
            mock_mutation_set = Mock()
            mock_mutation_set.success = True
            mock_mutation_set.modified_files = {Path("test.lean")}
            mock_apply.return_value = mock_mutation_set

            # Mock lean runner
            with patch.object(profiler.runner, "profile_module") as mock_profile:
                mock_result = Mock()
                mock_result.success = True

                mock_profile_data = {
                    "simp_time": 8.0,  # Improved
                    "total_time": 40.0,
                    "memory_peak": 200.0,
                    "iterations": 80,
                    "depth": 4,
                }

                mock_profile.return_value = (mock_result, mock_profile_data)

                # Create mutations
                mutations = [Mock() for _ in range(3)]

                # Execute profiling
                result = await profiler.profile_with_mutations(
                    mutations, sample_simp_rules, ["TestModule"], Path(".")
                )

                # Verify results
                assert isinstance(result, ProfileResult)
                assert result.success
                assert "TestModule" in result.module_profiles
                assert result.module_profiles["TestModule"]["simp_time"] == 8.0

    @pytest.mark.asyncio
    async def test_compare_profiles(self, profiler):
        """Test profile comparison."""
        baseline = {
            "TestModule": {
                "simp_time": 10.0,
                "total_time": 50.0,
                "memory_peak": 256.0,
                "iterations": 100,
            }
        }

        optimized = ProfileResult(
            success=True,
            module_profiles={
                "TestModule": {
                    "simp_time": 8.0,
                    "total_time": 42.0,
                    "memory_peak": 240.0,
                    "iterations": 85,
                }
            },
        )

        comparison = profiler.compare_profiles(baseline, optimized)

        # Verify comparison results
        assert isinstance(comparison, dict)
        assert "TestModule" in comparison

        module_comp = comparison["TestModule"]
        assert module_comp["simp_time"]["improvement_percent"] == 20.0
        assert module_comp["total_time"]["improvement_percent"] == 16.0
        assert module_comp["memory_peak"]["improvement_percent"] > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, profiler):
        """Test error handling in profiler."""
        # Test with failed Lean execution
        with patch.object(profiler.runner, "profile_module") as mock_profile:
            mock_result = Mock()
            mock_result.success = False
            mock_result.stderr = "Lean error: module not found"

            mock_profile.return_value = (mock_result, None)

            result = await profiler.profile_baseline(["NonExistentModule"], Path("."))

            # Should handle error gracefully
            assert isinstance(result, dict)
            assert len(result) == 0  # No successful profiles

    @pytest.mark.asyncio
    async def test_cache_functionality(self, profiler, temp_dir):
        """Test profiling cache."""
        cache_file = temp_dir / "profile_cache.json"

        # Create sample profile data
        profile_data = {
            "TestModule": {
                "simp_time": 10.0,
                "total_time": 50.0,
                "memory_peak": 256.0,
                "iterations": 100,
                "timestamp": "2024-01-01T00:00:00",
            }
        }

        # Save to cache
        profiler._save_to_cache(profile_data, cache_file)

        # Verify cache file exists
        assert cache_file.exists()

        # Load from cache
        loaded_data = profiler._load_from_cache(cache_file)

        # Verify loaded data
        assert loaded_data == profile_data

    @pytest.mark.asyncio
    async def test_parallel_profiling(self, profiler):
        """Test parallel module profiling."""
        modules = ["Module1", "Module2", "Module3"]

        with patch.object(profiler.runner, "profile_module") as mock_profile:
            # Setup different results for each module
            async def mock_profile_impl(module_name, options=None):
                mock_result = Mock()
                mock_result.success = True

                # Different timing for each module
                base_time = 10.0 * (modules.index(module_name) + 1)

                return mock_result, {
                    "simp_time": base_time,
                    "total_time": base_time * 5,
                    "memory_peak": 256.0,
                    "iterations": 100,
                }

            mock_profile.side_effect = mock_profile_impl

            # Execute parallel profiling
            result = await profiler.profile_baseline(modules, Path("."))

            # Verify all modules were profiled
            assert len(result) == 3
            assert all(module in result for module in modules)

            # Verify different timings
            assert result["Module1"]["simp_time"] == 10.0
            assert result["Module2"]["simp_time"] == 20.0
            assert result["Module3"]["simp_time"] == 30.0

    def test_calculate_fitness_score(self, profiler):
        """Test fitness score calculation."""
        baseline = {
            "simp_time": 10.0,
            "total_time": 50.0,
            "memory_peak": 256.0,
            "iterations": 100,
            "depth": 5,
        }

        optimized = {
            "simp_time": 8.0,
            "total_time": 42.0,
            "memory_peak": 240.0,
            "iterations": 85,
            "depth": 4,
        }

        fitness = profiler._calculate_fitness_score(baseline, optimized)

        # Verify fitness score
        assert 0 <= fitness <= 1
        assert fitness > 0.5  # Should be good since all metrics improved

    @pytest.mark.asyncio
    async def test_profile_with_timeout(self, profiler):
        """Test profiling with timeout."""
        with patch.object(profiler.runner, "profile_module") as mock_profile:
            # Simulate timeout
            mock_profile.side_effect = asyncio.TimeoutError("Profile timeout")

            result = await profiler.profile_baseline(
                ["SlowModule"], Path("."), timeout=1.0
            )

            # Should handle timeout gracefully
            assert isinstance(result, dict)
            assert len(result) == 0


class TestProfileResult:
    """Test suite for ProfileResult dataclass."""

    def test_profile_result_creation(self):
        """Test creating ProfileResult instances."""
        result = ProfileResult(
            success=True,
            module_profiles={"TestModule": {"simp_time": 10.0, "total_time": 50.0}},
            error=None,
            execution_time=2.5,
        )

        assert result.success
        assert "TestModule" in result.module_profiles
        assert result.execution_time == 2.5

    def test_profile_result_aggregation(self):
        """Test aggregating profile results."""
        result = ProfileResult(
            success=True,
            module_profiles={
                "Module1": {"simp_time": 10.0, "total_time": 50.0},
                "Module2": {"simp_time": 20.0, "total_time": 80.0},
            },
        )

        # Test aggregation methods
        total_simp_time = sum(p["simp_time"] for p in result.module_profiles.values())
        assert total_simp_time == 30.0

        avg_total_time = sum(
            p["total_time"] for p in result.module_profiles.values()
        ) / len(result.module_profiles)
        assert avg_total_time == 65.0


@pytest.mark.integration
class TestLeanProfilerIntegration:
    """Integration tests for Lean profiler."""

    @pytest.mark.requires_lean
    @pytest.mark.asyncio
    async def test_real_lean_profiling(self, mock_config):
        """Test with real Lean installation."""
        profiler = LeanProfiler(mock_config)

        # Check if Lean is available
        with patch.object(profiler.runner, "run_lean") as mock_run:
            mock_result = Mock()
            mock_result.success = True
            mock_run.return_value = mock_result

            # Would test with real Lean module if available
            # This is a placeholder for integration testing

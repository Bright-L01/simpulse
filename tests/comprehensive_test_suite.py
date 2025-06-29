"""
Comprehensive test suite for Simpulse.

This module provides the main test runner and fixtures for all Simpulse tests.
Individual test modules are organized by component.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, Mock

import pytest

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_lean_project(temp_dir: Path) -> Path:
    """Create a mock Lean project structure for testing."""
    project_dir = temp_dir / "test_project"
    project_dir.mkdir()

    # Create lakefile
    lakefile = project_dir / "lakefile.lean"
    lakefile.write_text(
        """
import Lake
open Lake DSL

package test where
  -- add any package configuration options here

lean_lib TestLib where
  -- add any library configuration options here

@[default_target]
lean_exe test where
  root := `Main
"""
    )

    # Create test Lean files
    test_lib = project_dir / "TestLib"
    test_lib.mkdir()

    basic_file = test_lib / "Basic.lean"
    basic_file.write_text(
        """
-- Test Lean file with simp rules

@[simp]
theorem test_simp_rule (n : Nat) : n + 0 = n := by simp

@[simp high]
theorem test_high_priority (n : Nat) : 0 + n = n := by simp

@[simp ↓]
theorem test_direction (n : Nat) : n * 1 = n := by simp
"""
    )

    algebra_file = test_lib / "Algebra.lean"
    algebra_file.write_text(
        """
-- Algebra-specific simp rules

@[simp 1000]
theorem mul_zero (n : Nat) : n * 0 = 0 := by simp

@[simp]
theorem add_assoc (a b c : Nat) : (a + b) + c = a + (b + c) := by simp
"""
    )

    return project_dir


@pytest.fixture
def mock_config(temp_dir: Path):
    """Create a mock configuration for testing."""
    from simpulse.config import ClaudeConfig, Config, OptimizationConfig, PathConfig

    return Config(
        optimization=OptimizationConfig(
            population_size=10,  # Small for testing
            generations=5,  # Short for testing
            time_budget=60,  # 1 minute for testing
            target_improvement=5.0,
            max_parallel_evaluations=1,  # Single-threaded for testing
        ),
        claude=ClaudeConfig(backend="mock", timeout_seconds=5),
        paths=PathConfig(
            output_dir=temp_dir / "output",
            cache_dir=temp_dir / "cache",
            log_dir=temp_dir / "logs",
        ),
    )


@pytest.fixture
def mock_claude_client():
    """Mock Claude client for testing."""
    from simpulse.claude.claude_code_client import ClaudeCodeClient, ClaudeResponse

    mock_client = Mock(spec=ClaudeCodeClient)

    # Mock successful response
    mock_response = ClaudeResponse(
        content="Mock mutation suggestion: Increase priority of test_rule to 1100",
        success=True,
        tokens_used=50,
        execution_time=0.5,
        model="mock-model",
        cached=False,
    )

    mock_client.query_claude = AsyncMock(return_value=mock_response)
    mock_client.is_available = Mock(return_value=True)

    return mock_client


@pytest.fixture
def mock_lean_runner():
    """Mock Lean runner for testing."""
    from simpulse.profiling.lean_runner import LeanRunner

    mock_runner = Mock(spec=LeanRunner)

    # Mock successful Lean execution
    mock_runner.run_lean = AsyncMock(return_value="Mock lean output with timing info")
    mock_runner.profile_module = AsyncMock(
        return_value={
            "simp_time": 1.5,
            "total_time": 5.0,
            "memory_peak": 100.0,
            "iterations": 25,
        }
    )
    mock_runner.is_available = Mock(return_value=True)

    return mock_runner


@pytest.fixture
def sample_simp_rules():
    """Create sample simp rules for testing."""
    from simpulse.evolution.models import SimpDirection, SimpPriority, SimpRule

    return [
        SimpRule(
            rule_name="TestLib.Basic.test_simp_rule",
            full_attribute="@[simp]",
            simp_priority=SimpPriority.DEFAULT,
            simp_direction=SimpDirection.POST,
            file_path="TestLib/Basic.lean",
        ),
        SimpRule(
            rule_name="TestLib.Basic.test_high_priority",
            full_attribute="@[simp high]",
            simp_priority=SimpPriority.HIGH,
            simp_direction=SimpDirection.POST,
            file_path="TestLib/Basic.lean",
        ),
        SimpRule(
            rule_name="TestLib.Algebra.mul_zero",
            full_attribute="@[simp 1000]",
            simp_priority=SimpPriority.CUSTOM,
            simp_direction=SimpDirection.POST,
            file_path="TestLib/Algebra.lean",
        ),
    ]


@pytest.fixture
def mock_optimization_result():
    """Create a mock optimization result for testing."""
    from datetime import datetime

    from simpulse.evolution.models import (
        AppliedMutation,
        Candidate,
        MutationType,
        OptimizationResult,
    )

    mutations = [
        AppliedMutation(
            rule_name="TestLib.Basic.test_simp_rule",
            mutation_type=MutationType.PRIORITY_ADJUSTMENT,
            old_attribute="@[simp]",
            new_attribute="@[simp 1100]",
        )
    ]

    candidate = Candidate(mutations=mutations)

    return OptimizationResult(
        modules=["TestLib.Basic"],
        improvement_percent=15.3,
        total_generations=10,
        total_evaluations=50,
        execution_time=120.0,
        success=True,
        best_candidate=candidate,
        optimization_start=datetime.now(),
    )


class TestFixtures:
    """Test that our fixtures work correctly."""

    def test_temp_dir_fixture(self, temp_dir: Path):
        """Test temporary directory fixture."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()

        # Test we can write to it
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.read_text() == "test content"

    def test_mock_lean_project_fixture(self, mock_lean_project: Path):
        """Test mock Lean project fixture."""
        assert mock_lean_project.exists()
        assert (mock_lean_project / "lakefile.lean").exists()
        assert (mock_lean_project / "TestLib" / "Basic.lean").exists()
        assert (mock_lean_project / "TestLib" / "Algebra.lean").exists()

        # Verify content
        basic_content = (mock_lean_project / "TestLib" / "Basic.lean").read_text()
        assert "@[simp]" in basic_content
        assert "test_simp_rule" in basic_content

    def test_mock_config_fixture(self, mock_config):
        """Test mock configuration fixture."""
        assert mock_config.optimization.population_size == 10
        assert mock_config.optimization.generations == 5
        assert mock_config.claude.backend == "mock"

    @pytest.mark.asyncio
    async def test_mock_claude_client_fixture(self, mock_claude_client):
        """Test mock Claude client fixture."""
        response = await mock_claude_client.query_claude("test prompt")
        assert response.success
        assert "mutation suggestion" in response.content
        assert mock_claude_client.is_available()

    @pytest.mark.asyncio
    async def test_mock_lean_runner_fixture(self, mock_lean_runner):
        """Test mock Lean runner fixture."""
        output = await mock_lean_runner.run_lean(Path("test.lean"))
        assert "Mock lean output" in output

        profile = await mock_lean_runner.profile_module("TestModule")
        assert profile["simp_time"] == 1.5
        assert mock_lean_runner.is_available()

    def test_sample_simp_rules_fixture(self, sample_simp_rules):
        """Test sample simp rules fixture."""
        assert len(sample_simp_rules) == 3
        assert all(rule.rule_name.startswith("TestLib") for rule in sample_simp_rules)
        assert any(rule.simp_priority.name == "HIGH" for rule in sample_simp_rules)

    def test_mock_optimization_result_fixture(self, mock_optimization_result):
        """Test mock optimization result fixture."""
        assert mock_optimization_result.success
        assert mock_optimization_result.improvement_percent == 15.3
        assert len(mock_optimization_result.best_candidate.mutations) == 1


# Test discovery and collection functions
def pytest_collect_file(parent, path):
    """Custom test collection for async tests."""
    return None


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take more than 1 second)"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "requires_lean: marks tests that require Lean installation"
    )
    config.addinivalue_line(
        "markers", "requires_claude: marks tests that require Claude API access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test characteristics."""
    for item in items:
        # Add slow marker for tests that might take a while
        if "evolution" in item.nodeid or "optimization" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Add integration marker for integration tests
        if "integration" in item.nodeid or "end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.integration)


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])

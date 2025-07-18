"""Pytest configuration and shared fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_lean_file(temp_dir: Path) -> Path:
    """Create a sample Lean file for testing."""
    lean_file = temp_dir / "sample.lean"
    lean_content = """
-- Sample Lean file for testing
@[simp] theorem list_append_nil (l : List a) : l ++ [] = l := by
  induction l with
  | nil => rfl
  | cons head tail ih => simp [ih]

@[simp, priority := 100] theorem zero_add (n : Nat) : 0 + n = n := by
  rfl

@[simp] theorem add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.succ_add, ih]

-- Non-simp theorem
theorem example_theorem (n : Nat) : n = n := rfl

-- Another simp rule
@[simp] theorem mul_one (n : Nat) : n * 1 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.succ_mul, ih]
"""
    lean_file.write_text(lean_content)
    return lean_file


@pytest.fixture
def sample_lean_project(temp_dir: Path) -> Path:
    """Create a sample Lean project structure for testing."""
    project_dir = temp_dir / "lean_project"
    project_dir.mkdir()

    # Create lakefile.lean
    lakefile = project_dir / "lakefile.lean"
    lakefile.write_text(
        """
import Lake
open Lake DSL

package test_project where
  version := v!"0.1.0"

lean_lib TestProject where
  roots := #[`TestProject]
"""
    )

    # Create main module
    lib_dir = project_dir / "TestProject"
    lib_dir.mkdir()

    basic_file = lib_dir / "Basic.lean"
    basic_file.write_text(
        """
@[simp] theorem basic_rule_1 : true = true := rfl
@[simp] theorem basic_rule_2 : false = false := rfl
@[simp, priority := 200] theorem basic_rule_3 : 1 + 1 = 2 := rfl
"""
    )

    advanced_file = lib_dir / "Advanced.lean"
    advanced_file.write_text(
        """
@[simp] theorem advanced_rule_1 (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.succ_add, ih]

@[simp] theorem advanced_rule_2 (n : Nat) : 0 + n = n := rfl
"""
    )

    return project_dir


@pytest.fixture
def lean_content_with_priorities() -> str:
    """Sample Lean content with various priority configurations."""
    return """
-- Default priority (1000)
@[simp] theorem default_prio : true = true := rfl

-- High priority
@[simp, priority := 100] theorem high_prio : false = false := rfl

-- Low priority
@[simp, priority := 2000] theorem low_prio : 1 = 1 := rfl

-- Medium priority
@[simp, priority := 500] theorem medium_prio : 2 = 2 := rfl

-- Another default
@[simp] theorem another_default : 3 = 3 := rfl
"""


@pytest.fixture
def analysis_result_sample():
    """Sample analysis result for testing."""
    return {
        "total_files": 3,
        "total_simp_rules": 15,
        "rules_with_custom_priority": 3,
        "default_priority_percent": 80.0,
        "rules_by_frequency": [],
        "optimization_opportunities": 12,
        "estimated_improvement": 0.18,
    }


@pytest.fixture
def optimization_suggestions_sample():
    """Sample optimization suggestions for testing."""
    from simpulse.optimization_engine import OptimizationRecommendation, OptimizationType

    return [
        OptimizationRecommendation(
            theorem_name="list_append_nil",
            file_path="Data/List/Basic.lean",
            optimization_type=OptimizationType.PRIORITY_INCREASE,
            current_priority=1000,
            suggested_priority=100,
            reason="High frequency rule (847 uses)",
            confidence=85.4,
            usage_count=847,
            success_rate=0.924,
        ),
        OptimizationRecommendation(
            theorem_name="zero_add",
            file_path="Algebra/Group/Basic.lean",
            optimization_type=OptimizationType.PRIORITY_INCREASE,
            current_priority=1000,
            suggested_priority=200,
            reason="Medium frequency rule (421 uses)",
            confidence=72.3,
            usage_count=421,
            success_rate=0.886,
        ),
    ]


# Test markers for different test types
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow (skip with -m 'not slow')")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Add unit marker to tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add integration marker to tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add performance marker to tests in performance/ directory
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

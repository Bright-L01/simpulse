"""Tests for evolution engine."""

from pathlib import Path

import pytest

from simpulse.evolution.evolution_engine import SimpleEvolutionEngine


class TestEvolutionEngine:
    """Test evolution engine."""

    def test_engine_initialization(self):
        """Test engine can be initialized."""
        engine = SimpleEvolutionEngine()
        assert engine is not None
        assert hasattr(engine, "extractor")
        assert hasattr(engine, "applicator")
        assert hasattr(engine, "runner")
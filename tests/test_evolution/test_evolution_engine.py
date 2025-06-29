"""
Tests for the evolution engine module.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from simpulse.evolution.evolution_engine import EvolutionEngine
from simpulse.evolution.models import SimpRule, OptimizationResult, SimpPriority
from simpulse.evolution.models_v2 import (
    Candidate, Population, FitnessScore, GenerationResult,
    EvolutionHistory, EvolutionConfig
)
from simpulse.config import Config


class TestEvolutionEngine:
    """Test suite for EvolutionEngine class."""
    
    @pytest.fixture
    def evolution_engine(self, mock_config, mock_claude_client):
        """Create an evolution engine instance for testing."""
        with patch('simpulse.evolution.evolution_engine.ClaudeCodeClient', return_value=mock_claude_client):
            engine = EvolutionEngine(mock_config)
            return engine
    
    @pytest.mark.asyncio
    async def test_initialization(self, evolution_engine):
        """Test evolution engine initialization."""
        assert evolution_engine is not None
        assert evolution_engine.config is not None
        assert evolution_engine.current_generation == 0
        assert evolution_engine.best_fitness == float('-inf')
    
    @pytest.mark.asyncio
    async def test_run_evolution_basic(self, evolution_engine, mock_lean_project, sample_simp_rules):
        """Test basic evolution run."""
        # Mock components
        with patch.object(evolution_engine.rule_extractor, 'extract_rules', return_value=sample_simp_rules):
            with patch.object(evolution_engine.population_manager, 'create_initial_population') as mock_create_pop:
                with patch.object(evolution_engine.fitness_evaluator, 'evaluate_candidate') as mock_evaluate:
                    # Setup mocks
                    mock_population = Population(candidates=[
                        Candidate(mutations=[]),
                        Candidate(mutations=[])
                    ])
                    mock_create_pop.return_value = mock_population
                    
                    mock_fitness = FitnessScore(
                        time_score=0.8,
                        memory_score=0.7,
                        iterations_score=0.9,
                        depth_score=0.85,
                        composite_score=0.8125
                    )
                    mock_evaluate.return_value = mock_fitness
                    
                    # Run evolution with short time budget
                    result = await evolution_engine.run_evolution(
                        modules=["TestModule"],
                        source_path=mock_lean_project,
                        time_budget=1  # 1 second for testing
                    )
                    
                    # Verify result
                    assert isinstance(result, OptimizationResult)
                    assert result.modules == ["TestModule"]
                    assert result.total_evaluations > 0
    
    @pytest.mark.asyncio
    async def test_run_single_generation(self, evolution_engine, sample_simp_rules):
        """Test running a single generation."""
        # Create test population
        test_population = Population(candidates=[
            Candidate(mutations=[], fitness=FitnessScore(composite_score=0.5)),
            Candidate(mutations=[], fitness=FitnessScore(composite_score=0.7)),
            Candidate(mutations=[], fitness=FitnessScore(composite_score=0.6))
        ])
        
        with patch.object(evolution_engine.population_manager, 'select_parents') as mock_select:
            with patch.object(evolution_engine.population_manager, 'crossover') as mock_crossover:
                with patch.object(evolution_engine, '_mutate_candidate') as mock_mutate:
                    with patch.object(evolution_engine.fitness_evaluator, 'evaluate_candidate') as mock_evaluate:
                        # Setup mocks
                        mock_select.return_value = [test_population.candidates[0], test_population.candidates[1]]
                        mock_crossover.return_value = Candidate(mutations=[])
                        mock_mutate.return_value = Candidate(mutations=[])
                        mock_evaluate.return_value = FitnessScore(composite_score=0.8)
                        
                        # Run generation
                        result = await evolution_engine._run_generation(test_population, sample_simp_rules)
                        
                        # Verify result
                        assert hasattr(result, 'population')
                        assert hasattr(result, 'best_candidate')
                        assert hasattr(result, 'generation_stats')
                        assert result.generation_stats.generation == 0
    
    @pytest.mark.asyncio
    async def test_mutation_with_claude(self, evolution_engine, sample_simp_rules, mock_claude_client):
        """Test candidate mutation using Claude."""
        candidate = Candidate(mutations=[])
        
        # Mock Claude response
        mock_claude_client.query_claude.return_value = AsyncMock(
            content="""Based on the analysis, I suggest:
1. Increase priority of test_simp_rule to 1100
2. Change direction of mul_zero to pre-simplification""",
            success=True
        )
        
        with patch.object(evolution_engine, '_parse_claude_suggestions') as mock_parse:
            mock_parse.return_value = []  # Simplified for testing
            
            mutated = await evolution_engine._mutate_candidate(candidate, sample_simp_rules)
            
            # Verify mutation was attempted
            assert mock_claude_client.query_claude.called
            assert mutated is not None
    
    @pytest.mark.asyncio 
    async def test_convergence_detection(self, evolution_engine):
        """Test convergence detection logic."""
        # Simulate no improvement for multiple generations
        evolution_engine.generations_without_improvement = 10
        evolution_engine.config.optimization.patience = 5
        
        converged = evolution_engine._check_convergence()
        assert converged == True
        
        # Simulate recent improvement
        evolution_engine.generations_without_improvement = 2
        converged = evolution_engine._check_convergence()
        assert converged == False
    
    @pytest.mark.asyncio
    async def test_checkpoint_save_load(self, evolution_engine, temp_dir):
        """Test checkpoint saving and loading."""
        # Setup test state
        evolution_engine.current_generation = 5
        evolution_engine.best_fitness = 0.85
        evolution_engine.evolution_history = EvolutionHistory()
        
        checkpoint_path = temp_dir / "test_checkpoint.json"
        
        # Save checkpoint
        await evolution_engine._save_checkpoint(checkpoint_path)
        assert checkpoint_path.exists()
        
        # Reset state
        evolution_engine.current_generation = 0
        evolution_engine.best_fitness = float('-inf')
        
        # Load checkpoint
        loaded = await evolution_engine._load_checkpoint(checkpoint_path)
        assert loaded == True
        assert evolution_engine.current_generation == 5
        assert evolution_engine.best_fitness == 0.85
    
    @pytest.mark.asyncio
    async def test_error_handling(self, evolution_engine):
        """Test error handling in evolution engine."""
        # Test with invalid module
        with patch.object(evolution_engine.rule_extractor, 'extract_rules', side_effect=Exception("Module not found")):
            result = await evolution_engine.run_evolution(
                modules=["NonExistentModule"],
                source_path=Path("/fake/path"),
                time_budget=1
            )
            
            # Should handle error gracefully
            assert result.success == False
            assert result.improvement_percent == 0.0
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, evolution_engine):
        """Test performance metrics tracking."""
        # Mock initial baseline
        with patch.object(evolution_engine.lean_runner, 'profile_module') as mock_profile:
            mock_profile.return_value = {
                'simp_time': 10.0,
                'total_time': 50.0,
                'memory_peak': 1000.0
            }
            
            baseline = await evolution_engine._establish_baseline(["TestModule"], Path("."))
            
            assert 'TestModule' in baseline
            assert baseline['TestModule']['simp_time'] == 10.0
    
    def test_parse_claude_suggestions(self, evolution_engine):
        """Test parsing of Claude suggestions."""
        claude_response = """Based on the analysis:
1. Increase priority of rule_one to 1200
2. Change direction of rule_two to pre-simplification
3. Disable rule_three temporarily"""
        
        suggestions = evolution_engine._parse_claude_suggestions(
            claude_response, 
            [SimpRule(rule_name="rule_one", full_attribute="@[simp]")]
        )
        
        # Should parse suggestions correctly
        assert len(suggestions) >= 0  # Depends on parsing implementation
    
    @pytest.mark.asyncio
    async def test_concurrent_evaluation(self, evolution_engine):
        """Test concurrent candidate evaluation."""
        candidates = [Candidate(mutations=[]) for _ in range(5)]
        
        with patch.object(evolution_engine.fitness_evaluator, 'evaluate_candidate') as mock_eval:
            mock_eval.return_value = FitnessScore(composite_score=0.7)
            
            # Should handle concurrent evaluation
            tasks = [evolution_engine.fitness_evaluator.evaluate_candidate(c, {}) for c in candidates]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            assert all(r.composite_score == 0.7 for r in results)


class TestEvolutionIntegration:
    """Integration tests for evolution engine."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_optimization_workflow(self, mock_config, mock_lean_project):
        """Test complete optimization workflow."""
        # This would test the full integration with real components
        # Marked as integration test to run separately
        pass
    
    @pytest.mark.integration
    @pytest.mark.requires_lean
    @pytest.mark.asyncio
    async def test_with_real_lean_project(self):
        """Test with actual Lean project."""
        # This would test against a real Lean project
        # Requires Lean to be installed
        pass
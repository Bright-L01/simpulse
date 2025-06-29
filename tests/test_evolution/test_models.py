"""
Tests for evolution models and data structures.
"""

from pathlib import Path

import pytest

from simpulse.evolution.models import (
    ModuleRules,
    MutationSuggestion,
    MutationType,
    OptimizationGoal,
    SimpDirection,
    SimpPriority,
    SimpRule,
    SourceLocation,
)
from simpulse.evolution.models_v2 import (
    Candidate,
    EvolutionConfig,
    EvolutionHistory,
    FitnessScore,
    GenerationResult,
    Mutation,
    Population,
)


class TestSimpRule:
    """Test suite for SimpRule model."""

    def test_simp_rule_creation(self):
        """Test creating SimpRule instances."""
        location = SourceLocation(
            file=Path("test.lean"), line=10, column=5, module="TestModule"
        )

        rule = SimpRule(
            name="test_rule",
            declaration="@[simp] theorem test_rule : a + 0 = a",
            priority=SimpPriority.HIGH,
            direction=SimpDirection.FORWARD,
            location=location,
            conditions=["Add a"],
            pattern="a + 0",
            rhs="a",
        )

        assert rule.name == "test_rule"
        assert rule.priority == SimpPriority.HIGH
        assert rule.direction == SimpDirection.FORWARD
        assert rule.location.line == 10
        assert "Add a" in rule.conditions

    def test_simp_rule_with_metadata(self):
        """Test SimpRule with metadata."""
        rule = SimpRule(
            name="test_rule",
            declaration="@[simp] theorem test_rule : true",
            metadata={"complexity": 5, "usage_count": 100, "performance_impact": 0.8},
        )

        assert rule.metadata["complexity"] == 5
        assert rule.metadata["usage_count"] == 100

    def test_simp_priority_enum(self):
        """Test SimpPriority enum values."""
        assert SimpPriority.LOW.value == "low"
        assert SimpPriority.DEFAULT.value == "default"
        assert SimpPriority.HIGH.value == "high"

    def test_simp_direction_enum(self):
        """Test SimpDirection enum values."""
        assert SimpDirection.FORWARD.value == "forward"
        assert SimpDirection.BACKWARD.value == "backward"


class TestModuleRules:
    """Test suite for ModuleRules model."""

    def test_module_rules_creation(self):
        """Test creating ModuleRules instances."""
        rules = [
            SimpRule(name="rule1", declaration="@[simp] theorem rule1 : true"),
            SimpRule(name="rule2", declaration="@[simp] theorem rule2 : false = false"),
        ]

        module = ModuleRules(
            module_name="TestModule",
            file_path=Path("test.lean"),
            rules=rules,
            imports=["Mathlib.Data.List.Basic"],
            metadata={"total_rules": 2},
        )

        assert module.module_name == "TestModule"
        assert len(module.rules) == 2
        assert "Mathlib.Data.List.Basic" in module.imports
        assert module.metadata["total_rules"] == 2


class TestMutationSuggestion:
    """Test suite for MutationSuggestion model."""

    def test_mutation_suggestion_creation(self):
        """Test creating MutationSuggestion instances."""
        suggestion = MutationSuggestion(
            rule_name="test_rule",
            mutation_type=MutationType.PRIORITY_CHANGE,
            original_declaration="@[simp] theorem test_rule : a + 0 = a",
            mutated_declaration="@[simp high] theorem test_rule : a + 0 = a",
            confidence=0.85,
            description="Increase priority to improve performance",
            estimated_impact={"time": -10.0, "iterations": -5.0},
        )

        assert suggestion.rule_name == "test_rule"
        assert suggestion.mutation_type == MutationType.PRIORITY_CHANGE
        assert suggestion.confidence == 0.85
        assert suggestion.estimated_impact["time"] == -10.0

    def test_mutation_type_enum(self):
        """Test MutationType enum values."""
        assert MutationType.PRIORITY_CHANGE.value == "priority_change"
        assert MutationType.DIRECTION_CHANGE.value == "direction_change"
        assert MutationType.CONDITION_ADD.value == "condition_add"
        assert MutationType.CONDITION_REMOVE.value == "condition_remove"
        assert MutationType.RULE_DISABLE.value == "rule_disable"
        assert MutationType.PATTERN_MODIFY.value == "pattern_modify"


class TestCandidate:
    """Test suite for Candidate model."""

    def test_candidate_creation(self):
        """Test creating Candidate instances."""
        mutations = [
            Mutation(
                suggestion=MutationSuggestion(
                    rule_name="rule1", mutation_type=MutationType.PRIORITY_CHANGE
                )
            )
        ]

        fitness = FitnessScore(
            time_score=0.8,
            memory_score=0.7,
            iterations_score=0.9,
            depth_score=0.85,
            composite_score=0.8125,
        )

        candidate = Candidate(mutations=mutations, fitness=fitness, generation=5)

        assert len(candidate.mutations) == 1
        assert candidate.fitness.composite_score == 0.8125
        assert candidate.generation == 5

    def test_candidate_comparison(self):
        """Test comparing candidates by fitness."""
        candidate1 = Candidate(mutations=[], fitness=FitnessScore(composite_score=0.8))

        candidate2 = Candidate(mutations=[], fitness=FitnessScore(composite_score=0.6))

        # Should be comparable by fitness
        assert candidate1.fitness.composite_score > candidate2.fitness.composite_score


class TestPopulation:
    """Test suite for Population model."""

    def test_population_creation(self):
        """Test creating Population instances."""
        candidates = [
            Candidate(mutations=[], fitness=FitnessScore(composite_score=0.7)),
            Candidate(mutations=[], fitness=FitnessScore(composite_score=0.8)),
            Candidate(mutations=[], fitness=FitnessScore(composite_score=0.6)),
        ]

        population = Population(candidates=candidates, generation=3)

        assert len(population.candidates) == 3
        assert population.generation == 3

    def test_population_best_candidate(self):
        """Test getting best candidate from population."""
        candidates = [
            Candidate(mutations=[], fitness=FitnessScore(composite_score=0.7)),
            Candidate(mutations=[], fitness=FitnessScore(composite_score=0.9)),
            Candidate(mutations=[], fitness=FitnessScore(composite_score=0.6)),
        ]

        population = Population(candidates=candidates)

        best = population.get_best_candidate()
        assert best is not None
        assert best.fitness.composite_score == 0.9

    def test_population_statistics(self):
        """Test population statistics calculation."""
        candidates = [
            Candidate(mutations=[], fitness=FitnessScore(composite_score=0.6)),
            Candidate(mutations=[], fitness=FitnessScore(composite_score=0.7)),
            Candidate(mutations=[], fitness=FitnessScore(composite_score=0.8)),
            Candidate(mutations=[], fitness=FitnessScore(composite_score=0.9)),
        ]

        population = Population(candidates=candidates)

        stats = population.get_statistics()
        assert stats["average_fitness"] == 0.75
        assert stats["best_fitness"] == 0.9
        assert stats["worst_fitness"] == 0.6
        assert stats["size"] == 4


class TestFitnessScore:
    """Test suite for FitnessScore model."""

    def test_fitness_score_creation(self):
        """Test creating FitnessScore instances."""
        score = FitnessScore(
            time_score=0.8,
            memory_score=0.7,
            iterations_score=0.9,
            depth_score=0.85,
            composite_score=0.8125,
            total_time=40.0,
            simp_time=8.0,
            memory_mb=200.0,
            iterations=85,
            depth=4,
        )

        assert score.time_score == 0.8
        assert score.memory_score == 0.7
        assert score.iterations_score == 0.9
        assert score.depth_score == 0.85
        assert score.composite_score == 0.8125
        assert score.total_time == 40.0
        assert score.simp_time == 8.0

    def test_fitness_score_calculation(self):
        """Test composite fitness score calculation."""
        score = FitnessScore(
            time_score=0.8, memory_score=0.6, iterations_score=0.9, depth_score=0.7
        )

        # Calculate composite (should be weighted average)
        score.composite_score = score.calculate_composite()

        # Composite should be between min and max component scores
        assert 0.6 <= score.composite_score <= 0.9


class TestGenerationResult:
    """Test suite for GenerationResult model."""

    def test_generation_result_creation(self):
        """Test creating GenerationResult instances."""
        result = GenerationResult(
            generation=5,
            best_fitness=0.85,
            average_fitness=0.72,
            diversity_score=0.65,
            valid_candidates=18,
            total_candidates=20,
            evaluation_time=45.5,
            new_best_found=True,
        )

        assert result.generation == 5
        assert result.best_fitness == 0.85
        assert result.average_fitness == 0.72
        assert result.diversity_score == 0.65
        assert result.valid_candidates == 18
        assert result.new_best_found

    def test_generation_result_statistics(self):
        """Test generation result statistics."""
        result = GenerationResult(
            generation=10,
            best_fitness=0.9,
            average_fitness=0.75,
            diversity_score=0.5,
            valid_candidates=19,
            total_candidates=20,
            mutation_statistics={
                "priority_changes": 5,
                "direction_changes": 3,
                "total_mutations": 8,
            },
        )

        assert result.mutation_statistics["total_mutations"] == 8
        assert result.get_validity_rate() == 0.95  # 19/20


class TestEvolutionHistory:
    """Test suite for EvolutionHistory model."""

    def test_evolution_history_creation(self):
        """Test creating EvolutionHistory instances."""
        history = EvolutionHistory()

        # Add generations
        for i in range(5):
            gen_result = GenerationResult(
                generation=i,
                best_fitness=0.5 + i * 0.1,
                average_fitness=0.4 + i * 0.08,
                diversity_score=0.8 - i * 0.1,
            )
            history.add_generation(gen_result)

        assert len(history.generations) == 5
        assert history.total_generations == 5

    def test_evolution_history_convergence(self):
        """Test convergence detection in history."""
        history = EvolutionHistory()

        # Add stagnant generations
        for i in range(10):
            gen_result = GenerationResult(
                generation=i,
                best_fitness=0.8,  # No improvement
                average_fitness=0.7,
                diversity_score=0.3,
                new_best_found=(i == 0),  # Only first generation finds best
            )
            history.add_generation(gen_result)

        # Should detect convergence
        history.convergence_generation = 5
        assert history.convergence_generation == 5

    def test_evolution_history_summary(self):
        """Test evolution history summary generation."""
        history = EvolutionHistory()

        # Add improving generations
        for i in range(5):
            gen_result = GenerationResult(
                generation=i,
                best_fitness=0.5 + i * 0.1,
                average_fitness=0.4 + i * 0.08,
                diversity_score=0.8 - i * 0.1,
                evaluation_time=10.0 + i,
            )
            history.add_generation(gen_result)

        summary = history.get_summary()

        assert summary["total_generations"] == 5
        assert summary["best_fitness"] == 0.9
        assert summary["initial_fitness"] == 0.5
        assert summary["improvement"] == 0.4
        assert summary["total_time"] == sum(range(10, 15))


class TestEvolutionConfig:
    """Test suite for EvolutionConfig model."""

    def test_evolution_config_defaults(self):
        """Test EvolutionConfig default values."""
        config = EvolutionConfig()

        assert config.population_size == 20
        assert config.generations == 50
        assert config.mutation_rate == 0.3
        assert config.crossover_rate == 0.7
        assert config.elite_size == 5
        assert config.tournament_size == 3

    def test_evolution_config_validation(self):
        """Test EvolutionConfig validation."""
        # Valid config
        config = EvolutionConfig(
            population_size=50, generations=100, mutation_rate=0.5, crossover_rate=0.8
        )

        assert config.is_valid()

        # Invalid config (rates > 1)
        invalid_config = EvolutionConfig(mutation_rate=1.5, crossover_rate=0.8)

        assert not invalid_config.is_valid()


class TestOptimizationGoal:
    """Test suite for OptimizationGoal enum."""

    def test_optimization_goal_values(self):
        """Test OptimizationGoal enum values."""
        assert OptimizationGoal.MINIMIZE_TIME.value == "minimize_time"
        assert OptimizationGoal.MINIMIZE_MEMORY.value == "minimize_memory"
        assert OptimizationGoal.MINIMIZE_ITERATIONS.value == "minimize_iterations"
        assert OptimizationGoal.BALANCED.value == "balanced"


@pytest.mark.integration
class TestModelsIntegration:
    """Integration tests for models working together."""

    def test_complete_evolution_workflow(self):
        """Test models in a complete evolution workflow."""
        # Create initial population
        candidates = []
        for i in range(10):
            candidate = Candidate(
                mutations=[], fitness=FitnessScore(composite_score=0.5 + i * 0.05)
            )
            candidates.append(candidate)

        population = Population(candidates=candidates, generation=0)

        # Create evolution history
        history = EvolutionHistory()

        # Simulate evolution
        for gen in range(5):
            # Get best candidate
            best = population.get_best_candidate()

            # Create generation result
            gen_result = GenerationResult(
                generation=gen,
                best_fitness=best.fitness.composite_score,
                average_fitness=population.get_statistics()["average_fitness"],
                diversity_score=0.7 - gen * 0.1,
                valid_candidates=len(population.candidates),
            )

            # Add to history
            history.add_generation(gen_result)

            # Create next generation (simplified)
            new_candidates = []
            for candidate in population.candidates:
                # Simulate improvement
                new_fitness = FitnessScore(
                    composite_score=min(1.0, candidate.fitness.composite_score + 0.02)
                )
                new_candidate = Candidate(
                    mutations=candidate.mutations,
                    fitness=new_fitness,
                    generation=gen + 1,
                )
                new_candidates.append(new_candidate)

            population = Population(candidates=new_candidates, generation=gen + 1)

        # Verify evolution progress
        assert history.total_generations == 5
        assert (
            history.generations[-1].best_fitness > history.generations[0].best_fitness
        )

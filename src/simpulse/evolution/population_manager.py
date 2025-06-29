"""Population manager for evolutionary algorithm.

This module implements the core evolutionary operations including
selection, crossover, mutation, and population management.
"""

import asyncio
import logging
import random
import statistics
import uuid
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from .models import SimpRule, MutationSuggestion, MutationType
from .models_v2 import (
    Candidate, FitnessScore, Population, SelectionMethod, 
    CrossoverMethod, MutationMethod, EvolutionConfig,
    PopulationStatistics, GenerationResult
)
from ..claude import ClaudeCodeClient

logger = logging.getLogger(__name__)


class PopulationManager:
    """Manages evolutionary operations on populations of candidates."""
    
    def __init__(self, 
                 config: EvolutionConfig,
                 claude_client: Optional[ClaudeCodeClient] = None):
        """Initialize population manager.
        
        Args:
            config: Evolution configuration
            claude_client: Claude client for intelligent mutations
        """
        self.config = config
        self.claude_client = claude_client
        self.generation = 0
        self.stagnation_counter = 0
        self.best_ever_fitness = 0.0
        
        # Adaptive parameters
        self.current_mutation_rate = config.mutation_rate
        self.current_crossover_rate = config.crossover_rate
        
        # Statistics tracking
        self.diversity_history: List[float] = []
        self.fitness_history: List[float] = []
        
    async def initialize_population(self, 
                                  rules: List[SimpRule],
                                  claude_suggestions: Optional[List[MutationSuggestion]] = None) -> Population:
        """Create initial population with diverse mutations.
        
        Args:
            rules: Available simp rules
            claude_suggestions: Optional Claude-generated suggestions
            
        Returns:
            Initial population of candidates
        """
        logger.info(f"Initializing population of size {self.config.population_size}")
        
        population = []
        
        # Create baseline candidate (no mutations)
        baseline = Candidate(
            id="baseline",
            mutations=[],
            generation=0,
            creation_method="baseline"
        )
        population.append(baseline)
        
        # Create candidates with Claude suggestions if available
        claude_candidates = 0
        if claude_suggestions:
            for i, suggestion in enumerate(claude_suggestions[:self.config.population_size // 3]):
                candidate = Candidate(
                    id=f"claude_{i}",
                    mutations=[],  # Will be converted to AppliedMutation later
                    generation=0,
                    creation_method="claude",
                    metadata={"suggestion": suggestion}
                )
                population.append(candidate)
                claude_candidates += 1
                
        # Fill remaining slots with random mutations
        remaining_slots = self.config.population_size - len(population)
        for i in range(remaining_slots):
            mutations = await self._generate_random_mutations(rules, max_mutations=3)
            candidate = Candidate(
                id=f"random_{i}",
                mutations=mutations,
                generation=0,
                creation_method="random"
            )
            population.append(candidate)
            
        logger.info(f"Created initial population: {claude_candidates} Claude-based, "
                   f"{remaining_slots} random, 1 baseline")
        
        return population
        
    def tournament_selection(self, 
                           population: Population,
                           tournament_size: Optional[int] = None) -> Candidate:
        """Select parent using tournament selection.
        
        Args:
            population: Population to select from
            tournament_size: Size of tournament (uses config default if None)
            
        Returns:
            Selected candidate
        """
        tournament_size = tournament_size or self.config.tournament_size
        valid_candidates = [c for c in population if c.is_valid]
        
        if not valid_candidates:
            # Fallback to any candidate if no valid ones
            return random.choice(population)
            
        # Select random candidates for tournament
        tournament = random.sample(
            valid_candidates, 
            min(tournament_size, len(valid_candidates))
        )
        
        # Return candidate with best fitness
        return max(tournament, key=lambda c: c.fitness_score)
        
    def roulette_selection(self, population: Population) -> Candidate:
        """Select parent using roulette wheel selection.
        
        Args:
            population: Population to select from
            
        Returns:
            Selected candidate
        """
        valid_candidates = [c for c in population if c.is_valid]
        
        if not valid_candidates:
            return random.choice(population)
            
        # Calculate selection probabilities
        fitness_scores = [c.fitness_score for c in valid_candidates]
        min_fitness = min(fitness_scores)
        
        # Ensure all probabilities are positive
        adjusted_scores = [score - min_fitness + 0.001 for score in fitness_scores]
        total_fitness = sum(adjusted_scores)
        
        if total_fitness == 0:
            return random.choice(valid_candidates)
            
        # Spin the wheel
        spin = random.uniform(0, total_fitness)
        current_sum = 0
        
        for candidate, score in zip(valid_candidates, adjusted_scores):
            current_sum += score
            if current_sum >= spin:
                return candidate
                
        return valid_candidates[-1]  # Fallback
        
    def select_parent(self, population: Population) -> Candidate:
        """Select a parent using configured selection method.
        
        Args:
            population: Population to select from
            
        Returns:
            Selected parent candidate
        """
        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return self.tournament_selection(population)
        elif self.config.selection_method == SelectionMethod.ROULETTE:
            return self.roulette_selection(population)
        elif self.config.selection_method == SelectionMethod.RANK:
            return self._rank_selection(population)
        else:
            return self.tournament_selection(population)  # Default
            
    def _rank_selection(self, population: Population) -> Candidate:
        """Select parent using rank-based selection.
        
        Args:
            population: Population to select from
            
        Returns:
            Selected candidate
        """
        valid_candidates = [c for c in population if c.is_valid]
        
        if not valid_candidates:
            return random.choice(population)
            
        # Sort by fitness and assign ranks
        sorted_candidates = sorted(valid_candidates, key=lambda c: c.fitness_score)
        ranks = list(range(1, len(sorted_candidates) + 1))
        
        # Select based on rank probability
        total_rank = sum(ranks)
        spin = random.uniform(0, total_rank)
        current_sum = 0
        
        for candidate, rank in zip(sorted_candidates, ranks):
            current_sum += rank
            if current_sum >= spin:
                return candidate
                
        return sorted_candidates[-1]
        
    async def crossover(self, parent1: Candidate, parent2: Candidate) -> Candidate:
        """Create offspring through crossover of two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Offspring candidate
        """
        if self.config.crossover_method == CrossoverMethod.UNIFORM:
            return await self._uniform_crossover(parent1, parent2)
        elif self.config.crossover_method == CrossoverMethod.SINGLE_POINT:
            return await self._single_point_crossover(parent1, parent2)
        elif self.config.crossover_method == CrossoverMethod.TWO_POINT:
            return await self._two_point_crossover(parent1, parent2)
        else:
            return await self._uniform_crossover(parent1, parent2)  # Default
            
    async def _uniform_crossover(self, parent1: Candidate, parent2: Candidate) -> Candidate:
        """Perform uniform crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Offspring candidate
        """
        # Combine mutations from both parents
        all_mutations = parent1.mutations + parent2.mutations
        
        # Remove duplicates based on rule name
        unique_mutations = []
        seen_rules = set()
        
        for mutation in all_mutations:
            rule_name = getattr(mutation, 'original_rule_name', str(hash(mutation)))
            if rule_name not in seen_rules:
                unique_mutations.append(mutation)
                seen_rules.add(rule_name)
                
        # Randomly select subset of mutations
        max_mutations = min(len(unique_mutations), 5)  # Limit complexity
        if max_mutations > 0:
            num_mutations = random.randint(1, max_mutations)
            selected_mutations = random.sample(unique_mutations, num_mutations)
        else:
            selected_mutations = []
            
        offspring = Candidate(
            id=str(uuid.uuid4())[:8],
            mutations=selected_mutations,
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id],
            creation_method="uniform_crossover"
        )
        
        return offspring
        
    async def _single_point_crossover(self, parent1: Candidate, parent2: Candidate) -> Candidate:
        """Perform single-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Offspring candidate
        """
        mutations1 = parent1.mutations
        mutations2 = parent2.mutations
        
        if not mutations1 and not mutations2:
            # Both parents have no mutations
            return Candidate(
                id=str(uuid.uuid4())[:8],
                mutations=[],
                generation=self.generation + 1,
                parent_ids=[parent1.id, parent2.id],
                creation_method="single_point_crossover"
            )
            
        # Choose crossover point
        max_len = max(len(mutations1), len(mutations2))
        if max_len == 0:
            crossover_point = 0
        else:
            crossover_point = random.randint(0, max_len)
            
        # Create offspring mutations
        offspring_mutations = []
        offspring_mutations.extend(mutations1[:crossover_point])
        offspring_mutations.extend(mutations2[crossover_point:])
        
        offspring = Candidate(
            id=str(uuid.uuid4())[:8],
            mutations=offspring_mutations,
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id],
            creation_method="single_point_crossover"
        )
        
        return offspring
        
    async def _two_point_crossover(self, parent1: Candidate, parent2: Candidate) -> Candidate:
        """Perform two-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Offspring candidate
        """
        mutations1 = parent1.mutations
        mutations2 = parent2.mutations
        
        max_len = max(len(mutations1), len(mutations2))
        if max_len <= 1:
            return await self._uniform_crossover(parent1, parent2)
            
        # Choose two crossover points
        point1 = random.randint(0, max_len - 1)
        point2 = random.randint(point1, max_len)
        
        # Create offspring mutations
        offspring_mutations = []
        offspring_mutations.extend(mutations1[:point1])
        offspring_mutations.extend(mutations2[point1:point2])
        offspring_mutations.extend(mutations1[point2:])
        
        offspring = Candidate(
            id=str(uuid.uuid4())[:8],
            mutations=offspring_mutations,
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id],
            creation_method="two_point_crossover"
        )
        
        return offspring
        
    async def mutate(self, 
                    candidate: Candidate,
                    rules: List[SimpRule],
                    mutation_rate: Optional[float] = None) -> Candidate:
        """Apply mutation to a candidate.
        
        Args:
            candidate: Candidate to mutate
            rules: Available rules for mutation
            mutation_rate: Override default mutation rate
            
        Returns:
            Mutated candidate
        """
        mutation_rate = mutation_rate or self.current_mutation_rate
        
        if random.random() > mutation_rate:
            # No mutation
            return candidate.clone()
            
        if self.config.mutation_method == MutationMethod.RANDOM:
            return await self._random_mutation(candidate, rules)
        elif self.config.mutation_method == MutationMethod.ADAPTIVE:
            return await self._adaptive_mutation(candidate, rules)
        elif self.config.mutation_method == MutationMethod.CLAUDE_GUIDED:
            return await self._claude_guided_mutation(candidate, rules)
        else:
            return await self._random_mutation(candidate, rules)
            
    async def _random_mutation(self, candidate: Candidate, rules: List[SimpRule]) -> Candidate:
        """Apply random mutation.
        
        Args:
            candidate: Candidate to mutate
            rules: Available rules
            
        Returns:
            Mutated candidate
        """
        mutated = candidate.clone()
        
        # Randomly choose mutation operation
        operations = ["add", "remove", "modify"]
        operation = random.choice(operations)
        
        if operation == "add" and len(mutated.mutations) < 5:
            # Add new random mutation
            new_mutations = await self._generate_random_mutations(rules, max_mutations=1)
            mutated.mutations.extend(new_mutations)
        elif operation == "remove" and mutated.mutations:
            # Remove random mutation
            mutated.mutations.pop(random.randint(0, len(mutated.mutations) - 1))
        elif operation == "modify" and mutated.mutations:
            # Modify existing mutation (simplified)
            idx = random.randint(0, len(mutated.mutations) - 1)
            # For now, just replace with a new random mutation
            new_mutations = await self._generate_random_mutations(rules, max_mutations=1)
            if new_mutations:
                mutated.mutations[idx] = new_mutations[0]
                
        mutated.creation_method = "random_mutation"
        return mutated
        
    async def _adaptive_mutation(self, candidate: Candidate, rules: List[SimpRule]) -> Candidate:
        """Apply adaptive mutation based on population diversity.
        
        Args:
            candidate: Candidate to mutate
            rules: Available rules
            
        Returns:
            Mutated candidate
        """
        # Adjust mutation rate based on diversity
        if self.diversity_history:
            recent_diversity = statistics.mean(self.diversity_history[-5:])
            if recent_diversity < self.config.diversity_threshold:
                # Low diversity, increase mutation rate
                adapted_rate = min(self.current_mutation_rate * 1.5, 0.8)
            else:
                # High diversity, maintain or reduce mutation rate
                adapted_rate = max(self.current_mutation_rate * 0.9, 0.05)
        else:
            adapted_rate = self.current_mutation_rate
            
        return await self._random_mutation(candidate, rules)
        
    async def _claude_guided_mutation(self, candidate: Candidate, rules: List[SimpRule]) -> Candidate:
        """Apply Claude-guided intelligent mutation.
        
        Args:
            candidate: Candidate to mutate
            rules: Available rules
            
        Returns:
            Mutated candidate
        """
        if not self.claude_client:
            return await self._random_mutation(candidate, rules)
            
        # For now, fall back to random mutation
        # In a full implementation, this would:
        # 1. Analyze current candidate's mutations
        # 2. Query Claude for improvement suggestions
        # 3. Apply intelligent modifications
        
        return await self._random_mutation(candidate, rules)
        
    async def _generate_random_mutations(self, 
                                       rules: List[SimpRule],
                                       max_mutations: int = 3) -> List:
        """Generate random mutations for rules.
        
        Args:
            rules: Available rules
            max_mutations: Maximum number of mutations
            
        Returns:
            List of random mutations (simplified for now)
        """
        # This is a simplified implementation
        # In practice, this would create proper AppliedMutation objects
        
        if not rules:
            return []
            
        num_mutations = random.randint(1, min(max_mutations, len(rules)))
        selected_rules = random.sample(rules, num_mutations)
        
        mutations = []
        for rule in selected_rules:
            # Create a simplified mutation placeholder
            mutation = {
                "rule_name": rule.name,
                "type": random.choice(list(MutationType)),
                "rule": rule
            }
            mutations.append(mutation)
            
        return mutations
        
    def update_population(self, 
                         population: Population,
                         offspring: Population) -> Population:
        """Update population using elitist replacement strategy.
        
        Args:
            population: Current population
            offspring: Generated offspring
            
        Returns:
            New population for next generation
        """
        # Combine current population and offspring
        combined = population + offspring
        
        # Sort by fitness (descending)
        valid_candidates = [c for c in combined if c.is_valid]
        invalid_candidates = [c for c in combined if not c.is_valid]
        
        # Sort valid candidates by fitness
        valid_candidates.sort(key=lambda c: c.fitness_score, reverse=True)
        
        # Select best candidates for next generation
        new_population = []
        
        # Elitism: preserve best candidates
        elite_count = min(self.config.elite_size, len(valid_candidates))
        new_population.extend(valid_candidates[:elite_count])
        
        # Fill remaining slots
        remaining_slots = self.config.population_size - len(new_population)
        
        # Add remaining valid candidates
        remaining_valid = valid_candidates[elite_count:]
        if remaining_valid and remaining_slots > 0:
            additional_count = min(remaining_slots, len(remaining_valid))
            new_population.extend(remaining_valid[:additional_count])
            remaining_slots -= additional_count
            
        # If still need more, add some invalid candidates (for diversity)
        if remaining_slots > 0 and invalid_candidates:
            additional_count = min(remaining_slots, len(invalid_candidates))
            new_population.extend(invalid_candidates[:additional_count])
            
        # Update generation counter
        self.generation += 1
        
        # Update adaptive parameters
        self._update_adaptive_parameters(new_population)
        
        return new_population[:self.config.population_size]
        
    def _update_adaptive_parameters(self, population: Population):
        """Update adaptive parameters based on population state.
        
        Args:
            population: Current population
        """
        # Calculate population diversity
        diversity = self._calculate_diversity(population)
        self.diversity_history.append(diversity)
        
        # Update best fitness
        valid_candidates = [c for c in population if c.is_valid]
        if valid_candidates:
            best_fitness = max(c.fitness_score for c in valid_candidates)
            self.fitness_history.append(best_fitness)
            
            if best_fitness > self.best_ever_fitness:
                self.best_ever_fitness = best_fitness
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
                
        # Adapt mutation rate based on stagnation
        if self.config.adaptive_mutation:
            if self.stagnation_counter > 5:
                self.current_mutation_rate = min(self.current_mutation_rate * 1.1, 0.8)
            elif self.stagnation_counter == 0:
                self.current_mutation_rate = max(self.current_mutation_rate * 0.95, 0.05)
                
    def _calculate_diversity(self, population: Population) -> float:
        """Calculate population diversity metric.
        
        Args:
            population: Population to analyze
            
        Returns:
            Diversity score (0.0 to 1.0)
        """
        if len(population) < 2:
            return 0.0
            
        # Calculate diversity based on mutation patterns
        mutation_signatures = []
        
        for candidate in population:
            # Create signature based on mutations
            if candidate.mutations:
                signature = tuple(sorted([
                    getattr(m, 'rule_name', str(hash(m))) 
                    for m in candidate.mutations
                ]))
            else:
                signature = ()
            mutation_signatures.append(signature)
            
        # Count unique signatures
        unique_signatures = len(set(mutation_signatures))
        max_possible_unique = len(population)
        
        return unique_signatures / max_possible_unique
        
    def get_population_statistics(self, population: Population) -> PopulationStatistics:
        """Calculate statistics for a population.
        
        Args:
            population: Population to analyze
            
        Returns:
            Population statistics
        """
        valid_candidates = [c for c in population if c.is_valid]
        
        if not valid_candidates:
            return PopulationStatistics(
                generation=self.generation,
                size=len(population),
                valid_count=0,
                mean_fitness=0.0,
                std_fitness=0.0,
                max_fitness=0.0,
                min_fitness=0.0,
                diversity_score=0.0
            )
            
        fitness_scores = [c.fitness_score for c in valid_candidates]
        diversity = self._calculate_diversity(population)
        
        # Calculate mutation type distribution
        mutation_counts = {}
        for candidate in valid_candidates:
            for mutation in candidate.mutations:
                mut_type = getattr(mutation, 'type', 'unknown')
                if hasattr(mut_type, 'value'):
                    mut_type = mut_type.value
                mutation_counts[str(mut_type)] = mutation_counts.get(str(mut_type), 0) + 1
                
        return PopulationStatistics(
            generation=self.generation,
            size=len(population),
            valid_count=len(valid_candidates),
            mean_fitness=statistics.mean(fitness_scores),
            std_fitness=statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0.0,
            max_fitness=max(fitness_scores),
            min_fitness=min(fitness_scores),
            diversity_score=diversity,
            mutation_diversity=mutation_counts
        )
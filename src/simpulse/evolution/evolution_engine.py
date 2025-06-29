"""Main evolution engine for Simpulse optimization.

This module implements the complete evolutionary algorithm loop,
coordinating all components to optimize simp rules.
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict

from .models import SimpRule, OptimizationResult, OptimizationSession, OptimizationGoal
from .models_v2 import (
    Candidate, Population, FitnessScore, GenerationResult, 
    EvolutionHistory, EvolutionConfig, Workspace
)
from .rule_extractor import RuleExtractor
from .mutation_applicator import MutationApplicator
from .population_manager import PopulationManager
from .workspace_manager import WorkspaceManager
from ..evaluation.fitness_evaluator import FitnessEvaluator
from ..profiling import LeanRunner
from ..claude import ClaudeCodeClient
from ..config import Config

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """Main evolutionary optimization engine."""
    
    def __init__(self, config: Config):
        """Initialize evolution engine.
        
        Args:
            config: Simpulse configuration
        """
        self.config = config
        
        # Initialize components
        self.lean_runner = LeanRunner(
            lake_path=config.profiling.lake_path,
            lean_path=config.profiling.lean_path
        )
        
        self.rule_extractor = RuleExtractor()
        self.mutation_applicator = MutationApplicator()
        self.fitness_evaluator = FitnessEvaluator(self.lean_runner)
        
        # Initialize evolution-specific components
        evolution_config = EvolutionConfig(
            population_size=config.optimization.population_size,
            elite_size=config.optimization.elite_size,
            max_generations=config.optimization.generations,
            crossover_rate=config.optimization.crossover_rate,
            mutation_rate=config.optimization.mutation_rate,
            max_workers=config.optimization.max_parallel_evaluations,
            evaluation_timeout=config.profiling.timeout
        )
        
        self.claude_client = ClaudeCodeClient(
            command=config.claude.command_path,
            timeout=config.claude.timeout
        ) if config.claude.backend.value == "claude_code" else None
        
        self.population_manager = PopulationManager(
            evolution_config, 
            self.claude_client
        )
        
        self.workspace_manager = WorkspaceManager(
            base_path=config.paths.output_dir / "workspaces",
            max_workspaces=evolution_config.max_workers
        )
        
        self.evolution_config = evolution_config
        self.history = EvolutionHistory(
            session_id=f"evolution_{int(time.time())}",
            start_time=datetime.now()
        )
        
        # State tracking
        self.current_generation = 0
        self.total_evaluations = 0
        self.start_time = 0.0
        self.best_candidate: Optional[Candidate] = None
        
    async def run_evolution(self, 
                          modules: List[str],
                          source_path: Path,
                          time_budget: int = 3600) -> OptimizationResult:
        """Run the complete evolutionary optimization process.
        
        Args:
            modules: List of Lean modules to optimize
            source_path: Path to source code directory
            time_budget: Time budget in seconds
            
        Returns:
            Optimization result with best candidate and statistics
        """
        logger.info(f"Starting evolution for modules: {modules}")
        logger.info(f"Time budget: {time_budget} seconds")
        
        self.start_time = time.time()
        
        try:
            # Initialize workspace manager
            if not await self.workspace_manager.initialize(source_path):
                raise RuntimeError("Failed to initialize workspace manager")
                
            # Extract rules from modules
            rules = await self._extract_rules_from_modules(modules, source_path)
            logger.info(f"Extracted {len(rules)} simp rules")
            
            if not rules:
                raise RuntimeError("No simp rules found in specified modules")
                
            # Get Claude suggestions for intelligent seeding
            claude_suggestions = await self._get_claude_suggestions(rules)
            
            # Initialize population
            population = await self.population_manager.initialize_population(
                rules, claude_suggestions
            )
            
            # Establish baseline performance
            await self._establish_baseline(population, modules)
            
            # Main evolution loop
            while not self._should_terminate(time_budget):
                generation_result = await self._run_generation(population, rules, modules)
                
                # Update population
                population = generation_result.population
                self.current_generation += 1
                
                # Track best candidate
                if generation_result.new_best_found:
                    self.best_candidate = generation_result.best_candidate
                    self.history.best_ever_candidate = self.best_candidate
                    
                # Add to history
                self.history.add_generation(generation_result.generation_stats)
                
                # Log progress
                logger.info(
                    f"Generation {self.current_generation}: "
                    f"best={generation_result.generation_stats.max_fitness:.4f}, "
                    f"avg={generation_result.generation_stats.mean_fitness:.4f}, "
                    f"diversity={generation_result.generation_stats.diversity_score:.3f}"
                )
                
                # Check for convergence
                convergence_info = self.history.get_convergence_info()
                if convergence_info["converged"]:
                    logger.info("Population converged, stopping evolution")
                    self.history.termination_reason = "convergence"
                    break
                    
            # Finalize results
            self.history.end_time = datetime.now()
            
            if not self.history.termination_reason:
                self.history.termination_reason = "time_budget_exceeded"
                
            # Create optimization result
            result = await self._create_optimization_result(modules)
            
            logger.info(f"Evolution completed in {time.time() - self.start_time:.2f} seconds")
            logger.info(f"Best fitness: {result.best_fitness:.4f}")
            logger.info(f"Total generations: {self.current_generation}")
            
            return result
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            raise
        finally:
            # Cleanup workspaces
            await self.workspace_manager.cleanup_all()
            
    async def _run_generation(self, 
                            population: Population,
                            rules: List[SimpRule],
                            modules: List[str]) -> 'GenerationResult':
        """Run a single generation of evolution.
        
        Args:
            population: Current population
            rules: Available simp rules
            modules: Modules to evaluate
            
        Returns:
            Results from this generation
        """
        generation_start = time.time()
        logger.debug(f"Running generation {self.current_generation}")
        
        # Evaluate fitness for all candidates
        fitness_scores = await self.fitness_evaluator.parallel_evaluate(
            population, 
            modules, 
            max_workers=self.evolution_config.max_workers,
            timeout=self.evolution_config.evaluation_timeout
        )
        
        # Update candidate fitness
        for candidate, fitness in zip(population, fitness_scores):
            candidate.fitness = fitness
            
        self.total_evaluations += len(population)
        
        # Generate offspring through selection, crossover, and mutation
        offspring = await self._generate_offspring(population, rules)
        
        # Evaluate offspring fitness
        offspring_fitness = await self.fitness_evaluator.parallel_evaluate(
            offspring,
            modules,
            max_workers=self.evolution_config.max_workers,
            timeout=self.evolution_config.evaluation_timeout
        )
        
        for candidate, fitness in zip(offspring, offspring_fitness):
            candidate.fitness = fitness
            
        self.total_evaluations += len(offspring)
        
        # Update population for next generation
        new_population = self.population_manager.update_population(population, offspring)
        
        # Calculate generation statistics
        stats = self.population_manager.get_population_statistics(new_population)
        
        # Check for new best
        valid_candidates = [c for c in new_population if c.is_valid]
        best_candidate = None
        new_best_found = False
        
        if valid_candidates:
            best_candidate = max(valid_candidates, key=lambda c: c.fitness_score)
            
            if (self.best_candidate is None or 
                best_candidate.fitness_score > self.best_candidate.fitness_score):
                new_best_found = True
                
        generation_time = time.time() - generation_start
        
        return GenerationRunResult(
            population=new_population,
            best_candidate=best_candidate,
            new_best_found=new_best_found,
            generation_stats=GenerationResult(
                generation=self.current_generation,
                population_size=len(new_population),
                best_fitness=stats.max_fitness,
                average_fitness=stats.mean_fitness,
                worst_fitness=stats.min_fitness,
                diversity_score=stats.diversity_score,
                new_best_found=new_best_found,
                evaluation_time=generation_time,
                valid_candidates=stats.valid_count,
                best_candidate_id=best_candidate.id if best_candidate else ""
            )
        )
        
    async def _generate_offspring(self, 
                                population: Population,
                                rules: List[SimpRule]) -> Population:
        """Generate offspring through crossover and mutation.
        
        Args:
            population: Parent population
            rules: Available rules for mutation
            
        Returns:
            Generated offspring
        """
        offspring = []
        offspring_target = self.evolution_config.population_size - self.evolution_config.elite_size
        
        import random  # Import needed for random operations
        
        while len(offspring) < offspring_target:
            # Selection
            parent1 = self.population_manager.select_parent(population)
            parent2 = self.population_manager.select_parent(population)
            
            # Crossover
            if random.random() < self.evolution_config.crossover_rate:
                child = await self.population_manager.crossover(parent1, parent2)
            else:
                child = parent1.clone()
                
            # Mutation
            child = await self.population_manager.mutate(child, rules)
            
            offspring.append(child)
            
        return offspring
        
    async def _extract_rules_from_modules(self, 
                                        modules: List[str],
                                        source_path: Path) -> List[SimpRule]:
        """Extract simp rules from specified modules.
        
        Args:
            modules: Module names to extract from
            source_path: Path to source code
            
        Returns:
            List of extracted simp rules
        """
        all_rules = []
        
        for module in modules:
            # Convert module name to file path
            module_path = source_path / module.replace('.', '/') + '.lean'
            
            if module_path.exists():
                module_rules = self.rule_extractor.extract_rules_from_file(module_path)
                all_rules.extend(module_rules.rules)
                logger.debug(f"Extracted {len(module_rules.rules)} rules from {module}")
            else:
                logger.warning(f"Module file not found: {module_path}")
                
        return all_rules
        
    async def _get_claude_suggestions(self, rules: List[SimpRule]) -> Optional[List]:
        """Get Claude suggestions for intelligent population seeding.
        
        Args:
            rules: Available simp rules
            
        Returns:
            List of Claude suggestions or None
        """
        if not self.claude_client or not rules:
            return None
            
        try:
            # Create mock profile data for Claude
            mock_profile = {
                "total_time_ms": 1000.0,
                "rule_applications": len(rules) * 10,
                "successful_rewrites": len(rules) * 8,
                "failed_rewrites": len(rules) * 2,
                "theorem_usage": {rule.name: 5 for rule in rules[:10]}
            }
            
            suggestions = await self.claude_client.suggest_mutations(
                profile_data=mock_profile,
                rules=[asdict(rule) for rule in rules[:10]],  # Limit for performance
                top_k=5
            )
            
            logger.info(f"Received {len(suggestions)} Claude suggestions")
            return suggestions
            
        except Exception as e:
            logger.warning(f"Failed to get Claude suggestions: {e}")
            return None
            
    async def _establish_baseline(self, population: Population, modules: List[str]) -> None:
        """Establish baseline performance metrics.
        
        Args:
            population: Initial population
            modules: Modules to evaluate
        """
        baseline_candidates = [c for c in population if c.is_baseline]
        
        if baseline_candidates:
            logger.info("Establishing baseline performance")
            baseline = baseline_candidates[0]
            
            baseline_fitness = await self.fitness_evaluator.evaluate_candidate(
                baseline, modules, timeout=self.evolution_config.evaluation_timeout
            )
            
            baseline.fitness = baseline_fitness
            
            if baseline_fitness.is_valid:
                logger.info(f"Baseline established: {baseline_fitness.total_time:.2f}ms")
            else:
                logger.warning("Failed to establish valid baseline")
                
    def _should_terminate(self, time_budget: int) -> bool:
        """Check if evolution should terminate.
        
        Args:
            time_budget: Time budget in seconds
            
        Returns:
            True if should terminate
        """
        elapsed_time = time.time() - self.start_time
        
        # Time budget exceeded
        if elapsed_time >= time_budget:
            logger.info("Time budget exceeded")
            return True
            
        # Maximum generations reached
        if self.current_generation >= self.evolution_config.max_generations:
            logger.info("Maximum generations reached")
            return True
            
        # Maximum evaluations reached
        if self.total_evaluations >= self.evolution_config.max_evaluations:
            logger.info("Maximum evaluations reached")
            return True
            
        # Target fitness reached
        if (self.best_candidate and self.best_candidate.fitness and 
            self.best_candidate.fitness.composite_score >= self.evolution_config.target_fitness):
            logger.info("Target fitness reached")
            return True
            
        return False
        
    async def _create_optimization_result(self, modules: List[str]) -> OptimizationResult:
        """Create final optimization result.
        
        Args:
            modules: Optimized modules
            
        Returns:
            Complete optimization result
        """
        if not self.best_candidate or not self.best_candidate.fitness:
            # No valid results
            return OptimizationResult(
                best_fitness=0.0,
                best_candidate=None,
                total_generations=self.current_generation,
                total_evaluations=self.total_evaluations,
                execution_time=time.time() - self.start_time,
                modules=modules,
                success=False,
                error_message="No valid candidates found"
            )
            
        # Calculate improvement
        baseline_time = 1000.0  # Default baseline
        if self.fitness_evaluator.baseline_metrics:
            baseline_time = self.fitness_evaluator.baseline_metrics.total_time_ms
            
        improvement = (baseline_time - self.best_candidate.fitness.total_time) / baseline_time
        
        return OptimizationResult(
            best_fitness=self.best_candidate.fitness.composite_score,
            best_candidate=self.best_candidate,
            improvement_percent=improvement * 100,
            total_generations=self.current_generation,
            total_evaluations=self.total_evaluations,
            execution_time=time.time() - self.start_time,
            modules=modules,
            success=True,
            history=self.history
        )
        
    async def save_checkpoint(self, checkpoint_path: Path) -> bool:
        """Save current evolution state to checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
            
        Returns:
            True if checkpoint saved successfully
        """
        try:
            checkpoint_data = {
                "current_generation": self.current_generation,
                "total_evaluations": self.total_evaluations,
                "start_time": self.start_time,
                "best_candidate": asdict(self.best_candidate) if self.best_candidate else None,
                "history": asdict(self.history),
                "config": asdict(self.evolution_config)
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
                
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
            
    async def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load evolution state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if checkpoint loaded successfully
        """
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                
            self.current_generation = checkpoint_data["current_generation"]
            self.total_evaluations = checkpoint_data["total_evaluations"]
            self.start_time = checkpoint_data["start_time"]
            
            # Note: Full reconstruction of complex objects would require more work
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False


# Helper classes for return values
class GenerationRunResult:
    """Results from running a single generation - includes population and stats."""
    
    def __init__(self, 
                 population: Population,
                 best_candidate: Optional[Candidate],
                 new_best_found: bool,
                 generation_stats: GenerationResult):
        self.population = population
        self.best_candidate = best_candidate
        self.new_best_found = new_best_found
        self.generation_stats = generation_stats


class OptimizationResult:
    """Final result from evolutionary optimization."""
    
    def __init__(self,
                 best_fitness: float,
                 best_candidate: Optional[Candidate],
                 total_generations: int,
                 total_evaluations: int,
                 execution_time: float,
                 modules: List[str],
                 success: bool,
                 improvement_percent: float = 0.0,
                 error_message: Optional[str] = None,
                 history: Optional[EvolutionHistory] = None):
        self.best_fitness = best_fitness
        self.best_candidate = best_candidate
        self.improvement_percent = improvement_percent
        self.total_generations = total_generations
        self.total_evaluations = total_evaluations
        self.execution_time = execution_time
        self.modules = modules
        self.success = success
        self.error_message = error_message
        self.history = history
"""Additional evolution-specific models for Phase 2.

This module extends the base models with evolution-specific data structures
for population management, fitness evaluation, and generation tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid

from .models import MutationSuggestion, PerformanceMetrics


class SelectionMethod(Enum):
    """Selection methods for evolutionary algorithm."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITE = "elite"


class CrossoverMethod(Enum):
    """Crossover methods for combining parents."""
    UNIFORM = "uniform"
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    SEMANTIC = "semantic"


class MutationMethod(Enum):
    """Mutation methods for generating diversity."""
    RANDOM = "random"
    ADAPTIVE = "adaptive"
    CLAUDE_GUIDED = "claude_guided"
    HEURISTIC = "heuristic"


@dataclass
class AppliedMutation:
    """Represents a mutation that has been applied to source code."""
    id: str
    original_rule_name: str
    suggestion: MutationSuggestion
    file_path: Path
    line_numbers: tuple[int, int]  # (start, end)
    original_content: str
    modified_content: str
    applied_successfully: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def rollback(self) -> bool:
        """Rollback this mutation by restoring original content."""
        try:
            with open(self.file_path, 'w') as f:
                f.write(self.original_content)
            return True
        except Exception:
            return False


@dataclass
class FitnessScore:
    """Comprehensive fitness score for a candidate."""
    total_time: float
    simp_time: float
    iterations: int
    max_depth: int
    memory_mb: float
    composite_score: float
    is_valid: bool
    baseline_improvement: float = 0.0
    individual_scores: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    evaluation_duration: float = 0.0
    
    @property
    def time_improvement_percent(self) -> float:
        """Get time improvement as percentage."""
        return self.baseline_improvement * 100.0
        
    def dominates(self, other: 'FitnessScore') -> bool:
        """Check if this score dominates another (Pareto dominance)."""
        if not self.is_valid or not other.is_valid:
            return self.is_valid and not other.is_valid
            
        # This dominates other if it's better in at least one objective
        # and not worse in any objective
        better_in_any = (
            self.total_time <= other.total_time or
            self.iterations <= other.iterations or
            self.memory_mb <= other.memory_mb
        )
        
        worse_in_any = (
            self.total_time > other.total_time or
            self.iterations > other.iterations or
            self.memory_mb > other.memory_mb
        )
        
        return better_in_any and not worse_in_any


@dataclass
class Candidate:
    """Represents a candidate solution in the evolutionary algorithm."""
    id: str
    mutations: List[AppliedMutation]
    fitness: Optional[FitnessScore] = None
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    creation_method: str = "unknown"  # random, crossover, mutation, claude
    evaluation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure candidate has a valid ID."""
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
    
    @property
    def is_baseline(self) -> bool:
        """Check if this is the baseline candidate (no mutations)."""
        return len(self.mutations) == 0
        
    @property
    def mutation_count(self) -> int:
        """Get number of mutations in this candidate."""
        return len(self.mutations)
        
    @property
    def is_valid(self) -> bool:
        """Check if candidate is valid (has fitness and is valid)."""
        return self.fitness is not None and self.fitness.is_valid
        
    @property
    def fitness_score(self) -> float:
        """Get composite fitness score."""
        return self.fitness.composite_score if self.fitness else 0.0
        
    def get_mutation_summary(self) -> Dict[str, int]:
        """Get summary of mutation types in this candidate."""
        from .models import MutationType
        
        summary = {}
        for mutation in self.mutations:
            mut_type = mutation.suggestion.mutation_type.value
            summary[mut_type] = summary.get(mut_type, 0) + 1
            
        return summary
        
    def clone(self, new_id: Optional[str] = None) -> 'Candidate':
        """Create a copy of this candidate."""
        return Candidate(
            id=new_id or str(uuid.uuid4())[:8],
            mutations=self.mutations.copy(),
            fitness=None,  # Fitness needs to be re-evaluated
            generation=self.generation,
            parent_ids=[self.id],
            creation_method="clone",
            metadata=self.metadata.copy()
        )


@dataclass
class GenerationResult:
    """Results from a single generation of evolution."""
    generation: int
    population_size: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    diversity_score: float
    new_best_found: bool
    evaluation_time: float
    valid_candidates: int
    best_candidate_id: str
    fitness_distribution: Dict[str, float] = field(default_factory=dict)
    mutation_statistics: Dict[str, int] = field(default_factory=dict)
    
    @property
    def fitness_range(self) -> float:
        """Get range of fitness scores."""
        return self.best_fitness - self.worst_fitness
        
    @property
    def valid_ratio(self) -> float:
        """Get ratio of valid candidates."""
        return self.valid_candidates / self.population_size if self.population_size > 0 else 0.0


@dataclass
class EvolutionHistory:
    """Tracks the history of an evolution run."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    generations: List[GenerationResult] = field(default_factory=list)
    best_ever_candidate: Optional[Candidate] = None
    total_evaluations: int = 0
    convergence_generation: Optional[int] = None
    termination_reason: str = ""
    
    def add_generation(self, result: GenerationResult):
        """Add a generation result to history."""
        self.generations.append(result)
        self.total_evaluations += result.population_size
        
        # Update best ever if this generation found a new best
        if result.new_best_found:
            self.convergence_generation = None  # Reset convergence
        
    def get_convergence_info(self, patience: int = 10) -> Dict[str, Any]:
        """Get information about convergence."""
        if len(self.generations) < patience:
            return {"converged": False, "generations_stable": len(self.generations)}
            
        recent_best = [g.best_fitness for g in self.generations[-patience:]]
        improvement = max(recent_best) - min(recent_best)
        
        return {
            "converged": improvement < 0.001,  # 0.1% improvement threshold
            "generations_stable": patience,
            "recent_improvement": improvement,
            "best_fitness_trend": recent_best
        }
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the evolution run."""
        if not self.generations:
            return {"error": "No generations completed"}
            
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        return {
            "session_id": self.session_id,
            "total_generations": len(self.generations),
            "total_evaluations": self.total_evaluations,
            "duration_seconds": duration,
            "best_fitness": max(g.best_fitness for g in self.generations),
            "final_fitness": self.generations[-1].best_fitness,
            "convergence_generation": self.convergence_generation,
            "termination_reason": self.termination_reason,
            "average_generation_time": sum(g.evaluation_time for g in self.generations) / len(self.generations)
        }


@dataclass
class Workspace:
    """Represents an isolated workspace for evaluation."""
    id: str
    path: Path
    is_active: bool = False
    last_used: datetime = field(default_factory=datetime.now)
    candidate_id: Optional[str] = None
    
    def activate(self, candidate_id: str):
        """Mark workspace as active for a candidate."""
        self.is_active = True
        self.candidate_id = candidate_id
        self.last_used = datetime.now()
        
    def deactivate(self):
        """Mark workspace as inactive."""
        self.is_active = False
        self.candidate_id = None
        self.last_used = datetime.now()


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithm parameters."""
    population_size: int = 30
    elite_size: int = 3
    tournament_size: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM
    mutation_method: MutationMethod = MutationMethod.ADAPTIVE
    
    # Termination criteria
    max_generations: int = 50
    max_evaluations: int = 1500
    target_fitness: float = 0.95
    convergence_patience: int = 10
    time_budget_seconds: int = 3600
    
    # Parallel evaluation
    max_workers: int = 4
    evaluation_timeout: float = 300.0
    
    # Adaptive parameters
    adaptive_mutation: bool = True
    adaptive_crossover: bool = True
    diversity_threshold: float = 0.1
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if self.population_size < 4:
            errors.append("Population size must be at least 4")
            
        if self.elite_size >= self.population_size:
            errors.append("Elite size must be less than population size")
            
        if not 0 <= self.crossover_rate <= 1:
            errors.append("Crossover rate must be between 0 and 1")
            
        if not 0 <= self.mutation_rate <= 1:
            errors.append("Mutation rate must be between 0 and 1")
            
        if self.tournament_size < 2:
            errors.append("Tournament size must be at least 2")
            
        if self.max_workers < 1:
            errors.append("Max workers must be at least 1")
            
        return errors


@dataclass
class OptimizationObjectives:
    """Defines optimization objectives and their weights."""
    minimize_time: float = 0.6
    minimize_memory: float = 0.1
    minimize_iterations: float = 0.2
    maximize_success_rate: float = 0.1
    
    def __post_init__(self):
        """Normalize weights to sum to 1.0."""
        total = self.minimize_time + self.minimize_memory + self.minimize_iterations + self.maximize_success_rate
        if total > 0:
            self.minimize_time /= total
            self.minimize_memory /= total
            self.minimize_iterations /= total
            self.maximize_success_rate /= total
            
    def get_weights(self) -> Dict[str, float]:
        """Get weights as dictionary."""
        return {
            "time": self.minimize_time,
            "memory": self.minimize_memory,
            "iterations": self.minimize_iterations,
            "success_rate": self.maximize_success_rate
        }


@dataclass
class PopulationStatistics:
    """Statistics for a population of candidates."""
    generation: int
    size: int
    valid_count: int
    mean_fitness: float
    std_fitness: float
    max_fitness: float
    min_fitness: float
    diversity_score: float
    mutation_diversity: Dict[str, int] = field(default_factory=dict)
    
    @property
    def validity_ratio(self) -> float:
        """Get ratio of valid candidates."""
        return self.valid_count / self.size if self.size > 0 else 0.0
        
    @property
    def fitness_range(self) -> float:
        """Get range of fitness values."""
        return self.max_fitness - self.min_fitness
        
    def is_converged(self, threshold: float = 0.001) -> bool:
        """Check if population has converged."""
        return self.std_fitness < threshold and self.diversity_score < 0.1


# Type aliases for clarity
Population = List[Candidate]
FitnessScores = List[FitnessScore]
MutationSet = List[AppliedMutation]
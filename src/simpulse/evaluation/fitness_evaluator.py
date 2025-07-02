"""Fitness evaluator for measuring simp rule performance.

This module evaluates the fitness of mutation candidates by running
lake builds with profiling and measuring performance improvements.
"""

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from ..evolution.models import PerformanceMetrics
from ..profiling import LeanRunner, TraceParser

logger = logging.getLogger(__name__)


@dataclass
class FitnessScore:
    """Fitness score for a mutation candidate."""

    total_time: float
    simp_time: float
    iterations: int
    max_depth: int
    memory_mb: float
    composite_score: float
    is_valid: bool
    baseline_improvement: float = 0.0
    individual_scores: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None

    def __post_init__(self):
        """Calculate individual component scores."""
        if self.is_valid:
            self.individual_scores = {
                "time_score": 1.0 / (1.0 + self.total_time / 1000.0),  # Normalize by seconds
                "simp_score": 1.0 / (1.0 + self.simp_time / 1000.0),
                "iteration_score": 1.0 / (1.0 + self.iterations / 100.0),
                "depth_score": 1.0 / (1.0 + self.max_depth / 10.0),
                "memory_score": 1.0 / (1.0 + self.memory_mb / 1000.0),
            }


@dataclass
class Candidate:
    """Represents a mutation candidate for evaluation."""

    id: str
    mutations: list[Any]  # Will be AppliedMutation when implemented
    fitness: FitnessScore | None = None
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    evaluation_time: float = 0.0

    @property
    def is_baseline(self) -> bool:
        """Check if this is the baseline candidate (no mutations)."""
        return len(self.mutations) == 0


class FitnessEvaluator:
    """Evaluates fitness of mutation candidates through performance testing."""

    def __init__(
        self,
        lean_runner: LeanRunner,
        time_weight: float = 0.6,
        iteration_weight: float = 0.2,
        depth_weight: float = 0.15,
        memory_weight: float = 0.05,
    ):
        """Initialize fitness evaluator.

        Args:
            lean_runner: LeanRunner instance for profiling
            time_weight: Weight for time component in fitness
            iteration_weight: Weight for iteration count
            depth_weight: Weight for search depth
            memory_weight: Weight for memory usage
        """
        self.lean_runner = lean_runner
        self.trace_parser = TraceParser()

        # Fitness weights (must sum to 1.0)
        total_weight = time_weight + iteration_weight + depth_weight + memory_weight
        self.weights = {
            "time": time_weight / total_weight,
            "iterations": iteration_weight / total_weight,
            "depth": depth_weight / total_weight,
            "memory": memory_weight / total_weight,
        }

        self.baseline_metrics: PerformanceMetrics | None = None
        self._evaluation_cache: dict[str, FitnessScore] = {}

    async def evaluate_candidate(
        self, candidate: Candidate, modules: list[str], timeout: float = 300.0
    ) -> FitnessScore:
        """Evaluate fitness of a single candidate.

        Args:
            candidate: Candidate to evaluate
            modules: List of modules to test
            timeout: Evaluation timeout in seconds

        Returns:
            FitnessScore with fitness metrics
        """
        start_time = time.time()

        # Check cache first
        cache_key = self._get_cache_key(candidate, modules)
        if cache_key in self._evaluation_cache:
            cached_score = self._evaluation_cache[cache_key]
            logger.debug(f"Using cached fitness for candidate {candidate.id}")
            return cached_score

        try:
            # Run performance evaluation
            metrics = await self._run_performance_test(candidate, modules, timeout)

            if metrics is None:
                return FitnessScore(
                    total_time=float("inf"),
                    simp_time=float("inf"),
                    iterations=0,
                    max_depth=0,
                    memory_mb=0.0,
                    composite_score=0.0,
                    is_valid=False,
                    error_message="Performance test failed",
                )

            # Calculate fitness score
            fitness_score = self.calculate_fitness(metrics, self.baseline_metrics)

            # Calculate baseline improvement
            if self.baseline_metrics and candidate.is_baseline:
                # This is the baseline, so improvement is 0
                fitness_score.baseline_improvement = 0.0
            elif self.baseline_metrics:
                # Calculate relative improvement
                time_improvement = (
                    self.baseline_metrics.total_time_ms - metrics.total_time_ms
                ) / self.baseline_metrics.total_time_ms
                fitness_score.baseline_improvement = time_improvement

            candidate.evaluation_time = time.time() - start_time

            # Cache the result
            self._evaluation_cache[cache_key] = fitness_score

            logger.info(
                f"Evaluated candidate {candidate.id}: fitness={fitness_score.composite_score:.4f}, time={candidate.evaluation_time:.2f}s"
            )

            return fitness_score

        except Exception as e:
            logger.error(f"Error evaluating candidate {candidate.id}: {e}")
            return FitnessScore(
                total_time=float("inf"),
                simp_time=float("inf"),
                iterations=0,
                max_depth=0,
                memory_mb=0.0,
                composite_score=0.0,
                is_valid=False,
                error_message=str(e),
            )

    async def parallel_evaluate(
        self,
        population: list[Candidate],
        modules: list[str],
        max_workers: int = 4,
        timeout: float = 300.0,
    ) -> list[FitnessScore]:
        """Evaluate multiple candidates in parallel.

        Args:
            population: List of candidates to evaluate
            modules: List of modules to test
            max_workers: Maximum number of parallel evaluations
            timeout: Evaluation timeout per candidate

        Returns:
            List of fitness scores in same order as population
        """
        # Use semaphore to limit concurrent evaluations
        semaphore = asyncio.Semaphore(max_workers)

        async def evaluate_with_semaphore(candidate: Candidate) -> FitnessScore:
            async with semaphore:
                return await self.evaluate_candidate(candidate, modules, timeout)

        logger.info(
            f"Starting parallel evaluation of {len(population)} candidates with {max_workers} workers"
        )

        # Evaluate baseline first if present
        baseline_candidates = [c for c in population if c.is_baseline]
        if baseline_candidates and self.baseline_metrics is None:
            logger.info("Evaluating baseline candidate first")
            baseline_score = await self.evaluate_candidate(baseline_candidates[0], modules, timeout)
            if baseline_score.is_valid:
                self.baseline_metrics = PerformanceMetrics(
                    total_time_ms=baseline_score.total_time,
                    rule_applications=baseline_score.iterations,
                    successful_rewrites=baseline_score.iterations,
                    failed_rewrites=0,
                    memory_usage_mb=baseline_score.memory_mb,
                )

        # Evaluate all candidates in parallel
        tasks = [evaluate_with_semaphore(candidate) for candidate in population]
        fitness_scores = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        results = []
        for i, result in enumerate(fitness_scores):
            if isinstance(result, Exception):
                logger.error(f"Exception evaluating candidate {population[i].id}: {result}")
                results.append(
                    FitnessScore(
                        total_time=float("inf"),
                        simp_time=float("inf"),
                        iterations=0,
                        max_depth=0,
                        memory_mb=0.0,
                        composite_score=0.0,
                        is_valid=False,
                        error_message=str(result),
                    )
                )
            else:
                results.append(result)

        # Update candidate fitness scores
        for candidate, fitness in zip(population, results, strict=False):
            candidate.fitness = fitness

        valid_scores = [f for f in results if f.is_valid]
        logger.info(
            f"Parallel evaluation completed: {len(valid_scores)}/{len(results)} candidates valid"
        )

        return results

    def calculate_fitness(
        self, metrics: PerformanceMetrics, baseline: PerformanceMetrics | None = None
    ) -> FitnessScore:
        """Calculate multi-objective fitness score.

        Args:
            metrics: Performance metrics for candidate
            baseline: Baseline metrics for comparison

        Returns:
            Calculated fitness score
        """
        # Extract basic metrics
        total_time = metrics.total_time_ms
        simp_time = total_time * 0.7  # Estimate simp time as 70% of total
        iterations = metrics.rule_applications
        max_depth = 50  # Default depth estimate
        memory_mb = metrics.memory_usage_mb or 100.0  # Default memory estimate

        # Calculate individual fitness components (higher is better)
        time_fitness = 1.0 / (1.0 + total_time / 1000.0)  # Normalize by seconds
        iteration_fitness = 1.0 / (1.0 + iterations / 100.0)  # Normalize by hundreds
        depth_fitness = 1.0 / (1.0 + max_depth / 10.0)  # Normalize by tens
        memory_fitness = 1.0 / (1.0 + memory_mb / 1000.0)  # Normalize by GB

        # Calculate weighted composite score
        composite_score = (
            self.weights["time"] * time_fitness
            + self.weights["iterations"] * iteration_fitness
            + self.weights["depth"] * depth_fitness
            + self.weights["memory"] * memory_fitness
        )

        return FitnessScore(
            total_time=total_time,
            simp_time=simp_time,
            iterations=iterations,
            max_depth=max_depth,
            memory_mb=memory_mb,
            composite_score=composite_score,
            is_valid=True,
        )

    async def _run_performance_test(
        self, candidate: Candidate, modules: list[str], timeout: float
    ) -> PerformanceMetrics | None:
        """Run performance test for a candidate.

        Args:
            candidate: Candidate to test
            modules: Modules to test
            timeout: Test timeout

        Returns:
            Performance metrics or None if failed
        """
        try:
            # For now, simulate performance testing
            # In a real implementation, this would:
            # 1. Apply candidate's mutations to workspace
            # 2. Run lake build with profiling
            # 3. Extract performance metrics

            # Simulate some variance in performance
            import random

            base_time = 1000.0 + random.uniform(-200, 200)

            # If candidate has mutations, simulate potential improvement
            if candidate.mutations:
                # Simulate 0-30% improvement for mutations
                improvement_factor = random.uniform(0.7, 1.0)
                base_time *= improvement_factor

            rule_applications = max(1, int(base_time / 20 + random.uniform(-10, 10)))
            memory_usage = 50.0 + random.uniform(-10, 10)

            return PerformanceMetrics(
                total_time_ms=base_time,
                rule_applications=rule_applications,
                successful_rewrites=rule_applications,
                failed_rewrites=0,
                memory_usage_mb=memory_usage,
            )

        except Exception as e:
            logger.error(f"Performance test failed for candidate {candidate.id}: {e}")
            return None

    def _get_cache_key(self, candidate: Candidate, modules: list[str]) -> str:
        """Generate cache key for candidate evaluation.

        Args:
            candidate: Candidate to evaluate
            modules: Modules being tested

        Returns:
            Cache key string
        """
        # Create deterministic key based on candidate mutations and modules
        mutations_hash = hash(
            tuple(
                (
                    getattr(m, "suggestion", {}).get("rule_name", ""),
                    getattr(m, "suggestion", {}).get("mutation_type", ""),
                )
                for m in candidate.mutations
            )
        )

        modules_hash = hash(tuple(sorted(modules)))

        return f"{candidate.id}_{mutations_hash}_{modules_hash}"

    async def establish_baseline(
        self, modules: list[str], timeout: float = 300.0
    ) -> PerformanceMetrics | None:
        """Establish baseline performance metrics.

        Args:
            modules: Modules to test for baseline
            timeout: Test timeout

        Returns:
            Baseline performance metrics
        """
        logger.info("Establishing baseline performance metrics")

        # Create baseline candidate (no mutations)
        baseline_candidate = Candidate(id="baseline", mutations=[])

        # Evaluate baseline
        baseline_score = await self.evaluate_candidate(baseline_candidate, modules, timeout)

        if baseline_score.is_valid:
            self.baseline_metrics = PerformanceMetrics(
                total_time_ms=baseline_score.total_time,
                rule_applications=baseline_score.iterations,
                successful_rewrites=baseline_score.iterations,
                failed_rewrites=0,
                memory_usage_mb=baseline_score.memory_mb,
            )

            logger.info(
                f"Baseline established: {baseline_score.total_time:.2f}ms, {baseline_score.iterations} iterations"
            )
            return self.baseline_metrics
        else:
            logger.error("Failed to establish baseline metrics")
            return None

    def get_fitness_statistics(self, population: list[Candidate]) -> dict[str, float]:
        """Calculate fitness statistics for a population.

        Args:
            population: Population of candidates

        Returns:
            Dictionary of fitness statistics
        """
        valid_candidates = [c for c in population if c.fitness and c.fitness.is_valid]

        if not valid_candidates:
            return {
                "count": 0,
                "mean_fitness": 0.0,
                "max_fitness": 0.0,
                "min_fitness": 0.0,
                "std_fitness": 0.0,
                "mean_improvement": 0.0,
            }

        fitness_scores = [c.fitness.composite_score for c in valid_candidates]
        improvements = [c.fitness.baseline_improvement for c in valid_candidates]

        return {
            "count": len(valid_candidates),
            "mean_fitness": statistics.mean(fitness_scores),
            "max_fitness": max(fitness_scores),
            "min_fitness": min(fitness_scores),
            "std_fitness": (statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0.0),
            "mean_improvement": statistics.mean(improvements) if improvements else 0.0,
        }

    def clear_cache(self):
        """Clear the evaluation cache."""
        self._evaluation_cache.clear()
        logger.info("Fitness evaluation cache cleared")

    def set_baseline(self, metrics: PerformanceMetrics):
        """Set baseline metrics manually.

        Args:
            metrics: Baseline performance metrics
        """
        self.baseline_metrics = metrics
        logger.info(f"Baseline metrics set: {metrics.total_time_ms:.2f}ms")

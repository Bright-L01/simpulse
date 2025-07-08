#!/usr/bin/env python3
"""
Adaptive Exploration: Optimizing the Optimizer's Learning

Sophisticated exploration strategies that balance learning speed
with user performance. No sacrificing user experience for learning.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExplorationStats:
    """Statistics for exploration behavior"""

    total_explorations: int = 0
    successful_explorations: int = 0
    exploration_regret: float = 0.0
    novel_discoveries: int = 0
    context_coverage: Dict[str, int] = field(default_factory=dict)

    @property
    def exploration_success_rate(self) -> float:
        return self.successful_explorations / max(1, self.total_explorations)

    @property
    def average_exploration_regret(self) -> float:
        return self.exploration_regret / max(1, self.total_explorations)


class CuriosityMechanism:
    """
    Implements "curiosity" for understudied contexts.

    Key insight: We should be more curious about contexts where:
    1. We have little data
    2. Recent results show high variance
    3. The context appears frequently but isn't well understood
    """

    def __init__(
        self,
        min_samples_threshold: int = 10,
        variance_threshold: float = 0.3,
        frequency_weight: float = 0.2,
    ):
        self.min_samples_threshold = min_samples_threshold
        self.variance_threshold = variance_threshold
        self.frequency_weight = frequency_weight

        # Track context frequency and recency
        self.context_frequency = defaultdict(int)
        self.context_last_seen = defaultdict(float)
        self.context_variance = defaultdict(float)

    def calculate_curiosity_score(self, context: str, strategy_stats: Dict[str, Any]) -> float:
        """
        Calculate how curious we should be about a context.

        Higher score = more curiosity = more exploration
        """
        # Update frequency and recency
        self.context_frequency[context] += 1
        self.context_last_seen[context] = time.time()

        # Factor 1: Data scarcity (more curiosity for less data)
        total_samples = sum(stats.pulls for stats in strategy_stats.values())
        if len(strategy_stats) == 0:
            scarcity_score = 1.0  # Maximum curiosity for unknown context
        else:
            scarcity_score = max(
                0, 1.0 - total_samples / (self.min_samples_threshold * len(strategy_stats))
            )

        # Factor 2: High variance (more curiosity for unpredictable contexts)
        variance_scores = []
        for stats in strategy_stats.values():
            if len(stats.speedups) > 1:
                variance = np.var(stats.speedups)
                variance_scores.append(variance)

        avg_variance = np.mean(variance_scores) if variance_scores else 0.5
        variance_score = min(1.0, avg_variance / self.variance_threshold)

        # Factor 3: Frequency (more curiosity for frequent but understudied contexts)
        frequency_score = min(1.0, self.context_frequency[context] / 100.0) * self.frequency_weight

        # Factor 4: Recency (less curiosity for recently explored)
        time_since_last = time.time() - self.context_last_seen.get(context, 0)
        recency_score = min(1.0, time_since_last / 3600.0)  # 1 hour decay

        # Combine factors (weighted average)
        curiosity_score = (
            0.4 * scarcity_score
            + 0.3 * variance_score
            + 0.2 * frequency_score
            + 0.1 * recency_score
        )

        self.context_variance[context] = avg_variance

        logger.debug(
            f"Curiosity for {context}: {curiosity_score:.2f} "
            f"(scarcity: {scarcity_score:.2f}, variance: {variance_score:.2f})"
        )

        return curiosity_score


class AdaptiveEpsilonGreedy:
    """
    Epsilon-greedy with intelligent decay and context awareness.

    Key features:
    1. Epsilon decays based on confidence, not just time
    2. Per-context epsilon adjustment
    3. Minimum exploration guarantee
    4. Curiosity-driven exploration boosts
    """

    def __init__(
        self,
        initial_epsilon: float = 0.3,
        min_epsilon: float = 0.05,
        decay_rate: float = 0.995,
        confidence_threshold: float = 0.8,
    ):
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.confidence_threshold = confidence_threshold

        # Per-context epsilon tracking
        self.context_epsilon = defaultdict(lambda: initial_epsilon)
        self.context_confidence = defaultdict(float)

        # Global exploration statistics
        self.exploration_stats = ExplorationStats()

    def get_epsilon(
        self, context: str, overall_confidence: float, curiosity_score: float = 0.0
    ) -> float:
        """Get adaptive epsilon for context"""

        # Base epsilon with decay
        base_epsilon = max(self.min_epsilon, self.context_epsilon[context] * self.decay_rate)

        # Confidence adjustment (lower confidence = more exploration)
        confidence_factor = max(0.5, 1.0 - overall_confidence)

        # Curiosity boost (high curiosity = more exploration)
        curiosity_boost = curiosity_score * 0.2

        # Calculate final epsilon
        epsilon = min(0.8, base_epsilon * confidence_factor + curiosity_boost)

        # Update tracking
        self.context_epsilon[context] = base_epsilon
        self.context_confidence[context] = overall_confidence

        logger.debug(
            f"Epsilon for {context}: {epsilon:.3f} "
            f"(base: {base_epsilon:.3f}, confidence: {overall_confidence:.2f})"
        )

        return epsilon

    def should_explore(
        self, context: str, overall_confidence: float, curiosity_score: float = 0.0
    ) -> bool:
        """Decide whether to explore"""
        epsilon = self.get_epsilon(context, overall_confidence, curiosity_score)
        return np.random.random() < epsilon


class IntelligentThompsonSampling:
    """
    Thompson Sampling with intelligent priors and curiosity integration.

    Improvements over basic Thompson Sampling:
    1. Informative priors based on domain knowledge
    2. Curiosity-adjusted sampling
    3. Exploration tracking and analysis
    """

    def __init__(self):
        # Strategy priors based on empirical knowledge
        self.strategy_priors = {
            "no_optimization": (1.0, 1.0),  # Neutral prior
            "conservative": (3.0, 1.0),  # Optimistic prior (usually safe)
            "moderate": (2.0, 1.0),  # Slightly optimistic
            "aggressive": (1.0, 2.0),  # Pessimistic prior (risky)
            "selective_top5": (4.0, 1.0),  # Very optimistic (selective is safe)
            "contextual_arithmetic": (1.0, 1.0),  # Context-dependent
            "inverse_reduction": (1.0, 3.0),  # Pessimistic (often backfires)
            "adaptive_threshold": (2.0, 1.0),  # Moderately optimistic
        }

        # Exploration tracking
        self.exploration_stats = ExplorationStats()

    def sample_strategy(
        self,
        context: str,
        strategy_stats: Dict[str, Any],
        strategies: List[str],
        curiosity_score: float = 0.0,
    ) -> Tuple[str, bool]:
        """
        Sample strategy using Thompson Sampling with curiosity.

        Returns: (strategy, is_exploration)
        """
        samples = {}

        for strategy in strategies:
            stats = strategy_stats.get(strategy)

            # Get priors
            alpha_prior, beta_prior = self.strategy_priors.get(strategy, (1.0, 1.0))

            # Update with observed data
            if stats and stats.pulls > 0:
                alpha = alpha_prior + stats.successes
                beta = beta_prior + (stats.pulls - stats.successes)
            else:
                alpha, beta = alpha_prior, beta_prior

            # Curiosity adjustment: boost exploration for understudied strategies
            if stats is None or stats.pulls < 5:
                curiosity_boost = curiosity_score * 0.3
                alpha += curiosity_boost

            # Sample from Beta distribution
            theta = np.random.beta(alpha, beta)

            # Expected reward = success probability * expected speedup
            if stats and stats.pulls > 0:
                expected_speedup = stats.mean_speedup
            else:
                # Use optimistic default based on strategy type
                expected_speedup = self._get_optimistic_default(strategy)

            samples[strategy] = theta * expected_speedup

        # Select strategy with highest sample
        selected_strategy = max(samples.items(), key=lambda x: x[1])[0]

        # Determine if this is exploration
        is_exploration = self._is_exploration(selected_strategy, strategy_stats)

        # Update exploration stats
        if is_exploration:
            self.exploration_stats.total_explorations += 1

        return selected_strategy, is_exploration

    def _get_optimistic_default(self, strategy: str) -> float:
        """Get optimistic default speedup for unknown strategies"""
        defaults = {
            "no_optimization": 1.0,
            "conservative": 1.15,
            "moderate": 1.3,
            "aggressive": 1.5,
            "selective_top5": 1.25,
            "contextual_arithmetic": 1.4,
            "inverse_reduction": 0.95,
            "adaptive_threshold": 1.35,
        }
        return defaults.get(strategy, 1.2)

    def _is_exploration(self, selected_strategy: str, strategy_stats: Dict[str, Any]) -> bool:
        """Determine if selection is exploration vs exploitation"""
        # Find best known strategy
        best_strategy = None
        best_performance = 0.0

        for strategy, stats in strategy_stats.items():
            if stats and stats.pulls > 0 and stats.mean_speedup > best_performance:
                best_performance = stats.mean_speedup
                best_strategy = strategy

        return selected_strategy != best_strategy


class CoverageOptimizer:
    """
    Ensures coverage of rare but important patterns.

    Problem: Some contexts are rare but when they appear, optimization
    is crucial (e.g., performance-critical code paths).

    Solution: Track coverage and boost exploration for under-sampled
    but important contexts.
    """

    def __init__(self, min_coverage_threshold: int = 5, importance_decay: float = 0.9):
        self.min_coverage_threshold = min_coverage_threshold
        self.importance_decay = importance_decay

        # Track context importance and coverage
        self.context_importance = defaultdict(float)
        self.context_coverage = defaultdict(int)
        self.context_last_boost = defaultdict(float)

    def update_context_importance(self, context: str, file_path: str, compilation_time: float):
        """Update importance score for context"""
        # Importance factors:
        # 1. Compilation time (longer = more important to optimize)
        # 2. File frequency (more frequent = more important)
        # 3. Recent activity

        time_importance = min(2.0, compilation_time / 1.0)  # 1 second baseline
        frequency_boost = 1.0  # Could be enhanced with file access patterns

        # Update importance with decay
        current_importance = self.context_importance[context]
        new_importance = (
            current_importance * self.importance_decay + time_importance * frequency_boost
        )

        self.context_importance[context] = new_importance

        logger.debug(f"Context importance for {context}: {new_importance:.2f}")

    def calculate_coverage_boost(self, context: str) -> float:
        """Calculate exploration boost for coverage"""
        coverage = self.context_coverage[context]
        importance = self.context_importance[context]

        # Under-covered contexts get exploration boost
        if coverage < self.min_coverage_threshold:
            coverage_factor = (self.min_coverage_threshold - coverage) / self.min_coverage_threshold
            importance_factor = min(2.0, importance)

            # Avoid spam: limit boost frequency
            time_since_boost = time.time() - self.context_last_boost[context]
            if time_since_boost > 300:  # 5 minutes
                boost = coverage_factor * importance_factor * 0.3
                self.context_last_boost[context] = time.time()
                return boost

        return 0.0

    def record_coverage(self, context: str):
        """Record that we've covered this context"""
        self.context_coverage[context] += 1


class AdaptiveExplorationManager:
    """
    Main manager that coordinates all exploration strategies.

    Key insight: Different situations require different exploration strategies.
    We intelligently combine multiple approaches.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Initialize exploration components
        self.curiosity = CuriosityMechanism()
        self.epsilon_greedy = AdaptiveEpsilonGreedy()
        self.thompson_sampling = IntelligentThompsonSampling()
        self.coverage_optimizer = CoverageOptimizer()

        # Configuration
        self.primary_algorithm = config.get("algorithm", "thompson")
        self.exploration_budget = config.get("exploration_budget", 0.1)  # 10% of decisions
        self.safety_threshold = config.get(
            "safety_threshold", 0.95
        )  # Never go below 95% of baseline

        # Performance tracking
        self.total_decisions = 0
        self.exploration_decisions = 0
        self.performance_history = []

    def select_strategy(
        self,
        context: str,
        strategy_stats: Dict[str, Any],
        strategies: List[str],
        file_path: str = "",
        baseline_time: float = 1.0,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Intelligently select optimization strategy with adaptive exploration.

        Returns: (strategy, metadata)
        """
        self.total_decisions += 1

        # Calculate curiosity score
        curiosity_score = self.curiosity.calculate_curiosity_score(context, strategy_stats)

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(strategy_stats)

        # Update context importance
        self.coverage_optimizer.update_context_importance(context, file_path, baseline_time)

        # Calculate coverage boost
        coverage_boost = self.coverage_optimizer.calculate_coverage_boost(context)

        # Decide exploration strategy
        if self.primary_algorithm == "thompson":
            strategy, is_exploration = self.thompson_sampling.sample_strategy(
                context, strategy_stats, strategies, curiosity_score + coverage_boost
            )
        elif self.primary_algorithm == "epsilon_greedy":
            should_explore = self.epsilon_greedy.should_explore(
                context, overall_confidence, curiosity_score + coverage_boost
            )
            if should_explore:
                strategy = self._select_exploration_strategy(context, strategy_stats, strategies)
                is_exploration = True
            else:
                strategy = self._select_best_known_strategy(strategy_stats, strategies)
                is_exploration = False
        else:
            # Fallback to safe strategy
            strategy = self._select_safe_strategy(strategies)
            is_exploration = False

        # Safety check: avoid catastrophic strategies
        if self._is_catastrophic_risk(strategy, strategy_stats):
            strategy = self._select_safe_strategy(strategies)
            is_exploration = False

        # Update tracking
        if is_exploration:
            self.exploration_decisions += 1

        self.coverage_optimizer.record_coverage(context)

        # Prepare metadata
        metadata = {
            "is_exploration": is_exploration,
            "curiosity_score": curiosity_score,
            "coverage_boost": coverage_boost,
            "overall_confidence": overall_confidence,
            "exploration_rate": self.exploration_decisions / self.total_decisions,
            "algorithm": self.primary_algorithm,
            "safety_check": strategy != self._get_original_selection(strategy_stats, strategies),
        }

        logger.info(
            f"Selected {strategy} for {context} "
            f"(exploration: {is_exploration}, curiosity: {curiosity_score:.2f})"
        )

        return strategy, metadata

    def record_exploration_result(
        self, context: str, strategy: str, speedup: float, was_exploration: bool
    ):
        """Record result of exploration for learning"""
        if was_exploration:
            # Update exploration statistics
            if speedup > 1.05:  # Successful exploration
                self.epsilon_greedy.exploration_stats.successful_explorations += 1
                self.thompson_sampling.exploration_stats.successful_explorations += 1

                # Check if this is a novel discovery
                if speedup > 1.5:  # Significant improvement
                    self.epsilon_greedy.exploration_stats.novel_discoveries += 1

            # Track exploration regret
            regret = max(0, 1.0 - speedup)  # Regret relative to baseline
            self.epsilon_greedy.exploration_stats.exploration_regret += regret
            self.thompson_sampling.exploration_stats.exploration_regret += regret

        # Track overall performance
        self.performance_history.append(speedup)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def _calculate_overall_confidence(self, strategy_stats: Dict[str, Any]) -> float:
        """Calculate overall confidence in our knowledge of this context"""
        if not strategy_stats:
            return 0.0

        confidences = []
        for stats in strategy_stats.values():
            if hasattr(stats, "ci_confidence"):
                confidences.append(stats.ci_confidence)
            elif stats.pulls > 0:
                # Simple confidence based on sample size
                confidence = min(1.0, stats.pulls / 30.0)
                confidences.append(confidence)

        return np.mean(confidences) if confidences else 0.0

    def _select_exploration_strategy(
        self, context: str, strategy_stats: Dict[str, Any], strategies: List[str]
    ) -> str:
        """Select strategy for exploration"""
        # Prefer strategies with little data
        under_explored = [
            s for s in strategies if (s not in strategy_stats or strategy_stats[s].pulls < 5)
        ]

        if under_explored:
            return np.random.choice(under_explored)

        # Otherwise, random exploration
        return np.random.choice(strategies)

    def _select_best_known_strategy(
        self, strategy_stats: Dict[str, Any], strategies: List[str]
    ) -> str:
        """Select best known strategy for exploitation"""
        best_strategy = None
        best_performance = 0.0

        for strategy in strategies:
            if strategy in strategy_stats:
                stats = strategy_stats[strategy]
                if stats.pulls > 0 and stats.mean_speedup > best_performance:
                    best_performance = stats.mean_speedup
                    best_strategy = strategy

        return best_strategy or "no_optimization"

    def _select_safe_strategy(self, strategies: List[str]) -> str:
        """Select safest strategy"""
        safe_strategies = ["no_optimization", "conservative", "selective_top5"]
        for strategy in safe_strategies:
            if strategy in strategies:
                return strategy
        return strategies[0]  # Fallback

    def _is_catastrophic_risk(self, strategy: str, strategy_stats: Dict[str, Any]) -> bool:
        """Check if strategy poses catastrophic risk"""
        if strategy not in strategy_stats:
            return False

        stats = strategy_stats[strategy]
        if stats.pulls < 3:  # Not enough data
            return False

        # Check if strategy consistently performs very poorly
        recent_speedups = stats.speedups[-5:] if len(stats.speedups) >= 5 else stats.speedups
        avg_recent = np.mean(recent_speedups)

        return avg_recent < self.safety_threshold

    def _get_original_selection(self, strategy_stats: Dict[str, Any], strategies: List[str]) -> str:
        """Get what would have been selected without safety checks"""
        return self._select_best_known_strategy(strategy_stats, strategies)

    def get_exploration_report(self) -> Dict[str, Any]:
        """Generate comprehensive exploration report"""
        epsilon_stats = self.epsilon_greedy.exploration_stats
        thompson_stats = self.thompson_sampling.exploration_stats

        return {
            "total_decisions": self.total_decisions,
            "exploration_decisions": self.exploration_decisions,
            "exploration_rate": self.exploration_decisions / max(1, self.total_decisions),
            "epsilon_greedy_stats": {
                "total_explorations": epsilon_stats.total_explorations,
                "successful_explorations": epsilon_stats.successful_explorations,
                "success_rate": epsilon_stats.exploration_success_rate,
                "average_regret": epsilon_stats.average_exploration_regret,
                "novel_discoveries": epsilon_stats.novel_discoveries,
            },
            "thompson_stats": {
                "total_explorations": thompson_stats.total_explorations,
                "successful_explorations": thompson_stats.successful_explorations,
                "success_rate": thompson_stats.exploration_success_rate,
                "average_regret": thompson_stats.average_exploration_regret,
            },
            "context_coverage": dict(self.coverage_optimizer.context_coverage),
            "context_importance": dict(self.coverage_optimizer.context_importance),
            "performance_trend": {
                "recent_average": (
                    np.mean(self.performance_history[-100:]) if self.performance_history else 1.0
                ),
                "overall_average": (
                    np.mean(self.performance_history) if self.performance_history else 1.0
                ),
                "improvement_trend": self._calculate_trend(),
            },
        }

    def _calculate_trend(self) -> str:
        """Calculate performance trend"""
        if len(self.performance_history) < 50:
            return "insufficient_data"

        early = np.mean(self.performance_history[:25])
        recent = np.mean(self.performance_history[-25:])

        if recent > early * 1.05:
            return "improving"
        elif recent < early * 0.95:
            return "declining"
        else:
            return "stable"

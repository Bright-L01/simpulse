"""
Advanced Hybrid Strategies for Mixed Contexts

Critical Target: 45% of files are mixed contexts with only 15% success rate
Goal: Achieve 40% success rate on mixed contexts → 18% contribution to overall 50%

Advanced Strategies:
1. Conditional Strategies based on file characteristics
2. Meta-Strategies for high-level decision making
3. Portfolio-based weighted combinations
4. Adaptive exploration for understudied patterns
5. Risk-aware ensemble methods
"""

import logging
from typing import Any, Dict, List

import numpy as np

from .hybrid_strategy_system import (
    BaseStrategy,
    ContextFeatures,
    OptimizationResult,
    StrategyPerformance,
)


class ConditionalHybridStrategy(BaseStrategy):
    """Different strategies based on file size, complexity, and pattern distribution"""

    def __init__(self, optimizers: Dict[str, Any]):
        self.optimizers = optimizers

        # Define decision thresholds
        self.size_thresholds = {
            "small": 1000,  # < 1000 lines
            "medium": 5000,  # 1000-5000 lines
            "large": 10000,  # 5000-10000 lines
        }

        self.complexity_thresholds = {
            "simple": 0.3,  # Low complexity
            "moderate": 0.6,  # Medium complexity
            "complex": 1.0,  # High complexity
        }

    def optimize(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Apply conditional optimization based on file characteristics"""

        # Determine file size category
        if context.line_count < self.size_thresholds["small"]:
            size_category = "small"
        elif context.line_count < self.size_thresholds["medium"]:
            size_category = "medium"
        else:
            size_category = "large"

        # Determine complexity category
        if context.complexity_score < self.complexity_thresholds["simple"]:
            complexity_category = "simple"
        elif context.complexity_score < self.complexity_thresholds["moderate"]:
            complexity_category = "moderate"
        else:
            complexity_category = "complex"

        # Select strategy based on conditions
        strategy_choice = self._select_conditional_strategy(
            size_category, complexity_category, context
        )

        logging.info(
            f"Conditional strategy: {strategy_choice} (size={size_category}, complexity={complexity_category})"
        )

        # Apply selected strategy
        if strategy_choice == "aggressive_arithmetic":
            result = self.optimizers["arithmetic"].optimize(file_path)
        elif strategy_choice == "balanced_algebraic":
            result = self.optimizers["algebraic"].optimize(file_path)
        elif strategy_choice == "conservative_structural":
            result = self.optimizers["structural"].optimize(file_path)
        elif strategy_choice == "weighted_blend":
            result = self._apply_weighted_blend(file_path, context)
        elif strategy_choice == "pattern_focused":
            result = self._apply_pattern_focused(file_path, context)
        else:
            # Default fallback
            result = self.optimizers["structural"].optimize(file_path)

        result.optimization_type = f"conditional_{strategy_choice}"
        return result

    def _select_conditional_strategy(
        self, size_category: str, complexity_category: str, context: ContextFeatures
    ) -> str:
        """Select strategy based on conditional rules"""

        # Small files: More aggressive
        if size_category == "small":
            if context.arithmetic_ratio > 0.7:
                return "aggressive_arithmetic"
            elif context.algebraic_ratio > 0.5:
                return "balanced_algebraic"
            else:
                return "weighted_blend"

        # Medium files: Balanced approach
        elif size_category == "medium":
            if complexity_category == "simple" and context.arithmetic_ratio > 0.5:
                return "balanced_algebraic"  # Moderate arithmetic
            elif context.mixed_context:
                return "weighted_blend"
            else:
                return "pattern_focused"

        # Large files: Conservative
        else:  # size_category == 'large'
            if complexity_category == "complex":
                return "conservative_structural"
            elif context.mixed_context:
                return "pattern_focused"  # Focus on dominant patterns
            else:
                return "conservative_structural"

    def _apply_weighted_blend(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Apply sophisticated weighted blending"""

        # Get results from all optimizers
        arithmetic_result = self.optimizers["arithmetic"].optimize(file_path)
        algebraic_result = self.optimizers["algebraic"].optimize(file_path)
        structural_result = self.optimizers["structural"].optimize(file_path)

        # Calculate adaptive weights based on context and confidence
        base_weights = [context.arithmetic_ratio, context.algebraic_ratio, context.structural_ratio]
        confidence_weights = [
            arithmetic_result.confidence_score,
            algebraic_result.confidence_score,
            structural_result.confidence_score,
        ]

        # Combine base weights with confidence weights
        combined_weights = []
        for i in range(3):
            combined_weight = 0.7 * base_weights[i] + 0.3 * confidence_weights[i]
            combined_weights.append(combined_weight)

        # Normalize weights
        total_weight = sum(combined_weights)
        if total_weight > 0:
            weights = [w / total_weight for w in combined_weights]
        else:
            weights = [1 / 3, 1 / 3, 1 / 3]

        # Apply weighted combination
        return self._combine_results_with_weights(
            [arithmetic_result, algebraic_result, structural_result], weights
        )

    def _apply_pattern_focused(
        self, file_path: str, context: ContextFeatures
    ) -> OptimizationResult:
        """Focus on the dominant pattern type"""

        # Find dominant pattern
        pattern_ratios = [
            context.arithmetic_ratio,
            context.algebraic_ratio,
            context.structural_ratio,
        ]
        dominant_idx = np.argmax(pattern_ratios)
        dominant_ratio = pattern_ratios[dominant_idx]

        # If no clear dominant pattern, use weighted blend
        if dominant_ratio < 0.4:
            return self._apply_weighted_blend(file_path, context)

        # Apply dominant pattern strategy with boosted confidence
        if dominant_idx == 0:  # Arithmetic dominant
            result = self.optimizers["arithmetic"].optimize(file_path)
        elif dominant_idx == 1:  # Algebraic dominant
            result = self.optimizers["algebraic"].optimize(file_path)
        else:  # Structural dominant
            result = self.optimizers["structural"].optimize(file_path)

        # Boost confidence for dominant patterns
        result.confidence_score = min(result.confidence_score * (1 + dominant_ratio), 1.0)
        return result

    def _combine_results_with_weights(
        self, results: List[OptimizationResult], weights: List[float]
    ) -> OptimizationResult:
        """Combine multiple optimization results with given weights"""

        combined_lemmas = []
        all_lemmas = set()

        # Collect all unique lemmas
        for result in results:
            all_lemmas.update(result.modified_lemmas)

        # Apply weighted combination
        for lemma in all_lemmas:
            weighted_priority = 0
            weight_sum = 0

            for i, result in enumerate(results):
                lemma_in_result = next(
                    (l for l in result.modified_lemmas if l.name == lemma.name), None
                )
                if lemma_in_result:
                    weighted_priority += weights[i] * lemma_in_result.priority
                    weight_sum += weights[i]
                else:
                    # Use original priority if not modified by this optimizer
                    weighted_priority += weights[i] * lemma.priority
                    weight_sum += weights[i]

            if weight_sum > 0:
                final_priority = weighted_priority / weight_sum
            else:
                final_priority = lemma.priority

            new_lemma = lemma._replace(priority=int(final_priority))
            combined_lemmas.append(new_lemma)

        # Combined confidence is weighted average
        combined_confidence = sum(
            weights[i] * results[i].confidence_score for i in range(len(results))
        )

        return OptimizationResult(
            modified_lemmas=combined_lemmas,
            optimization_type="weighted_combination",
            confidence_score=combined_confidence,
        )

    def get_name(self) -> str:
        return "conditional_hybrid"

    def get_risk_level(self) -> float:
        return 0.5  # Adaptive risk level


class MetaStrategy(BaseStrategy):
    """High-level meta-strategy that chooses which approach to use"""

    def __init__(
        self, optimizers: Dict[str, Any], performance_history: Dict[str, StrategyPerformance]
    ):
        self.optimizers = optimizers
        self.performance_history = performance_history

        # Available sub-strategies
        self.sub_strategies = {
            "pure_best": self._pure_best_strategy,
            "hybrid_weighted": self._hybrid_weighted_strategy,
            "exploration": self._exploration_strategy,
            "conservative": self._conservative_strategy,
            "aggressive": self._aggressive_strategy,
        }

    def optimize(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Meta-level strategy selection"""

        # Analyze context to determine meta-strategy
        meta_choice = self._select_meta_strategy(context)

        logging.info(f"Meta-strategy selected: {meta_choice}")

        # Apply chosen meta-strategy
        strategy_function = self.sub_strategies.get(meta_choice, self._conservative_strategy)
        result = strategy_function(file_path, context)

        result.optimization_type = f"meta_{meta_choice}"
        return result

    def _select_meta_strategy(self, context: ContextFeatures) -> str:
        """Select high-level meta-strategy"""

        # If we have good historical data for this context type
        if self._has_good_historical_data(context):
            if context.risk_tolerance > 0.7:
                return "pure_best"
            else:
                return "hybrid_weighted"

        # If context is well-understood (not mixed)
        elif not context.mixed_context:
            return "pure_best"

        # If user wants conservative approach
        elif context.risk_tolerance < 0.3:
            return "conservative"

        # If user wants aggressive approach
        elif context.risk_tolerance > 0.8:
            return "aggressive"

        # If we lack data, explore
        elif self._should_explore(context):
            return "exploration"

        # Default: hybrid approach
        else:
            return "hybrid_weighted"

    def _has_good_historical_data(self, context: ContextFeatures) -> bool:
        """Check if we have sufficient historical data for this context"""

        # Look for similar contexts in performance history
        similar_contexts = 0
        for key, perf in self.performance_history.items():
            if perf.total_attempts > 5:  # Minimum data threshold
                similar_contexts += 1

        return similar_contexts >= 3

    def _should_explore(self, context: ContextFeatures) -> bool:
        """Determine if we should explore new strategies"""

        # Explore if:
        # 1. We have limited data on mixed contexts
        # 2. Recent performance has been poor
        # 3. Context complexity suggests novel patterns

        if context.mixed_context and context.complexity_score > 0.7:
            return True

        # Check recent performance
        recent_success_rate = context.previous_success_rate
        if recent_success_rate < 0.2:  # Poor recent performance
            return True

        return False

    def _pure_best_strategy(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Choose the single best pure strategy"""

        # Determine best strategy based on context
        if context.arithmetic_ratio > max(context.algebraic_ratio, context.structural_ratio):
            return self.optimizers["arithmetic"].optimize(file_path)
        elif context.algebraic_ratio > context.structural_ratio:
            return self.optimizers["algebraic"].optimize(file_path)
        else:
            return self.optimizers["structural"].optimize(file_path)

    def _hybrid_weighted_strategy(
        self, file_path: str, context: ContextFeatures
    ) -> OptimizationResult:
        """Use sophisticated weighted combination"""

        # Similar to ConditionalHybridStrategy._apply_weighted_blend
        arithmetic_result = self.optimizers["arithmetic"].optimize(file_path)
        algebraic_result = self.optimizers["algebraic"].optimize(file_path)
        structural_result = self.optimizers["structural"].optimize(file_path)

        # Use pattern ratios as base weights
        weights = [context.arithmetic_ratio, context.algebraic_ratio, context.structural_ratio]

        # Normalize
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1 / 3, 1 / 3, 1 / 3]

        return self._combine_results(
            [arithmetic_result, algebraic_result, structural_result], weights
        )

    def _exploration_strategy(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Exploration strategy for learning"""

        # Try multiple strategies and combine with exploration weights
        arithmetic_result = self.optimizers["arithmetic"].optimize(file_path)
        algebraic_result = self.optimizers["algebraic"].optimize(file_path)
        structural_result = self.optimizers["structural"].optimize(file_path)

        # Use exploration weights (more equal distribution)
        exploration_weights = [0.4, 0.35, 0.25]  # Slight bias toward arithmetic

        result = self._combine_results(
            [arithmetic_result, algebraic_result, structural_result], exploration_weights
        )

        # Mark as exploration for learning purposes
        result.optimization_type = "exploration"
        return result

    def _conservative_strategy(
        self, file_path: str, context: ContextFeatures
    ) -> OptimizationResult:
        """Conservative strategy emphasizing safety"""

        # Use structural optimizer as base
        result = self.optimizers["structural"].optimize(file_path)

        # Apply conservative modifications
        conservative_lemmas = []
        for lemma in result.modified_lemmas:
            # Reduce priority changes by 50%
            original_priority = lemma.priority
            # Assuming original priority was some baseline, apply conservative adjustment
            conservative_priority = int(original_priority * 0.75)  # Less aggressive
            conservative_lemma = lemma._replace(priority=conservative_priority)
            conservative_lemmas.append(conservative_lemma)

        return OptimizationResult(
            modified_lemmas=conservative_lemmas,
            optimization_type="conservative",
            confidence_score=result.confidence_score * 0.9,  # Slightly less confident
        )

    def _aggressive_strategy(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Aggressive strategy for maximum optimization"""

        # Use arithmetic optimizer as base (most aggressive)
        result = self.optimizers["arithmetic"].optimize(file_path)

        # Apply aggressive modifications
        aggressive_lemmas = []
        for lemma in result.modified_lemmas:
            # Increase priority changes by 25%
            original_priority = lemma.priority
            aggressive_priority = int(original_priority * 1.25)  # More aggressive
            aggressive_lemma = lemma._replace(priority=aggressive_priority)
            aggressive_lemmas.append(aggressive_lemma)

        return OptimizationResult(
            modified_lemmas=aggressive_lemmas,
            optimization_type="aggressive",
            confidence_score=result.confidence_score * 0.8,  # Less confident due to risk
        )

    def _combine_results(
        self, results: List[OptimizationResult], weights: List[float]
    ) -> OptimizationResult:
        """Combine results with given weights"""

        combined_lemmas = []
        all_lemmas = set()

        # Collect all unique lemmas
        for result in results:
            all_lemmas.update(result.modified_lemmas)

        # Apply weighted combination
        for lemma in all_lemmas:
            weighted_priority = 0
            weight_sum = 0

            for i, result in enumerate(results):
                lemma_in_result = next(
                    (l for l in result.modified_lemmas if l.name == lemma.name), lemma
                )
                weighted_priority += weights[i] * lemma_in_result.priority
                weight_sum += weights[i]

            final_priority = weighted_priority / weight_sum if weight_sum > 0 else lemma.priority
            new_lemma = lemma._replace(priority=int(final_priority))
            combined_lemmas.append(new_lemma)

        # Combined confidence
        combined_confidence = sum(
            weights[i] * results[i].confidence_score for i in range(len(results))
        )

        return OptimizationResult(
            modified_lemmas=combined_lemmas,
            optimization_type="meta_combination",
            confidence_score=combined_confidence,
        )

    def get_name(self) -> str:
        return "meta_strategy"

    def get_risk_level(self) -> float:
        return 0.5  # Variable risk based on meta-decision


class PortfolioOptimizationStrategy(BaseStrategy):
    """Portfolio-based strategy using modern portfolio theory principles"""

    def __init__(
        self, optimizers: Dict[str, Any], performance_history: Dict[str, StrategyPerformance]
    ):
        self.optimizers = optimizers
        self.performance_history = performance_history

        # Strategy portfolio with expected returns and risks
        self.strategy_portfolio = {
            "arithmetic": {"expected_return": 0.7, "risk": 0.3},
            "algebraic": {"expected_return": 0.5, "risk": 0.2},
            "structural": {"expected_return": 0.3, "risk": 0.1},
        }

    def optimize(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Apply portfolio optimization principles to strategy selection"""

        # Calculate optimal portfolio weights using Markowitz-style optimization
        optimal_weights = self._calculate_optimal_weights(context)

        # Get results from all strategies
        arithmetic_result = self.optimizers["arithmetic"].optimize(file_path)
        algebraic_result = self.optimizers["algebraic"].optimize(file_path)
        structural_result = self.optimizers["structural"].optimize(file_path)

        # Apply portfolio combination
        result = self._portfolio_combination(
            [arithmetic_result, algebraic_result, structural_result], optimal_weights
        )

        result.optimization_type = (
            f"portfolio_{optimal_weights[0]:.2f}_{optimal_weights[1]:.2f}_{optimal_weights[2]:.2f}"
        )
        return result

    def _calculate_optimal_weights(self, context: ContextFeatures) -> List[float]:
        """Calculate optimal portfolio weights"""

        # Get expected returns and risks
        strategies = ["arithmetic", "algebraic", "structural"]
        expected_returns = []
        risks = []

        for strategy in strategies:
            portfolio_data = self.strategy_portfolio[strategy]

            # Adjust expected return based on context
            base_return = portfolio_data["expected_return"]
            context_adjustment = self._get_context_adjustment(strategy, context)
            adjusted_return = base_return + context_adjustment

            expected_returns.append(adjusted_return)
            risks.append(portfolio_data["risk"])

        # Simple portfolio optimization (equal risk contribution)
        risk_weights = [1 / r for r in risks]
        total_risk_weight = sum(risk_weights)
        normalized_weights = [w / total_risk_weight for w in risk_weights]

        # Adjust based on user risk tolerance
        if context.risk_tolerance > 0.7:
            # Increase weight on higher-return strategies
            for i, ret in enumerate(expected_returns):
                if ret > 0.5:
                    normalized_weights[i] *= 1.2
        elif context.risk_tolerance < 0.3:
            # Increase weight on lower-risk strategies
            for i, risk in enumerate(risks):
                if risk < 0.2:
                    normalized_weights[i] *= 1.3

        # Renormalize
        total_weight = sum(normalized_weights)
        final_weights = [w / total_weight for w in normalized_weights]

        return final_weights

    def _get_context_adjustment(self, strategy: str, context: ContextFeatures) -> float:
        """Adjust expected return based on context"""

        adjustments = {
            "arithmetic": context.arithmetic_ratio * 0.2,
            "algebraic": context.algebraic_ratio * 0.15,
            "structural": context.structural_ratio * 0.1,
        }

        return adjustments.get(strategy, 0.0)

    def _portfolio_combination(
        self, results: List[OptimizationResult], weights: List[float]
    ) -> OptimizationResult:
        """Combine results using portfolio weights"""

        combined_lemmas = []
        all_lemmas = set()

        # Collect all unique lemmas
        for result in results:
            all_lemmas.update(result.modified_lemmas)

        # Apply portfolio weights
        for lemma in all_lemmas:
            weighted_priority = 0

            for i, result in enumerate(results):
                lemma_in_result = next(
                    (l for l in result.modified_lemmas if l.name == lemma.name), lemma
                )
                weighted_priority += weights[i] * lemma_in_result.priority

            new_lemma = lemma._replace(priority=int(weighted_priority))
            combined_lemmas.append(new_lemma)

        # Portfolio confidence is weighted average
        portfolio_confidence = sum(
            weights[i] * results[i].confidence_score for i in range(len(results))
        )

        return OptimizationResult(
            modified_lemmas=combined_lemmas,
            optimization_type="portfolio_optimization",
            confidence_score=portfolio_confidence,
        )

    def get_name(self) -> str:
        return "portfolio_optimization"

    def get_risk_level(self) -> float:
        return 0.4  # Balanced portfolio approach


class AdaptiveExplorationStrategy(BaseStrategy):
    """Adaptive exploration strategy for understudied contexts"""

    def __init__(self, optimizers: Dict[str, Any], exploration_budget: float = 0.1):
        self.optimizers = optimizers
        self.exploration_budget = exploration_budget  # Fraction of time to explore
        self.exploration_count = 0
        self.total_count = 0

    def optimize(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Apply adaptive exploration"""

        self.total_count += 1
        exploration_rate = self.exploration_count / max(self.total_count, 1)

        # Decide whether to explore or exploit
        should_explore = (
            exploration_rate < self.exploration_budget  # Under exploration budget
            or context.complexity_score > 0.8  # Novel/complex context
            or context.previous_success_rate < 0.2  # Poor historical performance
        )

        if should_explore:
            self.exploration_count += 1
            return self._explore(file_path, context)
        else:
            return self._exploit(file_path, context)

    def _explore(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Exploration phase: try novel combinations"""

        # Try experimental strategy combinations
        experimental_strategies = [
            self._experimental_aggressive,
            self._experimental_conservative,
            self._experimental_hybrid,
        ]

        # Select exploration strategy based on context novelty
        if context.complexity_score > 0.9:
            strategy_func = experimental_strategies[0]  # Most aggressive
        elif context.mixed_context and context.arithmetic_ratio < 0.2:
            strategy_func = experimental_strategies[1]  # Conservative for unusual patterns
        else:
            strategy_func = experimental_strategies[2]  # Hybrid exploration

        result = strategy_func(file_path, context)
        result.optimization_type = f"exploration_{result.optimization_type}"

        logging.info(f"Exploration strategy applied: {result.optimization_type}")
        return result

    def _exploit(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Exploitation phase: use known good strategies"""

        # Use best known strategy for this context type
        if context.arithmetic_ratio > 0.6:
            return self.optimizers["arithmetic"].optimize(file_path)
        elif context.algebraic_ratio > 0.4:
            return self.optimizers["algebraic"].optimize(file_path)
        else:
            return self.optimizers["structural"].optimize(file_path)

    def _experimental_aggressive(
        self, file_path: str, context: ContextFeatures
    ) -> OptimizationResult:
        """Experimental aggressive strategy"""
        result = self.optimizers["arithmetic"].optimize(file_path)

        # Boost all priorities by experimental factor
        boosted_lemmas = []
        for lemma in result.modified_lemmas:
            boosted_priority = int(lemma.priority * 1.5)  # 50% boost
            boosted_lemma = lemma._replace(priority=boosted_priority)
            boosted_lemmas.append(boosted_lemma)

        return OptimizationResult(
            modified_lemmas=boosted_lemmas,
            optimization_type="experimental_aggressive",
            confidence_score=result.confidence_score * 0.7,  # Lower confidence for experimental
        )

    def _experimental_conservative(
        self, file_path: str, context: ContextFeatures
    ) -> OptimizationResult:
        """Experimental conservative strategy"""
        result = self.optimizers["structural"].optimize(file_path)

        # Apply minimal changes
        minimal_lemmas = []
        for lemma in result.modified_lemmas:
            minimal_priority = int(lemma.priority * 0.8)  # 20% reduction
            minimal_lemma = lemma._replace(priority=minimal_priority)
            minimal_lemmas.append(minimal_lemma)

        return OptimizationResult(
            modified_lemmas=minimal_lemmas,
            optimization_type="experimental_conservative",
            confidence_score=result.confidence_score * 0.9,
        )

    def _experimental_hybrid(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Experimental hybrid strategy"""

        # Try novel weighting scheme
        arithmetic_result = self.optimizers["arithmetic"].optimize(file_path)
        algebraic_result = self.optimizers["algebraic"].optimize(file_path)
        structural_result = self.optimizers["structural"].optimize(file_path)

        # Experimental weights based on complexity
        if context.complexity_score > 0.7:
            weights = [0.2, 0.3, 0.5]  # Favor structural for complex
        else:
            weights = [0.5, 0.3, 0.2]  # Favor arithmetic for simple

        return self._combine_with_weights(
            [arithmetic_result, algebraic_result, structural_result], weights, "experimental_hybrid"
        )

    def _combine_with_weights(
        self, results: List[OptimizationResult], weights: List[float], optimization_type: str
    ) -> OptimizationResult:
        """Combine results with weights"""

        combined_lemmas = []
        all_lemmas = set()

        for result in results:
            all_lemmas.update(result.modified_lemmas)

        for lemma in all_lemmas:
            weighted_priority = 0
            for i, result in enumerate(results):
                lemma_in_result = next(
                    (l for l in result.modified_lemmas if l.name == lemma.name), lemma
                )
                weighted_priority += weights[i] * lemma_in_result.priority

            new_lemma = lemma._replace(priority=int(weighted_priority))
            combined_lemmas.append(new_lemma)

        combined_confidence = sum(
            weights[i] * results[i].confidence_score for i in range(len(results))
        )

        return OptimizationResult(
            modified_lemmas=combined_lemmas,
            optimization_type=optimization_type,
            confidence_score=combined_confidence,
        )

    def get_name(self) -> str:
        return "adaptive_exploration"

    def get_risk_level(self) -> float:
        current_exploration_rate = self.exploration_count / max(self.total_count, 1)
        return 0.3 + 0.4 * current_exploration_rate  # Variable risk based on exploration

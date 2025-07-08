"""
Hybrid Strategy System: Context-Aware Optimization with Multi-Armed Bandits

Mathematical Foundation:
- Current: 30% success despite suboptimal context matching
- Target: 49.75% ≈ 50% with intelligent context-strategy mapping
- Key insight: Accept lower per-context rates for higher overall rate

Architecture:
1. Context-Strategy Mapping with Contextual Bandits
2. Hybrid Strategies for Mixed Contexts (45% of files!)
3. Phase-Based Optimization (conservative → adaptive)
4. Fallback Chains for safety guarantees
5. Meta-Strategies that choose strategies
6. Continuous learning with empirical feedback
"""

import json
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Import our existing optimizers
from .specialized_optimizers import (
    AlgebraicOptimizer,
    ArithmeticOptimizer,
    OptimizationResult,
    StructuralOptimizer,
)


@dataclass
class ContextFeatures:
    """Context features for contextual bandit decision making"""

    file_size: int
    line_count: int
    arithmetic_ratio: float  # 0-1
    algebraic_ratio: float  # 0-1
    structural_ratio: float  # 0-1
    complexity_score: float  # Derived metric
    mixed_context: bool  # True if multiple patterns present

    # Historical features
    previous_success_rate: float  # For this file type
    average_speedup: float  # Historical average
    risk_tolerance: float  # 0-1, higher = more aggressive

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for LinUCB"""
        return np.array(
            [
                self.file_size / 10000,  # Normalize
                self.line_count / 1000,
                self.arithmetic_ratio,
                self.algebraic_ratio,
                self.structural_ratio,
                self.complexity_score,
                float(self.mixed_context),
                self.previous_success_rate,
                self.average_speedup,
                self.risk_tolerance,
            ]
        )


@dataclass
class StrategyPerformance:
    """Track performance of strategies"""

    strategy_name: str
    context_type: str
    success_count: int = 0
    failure_count: int = 0
    total_speedup: float = 0.0
    total_attempts: int = 0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.5  # Neutral prior
        return self.success_count / self.total_attempts

    @property
    def average_speedup(self) -> float:
        if self.success_count == 0:
            return 1.0
        return self.total_speedup / self.success_count

    def update(self, success: bool, speedup: float = 1.0):
        """Update performance metrics"""
        self.total_attempts += 1
        if success:
            self.success_count += 1
            self.total_speedup += speedup
        else:
            self.failure_count += 1

        # Update confidence interval (Wilson score interval)
        self._update_confidence_interval()

    def _update_confidence_interval(self):
        """Calculate Wilson score confidence interval"""
        if self.total_attempts < 5:
            self.confidence_interval = (0.0, 1.0)
            return

        n = self.total_attempts
        p = self.success_rate
        z = 1.96  # 95% confidence

        denominator = 1 + z**2 / n
        centre_adjusted_probability = p + z**2 / (2 * n)
        adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)

        lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
        upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator

        self.confidence_interval = (max(0, lower_bound), min(1, upper_bound))


class StrategyType(Enum):
    """Available strategy types"""

    ARITHMETIC_PURE = "arithmetic_pure"
    ALGEBRAIC_PURE = "algebraic_pure"
    STRUCTURAL_PURE = "structural_pure"

    # Hybrid strategies for mixed contexts
    WEIGHTED_HYBRID = "weighted_hybrid"
    PHASE_BASED = "phase_based"
    CONDITIONAL = "conditional"
    FALLBACK_CHAIN = "fallback_chain"

    # Meta-strategies
    CONSERVATIVE_META = "conservative_meta"
    AGGRESSIVE_META = "aggressive_meta"
    ADAPTIVE_META = "adaptive_meta"


class BaseStrategy(ABC):
    """Base class for all optimization strategies"""

    @abstractmethod
    def optimize(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Apply optimization strategy"""

    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name"""

    @abstractmethod
    def get_risk_level(self) -> float:
        """Get risk level (0=conservative, 1=aggressive)"""


class PureStrategy(BaseStrategy):
    """Wrapper for existing pure optimizers"""

    def __init__(self, optimizer, strategy_type: StrategyType):
        self.optimizer = optimizer
        self.strategy_type = strategy_type

    def optimize(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        return self.optimizer.optimize(file_path)

    def get_name(self) -> str:
        return self.strategy_type.value

    def get_risk_level(self) -> float:
        risk_levels = {
            StrategyType.ARITHMETIC_PURE: 0.7,  # Moderately aggressive
            StrategyType.ALGEBRAIC_PURE: 0.5,  # Balanced
            StrategyType.STRUCTURAL_PURE: 0.2,  # Conservative
        }
        return risk_levels.get(self.strategy_type, 0.5)


class WeightedHybridStrategy(BaseStrategy):
    """Dynamically weighted combination of optimizers"""

    def __init__(self, optimizers: Dict[str, Any]):
        self.arithmetic_optimizer = optimizers["arithmetic"]
        self.algebraic_optimizer = optimizers["algebraic"]
        self.structural_optimizer = optimizers["structural"]

    def optimize(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Apply weighted combination based on context ratios"""

        # Determine weights from context ratios
        total_ratio = context.arithmetic_ratio + context.algebraic_ratio + context.structural_ratio
        if total_ratio == 0:
            # Equal weights if no clear pattern
            weights = [0.33, 0.33, 0.34]
        else:
            weights = [
                context.arithmetic_ratio / total_ratio,
                context.algebraic_ratio / total_ratio,
                context.structural_ratio / total_ratio,
            ]

        # Get results from each optimizer
        arithmetic_result = self.arithmetic_optimizer.optimize(file_path)
        algebraic_result = self.algebraic_optimizer.optimize(file_path)
        structural_result = self.structural_optimizer.optimize(file_path)

        # Weighted combination of lemma priorities
        combined_lemmas = []
        all_lemmas = set()

        # Collect all unique lemmas
        for result in [arithmetic_result, algebraic_result, structural_result]:
            all_lemmas.update(result.modified_lemmas)

        # Create weighted priority for each lemma
        for lemma in all_lemmas:
            arithmetic_priority = next(
                (l.priority for l in arithmetic_result.modified_lemmas if l.name == lemma.name),
                lemma.priority,
            )
            algebraic_priority = next(
                (l.priority for l in algebraic_result.modified_lemmas if l.name == lemma.name),
                lemma.priority,
            )
            structural_priority = next(
                (l.priority for l in structural_result.modified_lemmas if l.name == lemma.name),
                lemma.priority,
            )

            weighted_priority = (
                weights[0] * arithmetic_priority
                + weights[1] * algebraic_priority
                + weights[2] * structural_priority
            )

            new_lemma = lemma._replace(priority=int(weighted_priority))
            combined_lemmas.append(new_lemma)

        return OptimizationResult(
            modified_lemmas=combined_lemmas,
            optimization_type=f"weighted_hybrid_{weights[0]:.2f}_{weights[1]:.2f}_{weights[2]:.2f}",
            confidence_score=min(
                arithmetic_result.confidence_score,
                algebraic_result.confidence_score,
                structural_result.confidence_score,
            )
            * 0.9,  # Slight penalty for complexity
        )

    def get_name(self) -> str:
        return "weighted_hybrid"

    def get_risk_level(self) -> float:
        return 0.4  # Moderate risk due to combination


class PhaseBasedStrategy(BaseStrategy):
    """Start conservative, escalate based on early success"""

    def __init__(self, optimizers: Dict[str, Any]):
        self.conservative_optimizer = optimizers["structural"]
        self.moderate_optimizer = optimizers["algebraic"]
        self.aggressive_optimizer = optimizers["arithmetic"]

    def optimize(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Apply phase-based optimization"""

        # Phase 1: Conservative (always try first)
        conservative_result = self.conservative_optimizer.optimize(file_path)

        # If confidence is high and context suggests safety, stay conservative
        if conservative_result.confidence_score > 0.8 and context.risk_tolerance < 0.3:
            return conservative_result

        # Phase 2: Moderate (if context suggests it might work)
        if context.algebraic_ratio > 0.3 or context.mixed_context:
            moderate_result = self.moderate_optimizer.optimize(file_path)

            # If moderate has significantly better potential, use it
            if moderate_result.confidence_score > conservative_result.confidence_score + 0.1:
                # But still err conservative if context is risky
                if context.complexity_score > 0.7:
                    return conservative_result
                return moderate_result

        # Phase 3: Aggressive (only if arithmetic-heavy and high risk tolerance)
        if context.arithmetic_ratio > 0.6 and context.risk_tolerance > 0.7:
            aggressive_result = self.aggressive_optimizer.optimize(file_path)

            # Only use if significantly better and context supports it
            if (
                aggressive_result.confidence_score > 0.7
                and context.file_size < 5000  # Smaller files are safer
                and not context.mixed_context
            ):  # Avoid mixed contexts
                return aggressive_result

        # Default: Return the best of conservative/moderate
        return conservative_result

    def get_name(self) -> str:
        return "phase_based"

    def get_risk_level(self) -> float:
        return 0.3  # Conservative overall approach


class FallbackChainStrategy(BaseStrategy):
    """Try strategies in sequence until one succeeds or all fail"""

    def __init__(self, optimizers: Dict[str, Any]):
        self.strategy_chain = [
            optimizers["arithmetic"],  # Most aggressive first
            optimizers["algebraic"],  # Moderate
            optimizers["structural"],  # Conservative fallback
        ]

    def optimize(self, file_path: str, context: ContextFeatures) -> OptimizationResult:
        """Apply fallback chain optimization"""

        results = []

        for optimizer in self.strategy_chain:
            result = optimizer.optimize(file_path)
            results.append(result)

            # If confidence is high enough, use this result
            if result.confidence_score > 0.7:
                result.optimization_type = f"fallback_chain_success_{len(results)}"
                return result

            # For conservative strategy, use lower threshold
            if optimizer == self.strategy_chain[-1] and result.confidence_score > 0.5:
                result.optimization_type = f"fallback_chain_final"
                return result

        # If all failed, return the most conservative
        best_result = max(results, key=lambda r: r.confidence_score)
        best_result.optimization_type = "fallback_chain_best_effort"
        return best_result

    def get_name(self) -> str:
        return "fallback_chain"

    def get_risk_level(self) -> float:
        return 0.6  # Starts aggressive but has safety nets


class LinUCBContextualBandit:
    """LinUCB algorithm for contextual bandit optimization"""

    def __init__(self, feature_dim: int, alpha: float = 1.0):
        self.feature_dim = feature_dim
        self.alpha = alpha  # Exploration parameter

        # Initialize for each strategy
        self.strategies = {}
        self.reset_strategy_params()

    def reset_strategy_params(self):
        """Reset parameters for all strategies"""
        strategy_types = list(StrategyType)

        for strategy_type in strategy_types:
            self.strategies[strategy_type.value] = {
                "A": np.eye(self.feature_dim),  # A_a in LinUCB
                "b": np.zeros(self.feature_dim),  # b_a in LinUCB
                "rewards": [],
                "contexts": [],
            }

    def select_strategy(self, context: ContextFeatures, available_strategies: List[str]) -> str:
        """Select best strategy using LinUCB"""

        x = context.to_vector()
        best_strategy = None
        best_ucb = -np.inf

        for strategy_name in available_strategies:
            if strategy_name not in self.strategies:
                continue

            params = self.strategies[strategy_name]
            A_inv = np.linalg.inv(params["A"])
            theta = A_inv @ params["b"]

            # Calculate confidence interval
            confidence_radius = self.alpha * np.sqrt(x.T @ A_inv @ x)

            # Upper confidence bound
            ucb_value = theta.T @ x + confidence_radius

            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_strategy = strategy_name

        return best_strategy or available_strategies[0]  # Fallback

    def update(self, strategy_name: str, context: ContextFeatures, reward: float):
        """Update strategy parameters with observed reward"""

        if strategy_name not in self.strategies:
            return

        x = context.to_vector()
        params = self.strategies[strategy_name]

        # Update LinUCB parameters
        params["A"] += np.outer(x, x)
        params["b"] += reward * x
        params["rewards"].append(reward)
        params["contexts"].append(x)

    def get_strategy_stats(self) -> Dict[str, Dict]:
        """Get performance statistics for all strategies"""
        stats = {}

        for strategy_name, params in self.strategies.items():
            if len(params["rewards"]) > 0:
                stats[strategy_name] = {
                    "avg_reward": np.mean(params["rewards"]),
                    "total_trials": len(params["rewards"]),
                    "recent_reward": (
                        np.mean(params["rewards"][-10:])
                        if len(params["rewards"]) >= 10
                        else np.mean(params["rewards"])
                    ),
                }
            else:
                stats[strategy_name] = {"avg_reward": 0.0, "total_trials": 0, "recent_reward": 0.0}

        return stats


class HybridStrategySystem:
    """Main system orchestrating hybrid optimization strategies"""

    def __init__(self, db_path: str = "hybrid_strategy_system.db"):
        self.db_path = db_path
        self.setup_database()

        # Initialize optimizers
        self.optimizers = {
            "arithmetic": ArithmeticOptimizer(),
            "algebraic": AlgebraicOptimizer(),
            "structural": StructuralOptimizer(),
        }

        # Initialize strategies
        self.strategies: Dict[str, BaseStrategy] = {
            StrategyType.ARITHMETIC_PURE.value: PureStrategy(
                self.optimizers["arithmetic"], StrategyType.ARITHMETIC_PURE
            ),
            StrategyType.ALGEBRAIC_PURE.value: PureStrategy(
                self.optimizers["algebraic"], StrategyType.ALGEBRAIC_PURE
            ),
            StrategyType.STRUCTURAL_PURE.value: PureStrategy(
                self.optimizers["structural"], StrategyType.STRUCTURAL_PURE
            ),
            StrategyType.WEIGHTED_HYBRID.value: WeightedHybridStrategy(self.optimizers),
            StrategyType.PHASE_BASED.value: PhaseBasedStrategy(self.optimizers),
            StrategyType.FALLBACK_CHAIN.value: FallbackChainStrategy(self.optimizers),
        }

        # Initialize contextual bandit
        self.bandit = LinUCBContextualBandit(feature_dim=10, alpha=0.1)

        # Performance tracking
        self.performance_tracker: Dict[str, StrategyPerformance] = {}

        # Load historical data
        self.load_performance_data()

        logging.info(f"HybridStrategySystem initialized with {len(self.strategies)} strategies")

    def setup_database(self):
        """Setup SQLite database for tracking performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                file_path TEXT,
                strategy_name TEXT,
                context_type TEXT,
                success BOOLEAN,
                speedup REAL,
                context_features TEXT,
                optimization_result TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bandit_state (
                strategy_name TEXT PRIMARY KEY,
                matrix_A TEXT,
                vector_b TEXT,
                rewards TEXT,
                contexts TEXT
            )
        """
        )

        conn.commit()
        conn.close()

    def extract_context_features(self, file_path: str) -> ContextFeatures:
        """Extract context features from file"""
        # This would integrate with existing pattern detection
        # For now, create reasonable defaults

        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 1000
        line_count = 100  # Default

        # In real implementation, use existing pattern analyzers
        return ContextFeatures(
            file_size=file_size,
            line_count=line_count,
            arithmetic_ratio=0.4,  # Would be computed
            algebraic_ratio=0.3,  # Would be computed
            structural_ratio=0.3,  # Would be computed
            complexity_score=0.5,  # Would be computed
            mixed_context=True,  # Most files are mixed
            previous_success_rate=0.5,  # Historical data
            average_speedup=1.2,  # Historical data
            risk_tolerance=0.5,  # User setting
        )

    def optimize_with_context_awareness(
        self, file_path: str, user_risk_tolerance: float = 0.5
    ) -> Tuple[OptimizationResult, str]:
        """Main optimization method with context awareness"""

        # Extract context features
        context = self.extract_context_features(file_path)
        context.risk_tolerance = user_risk_tolerance

        # Determine available strategies based on context
        available_strategies = self._get_available_strategies(context)

        # Use contextual bandit to select strategy
        selected_strategy_name = self.bandit.select_strategy(context, available_strategies)

        # Apply selected strategy
        strategy = self.strategies[selected_strategy_name]
        result = strategy.optimize(file_path, context)

        # Log the selection
        logging.info(f"Selected strategy: {selected_strategy_name} for {file_path}")
        logging.info(
            f"Strategy risk level: {strategy.get_risk_level():.2f}, Context risk tolerance: {context.risk_tolerance:.2f}"
        )

        return result, selected_strategy_name

    def _get_available_strategies(self, context: ContextFeatures) -> List[str]:
        """Determine which strategies are available for this context"""
        available = []

        # Pure strategies always available
        available.extend(
            [
                StrategyType.ARITHMETIC_PURE.value,
                StrategyType.ALGEBRAIC_PURE.value,
                StrategyType.STRUCTURAL_PURE.value,
            ]
        )

        # Hybrid strategies for mixed contexts
        if context.mixed_context:
            available.extend(
                [
                    StrategyType.WEIGHTED_HYBRID.value,
                    StrategyType.PHASE_BASED.value,
                    StrategyType.FALLBACK_CHAIN.value,
                ]
            )

        # Filter by risk tolerance
        filtered = []
        for strategy_name in available:
            if strategy_name in self.strategies:
                strategy_risk = self.strategies[strategy_name].get_risk_level()
                if strategy_risk <= context.risk_tolerance + 0.2:  # Allow some tolerance
                    filtered.append(strategy_name)

        return filtered if filtered else available  # Fallback to all if none match

    def record_performance(
        self,
        file_path: str,
        strategy_name: str,
        context: ContextFeatures,
        result: OptimizationResult,
        success: bool,
        speedup: float,
    ):
        """Record performance for learning"""

        # Calculate reward (success + speedup bonus)
        reward = float(success)
        if success and speedup > 1.0:
            reward += min((speedup - 1.0) * 0.5, 1.0)  # Bonus for speedup, capped at 1.0

        # Update contextual bandit
        self.bandit.update(strategy_name, context, reward)

        # Update performance tracker
        key = f"{strategy_name}_{context.mixed_context}"
        if key not in self.performance_tracker:
            self.performance_tracker[key] = StrategyPerformance(
                strategy_name, "mixed" if context.mixed_context else "pure"
            )

        self.performance_tracker[key].update(success, speedup)

        # Save to database
        self._save_performance_to_db(file_path, strategy_name, context, result, success, speedup)

        logging.info(
            f"Recorded performance: {strategy_name} -> success={success}, speedup={speedup:.2f}x"
        )

    def _save_performance_to_db(
        self,
        file_path: str,
        strategy_name: str,
        context: ContextFeatures,
        result: OptimizationResult,
        success: bool,
        speedup: float,
    ):
        """Save performance data to database"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO strategy_performance 
            (timestamp, file_path, strategy_name, context_type, success, speedup, context_features, optimization_result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                time.time(),
                file_path,
                strategy_name,
                "mixed" if context.mixed_context else "pure",
                success,
                speedup,
                json.dumps(asdict(context)),
                json.dumps(asdict(result)) if hasattr(result, "__dict__") else str(result),
            ),
        )

        conn.commit()
        conn.close()

    def load_performance_data(self):
        """Load historical performance data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT strategy_name, context_type, success, speedup 
                FROM strategy_performance 
                ORDER BY timestamp DESC 
                LIMIT 1000
            """
            )

            for row in cursor.fetchall():
                strategy_name, context_type, success, speedup = row
                key = f"{strategy_name}_{context_type}"

                if key not in self.performance_tracker:
                    self.performance_tracker[key] = StrategyPerformance(strategy_name, context_type)

                self.performance_tracker[key].update(bool(success), float(speedup))

            conn.close()
            logging.info(
                f"Loaded performance data for {len(self.performance_tracker)} strategy-context combinations"
            )

        except Exception as e:
            logging.warning(f"Could not load performance data: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""

        summary = {
            "strategy_performance": {},
            "bandit_stats": self.bandit.get_strategy_stats(),
            "total_optimizations": sum(p.total_attempts for p in self.performance_tracker.values()),
            "overall_success_rate": 0.0,
        }

        total_successes = 0
        total_attempts = 0

        for key, perf in self.performance_tracker.items():
            summary["strategy_performance"][key] = {
                "success_rate": perf.success_rate,
                "average_speedup": perf.average_speedup,
                "total_attempts": perf.total_attempts,
                "confidence_interval": perf.confidence_interval,
            }

            total_successes += perf.success_count
            total_attempts += perf.total_attempts

        if total_attempts > 0:
            summary["overall_success_rate"] = total_successes / total_attempts

        return summary

    def predict_success_rate_for_target(self) -> Dict[str, float]:
        """Predict success rates for reaching 50% target"""

        # Based on mathematical model:
        # Arithmetic: 30% files × 85% success = 25.5%
        # Mixed: 45% files × 40% success = 18%
        # Complex: 25% files × 25% success = 6.25%
        # Total: 49.75% ≈ 50%

        current_performance = self.get_performance_summary()

        predictions = {
            "current_overall": current_performance["overall_success_rate"],
            "target_arithmetic": 0.85,
            "target_mixed": 0.40,
            "target_complex": 0.25,
            "projected_overall": 0.30 * 0.85 + 0.45 * 0.40 + 0.25 * 0.25,
            "improvement_needed": {
                "arithmetic": 0.85
                - current_performance.get("strategy_performance", {})
                .get("arithmetic_pure_pure", {})
                .get("success_rate", 0.3),
                "mixed": 0.40
                - current_performance.get("strategy_performance", {})
                .get("weighted_hybrid_mixed", {})
                .get("success_rate", 0.15),
                "complex": 0.25
                - current_performance.get("strategy_performance", {})
                .get("structural_pure_pure", {})
                .get("success_rate", 0.05),
            },
        }

        return predictions


if __name__ == "__main__":
    # Example usage
    system = HybridStrategySystem()

    # Test optimization
    test_file = "examples/test_mixed.lean"
    result, strategy_used = system.optimize_with_context_awareness(
        test_file, user_risk_tolerance=0.6
    )

    print(f"Selected strategy: {strategy_used}")
    print(f"Optimization result: {result.optimization_type}")
    print(f"Confidence: {result.confidence_score:.3f}")

    # Simulate feedback
    system.record_performance(
        test_file,
        strategy_used,
        system.extract_context_features(test_file),
        result,
        success=True,
        speedup=1.75,
    )

    # Show performance summary
    summary = system.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"Overall success rate: {summary['overall_success_rate']:.1%}")

    # Show predictions for 50% target
    predictions = system.predict_success_rate_for_target()
    print(f"\nProjected overall success rate: {predictions['projected_overall']:.1%}")

#!/usr/bin/env python3
"""
BanditOptimizer: Reinforcement Learning for Compilation Optimization

Applies multi-armed bandit algorithms to learn optimal optimization strategies
through online learning, not prediction.

Key innovations:
1. Each (context, strategy) pair is a bandit arm
2. Thompson Sampling for Bayesian exploration
3. UCB for optimistic exploration
4. Regret minimization through adaptive learning
"""

import json
import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ArmStatistics:
    """Statistics for a single bandit arm (context-strategy pair)"""

    context: str
    strategy: str
    pulls: int = 0
    successes: int = 0
    total_reward: float = 0.0
    rewards: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Empirical success rate"""
        return self.successes / self.pulls if self.pulls > 0 else 0.5

    @property
    def mean_reward(self) -> float:
        """Average reward (speedup)"""
        return self.total_reward / self.pulls if self.pulls > 0 else 1.0

    @property
    def std_reward(self) -> float:
        """Standard deviation of rewards"""
        if len(self.rewards) < 2:
            return 0.1  # Prior uncertainty
        return np.std(self.rewards)

    def update(self, reward: float, success: bool):
        """Update arm statistics with new observation"""
        self.pulls += 1
        self.total_reward += reward
        self.rewards.append(reward)
        if success:
            self.successes += 1


class BanditAlgorithm(ABC):
    """Abstract base class for bandit algorithms"""

    @abstractmethod
    def select_arm(self, arms: Dict[Tuple[str, str], ArmStatistics], context: str) -> str:
        """Select which strategy to use for given context"""

    @abstractmethod
    def update(self, arm_key: Tuple[str, str], reward: float, success: bool):
        """Update algorithm state after observing reward"""


class ThompsonSampling(BanditAlgorithm):
    """
    Thompson Sampling: Bayesian approach to exploration/exploitation

    Maintains Beta distribution for success probability of each arm.
    Samples from posterior and selects arm with highest sample.
    """

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        # Beta parameters for each arm
        self.alphas = defaultdict(lambda: alpha_prior)
        self.betas = defaultdict(lambda: beta_prior)

    def select_arm(self, arms: Dict[Tuple[str, str], ArmStatistics], context: str) -> str:
        """Select strategy using Thompson Sampling"""
        context_arms = {
            strategy: arms[(context, strategy)]
            for strategy in {key[1] for key in arms.keys() if key[0] == context}
        }

        if not context_arms:
            return "no_optimization"

        # Sample from Beta distribution for each arm
        samples = {}
        for strategy, arm_stats in context_arms.items():
            arm_key = (context, strategy)
            # Sample expected speedup from Beta distribution
            success_prob = np.random.beta(self.alphas[arm_key], self.betas[arm_key])
            # Expected reward = success_prob * mean_speedup_on_success
            expected_reward = success_prob * (arm_stats.mean_reward if arm_stats.pulls > 0 else 1.2)
            samples[strategy] = expected_reward

        # Select arm with highest sample
        return max(samples.items(), key=lambda x: x[1])[0]

    def update(self, arm_key: Tuple[str, str], reward: float, success: bool):
        """Update Beta distribution parameters"""
        if success:
            self.alphas[arm_key] += 1
        else:
            self.betas[arm_key] += 1


class UCB(BanditAlgorithm):
    """
    Upper Confidence Bound: Optimism in the face of uncertainty

    Selects arm with highest upper confidence bound on expected reward.
    Balances exploitation (mean reward) with exploration (uncertainty).
    """

    def __init__(self, confidence_level: float = 2.0):
        self.confidence_level = confidence_level
        self.total_pulls = 0

    def select_arm(self, arms: Dict[Tuple[str, str], ArmStatistics], context: str) -> str:
        """Select strategy using UCB algorithm"""
        context_arms = {
            strategy: arms[(context, strategy)]
            for strategy in {key[1] for key in arms.keys() if key[0] == context}
        }

        if not context_arms:
            return "no_optimization"

        # Calculate UCB for each arm
        ucb_scores = {}
        for strategy, arm_stats in context_arms.items():
            if arm_stats.pulls == 0:
                # Unexplored arm gets maximum priority
                ucb_scores[strategy] = float("inf")
            else:
                # UCB = mean + confidence * sqrt(2 * log(total) / pulls)
                mean_reward = arm_stats.mean_reward
                exploration_bonus = self.confidence_level * math.sqrt(
                    2 * math.log(max(1, self.total_pulls)) / arm_stats.pulls
                )
                ucb_scores[strategy] = mean_reward + exploration_bonus

        # Select arm with highest UCB
        return max(ucb_scores.items(), key=lambda x: x[1])[0]

    def update(self, arm_key: Tuple[str, str], reward: float, success: bool):
        """Update total pull count"""
        self.total_pulls += 1


class EpsilonGreedy(BanditAlgorithm):
    """
    Epsilon-Greedy: Simple exploration/exploitation tradeoff

    With probability epsilon, explore random arm.
    Otherwise, exploit best known arm.
    """

    def __init__(self, epsilon: float = 0.1, decay_rate: float = 0.995):
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.updates = 0

    def select_arm(self, arms: Dict[Tuple[str, str], ArmStatistics], context: str) -> str:
        """Select strategy using epsilon-greedy"""
        context_arms = {
            strategy: arms[(context, strategy)]
            for strategy in {key[1] for key in arms.keys() if key[0] == context}
        }

        if not context_arms:
            return "no_optimization"

        # Explore with probability epsilon
        if np.random.random() < self.epsilon:
            return np.random.choice(list(context_arms.keys()))

        # Exploit: choose best known arm
        best_strategy = max(context_arms.items(), key=lambda x: x[1].mean_reward)[0]

        return best_strategy

    def update(self, arm_key: Tuple[str, str], reward: float, success: bool):
        """Update epsilon with decay"""
        self.updates += 1
        self.epsilon = self.initial_epsilon * (self.decay_rate**self.updates)


class BanditOptimizer:
    """
    Main optimizer using multi-armed bandit algorithms for online learning.

    Key features:
    1. Maintains statistics for each (context, strategy) pair
    2. Uses bandit algorithms for exploration/exploitation
    3. Tracks regret and convergence metrics
    4. Persists learning across sessions
    """

    def __init__(
        self,
        algorithm: str = "thompson",
        confidence_level: float = 2.0,
        state_file: Optional[Path] = None,
    ):

        # Initialize bandit algorithm
        if algorithm == "thompson":
            self.algorithm = ThompsonSampling()
        elif algorithm == "ucb":
            self.algorithm = UCB(confidence_level)
        elif algorithm == "epsilon_greedy":
            self.algorithm = EpsilonGreedy()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.algorithm_name = algorithm

        # Arm statistics
        self.arms = {}

        # Available strategies
        self.strategies = [
            "no_optimization",
            "conservative",
            "moderate",
            "aggressive",
            "selective_top5",
            "contextual_arithmetic",
            "inverse_reduction",
            "adaptive_threshold",
        ]

        # Regret tracking
        self.regret_history = []
        self.optimal_arms = {}  # Best known arm per context

        # State persistence
        self.state_file = state_file
        if state_file and state_file.exists():
            self.load_state()

    def recommend_strategy(self, context: str) -> Tuple[str, Dict[str, Any]]:
        """
        Recommend optimization strategy for given context.

        Returns:
            strategy: Recommended strategy name
            metadata: Additional information (confidence, exploration factor, etc.)
        """
        # Initialize arms for new context if needed
        self._ensure_arms_initialized(context)

        # Select arm using bandit algorithm
        strategy = self.algorithm.select_arm(self.arms, context)

        # Calculate metadata
        arm_key = (context, strategy)
        arm_stats = self.arms[arm_key]

        # Confidence based on number of pulls
        if arm_stats.pulls == 0:
            confidence = "low"
        elif arm_stats.pulls < 10:
            confidence = "medium"
        else:
            confidence = "high"

        # Exploration vs exploitation indicator
        is_exploration = self._is_exploration(context, strategy)

        metadata = {
            "confidence": confidence,
            "pulls": arm_stats.pulls,
            "mean_reward": arm_stats.mean_reward,
            "success_rate": arm_stats.success_rate,
            "is_exploration": is_exploration,
            "algorithm": self.algorithm_name,
        }

        return strategy, metadata

    def update_reward(self, context: str, strategy: str, speedup: float, compilation_success: bool):
        """
        Update bandit with observed reward.

        Args:
            context: Context type
            strategy: Strategy that was used
            speedup: Observed speedup (1.0 = no change)
            compilation_success: Whether compilation succeeded
        """
        # Reward is speedup if successful, small penalty if failed
        reward = speedup if compilation_success else 0.8
        success = speedup > 1.05 and compilation_success

        # Update arm statistics
        arm_key = (context, strategy)
        if arm_key not in self.arms:
            self.arms[arm_key] = ArmStatistics(context, strategy)

        self.arms[arm_key].update(reward, success)

        # Update algorithm
        self.algorithm.update(arm_key, reward, success)

        # Update optimal arm tracking
        self._update_optimal_arm(context)

        # Calculate and store regret
        regret = self._calculate_regret(context, strategy, reward)
        self.regret_history.append(regret)

        # Persist state
        if self.state_file:
            self.save_state()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about bandit performance"""
        stats = {
            "total_pulls": sum(arm.pulls for arm in self.arms.values()),
            "contexts_seen": len({key[0] for key in self.arms.keys()}),
            "strategies_used": len({key[1] for key in self.arms.keys()}),
            "algorithm": self.algorithm_name,
            "cumulative_regret": sum(self.regret_history),
            "average_regret": np.mean(self.regret_history) if self.regret_history else 0,
        }

        # Per-context statistics
        context_stats = defaultdict(dict)
        for (context, strategy), arm in self.arms.items():
            if arm.pulls > 0:
                context_stats[context][strategy] = {
                    "pulls": arm.pulls,
                    "mean_reward": arm.mean_reward,
                    "success_rate": arm.success_rate,
                    "std_reward": arm.std_reward,
                }

        stats["context_statistics"] = dict(context_stats)

        # Convergence analysis
        if len(self.regret_history) > 100:
            recent_regret = np.mean(self.regret_history[-100:])
            early_regret = np.mean(self.regret_history[:100])
            stats["convergence_ratio"] = recent_regret / early_regret if early_regret > 0 else 1.0
            stats["is_converging"] = recent_regret < early_regret * 0.5

        return stats

    def _ensure_arms_initialized(self, context: str):
        """Initialize arms for a new context"""
        for strategy in self.strategies:
            arm_key = (context, strategy)
            if arm_key not in self.arms:
                self.arms[arm_key] = ArmStatistics(context, strategy)

    def _is_exploration(self, context: str, strategy: str) -> bool:
        """Determine if this selection is exploration vs exploitation"""
        # Get best known arm for context
        context_arms = {
            s: self.arms[(context, s)] for s in self.strategies if (context, s) in self.arms
        }

        if not context_arms:
            return True

        best_strategy = max(context_arms.items(), key=lambda x: x[1].mean_reward)[0]

        return strategy != best_strategy

    def _update_optimal_arm(self, context: str):
        """Update tracking of optimal arm for regret calculation"""
        context_arms = {
            strategy: self.arms[(context, strategy)]
            for strategy in self.strategies
            if (context, strategy) in self.arms and self.arms[(context, strategy)].pulls > 0
        }

        if context_arms:
            best_strategy = max(context_arms.items(), key=lambda x: x[1].mean_reward)[0]
            self.optimal_arms[context] = (best_strategy, context_arms[best_strategy].mean_reward)

    def _calculate_regret(self, context: str, strategy: str, reward: float) -> float:
        """Calculate instantaneous regret"""
        if context in self.optimal_arms:
            optimal_strategy, optimal_reward = self.optimal_arms[context]
            return max(0, optimal_reward - reward)
        return 0.0

    def save_state(self):
        """Persist bandit state to disk"""
        if not self.state_file:
            return

        state = {
            "algorithm": self.algorithm_name,
            "arms": {
                f"{context}|{strategy}": {
                    "pulls": arm.pulls,
                    "successes": arm.successes,
                    "total_reward": arm.total_reward,
                    "rewards": arm.rewards[-100:],  # Keep last 100 for efficiency
                }
                for (context, strategy), arm in self.arms.items()
            },
            "regret_history": self.regret_history[-1000:],  # Keep last 1000
            "optimal_arms": self.optimal_arms,
        }

        # Algorithm-specific state
        if isinstance(self.algorithm, ThompsonSampling):
            state["thompson_params"] = {
                f"{c}|{s}": {
                    "alpha": self.algorithm.alphas[(c, s)],
                    "beta": self.algorithm.betas[(c, s)],
                }
                for (c, s) in self.algorithm.alphas.keys()
            }

        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load bandit state from disk"""
        if not self.state_file or not self.state_file.exists():
            return

        with open(self.state_file) as f:
            state = json.load(f)

        # Restore arms
        for key, arm_data in state.get("arms", {}).items():
            context, strategy = key.split("|")
            arm = ArmStatistics(context, strategy)
            arm.pulls = arm_data["pulls"]
            arm.successes = arm_data["successes"]
            arm.total_reward = arm_data["total_reward"]
            arm.rewards = arm_data["rewards"]
            self.arms[(context, strategy)] = arm

        # Restore other state
        self.regret_history = state.get("regret_history", [])
        self.optimal_arms = state.get("optimal_arms", {})

        # Restore algorithm-specific state
        if isinstance(self.algorithm, ThompsonSampling) and "thompson_params" in state:
            for key, params in state["thompson_params"].items():
                context, strategy = key.split("|")
                self.algorithm.alphas[(context, strategy)] = params["alpha"]
                self.algorithm.betas[(context, strategy)] = params["beta"]


def demo_bandit_optimizer():
    """Demonstrate the bandit optimizer in action"""
    print("🎰 BANDIT OPTIMIZER DEMO")
    print("=" * 60)
    print("Online learning for compilation optimization")
    print()

    # Create optimizer
    optimizer = BanditOptimizer(algorithm="thompson")

    # Simulate optimization decisions
    contexts = ["arithmetic_uniform", "mixed_patterns", "pure_identity"]

    print("📊 SIMULATING 100 COMPILATION CYCLES")
    print("-" * 40)

    for i in range(100):
        # Random context
        context = np.random.choice(contexts)

        # Get recommendation
        strategy, metadata = optimizer.recommend_strategy(context)

        # Simulate reward (in reality, this would be actual compilation speedup)
        if context == "arithmetic_uniform" and strategy == "contextual_arithmetic":
            speedup = np.random.normal(2.1, 0.2)
        elif context == "pure_identity" and strategy == "aggressive":
            speedup = np.random.normal(1.8, 0.15)
        elif strategy == "no_optimization":
            speedup = 1.0
        else:
            speedup = np.random.normal(1.1, 0.3)

        speedup = max(0.5, speedup)  # Floor at 0.5x
        success = speedup > 0.9  # Compilation succeeds if not too slow

        # Update optimizer
        optimizer.update_reward(context, strategy, speedup, success)

        # Print periodic updates
        if (i + 1) % 20 == 0:
            print(f"\nAfter {i + 1} iterations:")
            print(f"  Context: {context}")
            print(f"  Recommended: {strategy}")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Exploration: {metadata['is_exploration']}")

    print("\n📈 FINAL STATISTICS")
    print("-" * 40)

    stats = optimizer.get_statistics()
    print(f"Total pulls: {stats['total_pulls']}")
    print(f"Average regret: {stats['average_regret']:.3f}")
    print(f"Convergence ratio: {stats.get('convergence_ratio', 'N/A')}")

    print("\n🎯 LEARNED OPTIMAL STRATEGIES")
    print("-" * 40)

    for context in contexts:
        context_stats = stats["context_statistics"].get(context, {})
        if context_stats:
            best_strategy = max(context_stats.items(), key=lambda x: x[1]["mean_reward"])[0]

            print(f"\n{context}:")
            print(f"  Best strategy: {best_strategy}")
            print(f"  Mean speedup: {context_stats[best_strategy]['mean_reward']:.2f}x")
            print(f"  Pulls: {context_stats[best_strategy]['pulls']}")

            # Show exploration vs exploitation
            total_pulls = sum(s["pulls"] for s in context_stats.values())
            exploit_ratio = context_stats[best_strategy]["pulls"] / total_pulls
            print(f"  Exploitation ratio: {exploit_ratio:.1%}")

    print("\n✅ The bandit optimizer learns optimal strategies through experience!")
    print("No predictions needed - just online learning and adaptation.")


if __name__ == "__main__":
    demo_bandit_optimizer()

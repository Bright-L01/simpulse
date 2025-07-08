#!/usr/bin/env python3
"""
Real-time Optimization Learner

Every compilation is a learning opportunity. The system continuously improves
through multi-armed bandit algorithms, tracking regret and building confidence.

Key principle: The system gets smarter with EVERY use!
"""

import json
import logging
import math
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..analysis.advanced_context_classifier import AdvancedContextClassifier

logger = logging.getLogger(__name__)


@dataclass
class CompilationEvent:
    """Single compilation event with full tracking"""

    timestamp: float
    file_path: str
    context_type: str
    strategy: str
    baseline_time: float
    optimized_time: float
    speedup: float
    success: bool
    regret: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyStats:
    """Real-time statistics for a strategy"""

    pulls: int = 0
    successes: int = 0
    total_speedup: float = 0.0
    speedups: List[float] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    # Confidence interval tracking
    ci_lower: float = 1.0
    ci_upper: float = 1.0
    ci_confidence: float = 0.0

    @property
    def mean_speedup(self) -> float:
        return self.total_speedup / self.pulls if self.pulls > 0 else 1.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.pulls if self.pulls > 0 else 0.5

    def update_confidence_interval(self, confidence_level: float = 0.95):
        """Update confidence interval using t-distribution"""
        if len(self.speedups) < 2:
            self.ci_confidence = 0.0
            return

        from scipy import stats

        mean = np.mean(self.speedups)
        sem = stats.sem(self.speedups)
        df = len(self.speedups) - 1

        ci = stats.t.interval(confidence_level, df, loc=mean, scale=sem)
        self.ci_lower = max(0.1, ci[0])  # Floor at 0.1x
        self.ci_upper = ci[1]
        self.ci_confidence = min(1.0, len(self.speedups) / 30.0)  # Full confidence at 30 samples


class RealtimeOptimizationLearner:
    """
    Real-time learning system that improves with every compilation.

    Features:
    - Multi-armed bandit learning (Thompson Sampling, UCB)
    - Persistent state across sessions
    - Regret tracking and minimization
    - Confidence interval calculation
    - Automatic exploration/exploitation balance
    """

    def __init__(
        self,
        db_path: Path = Path("optimization_history.db"),
        algorithm: str = "thompson",
        exploration_rate: float = 0.1,
    ):

        self.db_path = db_path
        self.algorithm = algorithm
        self.exploration_rate = exploration_rate

        # Context classifier
        self.classifier = AdvancedContextClassifier()

        # In-memory statistics cache
        self.stats: Dict[Tuple[str, str], StrategyStats] = defaultdict(StrategyStats)

        # Thompson Sampling parameters
        self.thompson_alpha = defaultdict(lambda: 1.0)  # Success prior
        self.thompson_beta = defaultdict(lambda: 1.0)  # Failure prior

        # UCB tracking
        self.total_pulls = 0

        # Regret tracking
        self.cumulative_regret = 0.0
        self.regret_history: List[float] = []
        self.optimal_strategies: Dict[str, Tuple[str, float]] = {}

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

        # Initialize database
        self._init_database()

        # Load historical data
        self._load_state()

    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Compilation events table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS compilation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                file_path TEXT NOT NULL,
                context_type TEXT NOT NULL,
                strategy TEXT NOT NULL,
                baseline_time REAL NOT NULL,
                optimized_time REAL NOT NULL,
                speedup REAL NOT NULL,
                success INTEGER NOT NULL,
                regret REAL DEFAULT 0.0,
                metadata TEXT
            )
        """
        )

        # Strategy statistics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_stats (
                context_type TEXT NOT NULL,
                strategy TEXT NOT NULL,
                pulls INTEGER DEFAULT 0,
                successes INTEGER DEFAULT 0,
                total_speedup REAL DEFAULT 0.0,
                mean_speedup REAL DEFAULT 1.0,
                ci_lower REAL DEFAULT 1.0,
                ci_upper REAL DEFAULT 1.0,
                ci_confidence REAL DEFAULT 0.0,
                last_updated REAL,
                PRIMARY KEY (context_type, strategy)
            )
        """
        )

        # Indexes for performance
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_context 
            ON compilation_events(context_type)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_timestamp 
            ON compilation_events(timestamp)
        """
        )

        conn.commit()
        conn.close()

    def _load_state(self):
        """Load historical data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load strategy statistics
        cursor.execute(
            """
            SELECT context_type, strategy, pulls, successes, total_speedup,
                   ci_lower, ci_upper, ci_confidence
            FROM strategy_stats
        """
        )

        for row in cursor.fetchall():
            context, strategy = row[0], row[1]
            stats = StrategyStats(
                pulls=row[2],
                successes=row[3],
                total_speedup=row[4],
                ci_lower=row[5],
                ci_upper=row[6],
                ci_confidence=row[7],
            )
            self.stats[(context, strategy)] = stats

            # Update Thompson Sampling parameters
            self.thompson_alpha[(context, strategy)] = stats.successes + 1
            self.thompson_beta[(context, strategy)] = (stats.pulls - stats.successes) + 1

        # Load recent speedup history for confidence intervals
        cursor.execute(
            """
            SELECT context_type, strategy, speedup
            FROM compilation_events
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            LIMIT 1000
        """,
            (time.time() - 86400 * 30,),
        )  # Last 30 days

        for row in cursor.fetchall():
            context, strategy, speedup = row
            if (context, strategy) in self.stats:
                self.stats[(context, strategy)].speedups.append(speedup)

        # Calculate total pulls
        cursor.execute("SELECT COUNT(*) FROM compilation_events")
        self.total_pulls = cursor.fetchone()[0]

        # Load regret history
        cursor.execute(
            """
            SELECT regret FROM compilation_events 
            ORDER BY timestamp DESC LIMIT 1000
        """
        )
        self.regret_history = [row[0] for row in cursor.fetchall()][::-1]
        self.cumulative_regret = sum(self.regret_history)

        conn.close()

        # Update optimal strategies
        self._update_optimal_strategies()

        logger.info(f"Loaded {len(self.stats)} strategy statistics")
        logger.info(f"Total historical compilations: {self.total_pulls}")

    def recommend_strategy(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Recommend optimization strategy for a file.

        Returns:
            strategy: Recommended strategy name
            metadata: Additional information including confidence
        """
        # Classify context
        classification = self.classifier.classify(file_path)
        context = classification.context_type

        # Ensure all strategies are initialized for this context
        for strategy in self.strategies:
            if (context, strategy) not in self.stats:
                self.stats[(context, strategy)] = StrategyStats()

        # Select strategy based on algorithm
        if self.algorithm == "thompson":
            selected_strategy = self._thompson_sampling(context)
        elif self.algorithm == "ucb":
            selected_strategy = self._ucb_selection(context)
        elif self.algorithm == "epsilon_greedy":
            selected_strategy = self._epsilon_greedy(context)
        else:
            selected_strategy = "no_optimization"

        # Get statistics and confidence
        stats = self.stats[(context, selected_strategy)]

        # Determine if this is exploration or exploitation
        is_exploration = self._is_exploration(context, selected_strategy)

        metadata = {
            "context_type": context,
            "confidence": self._calculate_confidence(stats),
            "expected_speedup": stats.mean_speedup,
            "ci_lower": stats.ci_lower,
            "ci_upper": stats.ci_upper,
            "success_rate": stats.success_rate,
            "pulls": stats.pulls,
            "is_exploration": is_exploration,
            "algorithm": self.algorithm,
            "classification_confidence": classification.confidence,
        }

        logger.info(
            f"Recommending {selected_strategy} for {context} "
            f"(confidence: {metadata['confidence']:.2f})"
        )

        return selected_strategy, metadata

    def record_result(
        self,
        file_path: Path,
        context_type: str,
        strategy: str,
        baseline_time: float,
        optimized_time: float,
        compilation_success: bool = True,
    ):
        """
        Record compilation result and update learning.

        This is where the system gets smarter!
        """
        # Calculate speedup
        speedup = baseline_time / optimized_time if optimized_time > 0 else 0.1
        if not compilation_success:
            speedup = 0.8  # Penalty for compilation failure

        # Determine success (>5% improvement and compilation succeeded)
        success = speedup > 1.05 and compilation_success

        # Calculate regret
        regret = self._calculate_regret(context_type, strategy, speedup)

        # Create event
        event = CompilationEvent(
            timestamp=time.time(),
            file_path=str(file_path),
            context_type=context_type,
            strategy=strategy,
            baseline_time=baseline_time,
            optimized_time=optimized_time,
            speedup=speedup,
            success=success,
            regret=regret,
        )

        # Update in-memory statistics
        stats = self.stats[(context_type, strategy)]
        stats.pulls += 1
        stats.total_speedup += speedup
        stats.speedups.append(speedup)
        if success:
            stats.successes += 1
        stats.last_updated = time.time()

        # Keep only recent speedups for memory efficiency
        if len(stats.speedups) > 100:
            stats.speedups = stats.speedups[-100:]

        # Update confidence interval
        stats.update_confidence_interval()

        # Update Thompson Sampling parameters
        if success:
            self.thompson_alpha[(context_type, strategy)] += 1
        else:
            self.thompson_beta[(context_type, strategy)] += 1

        # Update total pulls and regret
        self.total_pulls += 1
        self.cumulative_regret += regret
        self.regret_history.append(regret)

        # Update optimal strategies
        self._update_optimal_strategies()

        # Persist to database
        self._save_event(event)
        self._save_stats(context_type, strategy, stats)

        # Log learning progress
        logger.info(
            f"Recorded: {context_type}/{strategy} → {speedup:.2f}x "
            f"(success: {success}, regret: {regret:.3f})"
        )

        # Periodic analysis
        if self.total_pulls % 100 == 0:
            self._analyze_performance()

    def _thompson_sampling(self, context: str) -> str:
        """Thompson Sampling strategy selection"""
        samples = {}

        for strategy in self.strategies:
            key = (context, strategy)
            # Sample from Beta distribution
            theta = np.random.beta(self.thompson_alpha[key], self.thompson_beta[key])
            # Expected reward = success probability * mean speedup
            stats = self.stats[key]
            expected_reward = theta * (stats.mean_speedup if stats.pulls > 0 else 1.2)
            samples[strategy] = expected_reward

        # Select strategy with highest sample
        return max(samples.items(), key=lambda x: x[1])[0]

    def _ucb_selection(self, context: str) -> str:
        """Upper Confidence Bound strategy selection"""
        ucb_scores = {}

        for strategy in self.strategies:
            stats = self.stats[(context, strategy)]

            if stats.pulls == 0:
                # Unexplored strategy gets maximum priority
                ucb_scores[strategy] = float("inf")
            else:
                # UCB = mean + confidence * sqrt(2 * ln(total) / pulls)
                exploration_bonus = 2.0 * math.sqrt(
                    2 * math.log(max(1, self.total_pulls)) / stats.pulls
                )
                ucb_scores[strategy] = stats.mean_speedup + exploration_bonus

        # Select strategy with highest UCB
        return max(ucb_scores.items(), key=lambda x: x[1])[0]

    def _epsilon_greedy(self, context: str) -> str:
        """Epsilon-greedy strategy selection"""
        if np.random.random() < self.exploration_rate:
            # Explore: random strategy
            return np.random.choice(self.strategies)
        else:
            # Exploit: best known strategy
            best_strategy = None
            best_speedup = 0.0

            for strategy in self.strategies:
                stats = self.stats[(context, strategy)]
                if stats.mean_speedup > best_speedup:
                    best_speedup = stats.mean_speedup
                    best_strategy = strategy

            return best_strategy or "no_optimization"

    def _is_exploration(self, context: str, strategy: str) -> bool:
        """Determine if this selection is exploration"""
        # Find best known strategy for context
        best_strategy = None
        best_speedup = 0.0

        for s in self.strategies:
            stats = self.stats[(context, s)]
            if stats.pulls > 0 and stats.mean_speedup > best_speedup:
                best_speedup = stats.mean_speedup
                best_strategy = s

        return strategy != best_strategy

    def _calculate_confidence(self, stats: StrategyStats) -> float:
        """Calculate confidence level for a strategy"""
        if stats.pulls == 0:
            return 0.0
        elif stats.pulls < 5:
            return 0.3
        elif stats.pulls < 20:
            return 0.6
        else:
            # Asymptotic confidence based on CI width and pulls
            ci_width = stats.ci_upper - stats.ci_lower
            confidence = min(0.95, 0.6 + 0.35 * (1 - ci_width / 2.0))
            return confidence * stats.ci_confidence

    def _calculate_regret(self, context: str, strategy: str, speedup: float) -> float:
        """Calculate instantaneous regret"""
        if context in self.optimal_strategies:
            optimal_strategy, optimal_speedup = self.optimal_strategies[context]
            return max(0, optimal_speedup - speedup)
        return 0.0

    def _update_optimal_strategies(self):
        """Update tracking of optimal strategies per context"""
        contexts = {c for c, _ in self.stats.keys()}

        for context in contexts:
            best_strategy = None
            best_speedup = 0.0

            for strategy in self.strategies:
                stats = self.stats[(context, strategy)]
                if stats.pulls > 0 and stats.mean_speedup > best_speedup:
                    best_speedup = stats.mean_speedup
                    best_strategy = strategy

            if best_strategy:
                self.optimal_strategies[context] = (best_strategy, best_speedup)

    def _save_event(self, event: CompilationEvent):
        """Save compilation event to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO compilation_events 
            (timestamp, file_path, context_type, strategy, baseline_time,
             optimized_time, speedup, success, regret, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                event.timestamp,
                event.file_path,
                event.context_type,
                event.strategy,
                event.baseline_time,
                event.optimized_time,
                event.speedup,
                int(event.success),
                event.regret,
                json.dumps(event.metadata),
            ),
        )

        conn.commit()
        conn.close()

    def _save_stats(self, context: str, strategy: str, stats: StrategyStats):
        """Save strategy statistics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO strategy_stats
            (context_type, strategy, pulls, successes, total_speedup,
             mean_speedup, ci_lower, ci_upper, ci_confidence, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                context,
                strategy,
                stats.pulls,
                stats.successes,
                stats.total_speedup,
                stats.mean_speedup,
                stats.ci_lower,
                stats.ci_upper,
                stats.ci_confidence,
                stats.last_updated,
            ),
        )

        conn.commit()
        conn.close()

    def _analyze_performance(self):
        """Periodic performance analysis"""
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE ANALYSIS")
        logger.info("=" * 60)

        # Overall statistics
        logger.info(f"Total compilations: {self.total_pulls}")
        logger.info(f"Cumulative regret: {self.cumulative_regret:.2f}")
        logger.info(f"Average regret: {self.cumulative_regret/self.total_pulls:.3f}")

        # Per-context optimal strategies
        logger.info("\nOptimal strategies by context:")
        for context, (strategy, speedup) in sorted(self.optimal_strategies.items()):
            stats = self.stats[(context, strategy)]
            logger.info(
                f"  {context}: {strategy} ({speedup:.2f}x, "
                f"{stats.pulls} pulls, {stats.ci_confidence:.1%} confidence)"
            )

        # Convergence analysis
        if len(self.regret_history) > 100:
            recent_regret = np.mean(self.regret_history[-100:])
            early_regret = np.mean(self.regret_history[: min(100, len(self.regret_history))])
            convergence_ratio = recent_regret / early_regret if early_regret > 0 else 1.0
            logger.info(f"\nConvergence ratio: {convergence_ratio:.3f}")
            logger.info(f"Is converging: {convergence_ratio < 0.5}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the learner"""
        return {
            "total_compilations": self.total_pulls,
            "cumulative_regret": self.cumulative_regret,
            "average_regret": (
                self.cumulative_regret / self.total_pulls if self.total_pulls > 0 else 0
            ),
            "contexts_seen": len({c for c, _ in self.stats.keys()}),
            "optimal_strategies": dict(self.optimal_strategies),
            "algorithm": self.algorithm,
            "db_path": str(self.db_path),
        }

    def get_strategy_report(self, context: str) -> Dict[str, Any]:
        """Get detailed report for a specific context"""
        report = {"context": context, "strategies": {}}

        for strategy in self.strategies:
            stats = self.stats[(context, strategy)]
            report["strategies"][strategy] = {
                "pulls": stats.pulls,
                "mean_speedup": stats.mean_speedup,
                "success_rate": stats.success_rate,
                "confidence_interval": [stats.ci_lower, stats.ci_upper],
                "confidence": self._calculate_confidence(stats),
            }

        # Add recommendation
        if context in self.optimal_strategies:
            report["recommendation"] = self.optimal_strategies[context][0]

        return report


def create_optimizer(config: Optional[Dict[str, Any]] = None) -> RealtimeOptimizationLearner:
    """Factory function to create optimizer with configuration"""
    config = config or {}

    return RealtimeOptimizationLearner(
        db_path=Path(config.get("db_path", "optimization_history.db")),
        algorithm=config.get("algorithm", "thompson"),
        exploration_rate=config.get("exploration_rate", 0.1),
    )

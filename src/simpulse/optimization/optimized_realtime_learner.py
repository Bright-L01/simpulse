#!/usr/bin/env python3
"""
Optimized Real-time Learner

Enhanced version with sophisticated exploration strategies that balance
learning speed with user performance. No sacrificing user experience.
"""

import logging
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..analysis.advanced_context_classifier import AdvancedContextClassifier
from .adaptive_exploration import AdaptiveExplorationManager

logger = logging.getLogger(__name__)


class OptimizedRealtimeLearner:
    """
    Enhanced real-time learner with optimized exploration strategies.

    Key improvements:
    1. Adaptive epsilon-greedy with decay
    2. Intelligent Thompson Sampling with priors
    3. Curiosity mechanism for understudied contexts
    4. Coverage optimization for rare patterns
    5. Safety guarantees to protect user performance
    """

    def __init__(
        self, db_path: Path = Path("optimized_learning.db"), config: Optional[Dict[str, Any]] = None
    ):

        self.db_path = db_path
        self.config = config or {}

        # Context classifier
        self.classifier = AdvancedContextClassifier()

        # Enhanced exploration manager
        self.exploration_manager = AdaptiveExplorationManager(self.config)

        # Strategy statistics (from original learner)
        from .realtime_optimizer import StrategyStats

        self.stats: Dict[Tuple[str, str], StrategyStats] = defaultdict(StrategyStats)

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

        # Performance tracking
        self.total_compilations = 0
        self.successful_compilations = 0
        self.cumulative_speedup = 0.0
        self.regret_history = []

        # Initialize database
        self._init_database()

        # Load historical data
        self._load_state()

    def _init_database(self):
        """Initialize database with exploration tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Original tables
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
                was_exploration INTEGER DEFAULT 0,
                curiosity_score REAL DEFAULT 0.0,
                metadata TEXT
            )
        """
        )

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

        # New exploration tracking table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS exploration_stats (
                context_type TEXT PRIMARY KEY,
                total_explorations INTEGER DEFAULT 0,
                successful_explorations INTEGER DEFAULT 0,
                last_exploration REAL DEFAULT 0.0,
                curiosity_score REAL DEFAULT 0.0,
                coverage_score REAL DEFAULT 0.0,
                importance_score REAL DEFAULT 0.0
            )
        """
        )

        # Performance tracking table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                timestamp REAL PRIMARY KEY,
                total_compilations INTEGER,
                exploration_rate REAL,
                average_speedup REAL,
                safety_violations INTEGER DEFAULT 0,
                novel_discoveries INTEGER DEFAULT 0
            )
        """
        )

        conn.commit()
        conn.close()

    def _load_state(self):
        """Load historical data and initialize exploration manager"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load strategy statistics (same as original)
        cursor.execute(
            """
            SELECT context_type, strategy, pulls, successes, total_speedup,
                   ci_lower, ci_upper, ci_confidence
            FROM strategy_stats
        """
        )

        for row in cursor.fetchall():
            context, strategy = row[0], row[1]
            from .realtime_optimizer import StrategyStats

            stats = StrategyStats(
                pulls=row[2],
                successes=row[3],
                total_speedup=row[4],
                ci_lower=row[5],
                ci_upper=row[6],
                ci_confidence=row[7],
            )
            self.stats[(context, strategy)] = stats

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

        # Load performance metrics
        cursor.execute("SELECT COUNT(*) FROM compilation_events")
        self.total_compilations = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*) FROM compilation_events 
            WHERE success = 1
        """
        )
        self.successful_compilations = cursor.fetchone()[0] or 0

        cursor.execute(
            """
            SELECT AVG(speedup) FROM compilation_events
            WHERE timestamp > ?
        """,
            (time.time() - 86400 * 7,),
        )  # Last week

        avg_speedup = cursor.fetchone()[0]
        if avg_speedup:
            self.cumulative_speedup = avg_speedup * self.total_compilations

        conn.close()

        logger.info(f"Loaded {len(self.stats)} strategy statistics")
        logger.info(f"Total compilations: {self.total_compilations}")
        logger.info(
            f"Success rate: {self.successful_compilations/max(1, self.total_compilations):.1%}"
        )

    def recommend_strategy(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Get intelligent optimization recommendation with adaptive exploration.
        """
        # Classify context
        classification = self.classifier.classify(file_path)
        context = classification.context_type

        # Get strategy statistics for this context
        context_stats = {}
        for strategy in self.strategies:
            key = (context, strategy)
            if key in self.stats:
                context_stats[strategy] = self.stats[key]

        # Use adaptive exploration manager for selection
        strategy, exploration_metadata = self.exploration_manager.select_strategy(
            context=context,
            strategy_stats=context_stats,
            strategies=self.strategies,
            file_path=str(file_path),
            baseline_time=1.0,  # Default baseline
        )

        # Get detailed statistics for selected strategy
        stats = self.stats.get((context, strategy))
        if stats:
            expected_speedup = stats.mean_speedup
            ci_lower = stats.ci_lower
            ci_upper = stats.ci_upper
            confidence = self._calculate_confidence(stats)
            pulls = stats.pulls
            success_rate = stats.success_rate
        else:
            expected_speedup = 1.0
            ci_lower = 0.9
            ci_upper = 1.1
            confidence = 0.0
            pulls = 0
            success_rate = 0.5

        # Combine metadata
        metadata = {
            "context_type": context,
            "expected_speedup": expected_speedup,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence": confidence,
            "success_rate": success_rate,
            "pulls": pulls,
            "classification_confidence": classification.confidence,
            **exploration_metadata,
        }

        logger.info(
            f"Recommending {strategy} for {context} "
            f"(expected: {expected_speedup:.2f}x, "
            f"exploration: {exploration_metadata['is_exploration']})"
        )

        return strategy, metadata

    def record_result(
        self,
        file_path: Path,
        context_type: str,
        strategy: str,
        baseline_time: float,
        optimized_time: float,
        compilation_success: bool = True,
        was_exploration: bool = False,
        curiosity_score: float = 0.0,
    ):
        """
        Record compilation result with enhanced tracking.
        """
        # Calculate speedup
        speedup = baseline_time / optimized_time if optimized_time > 0 else 0.1
        if not compilation_success:
            speedup = 0.8  # Penalty for compilation failure

        # Determine success
        success = speedup > 1.05 and compilation_success

        # Calculate regret (simplified for now)
        regret = max(0, 1.0 - speedup)

        # Update strategy statistics
        stats = self.stats[(context_type, strategy)]
        stats.pulls += 1
        stats.total_speedup += speedup
        stats.speedups.append(speedup)
        if success:
            stats.successes += 1
        stats.last_updated = time.time()

        # Keep only recent speedups
        if len(stats.speedups) > 100:
            stats.speedups = stats.speedups[-100:]

        # Update confidence interval
        stats.update_confidence_interval()

        # Update exploration manager
        self.exploration_manager.record_exploration_result(
            context_type, strategy, speedup, was_exploration
        )

        # Update global statistics
        self.total_compilations += 1
        if success:
            self.successful_compilations += 1
        self.cumulative_speedup += speedup
        self.regret_history.append(regret)

        # Persist to database
        self._save_event(
            file_path,
            context_type,
            strategy,
            baseline_time,
            optimized_time,
            speedup,
            success,
            regret,
            was_exploration,
            curiosity_score,
        )

        self._save_stats(context_type, strategy, stats)

        # Periodic analysis
        if self.total_compilations % 50 == 0:
            self._analyze_performance()

        logger.info(
            f"Recorded: {context_type}/{strategy} → {speedup:.2f}x "
            f"(success: {success}, exploration: {was_exploration})"
        )

    def _save_event(
        self,
        file_path: Path,
        context_type: str,
        strategy: str,
        baseline_time: float,
        optimized_time: float,
        speedup: float,
        success: bool,
        regret: float,
        was_exploration: bool,
        curiosity_score: float,
    ):
        """Save enhanced compilation event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO compilation_events 
            (timestamp, file_path, context_type, strategy, baseline_time,
             optimized_time, speedup, success, regret, was_exploration,
             curiosity_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                time.time(),
                str(file_path),
                context_type,
                strategy,
                baseline_time,
                optimized_time,
                speedup,
                int(success),
                regret,
                int(was_exploration),
                curiosity_score,
                "{}",
            ),
        )

        conn.commit()
        conn.close()

    def _save_stats(self, context: str, strategy: str, stats):
        """Save strategy statistics"""
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

    def _calculate_confidence(self, stats) -> float:
        """Calculate confidence level for a strategy"""
        if stats.pulls == 0:
            return 0.0
        elif stats.pulls < 5:
            return 0.3
        elif stats.pulls < 20:
            return 0.6
        else:
            # Asymptotic confidence
            ci_width = stats.ci_upper - stats.ci_lower
            confidence = min(0.95, 0.6 + 0.35 * (1 - ci_width / 2.0))
            return confidence * stats.ci_confidence

    def _analyze_performance(self):
        """Enhanced performance analysis"""
        logger.info("\n" + "=" * 70)
        logger.info("OPTIMIZED LEARNING PERFORMANCE ANALYSIS")
        logger.info("=" * 70)

        # Overall statistics
        success_rate = self.successful_compilations / max(1, self.total_compilations)
        avg_speedup = self.cumulative_speedup / max(1, self.total_compilations)

        logger.info(f"Total compilations: {self.total_compilations}")
        logger.info(f"Success rate: {success_rate:.1%}")
        logger.info(f"Average speedup: {avg_speedup:.2f}x")

        # Exploration analysis
        exploration_report = self.exploration_manager.get_exploration_report()
        logger.info(f"Exploration rate: {exploration_report['exploration_rate']:.1%}")
        logger.info(
            f"Exploration success rate: {exploration_report['epsilon_greedy_stats']['success_rate']:.1%}"
        )
        logger.info(
            f"Novel discoveries: {exploration_report['epsilon_greedy_stats']['novel_discoveries']}"
        )

        # Performance trend
        trend = exploration_report["performance_trend"]["improvement_trend"]
        logger.info(f"Performance trend: {trend}")

        # Context coverage
        coverage = exploration_report["context_coverage"]
        logger.info(f"Contexts covered: {len(coverage)}")

        # Save performance metrics
        self._save_performance_metrics(exploration_report)

    def _save_performance_metrics(self, exploration_report: Dict[str, Any]):
        """Save performance metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO performance_metrics
            (timestamp, total_compilations, exploration_rate, average_speedup,
             novel_discoveries)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                time.time(),
                self.total_compilations,
                exploration_report["exploration_rate"],
                self.cumulative_speedup / max(1, self.total_compilations),
                exploration_report["epsilon_greedy_stats"]["novel_discoveries"],
            ),
        )

        conn.commit()
        conn.close()

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive learning and exploration report"""
        exploration_report = self.exploration_manager.get_exploration_report()

        return {
            "learning_stats": {
                "total_compilations": self.total_compilations,
                "successful_compilations": self.successful_compilations,
                "success_rate": self.successful_compilations / max(1, self.total_compilations),
                "average_speedup": self.cumulative_speedup / max(1, self.total_compilations),
                "contexts_learned": len({c for c, _ in self.stats.keys()}),
            },
            "exploration_stats": exploration_report,
            "strategy_performance": self._get_strategy_performance(),
            "recommendations": self._generate_recommendations(),
        }

    def _get_strategy_performance(self) -> Dict[str, Any]:
        """Get per-strategy performance summary"""
        performance = {}

        for (context, strategy), stats in self.stats.items():
            if stats.pulls > 0:
                if context not in performance:
                    performance[context] = {}

                performance[context][strategy] = {
                    "mean_speedup": stats.mean_speedup,
                    "success_rate": stats.success_rate,
                    "confidence": self._calculate_confidence(stats),
                    "pulls": stats.pulls,
                }

        return performance

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        exploration_report = self.exploration_manager.get_exploration_report()

        # Exploration recommendations
        exploration_rate = exploration_report["exploration_rate"]
        if exploration_rate < 0.05:
            recommendations.append("Consider increasing exploration to discover better strategies")
        elif exploration_rate > 0.3:
            recommendations.append(
                "High exploration rate - good for learning but may impact performance"
            )

        # Performance recommendations
        avg_speedup = self.cumulative_speedup / max(1, self.total_compilations)
        if avg_speedup < 1.1:
            recommendations.append("Low average speedup - consider more aggressive exploration")
        elif avg_speedup > 1.5:
            recommendations.append("Excellent performance - current strategy is working well")

        # Coverage recommendations
        coverage = exploration_report["context_coverage"]
        under_explored = [c for c, count in coverage.items() if count < 5]
        if under_explored:
            recommendations.append(f"Under-explored contexts: {', '.join(under_explored[:3])}")

        return recommendations


def create_optimized_learner(config: Optional[Dict[str, Any]] = None) -> OptimizedRealtimeLearner:
    """Factory function for optimized learner"""
    return OptimizedRealtimeLearner(config=config)

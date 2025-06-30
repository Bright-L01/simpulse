"""
Runtime adapter for Simpulse JIT profiler.

Analyzes statistics from Lean's JIT profiler and provides
Python interface for priority optimization.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

# import numpy as np  # Not needed for this implementation


@dataclass
class RuleStatistics:
    """Statistics for a single simp rule."""

    rule_name: str
    attempts: int = 0
    successes: int = 0
    total_time: float = 0.0
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successes / self.attempts if self.attempts > 0 else 0.0

    @property
    def avg_time(self) -> float:
        """Calculate average execution time."""
        return self.total_time / self.attempts if self.attempts > 0 else 0.0

    def apply_decay(self, factor: float, current_time: float) -> None:
        """Apply exponential decay based on time since last use."""
        time_diff = current_time - self.last_used
        decay_multiplier = factor ** (time_diff / 60.0)  # Decay per minute

        self.attempts = int(self.attempts * decay_multiplier)
        self.successes = int(self.successes * decay_multiplier)
        self.total_time *= decay_multiplier


@dataclass
class AdapterConfig:
    """Configuration for runtime adapter."""

    stats_file: str = "simp_stats.json"
    priority_file: str = "simp_priorities.json"
    log_file: Optional[str] = "jit_adapter.log"

    # Optimization parameters
    adaptation_interval: int = 100  # Analyze every N simp calls
    decay_factor: float = 0.95  # Exponential decay rate
    min_samples: int = 10  # Minimum attempts before optimization

    # Priority calculation
    priority_range: Tuple[int, int] = (100, 5000)
    boost_factor: float = 2.0

    # Performance thresholds
    high_success_threshold: float = 0.8
    low_success_threshold: float = 0.2


class RuntimeAdapter:
    """Runtime adapter for JIT priority optimization."""

    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()
        self.statistics: Dict[str, RuleStatistics] = {}
        self.call_count = 0
        self.last_optimization = time.time()

        # Create necessary directories
        for file_path in [self.config.stats_file, self.config.priority_file]:
            if file_path:
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    def load_statistics(self) -> None:
        """Load statistics from Lean's output file."""
        if not os.path.exists(self.config.stats_file):
            return

        try:
            with open(self.config.stats_file) as f:
                data = json.load(f)

            for rule_name, stats in data.items():
                self.statistics[rule_name] = RuleStatistics(
                    rule_name=rule_name,
                    attempts=stats.get("attempts", 0),
                    successes=stats.get("successes", 0),
                    total_time=stats.get("total_time", 0.0),
                    last_used=stats.get("last_used", time.time()),
                )
        except Exception as e:
            self._log(f"Error loading statistics: {e}")

    def save_statistics(self) -> None:
        """Save current statistics to file."""
        data = {
            name: {
                "attempts": stats.attempts,
                "successes": stats.successes,
                "total_time": stats.total_time,
                "last_used": stats.last_used,
            }
            for name, stats in self.statistics.items()
        }

        try:
            with open(self.config.stats_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self._log(f"Error saving statistics: {e}")

    def update_statistics(self, rule_name: str, success: bool, duration: float) -> None:
        """Update statistics for a rule attempt."""
        if rule_name not in self.statistics:
            self.statistics[rule_name] = RuleStatistics(rule_name=rule_name)

        stats = self.statistics[rule_name]
        stats.attempts += 1
        if success:
            stats.successes += 1
        stats.total_time += duration
        stats.last_used = time.time()

        self.call_count += 1

        # Check if we should optimize
        if self.call_count % self.config.adaptation_interval == 0:
            self.optimize_priorities()

    def apply_decay(self) -> None:
        """Apply exponential decay to all statistics."""
        current_time = time.time()

        for stats in self.statistics.values():
            stats.apply_decay(self.config.decay_factor, current_time)

        # Remove rules with too few samples after decay
        self.statistics = {
            name: stats
            for name, stats in self.statistics.items()
            if stats.attempts >= 1
        }

    def calculate_priority(self, stats: RuleStatistics) -> int:
        """Calculate dynamic priority for a rule based on statistics."""
        if stats.attempts < self.config.min_samples:
            return 1000  # Default priority

        # Success rate component (0-1)
        success_component = stats.success_rate

        # Speed component (inverse of avg time, normalized)
        speed_component = 1.0 / (stats.avg_time + 0.001)

        # Frequency component (how often it's attempted)
        total_attempts = sum(s.attempts for s in self.statistics.values())
        frequency_component = (
            stats.attempts / total_attempts if total_attempts > 0 else 0
        )

        # Combined score with weights
        score = (
            0.5 * success_component + 0.3 * speed_component + 0.2 * frequency_component
        )

        # Apply boost factor and map to priority range
        min_prio, max_prio = self.config.priority_range
        priority = int(
            min_prio + score * self.config.boost_factor * (max_prio - min_prio)
        )

        return max(min_prio, min(priority, max_prio))

    def optimize_priorities(self) -> Dict[str, int]:
        """Optimize priorities based on current statistics."""
        self._log(f"Optimizing priorities (call count: {self.call_count})")

        # Apply decay to old statistics
        self.apply_decay()

        # Calculate new priorities
        priorities = {}

        for rule_name, stats in self.statistics.items():
            if stats.attempts >= self.config.min_samples:
                priority = self.calculate_priority(stats)
                priorities[rule_name] = priority

                # Log significant changes
                if stats.success_rate > self.config.high_success_threshold:
                    self._log(
                        f"High performer: {rule_name} ({stats.success_rate:.1%} success)"
                    )
                elif stats.success_rate < self.config.low_success_threshold:
                    self._log(
                        f"Low performer: {rule_name} ({stats.success_rate:.1%} success)"
                    )

        # Save priorities
        self.save_priorities(priorities)

        # Update optimization time
        self.last_optimization = time.time()

        return priorities

    def save_priorities(self, priorities: Dict[str, int]) -> None:
        """Save optimized priorities to file."""
        try:
            with open(self.config.priority_file, "w") as f:
                json.dump(priorities, f, indent=2)
            self._log(f"Saved {len(priorities)} optimized priorities")
        except Exception as e:
            self._log(f"Error saving priorities: {e}")

    def load_priorities(self) -> Dict[str, int]:
        """Load saved priorities from file."""
        if not os.path.exists(self.config.priority_file):
            return {}

        try:
            with open(self.config.priority_file) as f:
                return json.load(f)
        except Exception as e:
            self._log(f"Error loading priorities: {e}")
            return {}

    def get_statistics_summary(self) -> str:
        """Get summary of current statistics."""
        if not self.statistics:
            return "No statistics collected yet."

        total_attempts = sum(s.attempts for s in self.statistics.values())
        total_successes = sum(s.successes for s in self.statistics.values())

        # Sort by attempts
        sorted_stats = sorted(
            self.statistics.items(), key=lambda x: x[1].attempts, reverse=True
        )

        summary = [
            "=== JIT Runtime Adapter Statistics ===",
            f"Total rules tracked: {len(self.statistics)}",
            f"Total attempts: {total_attempts}",
            f"Total successes: {total_successes}",
            (
                f"Overall success rate: {total_successes/total_attempts:.1%}"
                if total_attempts > 0
                else "N/A"
            ),
            f"Calls since last optimization: {self.call_count % self.config.adaptation_interval}",
            "",
            "Top 10 rules by attempts:",
        ]

        for rule_name, stats in sorted_stats[:10]:
            priority = self.calculate_priority(stats)
            summary.append(
                f"  {rule_name}: {stats.attempts} attempts, "
                f"{stats.success_rate:.1%} success, "
                f"{stats.avg_time*1000:.2f}ms avg, "
                f"priority={priority}"
            )

        return "\n".join(summary)

    def export_analysis(self, output_file: str) -> None:
        """Export detailed analysis to file."""
        analysis = {
            "metadata": {
                "timestamp": time.time(),
                "total_rules": len(self.statistics),
                "total_calls": self.call_count,
                "config": {
                    "decay_factor": self.config.decay_factor,
                    "adaptation_interval": self.config.adaptation_interval,
                    "min_samples": self.config.min_samples,
                },
            },
            "rules": {},
        }

        for rule_name, stats in self.statistics.items():
            analysis["rules"][rule_name] = {
                "attempts": stats.attempts,
                "successes": stats.successes,
                "success_rate": stats.success_rate,
                "avg_time_ms": stats.avg_time * 1000,
                "total_time_ms": stats.total_time * 1000,
                "calculated_priority": self.calculate_priority(stats),
                "last_used": stats.last_used,
            }

        with open(output_file, "w") as f:
            json.dump(analysis, f, indent=2)

        self._log(f"Exported analysis to {output_file}")

    def _log(self, message: str) -> None:
        """Log message to file if configured."""
        if self.config.log_file:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.config.log_file, "a") as f:
                f.write(f"[{timestamp}] {message}\n")


def monitor_lean_process(
    adapter: RuntimeAdapter, stats_file: str, interval: float = 1.0
):
    """Monitor Lean process and update adapter with new statistics."""
    import watchdog.events
    import watchdog.observers

    class StatsHandler(watchdog.events.FileSystemEventHandler):
        def on_modified(self, event):
            if event.src_path == stats_file:
                adapter.load_statistics()
                adapter._log("Reloaded statistics from Lean")

    observer = watchdog.observers.Observer()
    observer.schedule(StatsHandler(), os.path.dirname(stats_file), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(interval)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    # Example usage
    config = AdapterConfig(
        stats_file="simp_stats.json",
        priority_file="simp_priorities.json",
        adaptation_interval=50,
    )

    adapter = RuntimeAdapter(config)

    # Simulate some rule attempts
    rules = ["add_zero", "zero_add", "mul_one", "complex_rule"]

    for i in range(200):
        rule = rules[i % len(rules)]
        success = i % 3 != 0  # Simulate 66% success rate
        duration = 0.001 if rule != "complex_rule" else 0.01

        adapter.update_statistics(rule, success, duration)

    # Show summary
    print(adapter.get_statistics_summary())

    # Export analysis
    adapter.export_analysis("jit_analysis.json")

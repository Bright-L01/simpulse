#!/usr/bin/env python3
"""
JIT-style dynamic optimization for Lean's simp tactic.

Monitors runtime behavior and adapts priorities in real-time.
"""

import json
import random
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RuleStats:
    """Runtime statistics for a simp rule."""

    name: str
    attempts: int = 0
    successes: int = 0
    total_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_used: float = field(default_factory=time.time)
    contexts: dict[str, int] = field(default_factory=dict)  # Context -> usage count

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successes / self.attempts if self.attempts > 0 else 0.0

    @property
    def avg_time(self) -> float:
        """Average execution time."""
        return self.total_time / self.attempts if self.attempts > 0 else 0.0

    @property
    def recent_avg_time(self) -> float:
        """Recent average time (adaptive)."""
        if self.recent_times:
            return sum(self.recent_times) / len(self.recent_times)
        return self.avg_time

    def decay(self, decay_factor: float = 0.95):
        """Apply time decay to statistics."""
        self.attempts = int(self.attempts * decay_factor)
        self.successes = int(self.successes * decay_factor)
        self.total_time *= decay_factor


@dataclass
class OptimizationContext:
    """Context for optimization decisions."""

    module_name: str
    goal_type: str  # arithmetic, list, logic, etc.
    proof_depth: int
    previous_tactics: list[str]


class DynamicSimpOptimizer:
    """JIT-style runtime optimizer for simp priorities."""

    def __init__(
        self,
        adaptation_interval: int = 100,
        decay_interval: int = 1000,
        cache_size: int = 10000,
    ):
        """Initialize dynamic optimizer."""
        # Runtime statistics
        self.rule_stats: dict[str, RuleStats] = {}
        self.global_attempts = 0

        # Configuration
        self.adaptation_interval = adaptation_interval
        self.decay_interval = decay_interval
        self.last_adaptation = 0
        self.last_decay = 0

        # Priority cache
        self.priority_cache: dict[str, int] = {}
        self.cache_size = cache_size

        # Context-aware optimization
        self.context_priorities: dict[str, dict[str, int]] = defaultdict(dict)

        # Performance tracking
        self.baseline_time = 0.0
        self.optimized_time = 0.0
        self.improvements: deque = deque(maxlen=1000)

        # Hot path detection
        self.hot_paths: dict[str, int] = defaultdict(int)
        self.compiled_paths: dict[str, Any] = {}

    def instrument_simp_attempt(
        self, rule_name: str, context: OptimizationContext | None = None
    ) -> "InstrumentedAttempt":
        """Instrument a simp rule attempt."""
        return InstrumentedAttempt(self, rule_name, context)

    def record_attempt(
        self,
        rule_name: str,
        success: bool,
        duration: float,
        context: OptimizationContext | None = None,
    ):
        """Record statistics for a rule attempt."""
        # Initialize stats if needed
        if rule_name not in self.rule_stats:
            self.rule_stats[rule_name] = RuleStats(name=rule_name)

        stats = self.rule_stats[rule_name]

        # Update statistics
        stats.attempts += 1
        if success:
            stats.successes += 1
        stats.total_time += duration
        stats.recent_times.append(duration)
        stats.last_used = time.time()

        # Track context
        if context:
            context_key = f"{context.module_name}:{context.goal_type}"
            stats.contexts[context_key] = stats.contexts.get(context_key, 0) + 1

            # Track hot paths
            path_key = f"{context.module_name}:{'.'.join(context.previous_tactics[-3:])}"
            self.hot_paths[path_key] += 1

        self.global_attempts += 1

        # Trigger adaptation if needed
        if self.global_attempts - self.last_adaptation >= self.adaptation_interval:
            self.adapt_priorities()

        # Apply decay periodically
        if self.global_attempts - self.last_decay >= self.decay_interval:
            self.apply_decay()

    def adapt_priorities(self):
        """Dynamically adjust priorities based on runtime statistics."""
        print(f"Adapting priorities after {self.global_attempts} attempts...")

        # Calculate priority scores
        priority_scores = []

        for rule_name, stats in self.rule_stats.items():
            # Multi-factor scoring
            success_score = stats.success_rate * 100
            speed_score = 1.0 / (stats.recent_avg_time + 0.001) if stats.recent_avg_time > 0 else 0
            frequency_score = stats.attempts / max(self.global_attempts, 1) * 100
            recency_score = 1.0 / (time.time() - stats.last_used + 1.0)

            # Weighted combination
            total_score = (
                0.4 * success_score
                + 0.3 * frequency_score
                + 0.2 * speed_score
                + 0.1 * recency_score
            )

            priority_scores.append((rule_name, total_score))

        # Sort by score
        priority_scores.sort(key=lambda x: x[1], reverse=True)

        # Assign priorities
        new_priorities = {}
        for i, (rule_name, score) in enumerate(priority_scores):
            # High scores get low priority numbers (higher precedence)
            priority = 100 + i * 10
            new_priorities[rule_name] = priority

        # Update cache
        self.priority_cache = new_priorities
        self.last_adaptation = self.global_attempts

        # Track improvement
        self._measure_improvement()

        # Compile hot paths
        self._compile_hot_paths()

    def apply_decay(self):
        """Apply time decay to statistics."""
        for stats in self.rule_stats.values():
            stats.decay(0.95)

        self.last_decay = self.global_attempts

    def get_priority(self, rule_name: str, context: OptimizationContext | None = None) -> int:
        """Get current priority for a rule."""
        # Check context-specific priority
        if context:
            context_key = f"{context.module_name}:{context.goal_type}"
            if context_key in self.context_priorities:
                if rule_name in self.context_priorities[context_key]:
                    return self.context_priorities[context_key][rule_name]

        # Use cached priority
        return self.priority_cache.get(rule_name, 1000)  # Default priority

    def compile_optimized_simp(self, output_path: Path):
        """Generate optimized simp tactic configuration."""
        config = {
            "version": "1.0",
            "generated_at": time.time(),
            "statistics": {
                "total_attempts": self.global_attempts,
                "unique_rules": len(self.rule_stats),
                "avg_improvement": (statistics.mean(self.improvements) if self.improvements else 0),
            },
            "priorities": self.priority_cache,
            "hot_paths": dict(self.hot_paths),
            "rule_statistics": {
                rule_name: {
                    "success_rate": stats.success_rate,
                    "avg_time": stats.avg_time,
                    "attempts": stats.attempts,
                }
                for rule_name, stats in self.rule_stats.items()
                if stats.attempts > 10  # Only include well-tested rules
            },
        }

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Optimized configuration saved to {output_path}")

        # Generate Lean integration
        self._generate_lean_integration(output_path.with_suffix(".lean"))

    def _measure_improvement(self):
        """Measure performance improvement."""
        if not self.rule_stats:
            return

        # Simulate default vs optimized execution
        rules = list(self.rule_stats.keys())

        # Default: arbitrary order
        default_time = 0.0
        for i, rule in enumerate(rules):
            stats = self.rule_stats[rule]
            # Expected attempts before success
            expected_attempts = i * (1 - stats.success_rate) + 1
            default_time += expected_attempts * stats.avg_time

        # Optimized: by priority
        optimized_rules = sorted(rules, key=lambda r: self.get_priority(r))
        optimized_time = 0.0
        for i, rule in enumerate(optimized_rules):
            stats = self.rule_stats[rule]
            expected_attempts = i * (1 - stats.success_rate) + 1
            optimized_time += expected_attempts * stats.avg_time

        if default_time > 0:
            improvement = (default_time - optimized_time) / default_time * 100
            self.improvements.append(improvement)

            if len(self.improvements) % 10 == 0:
                avg_improvement = statistics.mean(self.improvements)
                print(f"Average improvement: {avg_improvement:.1f}%")

    def _compile_hot_paths(self):
        """Compile frequently executed proof paths."""
        # Identify top hot paths
        top_paths = sorted(self.hot_paths.items(), key=lambda x: x[1], reverse=True)[:10]

        for path_key, count in top_paths:
            if count > 50:  # Threshold for compilation
                # In real implementation, would generate optimized code
                self.compiled_paths[path_key] = {
                    "count": count,
                    "optimized": True,
                    "rules": self._get_path_rules(path_key),
                }

    def _get_path_rules(self, path_key: str) -> list[str]:
        """Get frequently used rules for a path."""
        # Extract rules used in this context
        module, tactics = path_key.split(":", 1)
        relevant_rules = []

        for rule_name, stats in self.rule_stats.items():
            for context in stats.contexts:
                if context.startswith(module):
                    relevant_rules.append((rule_name, stats.contexts[context]))

        # Sort by usage in this context
        relevant_rules.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in relevant_rules[:20]]

    def _generate_lean_integration(self, output_path: Path):
        """Generate Lean 4 integration code."""
        lean_code = """-- Auto-generated by Simpulse JIT Optimizer
import Lean

namespace Simpulse.JIT

-- Priority configuration from runtime profiling
def optimizedPriorities : List (Name × Nat) := ["""

        # Add top 50 rules with custom priorities
        rules = []
        for rule, priority in sorted(self.priority_cache.items(), key=lambda x: x[1])[:50]:
            rules.append(f"  (`{rule}, {priority})")

        lean_code += ",\n".join(rules)
        lean_code += """
]

-- Apply optimized priorities at initialization
def applyOptimizedPriorities : IO Unit := do
  for (rule, priority) in optimizedPriorities do
    -- Set simp priority for rule
    modifyEnv fun env => 
      simpExtension.modifyState env fun s =>
        s.insert rule priority

-- Initialize on import
initialize applyOptimizedPriorities

end Simpulse.JIT
"""

        output_path.write_text(lean_code)
        print(f"Lean integration saved to {output_path}")


class InstrumentedAttempt:
    """Context manager for instrumenting rule attempts."""

    def __init__(
        self,
        optimizer: DynamicSimpOptimizer,
        rule_name: str,
        context: OptimizationContext | None = None,
    ):
        self.optimizer = optimizer
        self.rule_name = rule_name
        self.context = context
        self.start_time = None
        self.success = False

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        # If no exception, consider it successful
        self.success = exc_type is None
        self.optimizer.record_attempt(self.rule_name, self.success, duration, self.context)

    def mark_success(self):
        """Explicitly mark attempt as successful."""
        self.success = True


def create_benchmark_scenario():
    """Create a benchmark scenario for testing."""
    optimizer = DynamicSimpOptimizer(adaptation_interval=50)

    # Simulate realistic simp usage patterns
    rules = [
        # High-frequency, high-success rules
        ("List.append_nil", 0.95, 0.001),
        ("List.nil_append", 0.90, 0.001),
        ("Nat.add_zero", 0.85, 0.001),
        ("Nat.zero_add", 0.85, 0.001),
        # Medium-frequency rules
        ("List.map_append", 0.60, 0.002),
        ("List.length_append", 0.55, 0.002),
        ("Nat.add_comm", 0.50, 0.001),
        ("Nat.add_assoc", 0.45, 0.002),
        # Low-frequency, expensive rules
        ("ComplexRule1", 0.10, 0.010),
        ("ComplexRule2", 0.08, 0.015),
        ("ComplexRule3", 0.05, 0.020),
    ]

    # Simulate 1000 simp executions
    print("Simulating JIT optimization...")
    for i in range(1000):
        # Create context
        context = OptimizationContext(
            module_name="TestModule",
            goal_type="arithmetic" if i % 2 == 0 else "list",
            proof_depth=i % 5,
            previous_tactics=["intro", "rw", "simp"],
        )

        # Try rules in current priority order
        rule_order = sorted(rules, key=lambda r: optimizer.get_priority(r[0], context))

        # Simulate rule attempts
        for rule_name, success_rate, exec_time in rule_order:
            with optimizer.instrument_simp_attempt(rule_name, context) as attempt:
                # Simulate execution time
                time.sleep(exec_time)

                # Simulate success/failure
                if random.random() < success_rate:
                    attempt.mark_success()
                    break  # Rule succeeded, stop trying others

        # Show progress
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1} simulations")
            avg_improvement = (
                statistics.mean(optimizer.improvements) if optimizer.improvements else 0
            )
            print(f"Current average improvement: {avg_improvement:.1f}%")

    # Save optimized configuration
    optimizer.compile_optimized_simp(Path("jit_optimized_simp.json"))

    # Show final statistics
    print("\nFinal Statistics:")
    print(f"Total attempts: {optimizer.global_attempts}")
    print(f"Unique rules: {len(optimizer.rule_stats)}")
    print(
        f"Average improvement: {statistics.mean(optimizer.improvements) if optimizer.improvements else 0:.1f}%"
    )

    print("\nTop 5 rules by adapted priority:")
    top_rules = sorted(optimizer.priority_cache.items(), key=lambda x: x[1])[:5]
    for rule, priority in top_rules:
        stats = optimizer.rule_stats[rule]
        print(
            f"  {rule}: priority={priority}, "
            f"success={stats.success_rate:.1%}, "
            f"attempts={stats.attempts}"
        )


if __name__ == "__main__":
    create_benchmark_scenario()

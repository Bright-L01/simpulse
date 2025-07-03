"""Fast optimizer implementation with performance optimizations."""

import heapq
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..errors import ErrorHandler
from ..evolution.optimized_rule_extractor import OptimizedRuleExtractor


@dataclass
class RuleScore:
    """Efficient rule scoring structure."""

    rule_name: str
    file_path: str
    score: float
    priority: int

    def __lt__(self, other):
        # For heap - higher score = higher priority (negate for min heap)
        return -self.score < -other.score


class FastOptimizer:
    """High-performance optimizer with algorithmic improvements."""

    def __init__(self, strategy: str = "balanced"):
        self.strategy = strategy
        self.extractor = OptimizedRuleExtractor()
        self.error_handler = ErrorHandler(logging.getLogger(__name__))

        # Pre-allocate data structures
        self.rule_scores: list[RuleScore] = []
        self.rule_index: dict[str, int] = {}  # name -> index mapping
        self.file_rules: defaultdict[str, list] = defaultdict(list)  # file -> rules

        # Strategy weights (vectorized for numpy)
        self.weights = self._get_strategy_weights(strategy)

    def analyze(self, project_path: Path, use_parallel: bool = True) -> dict:
        """Analyze project with optimized extraction."""
        if not project_path.exists():
            return {"project_path": project_path, "rules": [], "analysis_stats": {}}

        # Extract rules in parallel
        if use_parallel:
            results = self.extractor.extract_rules_from_project(project_path)

            # Flatten results
            all_rules = []
            for module_rules in results.values():
                all_rules.extend(module_rules.rules)
                # Build index for fast lookup
                for rule in module_rules.rules:
                    self.file_rules[str(module_rules.file_path)].append(rule)
        else:
            # Sequential extraction (for comparison)
            all_rules = []
            lean_files = list(project_path.glob("**/*.lean"))
            for lean_file in lean_files:
                if "lake-packages" not in str(lean_file) and ".lake" not in str(lean_file):
                    module_rules = self.extractor.extract_rules_from_file(lean_file)
                    all_rules.extend(module_rules.rules)
                    for rule in module_rules.rules:
                        self.file_rules[str(lean_file)].append(rule)

        # Build rule index
        for i, rule in enumerate(all_rules):
            self.rule_index[rule.name] = i

        stats = self.extractor.get_statistics()

        return {
            "project_path": project_path,
            "rules": all_rules,
            "analysis_stats": {
                "total_files": stats.get("files_processed", 0),
                "total_rules": len(all_rules),
                "cache_hit_rate": stats.get("cache_hit_rate", 0),
                **stats,
            },
        }

    def optimize(self, analysis: dict) -> dict:
        """Optimize with efficient algorithms."""
        rules = analysis.get("rules", [])
        if not rules:
            return self._empty_result(analysis["project_path"])

        # Vectorized scoring for performance
        scores = self._compute_scores_vectorized(rules)

        # Use heap for efficient top-k selection
        top_rules = self._select_top_rules_heap(rules, scores, k=50)

        # Generate optimization changes
        changes = self._generate_changes_fast(top_rules)

        # Estimate improvement
        estimated_improvement = self._estimate_improvement_fast(len(changes), len(rules))

        return {
            "project_path": analysis["project_path"],
            "rules_changed": len(changes),
            "estimated_improvement": estimated_improvement,
            "changes": changes,
        }

    def _get_strategy_weights(self, strategy: str) -> np.ndarray:
        """Get strategy weights as numpy array for vectorized operations."""
        weights_map = {
            "balanced": np.array([0.3, 0.25, 0.25, 0.2]),  # freq, complexity, success, recency
            "performance": np.array([0.1, 0.2, 0.5, 0.2]),  # focus on success rate
            "frequency": np.array([0.6, 0.1, 0.2, 0.1]),  # focus on frequency
            "complexity": np.array([0.1, 0.6, 0.2, 0.1]),  # focus on simplicity
            "conservative": np.array([0.25, 0.25, 0.25, 0.25]),  # equal weights
        }
        return weights_map.get(strategy, weights_map["balanced"])

    def _compute_scores_vectorized(self, rules: list) -> np.ndarray:
        """Compute rule scores using vectorized operations."""
        n_rules = len(rules)

        # Pre-allocate feature matrix
        features = np.zeros((n_rules, 4))

        # Extract features in batch
        for i, rule in enumerate(rules):
            # Frequency (normalized)
            features[i, 0] = min(getattr(rule, "frequency", 1) / 10.0, 1.0)

            # Complexity (inverse, normalized)
            decl_len = len(getattr(rule, "declaration", ""))
            features[i, 1] = max(0, 1.0 - decl_len / 200.0)

            # Success rate
            features[i, 2] = getattr(rule, "success_rate", 0.8)

            # Recency (placeholder)
            features[i, 3] = 0.5

        # Vectorized scoring
        scores = features @ self.weights  # Matrix multiplication

        return scores

    def _select_top_rules_heap(self, rules: list, scores: np.ndarray, k: int) -> list[tuple]:
        """Select top-k rules using heap for O(n log k) complexity."""
        # Build heap of (score, index) tuples
        heap = []

        for i, (rule, score) in enumerate(zip(rules, scores)):
            # Only consider rules with default priority
            if getattr(rule, "priority", 1000) == 1000:
                if len(heap) < k:
                    heapq.heappush(heap, (score, i, rule))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, i, rule))

        # Extract rules in score order
        top_rules = []
        while heap:
            score, idx, rule = heapq.heappop(heap)
            top_rules.append((rule, score))

        top_rules.reverse()  # Highest scores first
        return top_rules

    def _generate_changes_fast(self, top_rules: list[tuple]) -> list[dict]:
        """Generate optimization changes efficiently."""
        changes = []

        # Pre-compute priority assignments
        base_priority = 100
        priority_step = 25

        for i, (rule, score) in enumerate(top_rules):
            new_priority = base_priority + i * priority_step

            change = {
                "rule_name": rule.name,
                "file_path": str(getattr(rule.location, "file", "unknown")),
                "old_priority": 1000,
                "new_priority": new_priority,
                "reason": f"{self.strategy} optimization (score: {score:.3f})",
            }
            changes.append(change)

        return changes

    def _estimate_improvement_fast(self, n_changes: int, n_total_rules: int) -> int:
        """Fast improvement estimation."""
        if n_changes == 0:
            return 0

        # Strategy-specific multipliers
        multipliers = {
            "balanced": 0.6,
            "performance": 0.8,
            "frequency": 0.5,
            "complexity": 0.4,
            "conservative": 0.3,
        }

        multiplier = multipliers.get(self.strategy, 0.4)
        base_improvement = min(n_changes * multiplier, 50)

        # Adjust for rule coverage
        coverage = n_changes / max(n_total_rules, 1)
        adjusted_improvement = base_improvement * (1 + coverage * 0.2)

        return int(min(adjusted_improvement, 60))

    def _empty_result(self, project_path: Path) -> dict:
        """Return empty optimization result."""
        return {
            "project_path": project_path,
            "rules_changed": 0,
            "estimated_improvement": 0,
            "changes": [],
        }

    def apply_changes_streaming(self, changes: list[dict], project_path: Path) -> int:
        """Apply changes with streaming to handle large files efficiently."""
        applied = 0

        # Group changes by file for batch processing
        file_changes = defaultdict(list)
        for change in changes:
            file_changes[change["file_path"]].append(change)

        for file_path, file_change_list in file_changes.items():
            full_path = project_path / file_path
            if not full_path.exists():
                continue

            try:
                # Read file once
                content = full_path.read_text()

                # Apply all changes for this file
                for change in file_change_list:
                    old_pattern = f"@[simp] theorem {change['rule_name']}"
                    new_pattern = f"@[simp, priority := {change['new_priority']}] theorem {change['rule_name']}"

                    if old_pattern in content:
                        content = content.replace(old_pattern, new_pattern)
                        applied += 1

                # Write once
                full_path.write_text(content)

            except Exception as e:
                logging.error(f"Failed to apply changes to {full_path}: {e}")

        return applied


class StreamingOptimizer:
    """Memory-efficient optimizer for very large projects."""

    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.extractor = OptimizedRuleExtractor()

    def optimize_streaming(self, project_path: Path) -> Iterator[dict]:
        """Process project in chunks to minimize memory usage."""
        lean_files = list(project_path.glob("**/*.lean"))

        # Process files in chunks
        for i in range(0, len(lean_files), self.chunk_size):
            chunk_files = lean_files[i : i + self.chunk_size]

            # Extract rules for chunk
            chunk_rules = []
            for file_path in chunk_files:
                if "lake-packages" not in str(file_path):
                    module_rules = self.extractor.extract_rules_from_file(file_path)
                    chunk_rules.extend(module_rules.rules)

            # Optimize chunk
            if chunk_rules:
                optimizer = FastOptimizer()
                analysis = {"project_path": project_path, "rules": chunk_rules}
                result = optimizer.optimize(analysis)

                yield {
                    "chunk": i // self.chunk_size,
                    "files_processed": len(chunk_files),
                    "optimization": result,
                }

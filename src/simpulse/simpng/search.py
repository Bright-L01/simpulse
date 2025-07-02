"""
Neural proof search using beam search in embedding space.

This module implements the core innovation of SimpNG: treating proof search
as navigation through semantic embedding space rather than syntactic matching.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from .embeddings import GoalEmbedder, RuleEmbedder


@dataclass
class SearchNode:
    """Node in the proof search tree."""

    goal: str
    applied_rules: list[str]
    rule_sequence: list[dict[str, Any]]
    score: float
    depth: int
    parent: Optional["SearchNode"] = None

    def __lt__(self, other):
        """For priority queue - higher score is better."""
        return self.score > other.score


@dataclass
class SearchStatistics:
    """Statistics about the search process."""

    nodes_explored: int = 0
    nodes_pruned: int = 0
    embeddings_computed: int = 0
    cache_hits: int = 0
    max_depth_reached: int = 0
    beam_pruning_events: int = 0


class NeuralProofSearch:
    """
    Implements beam search through proof space guided by embeddings.

    Key innovations:
    1. Semantic similarity drives rule selection
    2. Beam search maintains multiple promising paths
    3. Learned heuristics guide exploration
    """

    def __init__(
        self,
        rule_embedder: RuleEmbedder,
        goal_embedder: GoalEmbedder,
        max_depth: int = 10,
        beam_width: int = 5,
        similarity_threshold: float = 0.3,
    ):
        self.rule_embedder = rule_embedder
        self.goal_embedder = goal_embedder
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)

        # Global statistics
        self.global_stats = defaultdict(int)

    def search(
        self,
        initial_state,
        available_rules: list[dict[str, Any]],
        rule_embeddings: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """
        Main search entry point.

        Returns the best simplification found.
        """
        start_time = time.time()
        stats = SearchStatistics()

        # Embed rules if not provided
        if rule_embeddings is None:
            rule_embeddings = self.rule_embedder.embed_rules(available_rules)
            stats.embeddings_computed += len(available_rules)

        # Initialize beam with starting state
        initial_node = SearchNode(
            goal=initial_state.goal,
            applied_rules=[],
            rule_sequence=[],
            score=1.0,
            depth=0,
        )

        beam = [initial_node]
        best_terminal = None
        explored_goals = set()

        # Beam search
        for depth in range(self.max_depth):
            if not beam:
                break

            stats.max_depth_reached = depth
            next_beam = []

            # Expand each node in current beam
            for node in beam:
                if node.goal in explored_goals:
                    stats.nodes_pruned += 1
                    continue

                explored_goals.add(node.goal)
                stats.nodes_explored += 1

                # Check if terminal (simplified enough)
                if self._is_terminal(node):
                    if best_terminal is None or node.score > best_terminal.score:
                        best_terminal = node
                    continue

                # Expand node
                expansions = self._expand_node(
                    node, initial_state.context, available_rules, rule_embeddings, stats
                )

                next_beam.extend(expansions)

            # Prune beam to width
            if len(next_beam) > self.beam_width:
                next_beam.sort()
                next_beam = next_beam[: self.beam_width]
                stats.beam_pruning_events += 1

            beam = next_beam

        # Find best result
        if best_terminal is None and beam:
            best_terminal = max(beam, key=lambda n: n.score)

        # Build result
        result = self._build_result(
            best_terminal if best_terminal else initial_node,
            stats,
            time.time() - start_time,
        )

        # Update global statistics
        self._update_global_stats(stats)

        return result

    def search_with_embeddings(
        self,
        initial_state,
        available_rules: list[dict[str, Any]],
        rule_embeddings: list[list[float]],
        goal_embedding: list[float],
    ) -> dict[str, Any]:
        """
        Search with pre-computed embeddings for efficiency.
        """
        # Store goal embedding for reuse
        self._cached_goal_embedding = goal_embedding

        result = self.search(initial_state, available_rules, rule_embeddings)

        # Clear cache
        self._cached_goal_embedding = None

        return result

    def _expand_node(
        self,
        node: SearchNode,
        context: list[str],
        available_rules: list[dict[str, Any]],
        rule_embeddings: list[list[float]],
        stats: SearchStatistics,
    ) -> list[SearchNode]:
        """
        Expand a node by finding applicable rules using embeddings.
        """
        # Embed current goal
        if hasattr(self, "_cached_goal_embedding"):
            goal_embedding = self._cached_goal_embedding
        else:
            goal_embedding = self.goal_embedder.embed_goal(node.goal, context)
            stats.embeddings_computed += 1

        # Find similar rules
        rule_scores = []
        for i, (rule, rule_emb) in enumerate(zip(available_rules, rule_embeddings, strict=False)):
            similarity = self.goal_embedder.compute_similarity(goal_embedding, rule_emb)

            if similarity >= self.similarity_threshold:
                # Adjust score based on various factors
                adjusted_score = self._compute_rule_score(rule, similarity, node)
                rule_scores.append((adjusted_score, i, similarity))

        # Sort by score and take top candidates
        rule_scores.sort(reverse=True)
        top_rules = rule_scores[: self.beam_width * 2]  # Consider more than beam width

        # Create child nodes
        children = []
        for score, rule_idx, similarity in top_rules:
            rule = available_rules[rule_idx]

            # Simulate rule application (in real implementation, would call Lean)
            new_goal = self._apply_rule_simulation(node.goal, rule)

            if new_goal != node.goal:  # Rule made progress
                child = SearchNode(
                    goal=new_goal,
                    applied_rules=node.applied_rules + [rule["name"]],
                    rule_sequence=node.rule_sequence
                    + [
                        {
                            "rule": rule["name"],
                            "confidence": score,
                            "similarity_score": similarity,
                        }
                    ],
                    score=node.score * score,  # Cumulative score
                    depth=node.depth + 1,
                    parent=node,
                )
                children.append(child)

        return children

    def _compute_rule_score(
        self, rule: dict[str, Any], similarity: float, node: SearchNode
    ) -> float:
        """
        Compute adjusted score for rule application.

        Considers:
        - Semantic similarity
        - Rule priority/frequency
        - Depth penalty
        - Repetition penalty
        """
        score = similarity

        # Priority bonus
        priority = rule.get("priority", 1000)
        priority_factor = 1.0 / (1 + priority / 1000)
        score *= 1 + priority_factor * 0.2

        # Depth penalty (prefer shorter proofs)
        depth_penalty = 0.95**node.depth
        score *= depth_penalty

        # Repetition penalty
        if rule["name"] in node.applied_rules:
            score *= 0.7

        # Learned adjustments (would come from learning system)
        learned_factor = rule.get("learned_score", 1.0)
        score *= learned_factor

        return score

    def _apply_rule_simulation(self, goal: str, rule: dict[str, Any]) -> str:
        """
        Simulate rule application for demonstration.

        In production, this would call Lean to actually apply the rule.
        """
        # Simple pattern replacement simulation
        lhs = rule.get("lhs", "")
        rhs = rule.get("rhs", "")

        if lhs and lhs in goal:
            # Direct pattern match
            return goal.replace(lhs, rhs, 1)

        # Probabilistic simplification based on rule type
        import random

        random.seed(hash(goal + rule["name"]))

        if random.random() < 0.3:  # 30% chance of progress
            # Simulate simplification
            if "algebra" in rule["name"]:
                return goal.replace(" + 0", "").replace("0 + ", "")
            elif "list" in rule["name"]:
                return goal.replace("++ []", "").replace("[] ++", "")
            elif "logic" in rule["name"]:
                return goal.replace("∧ True", "").replace("True ∧", "")

        return goal

    def _is_terminal(self, node: SearchNode) -> bool:
        """
        Check if a node represents a terminal (simplified) state.
        """
        goal = node.goal

        # Terminal conditions
        if goal in ["True", "rfl", "trivial"]:
            return True

        # Depth limit
        if node.depth >= self.max_depth:
            return True

        # No operators left
        operators = ["+", "-", "*", "/", "++", "∧", "∨", "→"]
        if not any(op in goal for op in operators):
            return True

        return False

    def _build_result(
        self, final_node: SearchNode, stats: SearchStatistics, time_taken: float
    ) -> dict[str, Any]:
        """Build search result dictionary."""
        # Reconstruct proof path
        path = []
        node = final_node
        while node.parent is not None:
            path.append(node)
            node = node.parent
        path.reverse()

        return {
            "success": self._is_terminal(final_node),
            "final_goal": final_node.goal,
            "applied_rules": final_node.applied_rules,
            "proof_steps": final_node.rule_sequence,
            "confidence": final_node.score,
            "depth": final_node.depth,
            "embeddings_computed": stats.embeddings_computed,
            "nodes_explored": stats.nodes_explored,
            "time_taken": time_taken,
            "path": [
                {
                    "goal": n.goal,
                    "rule": n.applied_rules[-1] if n.applied_rules else None,
                    "score": n.score,
                }
                for n in path
            ],
        }

    def _update_global_stats(self, stats: SearchStatistics):
        """Update global statistics."""
        self.global_stats["searches"] += 1
        self.global_stats["total_nodes_explored"] += stats.nodes_explored
        self.global_stats["total_nodes_pruned"] += stats.nodes_pruned
        self.global_stats["total_embeddings"] += stats.embeddings_computed
        self.global_stats["max_depth_seen"] = max(
            self.global_stats["max_depth_seen"], stats.max_depth_reached
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get search statistics."""
        stats = dict(self.global_stats)

        if stats["searches"] > 0:
            stats["avg_nodes_per_search"] = stats["total_nodes_explored"] / stats["searches"]
            stats["avg_embeddings_per_search"] = stats["total_embeddings"] / stats["searches"]
            stats["pruning_ratio"] = stats["total_nodes_pruned"] / (
                stats["total_nodes_explored"] + stats["total_nodes_pruned"]
            )

        return stats

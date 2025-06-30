"""
Self-learning system for SimpNG that improves from experience.

This module implements continuous learning from successful proofs,
allowing SimpNG to adapt and improve over time.
"""

import json
import logging
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .embeddings import GoalEmbedder, RuleEmbedder


@dataclass
class ProofExample:
    """A successful proof for learning."""

    initial_goal: str
    final_goal: str
    applied_rules: List[str]
    rule_sequence: List[Dict[str, Any]]
    success_score: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class RuleLearning:
    """Learned statistics about a rule."""

    name: str
    success_count: int = 0
    failure_count: int = 0
    total_time: float = 0.0
    avg_position: float = 0.0  # Average position in successful proofs
    co_occurrence: Dict[str, int] = field(default_factory=dict)
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def avg_score(self) -> float:
        """Average recent score."""
        return (
            sum(self.recent_scores) / len(self.recent_scores)
            if self.recent_scores
            else 0.5
        )

    def update_score(self, score: float):
        """Update with new score."""
        self.recent_scores.append(score)


class SelfLearningSystem:
    """
    Implements continuous learning from proof experiences.

    Key features:
    1. Learn rule effectiveness from successful proofs
    2. Discover rule combinations that work well together
    3. Adapt search heuristics based on domain
    4. Transfer learning across similar problems
    """

    def __init__(
        self,
        rule_embedder: RuleEmbedder,
        goal_embedder: GoalEmbedder,
        learning_rate: float = 0.01,
        memory_size: int = 10000,
    ):
        self.rule_embedder = rule_embedder
        self.goal_embedder = goal_embedder
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.logger = logging.getLogger(__name__)

        # Learning data
        self.rule_stats = {}  # name -> RuleLearning
        self.proof_memory = deque(maxlen=memory_size)
        self.pattern_library = {}  # goal_pattern -> effective_rules

        # Meta-learning
        self.domain_models = {}  # domain -> specialized parameters
        self.transfer_knowledge = {}  # similarity mappings

        # Statistics
        self.stats = defaultdict(int)

    def learn_from_proof(
        self, initial_goal: str, final_goal: str, proof_steps: List[Dict[str, Any]]
    ):
        """
        Learn from a successful proof.

        Updates:
        - Rule effectiveness scores
        - Rule co-occurrence patterns
        - Goal pattern library
        """
        self.stats["proofs_learned"] += 1

        # Create proof example
        example = ProofExample(
            initial_goal=initial_goal,
            final_goal=final_goal,
            applied_rules=[step["rule"] for step in proof_steps],
            rule_sequence=proof_steps,
            success_score=self._compute_proof_score(proof_steps),
        )

        # Store in memory
        self.proof_memory.append(example)

        # Update rule statistics
        self._update_rule_stats(example)

        # Learn patterns
        self._learn_patterns(example)

        # Update domain model
        domain = self._identify_domain(initial_goal)
        self._update_domain_model(domain, example)

    def suggest_rules(
        self, goal: str, context: List[str], available_rules: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Suggest rules based on learned knowledge.

        Returns rules with learned confidence scores.
        """
        # Find similar past proofs
        similar_proofs = self._find_similar_proofs(goal, context)

        # Get pattern-based suggestions
        pattern_rules = self._get_pattern_suggestions(goal)

        # Score each available rule
        scored_rules = []
        for rule in available_rules:
            score = self._score_rule_for_goal(
                rule, goal, context, similar_proofs, pattern_rules
            )
            scored_rules.append((rule, score))

        # Sort by score
        scored_rules.sort(key=lambda x: x[1], reverse=True)

        return scored_rules

    def _compute_proof_score(self, proof_steps: List[Dict[str, Any]]) -> float:
        """
        Compute quality score for a proof.

        Considers:
        - Length (shorter is better)
        - Confidence of steps
        - Simplicity of final result
        """
        if not proof_steps:
            return 0.0

        # Length penalty
        length_score = 1.0 / (1 + len(proof_steps) * 0.1)

        # Average confidence
        avg_confidence = sum(step.get("confidence", 0.5) for step in proof_steps) / len(
            proof_steps
        )

        # Combined score
        return length_score * avg_confidence

    def _update_rule_stats(self, example: ProofExample):
        """Update statistics for rules used in proof."""
        # Update individual rule stats
        for i, rule_name in enumerate(example.applied_rules):
            if rule_name not in self.rule_stats:
                self.rule_stats[rule_name] = RuleLearning(name=rule_name)

            stats = self.rule_stats[rule_name]
            stats.success_count += 1
            stats.avg_position = (
                stats.avg_position * 0.9 + i * 0.1  # Exponential average
            )
            stats.update_score(example.success_score)

            # Update co-occurrence
            for other_rule in example.applied_rules:
                if other_rule != rule_name:
                    stats.co_occurrence[other_rule] = (
                        stats.co_occurrence.get(other_rule, 0) + 1
                    )

    def _learn_patterns(self, example: ProofExample):
        """Learn goal patterns and effective rules."""
        # Extract goal pattern
        pattern = self._extract_goal_pattern(example.initial_goal)

        if pattern not in self.pattern_library:
            self.pattern_library[pattern] = defaultdict(float)

        # Update pattern library
        for rule in example.applied_rules:
            self.pattern_library[pattern][rule] += example.success_score

    def _extract_goal_pattern(self, goal: str) -> str:
        """
        Extract abstract pattern from goal.

        Replaces specific values with placeholders.
        """
        import re

        # Replace numbers with placeholder
        pattern = re.sub(r"\b\d+\b", "N", goal)

        # Replace identifiers with placeholder
        pattern = re.sub(r"\b[a-z]\w*\b", "X", pattern)

        # Normalize whitespace
        pattern = " ".join(pattern.split())

        return pattern

    def _identify_domain(self, goal: str) -> str:
        """Identify mathematical domain of goal."""
        goal_lower = goal.lower()

        if any(term in goal_lower for term in ["group", "ring", "field"]):
            return "algebra"
        elif any(term in goal_lower for term in ["list", "array", "seq"]):
            return "data_structures"
        elif any(term in goal_lower for term in ["nat", "int", "real"]):
            return "arithmetic"
        elif any(term in goal_lower for term in ["∧", "∨", "¬", "→"]):
            return "logic"
        else:
            return "general"

    def _update_domain_model(self, domain: str, example: ProofExample):
        """Update domain-specific model."""
        if domain not in self.domain_models:
            self.domain_models[domain] = {
                "rule_weights": defaultdict(float),
                "proof_count": 0,
            }

        model = self.domain_models[domain]
        model["proof_count"] += 1

        # Update rule weights for domain
        for rule in example.applied_rules:
            model["rule_weights"][rule] += example.success_score

    def _find_similar_proofs(
        self, goal: str, context: List[str], max_results: int = 10
    ) -> List[ProofExample]:
        """Find similar proofs from memory."""
        if not self.proof_memory:
            return []

        # Embed query goal
        goal_embedding = self.goal_embedder.embed_goal(goal, context)

        # Score all proofs by similarity
        scored_proofs = []
        for proof in self.proof_memory:
            # Embed proof's initial goal
            proof_embedding = self.goal_embedder.embed_goal(
                proof.initial_goal, []  # No context stored
            )

            similarity = self.goal_embedder.compute_similarity(
                goal_embedding, proof_embedding
            )

            scored_proofs.append((similarity, proof))

        # Sort and return top results
        scored_proofs.sort(reverse=True)
        return [proof for _, proof in scored_proofs[:max_results]]

    def _get_pattern_suggestions(self, goal: str) -> Dict[str, float]:
        """Get rule suggestions based on goal pattern."""
        pattern = self._extract_goal_pattern(goal)

        if pattern in self.pattern_library:
            return dict(self.pattern_library[pattern])
        else:
            return {}

    def _score_rule_for_goal(
        self,
        rule: Dict[str, Any],
        goal: str,
        context: List[str],
        similar_proofs: List[ProofExample],
        pattern_rules: Dict[str, float],
    ) -> float:
        """
        Score a rule for a specific goal using learned knowledge.
        """
        rule_name = rule["name"]
        score = 0.5  # Base score

        # Factor 1: General rule statistics
        if rule_name in self.rule_stats:
            stats = self.rule_stats[rule_name]
            score *= 1 + stats.success_rate * 0.3
            score *= 1 + stats.avg_score * 0.2

        # Factor 2: Pattern-based score
        if rule_name in pattern_rules:
            pattern_score = pattern_rules[rule_name]
            score *= 1 + pattern_score * 0.4

        # Factor 3: Similar proof evidence
        similar_uses = sum(
            1 for proof in similar_proofs if rule_name in proof.applied_rules
        )
        if similar_uses > 0:
            score *= 1 + similar_uses / len(similar_proofs) * 0.5

        # Factor 4: Domain-specific weight
        domain = self._identify_domain(goal)
        if domain in self.domain_models:
            model = self.domain_models[domain]
            if rule_name in model["rule_weights"]:
                domain_weight = model["rule_weights"][rule_name]
                score *= 1 + domain_weight / model["proof_count"] * 0.3

        # Factor 5: Co-occurrence bonus
        if similar_proofs and rule_name in self.rule_stats:
            stats = self.rule_stats[rule_name]
            for proof in similar_proofs[:3]:  # Top 3 similar
                for used_rule in proof.applied_rules:
                    if used_rule in stats.co_occurrence:
                        score *= 1.1  # Bonus for co-occurrence

        return min(score, 2.0)  # Cap maximum score

    def train_on_corpus(self, corpus_path: Path):
        """
        Train on a corpus of proofs.

        Expects JSONL format with proof examples.
        """
        self.logger.info(f"Training on corpus: {corpus_path}")

        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        with open(corpus_path) as f:
            for line in f:
                if line.strip():
                    proof_data = json.loads(line)
                    self.learn_from_proof(
                        initial_goal=proof_data["initial_goal"],
                        final_goal=proof_data["final_goal"],
                        proof_steps=proof_data["proof_steps"],
                    )

        self.logger.info(f"Trained on {self.stats['proofs_learned']} proofs")

    def save_model(self, model_path: Path):
        """Save learned model."""
        self.logger.info(f"Saving model to: {model_path}")

        model_data = {
            "rule_stats": {
                name: {
                    "success_count": stats.success_count,
                    "failure_count": stats.failure_count,
                    "total_time": stats.total_time,
                    "avg_position": stats.avg_position,
                    "co_occurrence": dict(stats.co_occurrence),
                    "recent_scores": list(stats.recent_scores),
                }
                for name, stats in self.rule_stats.items()
            },
            "pattern_library": {
                pattern: dict(rules) for pattern, rules in self.pattern_library.items()
            },
            "domain_models": self.domain_models,
            "stats": dict(self.stats),
        }

        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, model_path: Path):
        """Load learned model."""
        self.logger.info(f"Loading model from: {model_path}")

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Restore rule stats
        self.rule_stats = {}
        for name, data in model_data["rule_stats"].items():
            stats = RuleLearning(name=name)
            stats.success_count = data["success_count"]
            stats.failure_count = data["failure_count"]
            stats.total_time = data["total_time"]
            stats.avg_position = data["avg_position"]
            stats.co_occurrence = data["co_occurrence"]
            stats.recent_scores = deque(data["recent_scores"], maxlen=100)
            self.rule_stats[name] = stats

        self.pattern_library = model_data["pattern_library"]
        self.domain_models = model_data["domain_models"]
        self.stats = defaultdict(int, model_data["stats"])

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics."""
        stats = dict(self.stats)

        stats["rules_learned"] = len(self.rule_stats)
        stats["patterns_learned"] = len(self.pattern_library)
        stats["domains_specialized"] = len(self.domain_models)
        stats["memory_usage"] = len(self.proof_memory)

        # Top rules by success rate
        if self.rule_stats:
            top_rules = sorted(
                self.rule_stats.items(), key=lambda x: x[1].success_rate, reverse=True
            )[:5]
            stats["top_rules"] = [
                {
                    "name": name,
                    "success_rate": stats.success_rate,
                    "uses": stats.success_count,
                }
                for name, stats in top_rules
            ]

        return stats

    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {"stats": dict(self.stats), "memory_size": len(self.proof_memory)}

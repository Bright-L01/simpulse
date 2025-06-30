"""
Core SimpNG engine - orchestrates the revolutionary simplification approach.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .embeddings import GoalEmbedder, RuleEmbedder
from .learning import SelfLearningSystem
from .search import NeuralProofSearch


@dataclass
class SimpNGConfig:
    """Configuration for SimpNG engine."""

    embedding_dim: int = 768
    max_search_depth: int = 10
    beam_width: int = 5
    learning_rate: float = 0.001
    cache_embeddings: bool = True
    enable_self_learning: bool = True
    proof_corpus_path: Optional[Path] = None
    model_checkpoint_path: Optional[Path] = None


@dataclass
class ProofState:
    """Represents a proof state during simplification."""

    goal: str
    context: List[str]
    applied_rules: List[str] = field(default_factory=list)
    score: float = 0.0
    depth: int = 0


@dataclass
class SimplificationResult:
    """Result of SimpNG simplification."""

    simplified_goal: str
    applied_rules: List[str]
    proof_steps: List[Dict[str, Any]]
    confidence: float
    time_taken: float
    embeddings_used: int


class SimpNGEngine:
    """
    Revolutionary simplification engine using deep learning.

    This represents a complete departure from traditional rule-based
    simplification, using transformer embeddings and neural search.
    """

    def __init__(self, config: Optional[SimpNGConfig] = None):
        """Initialize SimpNG with configuration."""
        self.config = config or SimpNGConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.rule_embedder = RuleEmbedder(
            embedding_dim=self.config.embedding_dim,
            cache_enabled=self.config.cache_embeddings,
        )

        self.goal_embedder = GoalEmbedder(
            embedding_dim=self.config.embedding_dim,
            cache_enabled=self.config.cache_embeddings,
        )

        self.proof_search = NeuralProofSearch(
            rule_embedder=self.rule_embedder,
            goal_embedder=self.goal_embedder,
            max_depth=self.config.max_search_depth,
            beam_width=self.config.beam_width,
        )

        self.learning_system = None
        if self.config.enable_self_learning:
            self.learning_system = SelfLearningSystem(
                rule_embedder=self.rule_embedder,
                goal_embedder=self.goal_embedder,
                learning_rate=self.config.learning_rate,
            )

        # Statistics
        self.stats = defaultdict(int)

    def simplify(
        self, goal: str, context: List[str], available_rules: List[Dict[str, Any]]
    ) -> SimplificationResult:
        """
        Simplify a goal using neural proof search.

        This is the main entry point that replaces traditional simp.
        """
        start_time = time.time()
        self.logger.info(f"SimpNG: Simplifying goal: {goal[:100]}...")

        # Create initial proof state
        initial_state = ProofState(goal=goal, context=context)

        # Embed available rules
        self.logger.debug("Embedding rules...")
        rule_embeddings = self.rule_embedder.embed_rules(available_rules)
        self.stats["rules_embedded"] += len(available_rules)

        # Run neural proof search
        self.logger.debug("Starting neural proof search...")
        search_result = self.proof_search.search(
            initial_state=initial_state,
            available_rules=available_rules,
            rule_embeddings=rule_embeddings,
        )

        # Extract best simplification
        simplified_goal = search_result["final_goal"]
        applied_rules = search_result["applied_rules"]
        proof_steps = search_result["proof_steps"]
        confidence = search_result["confidence"]

        # Update learning system if enabled
        if self.learning_system and search_result["success"]:
            self.logger.debug("Updating learning system...")
            self.learning_system.learn_from_proof(
                initial_goal=goal, final_goal=simplified_goal, proof_steps=proof_steps
            )

        # Record statistics
        time_taken = time.time() - start_time
        self.stats["simplifications"] += 1
        self.stats["total_time"] += time_taken
        self.stats["embeddings_used"] += search_result["embeddings_computed"]

        return SimplificationResult(
            simplified_goal=simplified_goal,
            applied_rules=applied_rules,
            proof_steps=proof_steps,
            confidence=confidence,
            time_taken=time_taken,
            embeddings_used=search_result["embeddings_computed"],
        )

    def batch_simplify(
        self,
        goals: List[str],
        context: List[str],
        available_rules: List[Dict[str, Any]],
    ) -> List[SimplificationResult]:
        """
        Simplify multiple goals efficiently using batched embeddings.
        """
        self.logger.info(f"SimpNG: Batch simplifying {len(goals)} goals")

        # Pre-compute rule embeddings once
        rule_embeddings = self.rule_embedder.embed_rules(available_rules)

        # Batch embed all goals
        goal_embeddings = self.goal_embedder.batch_embed_goals(goals)

        results = []
        for i, goal in enumerate(goals):
            # Use pre-computed embeddings
            initial_state = ProofState(goal=goal, context=context)

            search_result = self.proof_search.search_with_embeddings(
                initial_state=initial_state,
                available_rules=available_rules,
                rule_embeddings=rule_embeddings,
                goal_embedding=goal_embeddings[i],
            )

            results.append(
                SimplificationResult(
                    simplified_goal=search_result["final_goal"],
                    applied_rules=search_result["applied_rules"],
                    proof_steps=search_result["proof_steps"],
                    confidence=search_result["confidence"],
                    time_taken=search_result["time_taken"],
                    embeddings_used=search_result["embeddings_computed"],
                )
            )

        return results

    def train_on_corpus(self, proof_corpus_path: Path):
        """
        Train SimpNG on a corpus of successful proofs.
        """
        if not self.learning_system:
            raise ValueError("Learning system not enabled")

        self.logger.info(f"Training on proof corpus: {proof_corpus_path}")
        self.learning_system.train_on_corpus(proof_corpus_path)

    def save_checkpoint(self, checkpoint_path: Path):
        """Save model checkpoint."""
        self.logger.info(f"Saving checkpoint to: {checkpoint_path}")

        checkpoint = {
            "config": self.config.__dict__,
            "stats": dict(self.stats),
            "rule_embedder_state": self.rule_embedder.get_state(),
            "goal_embedder_state": self.goal_embedder.get_state(),
        }

        if self.learning_system:
            checkpoint["learning_system_state"] = self.learning_system.get_state()

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")

        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        self.stats = defaultdict(int, checkpoint["stats"])
        self.rule_embedder.load_state(checkpoint["rule_embedder_state"])
        self.goal_embedder.load_state(checkpoint["goal_embedder_state"])

        if self.learning_system and "learning_system_state" in checkpoint:
            self.learning_system.load_state(checkpoint["learning_system_state"])

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = dict(self.stats)

        if stats["simplifications"] > 0:
            stats["avg_time"] = stats["total_time"] / stats["simplifications"]
            stats["avg_embeddings"] = (
                stats["embeddings_used"] / stats["simplifications"]
            )

        # Add component statistics
        stats["rule_embedder"] = self.rule_embedder.get_statistics()
        stats["goal_embedder"] = self.goal_embedder.get_statistics()
        stats["proof_search"] = self.proof_search.get_statistics()

        if self.learning_system:
            stats["learning_system"] = self.learning_system.get_statistics()

        return stats

    def explain_simplification(self, result: SimplificationResult) -> Dict[str, Any]:
        """
        Provide interpretable explanation of simplification decisions.
        """
        explanation = {
            "summary": f"Applied {len(result.applied_rules)} rules with {result.confidence:.1%} confidence",
            "time_taken": f"{result.time_taken:.3f}s",
            "embeddings_used": result.embeddings_used,
            "steps": [],
        }

        for step in result.proof_steps:
            explanation["steps"].append(
                {
                    "rule": step["rule"],
                    "confidence": step["confidence"],
                    "similarity_score": step["similarity_score"],
                    "rationale": self._generate_rationale(step),
                }
            )

        return explanation

    def _generate_rationale(self, step: Dict[str, Any]) -> str:
        """Generate human-readable rationale for a proof step."""
        rule = step["rule"]
        confidence = step["confidence"]
        similarity = step["similarity_score"]

        if similarity > 0.9:
            match_quality = "excellent"
        elif similarity > 0.7:
            match_quality = "good"
        else:
            match_quality = "moderate"

        return (
            f"Selected rule '{rule}' with {match_quality} semantic match "
            f"(similarity: {similarity:.3f}, confidence: {confidence:.1%})"
        )

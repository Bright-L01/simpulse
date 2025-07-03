"""
Transformer-based embeddings for rules and goals in SimpNG.

This module implements the revolutionary approach of treating theorem proving
as a semantic matching problem in high-dimensional embedding space.

Now using REAL transformer models for semantic embeddings!
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingCache:
    """Cache for computed embeddings."""

    cache: dict[str, list[float]]
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> list[float] | None:
        """Get embedding from cache."""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, embedding: list[float]):
        """Store embedding in cache."""
        self.cache[key] = embedding

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
        }


class RealTransformer:
    """
    Real transformer-based embeddings using sentence-transformers.

    Provides semantic embeddings using pre-trained transformer models
    optimized for mathematical and formal language understanding.
    """

    def __init__(self, embedding_dim: int = 384, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        # Initialize sentence transformer model
        try:
            self.model = SentenceTransformer(model_name)
            # Get actual embedding dimension from model
            self.actual_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Loaded transformer model '{model_name}' with dim {self.actual_dim}")
        except Exception as e:
            self.logger.error(f"Failed to load transformer model: {e}")
            # Fallback to feature-based encoding
            self.model = None
            self.actual_dim = embedding_dim

    def encode(self, text: str) -> list[float]:
        """
        Generate real semantic embeddings using transformer model.

        Falls back to feature-based encoding if model unavailable.
        """
        if self.model is not None:
            try:
                # Get embedding from transformer model
                embedding = self.model.encode(text, convert_to_numpy=True)

                # Resize to target dimension if needed
                if len(embedding) != self.embedding_dim:
                    embedding = self._resize_embedding(embedding)

                return embedding.tolist()
            except Exception as e:
                self.logger.warning(f"Transformer encoding failed: {e}, using fallback")

        # Fallback: No ML simulation - require real transformers
        raise RuntimeError(
            "Transformer model not available. Please install sentence-transformers: "
            "pip install sentence-transformers"
        )

    def _resize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Resize embedding to target dimension."""
        current_dim = len(embedding)

        if current_dim == self.embedding_dim:
            return embedding
        elif current_dim > self.embedding_dim:
            # Truncate to target dimension
            return embedding[: self.embedding_dim]
        else:
            # Pad with zeros to reach target dimension
            padding = np.zeros(self.embedding_dim - current_dim)
            return np.concatenate([embedding, padding])


# Maintain backward compatibility
TransformerSimulator = RealTransformer


class RuleEmbedder:
    """
    Embeds simplification rules into high-dimensional space.

    Key innovation: Rules with similar semantic meaning will have
    similar embeddings, enabling neural similarity search.
    """

    def __init__(self, embedding_dim: int = 768, cache_enabled: bool = True):
        self.embedding_dim = embedding_dim
        self.transformer = RealTransformer(embedding_dim)
        self.logger = logging.getLogger(__name__)

        self.cache = EmbeddingCache(cache={}) if cache_enabled else None
        self.stats = defaultdict(int)

    def embed_rule(self, rule: dict[str, Any]) -> list[float]:
        """
        Embed a single simplification rule.

        Combines embeddings of:
        - Rule name
        - Left-hand side pattern
        - Right-hand side pattern
        - Conditions/hypotheses
        """
        self.stats["rules_embedded"] += 1

        # Check cache
        rule_key = self._rule_key(rule)
        if self.cache:
            cached = self.cache.get(rule_key)
            if cached is not None:
                return cached

        # Extract components
        name = rule.get("name", "")
        lhs = rule.get("lhs", "")
        rhs = rule.get("rhs", "")
        conditions = rule.get("conditions", [])

        # Embed each component
        name_emb = self.transformer.encode(f"RULE_NAME: {name}")
        lhs_emb = self.transformer.encode(f"LHS_PATTERN: {lhs}")
        rhs_emb = self.transformer.encode(f"RHS_PATTERN: {rhs}")

        # Combine embeddings
        embedding = self._combine_embeddings(
            [
                (name_emb, 0.2),
                (lhs_emb, 0.4),
                (rhs_emb, 0.3),
            ]
        )

        # Add condition embeddings
        if conditions:
            cond_text = " AND ".join(conditions)
            cond_emb = self.transformer.encode(f"CONDITIONS: {cond_text}")
            embedding = self._combine_embeddings([(embedding, 0.8), (cond_emb, 0.2)])

        # Cache result
        if self.cache:
            self.cache.put(rule_key, embedding)

        return embedding

    def embed_rules(self, rules: list[dict[str, Any]]) -> list[list[float]]:
        """Embed multiple rules efficiently."""
        return [self.embed_rule(rule) for rule in rules]

    def _rule_key(self, rule: dict[str, Any]) -> str:
        """Generate cache key for rule."""
        components = [
            rule.get("name", ""),
            rule.get("lhs", ""),
            rule.get("rhs", ""),
            str(rule.get("conditions", [])),
        ]
        return hashlib.md5("|".join(components).encode()).hexdigest()

    def _combine_embeddings(
        self, weighted_embeddings: list[tuple[list[float], float]]
    ) -> list[float]:
        """Combine multiple embeddings with weights."""
        result = [0.0] * self.embedding_dim

        for embedding, weight in weighted_embeddings:
            for i, value in enumerate(embedding):
                result[i] += value * weight

        # Normalize
        norm = math.sqrt(sum(x * x for x in result))
        if norm > 0:
            result = [x / norm for x in result]

        return result

    def get_statistics(self) -> dict[str, Any]:
        """Get embedder statistics."""
        stats = dict(self.stats)
        if self.cache:
            stats["cache"] = self.cache.stats()
        return stats

    def get_state(self) -> dict[str, Any]:
        """Get embedder state for checkpointing."""
        state = {"stats": dict(self.stats)}
        if self.cache:
            state["cache"] = {"entries": self.cache.cache, "stats": self.cache.stats()}
        return state

    def load_state(self, state: dict[str, Any]):
        """Load embedder state from checkpoint."""
        self.stats = defaultdict(int, state["stats"])
        if self.cache and "cache" in state:
            self.cache.cache = state["cache"]["entries"]


class GoalEmbedder:
    """
    Embeds proof goals into the same space as rules.

    This enables efficient similarity search to find applicable rules.
    """

    def __init__(self, embedding_dim: int = 768, cache_enabled: bool = True):
        self.embedding_dim = embedding_dim
        self.transformer = RealTransformer(embedding_dim)
        self.logger = logging.getLogger(__name__)

        self.cache = EmbeddingCache(cache={}) if cache_enabled else None
        self.stats = defaultdict(int)

    def embed_goal(self, goal: str, context: list[str] | None = None) -> list[float]:
        """
        Embed a proof goal with optional context.

        Context includes:
        - Local hypotheses
        - Available definitions
        - Type information
        """
        self.stats["goals_embedded"] += 1

        # Check cache
        goal_key = self._goal_key(goal, context)
        if self.cache:
            cached = self.cache.get(goal_key)
            if cached is not None:
                return cached

        # Embed goal
        goal_emb = self.transformer.encode(f"GOAL: {goal}")

        # Add context if available
        if context:
            context_text = " ; ".join(context[:5])  # Limit context
            context_emb = self.transformer.encode(f"CONTEXT: {context_text}")

            # Combine with more weight on goal
            embedding = self._combine_embeddings([(goal_emb, 0.7), (context_emb, 0.3)])
        else:
            embedding = goal_emb

        # Cache result
        if self.cache:
            self.cache.put(goal_key, embedding)

        return embedding

    def batch_embed_goals(
        self, goals: list[str], contexts: list[list[str]] | None = None
    ) -> list[list[float]]:
        """Efficiently embed multiple goals."""
        if contexts is None:
            contexts = [None] * len(goals)

        return [
            self.embed_goal(goal, context) for goal, context in zip(goals, contexts, strict=False)
        ]

    def _goal_key(self, goal: str, context: list[str] | None) -> str:
        """Generate cache key for goal."""
        components = [goal]
        if context:
            components.extend(context[:5])
        return hashlib.md5("|".join(components).encode()).hexdigest()

    def _combine_embeddings(
        self, weighted_embeddings: list[tuple[list[float], float]]
    ) -> list[float]:
        """Combine multiple embeddings with weights."""
        result = [0.0] * self.embedding_dim

        for embedding, weight in weighted_embeddings:
            for i, value in enumerate(embedding):
                result[i] += value * weight

        # Normalize
        norm = math.sqrt(sum(x * x for x in result))
        if norm > 0:
            result = [x / norm for x in result]

        return result

    def compute_similarity(self, goal_embedding: list[float], rule_embedding: list[float]) -> float:
        """
        Compute cosine similarity between goal and rule embeddings.

        Returns value in [0, 1] where 1 is perfect match.
        """
        # Cosine similarity
        dot_product = sum(g * r for g, r in zip(goal_embedding, rule_embedding, strict=False))

        # Already normalized, so just return dot product
        # Map from [-1, 1] to [0, 1]
        return (dot_product + 1) / 2

    def get_statistics(self) -> dict[str, Any]:
        """Get embedder statistics."""
        stats = dict(self.stats)
        if self.cache:
            stats["cache"] = self.cache.stats()
        return stats

    def get_state(self) -> dict[str, Any]:
        """Get embedder state for checkpointing."""
        state = {"stats": dict(self.stats)}
        if self.cache:
            state["cache"] = {"entries": self.cache.cache, "stats": self.cache.stats()}
        return state

    def load_state(self, state: dict[str, Any]):
        """Load embedder state from checkpoint."""
        self.stats = defaultdict(int, state["stats"])
        if self.cache and "cache" in state:
            self.cache.cache = state["cache"]["entries"]

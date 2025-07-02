"""
Feature extractor for Lean goals.

Analyzes goal structure and extracts features for ML-based tactic selection.
"""

import re
from dataclasses import dataclass


@dataclass
class GoalFeatures:
    """Features extracted from a Lean goal."""

    # Basic structure
    goal_type: str  # e.g., "equality", "inequality", "membership"
    depth: int  # AST depth
    num_subgoals: int

    # Operators and patterns
    operators: dict[str, int]  # Operator counts
    has_arithmetic: bool
    has_algebra: bool
    has_linear: bool
    has_logic: bool
    has_sets: bool

    # Complexity metrics
    num_variables: int
    num_constants: int
    num_functions: int
    max_nesting: int
    total_terms: int

    # Specific patterns
    is_equation: bool
    is_inequality: bool
    has_addition: bool
    has_multiplication: bool
    has_subtraction: bool
    has_division: bool
    has_exponentiation: bool
    has_modulo: bool

    # Type information
    involves_nat: bool
    involves_int: bool
    involves_real: bool
    involves_complex: bool
    involves_list: bool
    involves_set: bool

    # Historical hint (if available)
    previous_tactics: list[str] = None

    def to_vector(self) -> list[float]:
        """Convert features to numerical vector for ML."""
        vector = []

        # Binary features
        vector.extend(
            [
                float(self.has_arithmetic),
                float(self.has_algebra),
                float(self.has_linear),
                float(self.has_logic),
                float(self.has_sets),
                float(self.is_equation),
                float(self.is_inequality),
                float(self.has_addition),
                float(self.has_multiplication),
                float(self.has_subtraction),
                float(self.has_division),
                float(self.has_exponentiation),
                float(self.has_modulo),
                float(self.involves_nat),
                float(self.involves_int),
                float(self.involves_real),
                float(self.involves_complex),
                float(self.involves_list),
                float(self.involves_set),
            ]
        )

        # Numerical features (normalized)
        vector.extend(
            [
                min(self.depth / 10.0, 1.0),
                min(self.num_subgoals / 5.0, 1.0),
                min(self.num_variables / 20.0, 1.0),
                min(self.num_constants / 10.0, 1.0),
                min(self.num_functions / 15.0, 1.0),
                min(self.max_nesting / 8.0, 1.0),
                min(self.total_terms / 50.0, 1.0),
            ]
        )

        # Top operator frequencies
        top_operators = [
            "add",
            "mul",
            "sub",
            "div",
            "pow",
            "eq",
            "le",
            "lt",
            "and",
            "or",
        ]
        for op in top_operators:
            vector.append(min(self.operators.get(op, 0) / 5.0, 1.0))

        return vector


class LeanGoalParser:
    """Parser for Lean goal expressions."""

    # Common Lean operators and their categories
    ARITHMETIC_OPS = {
        "+",
        "-",
        "*",
        "/",
        "^",
        "%",
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "mod",
    }
    COMPARISON_OPS = {"=", "≠", "<", ">", "≤", "≥", "le", "lt", "eq", "ne"}
    LOGICAL_OPS = {"∧", "∨", "¬", "→", "↔", "and", "or", "not", "implies", "iff"}
    SET_OPS = {"∈", "∉", "⊆", "⊂", "∪", "∩", "mem", "subset", "union", "inter"}

    # Type patterns
    TYPE_PATTERNS = {
        "nat": re.compile(r"\bNat\b|\bℕ\b"),
        "int": re.compile(r"\bInt\b|\bℤ\b"),
        "real": re.compile(r"\bReal\b|\bℝ\b"),
        "complex": re.compile(r"\bComplex\b|\bℂ\b"),
        "list": re.compile(r"\bList\b"),
        "set": re.compile(r"\bSet\b"),
    }

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset parser state."""
        self.operators = {}
        self.variables = set()
        self.constants = set()
        self.functions = set()
        self.max_depth = 0
        self.term_count = 0

    def parse_goal(self, goal_text: str) -> GoalFeatures:
        """Parse a Lean goal and extract features."""
        self.reset()

        # Clean and tokenize
        tokens = self._tokenize(goal_text)

        # Analyze structure
        ast_info = self._analyze_structure(tokens)

        # Detect patterns
        patterns = self._detect_patterns(goal_text, tokens)

        # Detect types
        types = self._detect_types(goal_text)

        # Create features
        features = GoalFeatures(
            goal_type=self._classify_goal_type(goal_text, patterns),
            depth=ast_info["depth"],
            num_subgoals=ast_info["subgoals"],
            operators=self.operators,
            has_arithmetic=patterns["arithmetic"],
            has_algebra=patterns["algebra"],
            has_linear=patterns["linear"],
            has_logic=patterns["logic"],
            has_sets=patterns["sets"],
            num_variables=len(self.variables),
            num_constants=len(self.constants),
            num_functions=len(self.functions),
            max_nesting=ast_info["max_nesting"],
            total_terms=self.term_count,
            is_equation=patterns["is_equation"],
            is_inequality=patterns["is_inequality"],
            has_addition="+" in self.operators or "add" in self.operators,
            has_multiplication="*" in self.operators or "mul" in self.operators,
            has_subtraction="-" in self.operators or "sub" in self.operators,
            has_division="/" in self.operators or "div" in self.operators,
            has_exponentiation="^" in self.operators or "pow" in self.operators,
            has_modulo="%" in self.operators or "mod" in self.operators,
            involves_nat=types["nat"],
            involves_int=types["int"],
            involves_real=types["real"],
            involves_complex=types["complex"],
            involves_list=types["list"],
            involves_set=types["set"],
        )

        return features

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize Lean goal text."""
        # Simple tokenization - can be improved with proper Lean parser
        # Replace common symbols with spaced versions
        replacements = [
            ("≤", " ≤ "),
            ("≥", " ≥ "),
            ("≠", " ≠ "),
            ("∧", " ∧ "),
            ("∨", " ∨ "),
            ("¬", " ¬ "),
            ("→", " → "),
            ("↔", " ↔ "),
            ("∈", " ∈ "),
            ("∉", " ∉ "),
            ("⊆", " ⊆ "),
            ("⊂", " ⊂ "),
            ("∪", " ∪ "),
            ("∩", " ∩ "),
            ("(", " ( "),
            (")", " ) "),
            ("[", " [ "),
            ("]", " ] "),
            ("{", " { "),
            ("}", " } "),
            (",", " , "),
            ("+", " + "),
            ("-", " - "),
            ("*", " * "),
            ("/", " / "),
            ("^", " ^ "),
            ("=", " = "),
            ("<", " < "),
            (">", " > "),
        ]

        for old, new in replacements:
            text = text.replace(old, new)

        # Split and filter
        tokens = [t for t in text.split() if t]

        return tokens

    def _analyze_structure(self, tokens: list[str]) -> dict:
        """Analyze AST structure from tokens."""
        depth = 0
        max_depth = 0
        nesting_stack = []
        subgoal_count = 1  # At least one goal

        for token in tokens:
            if token in "([{":
                nesting_stack.append(token)
                depth += 1
                max_depth = max(max_depth, depth)
            elif token in ")]}":
                if nesting_stack:
                    nesting_stack.pop()
                    depth -= 1
            elif token == "⊢":  # Goal separator
                subgoal_count += 1

            # Track operators
            if token in self.ARITHMETIC_OPS | self.COMPARISON_OPS | self.LOGICAL_OPS | self.SET_OPS:
                self.operators[token] = self.operators.get(token, 0) + 1

            # Simple heuristic for variables/constants/functions
            if token.isalpha() and len(token) <= 3:
                self.variables.add(token)
            elif token.isdigit() or token in ["0", "1", "-1"]:
                self.constants.add(token)
            elif token.isalpha() and len(token) > 3:
                self.functions.add(token)

            self.term_count += 1

        return {
            "depth": len(nesting_stack),  # Remaining open parens indicate depth
            "max_nesting": max_depth,
            "subgoals": subgoal_count,
        }

    def _detect_patterns(self, text: str, tokens: list[str]) -> dict[str, bool]:
        """Detect specific patterns in the goal."""
        token_set = set(tokens)

        patterns = {
            "arithmetic": bool(token_set & self.ARITHMETIC_OPS),
            "algebra": any(t in text for t in ["ring", "field", "group", "monoid"]),
            "linear": any(t in text for t in ["linear", "vector", "matrix"])
            or self._is_linear_expression(tokens),
            "logic": bool(token_set & self.LOGICAL_OPS),
            "sets": bool(token_set & self.SET_OPS),
            "is_equation": "=" in tokens or "eq" in tokens,
            "is_inequality": any(op in tokens for op in ["<", ">", "≤", "≥", "le", "lt"]),
        }

        return patterns

    def _is_linear_expression(self, tokens: list[str]) -> bool:
        """Check if expression appears to be linear."""
        # Simple heuristic: has addition/subtraction but no multiplication between variables
        has_add_sub = any(op in tokens for op in ["+", "-", "add", "sub"])

        # Check for non-linear patterns
        for i, token in enumerate(tokens[:-1]):
            if token in self.variables and tokens[i + 1] in ["*", "mul"] and i + 2 < len(tokens):
                if tokens[i + 2] in self.variables:
                    return False  # Variable * Variable = non-linear

        return has_add_sub

    def _detect_types(self, text: str) -> dict[str, bool]:
        """Detect type information in the goal."""
        types = {}

        for type_name, pattern in self.TYPE_PATTERNS.items():
            types[type_name] = bool(pattern.search(text))

        return types

    def _classify_goal_type(self, text: str, patterns: dict[str, bool]) -> str:
        """Classify the overall goal type."""
        if patterns["is_equation"]:
            if patterns["linear"] and not patterns["algebra"]:
                return "linear_equation"
            elif patterns["algebra"]:
                return "algebraic_equation"
            else:
                return "equation"
        elif patterns["is_inequality"]:
            if patterns["linear"]:
                return "linear_inequality"
            else:
                return "inequality"
        elif patterns["logic"]:
            return "logical"
        elif patterns["sets"]:
            return "set_theory"
        else:
            return "general"


class FeatureCache:
    """Cache for computed goal features."""

    def __init__(self, max_size: int = 1000):
        self.cache: dict[str, GoalFeatures] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, goal_text: str) -> GoalFeatures | None:
        """Get cached features if available."""
        if goal_text in self.cache:
            self.hits += 1
            return self.cache[goal_text]
        self.misses += 1
        return None

    def put(self, goal_text: str, features: GoalFeatures):
        """Cache computed features."""
        if len(self.cache) >= self.max_size:
            # Simple FIFO eviction
            oldest = next(iter(self.cache))
            del self.cache[oldest]

        self.cache[goal_text] = features

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def extract_features(goal_text: str, cache: FeatureCache | None = None) -> GoalFeatures:
    """Main entry point for feature extraction."""
    # Check cache
    if cache:
        cached = cache.get(goal_text)
        if cached:
            return cached

    # Parse and extract
    parser = LeanGoalParser()
    features = parser.parse_goal(goal_text)

    # Cache result
    if cache:
        cache.put(goal_text, features)

    return features


if __name__ == "__main__":
    # Example usage
    test_goals = [
        "⊢ ∀ x y : ℕ, x + y = y + x",
        "⊢ ∀ a b c : ℝ, a * (b + c) = a * b + a * c",
        "⊢ ∀ x : ℤ, x < x + 1",
        "⊢ ∀ A B : Set α, A ∪ B = B ∪ A",
        "⊢ ∀ p q : Prop, p ∧ q → p",
    ]

    parser = LeanGoalParser()

    for goal in test_goals:
        print(f"\nGoal: {goal}")
        features = parser.parse_goal(goal)
        print(f"Type: {features.goal_type}")
        print(
            f"Features: arithmetic={features.has_arithmetic}, "
            f"linear={features.has_linear}, "
            f"logic={features.has_logic}"
        )
        print(
            f"Complexity: depth={features.depth}, "
            f"terms={features.total_terms}, "
            f"variables={features.num_variables}"
        )
        print(f"Vector: {features.to_vector()[:10]}...")  # First 10 features

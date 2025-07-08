#!/usr/bin/env python3
"""
Sophisticated Pattern Analyzer - Advanced AST-based pattern detection

Based on research from:
- Discrimination trees in theorem provers (McCune 1992)
- AST edit distance for code similarity (arXiv:2404.08817)
- Tree pattern matching (Hoffmann & O'Donnell 1982)
- Syntactic pattern analysis in formal verification

This analyzer goes beyond simple regex counting to provide deep structural
analysis of Lean 4 proof patterns.
"""

import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class NodeType(Enum):
    """AST node types for Lean 4 patterns"""

    THEOREM = "theorem"
    LEMMA = "lemma"
    DEFINITION = "definition"
    TACTIC = "tactic"
    EXPRESSION = "expression"
    IDENTIFIER = "identifier"
    OPERATOR = "operator"
    LITERAL = "literal"
    QUANTIFIER = "quantifier"
    IMPLICATION = "implication"
    APPLICATION = "application"
    TYPE = "type"
    IDENTITY_PATTERN = "identity_pattern"  # For identity patterns
    LIST_PATTERN = "list_pattern"  # For list operations
    UNKNOWN = "unknown"


@dataclass
class ASTNode:
    """Representation of an AST node with structural information"""

    node_type: NodeType
    value: str
    depth: int
    children: List["ASTNode"] = field(default_factory=list)
    parent: Optional["ASTNode"] = None
    position: int = 0  # Position among siblings

    @property
    def branching_factor(self) -> int:
        """Number of children this node has"""
        return len(self.children)

    @property
    def subtree_size(self) -> int:
        """Total number of nodes in this subtree"""
        return 1 + sum(child.subtree_size for child in self.children)

    @property
    def max_depth(self) -> int:
        """Maximum depth of this subtree"""
        if not self.children:
            return self.depth
        return max(child.max_depth for child in self.children)

    def to_string(self, prefix="") -> str:
        """Convert subtree to string representation"""
        result = f"{prefix}{self.node_type.value}:{self.value}\n"
        for i, child in enumerate(self.children):
            is_last = i == len(self.children) - 1
            child_prefix = prefix + ("└── " if is_last else "├── ")
            next_prefix = prefix + ("    " if is_last else "│   ")
            result += child.to_string(next_prefix)
        return result


@dataclass
class PatternFingerprint:
    """Multi-dimensional fingerprint of a pattern"""

    # Structural features
    depth_distribution: List[int]  # Nodes at each depth level
    branching_distribution: List[int]  # Distribution of branching factors
    node_type_distribution: Dict[NodeType, int]

    # Complexity metrics
    avg_depth: float
    max_depth: int
    avg_branching_factor: float
    total_nodes: int

    # Pattern-specific features
    identity_pattern_count: int
    nested_pattern_depth: int
    operator_diversity: float  # Number of unique operators / total operators
    quantifier_nesting: int

    # Edit distance features
    structural_hash: str  # Hash of tree structure for fast comparison
    canonical_form: str  # Normalized representation

    def similarity_to(self, other: "PatternFingerprint") -> float:
        """Calculate similarity score between two fingerprints (0-1)"""
        # Structural similarity
        depth_sim = 1 - abs(self.avg_depth - other.avg_depth) / max(
            self.avg_depth, other.avg_depth, 1
        )
        branch_sim = 1 - abs(self.avg_branching_factor - other.avg_branching_factor) / max(
            self.avg_branching_factor, other.avg_branching_factor, 1
        )

        # Node type distribution similarity (cosine similarity)
        all_types = set(self.node_type_distribution.keys()) | set(
            other.node_type_distribution.keys()
        )
        vec1 = [self.node_type_distribution.get(t, 0) for t in all_types]
        vec2 = [other.node_type_distribution.get(t, 0) for t in all_types]

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        type_sim = dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0

        # Pattern similarity
        pattern_sim = 1 - abs(self.identity_pattern_count - other.identity_pattern_count) / max(
            self.identity_pattern_count, other.identity_pattern_count, 1
        )

        # Weighted combination
        return 0.3 * depth_sim + 0.3 * branch_sim + 0.3 * type_sim + 0.1 * pattern_sim


@dataclass
class StructuralComplexity:
    """Comprehensive structural complexity metrics"""

    cyclomatic_complexity: int  # Number of independent paths
    halstead_volume: float  # Information content metric
    maintainability_index: float  # Composite metric (0-100)
    cognitive_complexity: int  # How hard to understand
    nesting_complexity: float  # Average nesting depth
    pattern_diversity: float  # Uniqueness of patterns (0-1)


class LeanASTParser:
    """Parser for Lean 4 syntax into AST representation"""

    def __init__(self):
        # Pattern matchers for different Lean constructs
        self.patterns = {
            "theorem": re.compile(
                r"(?:theorem|lemma)\s+(\w+).*?:=\s*(.+?)(?=theorem|lemma|def|example|$)",
                re.MULTILINE | re.DOTALL,
            ),
            "definition": re.compile(
                r"def\s+(\w+).*?:=\s*(.+?)(?=theorem|lemma|def|example|$)", re.MULTILINE | re.DOTALL
            ),
            "tactic": re.compile(r"by\s+(\w+)(?:\s+\[([^\]]+)\])?"),
            "operator": re.compile(r"([+\-*/∧∨¬→↔∀∃∈∉⊆⊇∪∩]|::|==|!=|<=|>=|<|>|\+\+)"),
            "quantifier": re.compile(r"(∀|∃|λ)\s*(\w+)"),
            "identifier": re.compile(r"\b([a-zA-Z_]\w*)\b"),
            "literal": re.compile(r"\b(\d+|True|False|nil)\b"),
        }

    def parse_expression(self, expr: str, depth: int = 0) -> ASTNode:
        """Parse a Lean expression into an AST"""
        expr = expr.strip()

        # Handle tactics
        tactic_match = self.patterns["tactic"].match(expr)
        if tactic_match:
            node = ASTNode(NodeType.TACTIC, tactic_match.group(1), depth)
            if tactic_match.group(2):  # Tactic arguments
                args = tactic_match.group(2).split(",")
                for i, arg in enumerate(args):
                    child = self.parse_expression(arg.strip(), depth + 1)
                    child.position = i
                    child.parent = node
                    node.children.append(child)
            return node

        # Handle operators (binary)
        for op_match in self.patterns["operator"].finditer(expr):
            op = op_match.group(1)
            op_pos = op_match.start()

            # Split expression at operator
            left = expr[:op_pos].strip()
            right = expr[op_pos + len(op) :].strip()

            if left and right:  # Binary operator
                node = ASTNode(NodeType.OPERATOR, op, depth)
                left_child = self.parse_expression(left, depth + 1)
                right_child = self.parse_expression(right, depth + 1)
                left_child.position = 0
                right_child.position = 1
                left_child.parent = node
                right_child.parent = node
                node.children = [left_child, right_child]
                return node

        # Handle quantifiers
        quant_match = self.patterns["quantifier"].match(expr)
        if quant_match:
            node = ASTNode(NodeType.QUANTIFIER, quant_match.group(1), depth)
            var_node = ASTNode(NodeType.IDENTIFIER, quant_match.group(2), depth + 1)
            var_node.parent = node
            var_node.position = 0
            node.children.append(var_node)

            # Parse the body
            body = expr[quant_match.end() :].strip()
            if body.startswith(":"):
                body = body[1:].strip()
            if body:
                body_node = self.parse_expression(body, depth + 1)
                body_node.parent = node
                body_node.position = 1
                node.children.append(body_node)

            return node

        # Handle literals
        literal_match = self.patterns["literal"].match(expr)
        if literal_match:
            return ASTNode(NodeType.LITERAL, literal_match.group(1), depth)

        # Handle identifiers
        ident_match = self.patterns["identifier"].match(expr)
        if ident_match:
            return ASTNode(NodeType.IDENTIFIER, ident_match.group(1), depth)

        # Default to expression node
        return ASTNode(NodeType.EXPRESSION, expr[:20] + "..." if len(expr) > 20 else expr, depth)

    def parse_file(self, content: str) -> List[ASTNode]:
        """Parse entire Lean file into AST forest"""
        trees = []

        # Parse theorems and lemmas
        for match in self.patterns["theorem"].finditer(content):
            name = match.group(1)
            body = match.group(2)

            theorem_node = ASTNode(NodeType.THEOREM, name, 0)
            body_ast = self.parse_expression(body, 1)
            body_ast.parent = theorem_node
            theorem_node.children.append(body_ast)
            trees.append(theorem_node)

        # Parse definitions
        for match in self.patterns["definition"].finditer(content):
            name = match.group(1)
            body = match.group(2)

            def_node = ASTNode(NodeType.DEFINITION, name, 0)
            body_ast = self.parse_expression(body, 1)
            body_ast.parent = def_node
            def_node.children.append(body_ast)
            trees.append(def_node)

        return trees


class TreeEditDistance:
    """Calculate tree edit distance between AST patterns"""

    @staticmethod
    def calculate(tree1: ASTNode, tree2: ASTNode) -> int:
        """
        Calculate tree edit distance using dynamic programming.
        Based on Zhang-Shasha algorithm for ordered trees.
        """

        def postorder(node: ASTNode, nodes: List[ASTNode]) -> None:
            """Get nodes in postorder traversal"""
            for child in node.children:
                postorder(child, nodes)
            nodes.append(node)

        # Get postorder traversals
        nodes1 = []
        nodes2 = []
        postorder(tree1, nodes1)
        postorder(tree2, nodes2)

        n1 = len(nodes1)
        n2 = len(nodes2)

        # Initialize distance matrix
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]

        # Base cases
        for i in range(1, n1 + 1):
            dp[i][0] = i
        for j in range(1, n2 + 1):
            dp[0][j] = j

        # Fill matrix
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if (
                    nodes1[i - 1].node_type == nodes2[j - 1].node_type
                    and nodes1[i - 1].value == nodes2[j - 1].value
                ):
                    cost = 0
                else:
                    cost = 1

                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Delete
                    dp[i][j - 1] + 1,  # Insert
                    dp[i - 1][j - 1] + cost,  # Replace
                )

        return dp[n1][n2]

    @staticmethod
    def normalized_distance(tree1: ASTNode, tree2: ASTNode) -> float:
        """Calculate normalized tree edit distance (0-1)"""
        distance = TreeEditDistance.calculate(tree1, tree2)
        max_size = max(tree1.subtree_size, tree2.subtree_size)
        return distance / max_size if max_size > 0 else 0


class SophisticatedPatternAnalyzer:
    """
    Advanced pattern analyzer using AST analysis, tree kernels, and edit distance.

    Based on research from discrimination trees, syntactic pattern matching,
    and tree edit distance algorithms.
    """

    def __init__(self):
        # Import and use the improved parser
        try:
            from simpulse.analysis.improved_lean_parser import ImprovedLeanParser

            self.parser = ImprovedLeanParser()
        except ImportError:
            # Fallback to basic parser if improved not available
            self.parser = LeanASTParser()
        self.pattern_library = {}  # Store representative patterns
        self.complexity_weights = {
            "depth": 0.2,
            "branching": 0.2,
            "diversity": 0.2,
            "nesting": 0.2,
            "cognitive": 0.2,
        }

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Perform comprehensive pattern analysis on a Lean file"""
        content = file_path.read_text()

        # Parse into AST forest
        ast_trees = self.parser.parse_file(content)

        if not ast_trees:
            return self._empty_analysis()

        # Extract fingerprints for each pattern
        fingerprints = [self._extract_fingerprint(tree) for tree in ast_trees]

        # Calculate structural complexity
        complexity = self._calculate_structural_complexity(ast_trees, fingerprints)

        # Identify dominant patterns
        dominant_patterns = self._identify_dominant_patterns(fingerprints)

        # Calculate pattern mixing coefficient
        mixing_coefficient = self._calculate_mixing_coefficient(fingerprints)

        # Generate comprehensive report
        return {
            "file_path": str(file_path),
            "pattern_complexity_score": self._calculate_overall_complexity_score(complexity),
            "dominant_patterns": dominant_patterns,
            "structural_complexity": {
                "cyclomatic_complexity": complexity.cyclomatic_complexity,
                "halstead_volume": complexity.halstead_volume,
                "maintainability_index": complexity.maintainability_index,
                "cognitive_complexity": complexity.cognitive_complexity,
                "nesting_complexity": complexity.nesting_complexity,
                "pattern_diversity": complexity.pattern_diversity,
            },
            "pattern_mixing_coefficient": mixing_coefficient,
            "ast_metrics": {
                "total_theorems": len(ast_trees),
                "avg_tree_depth": (
                    statistics.mean([t.max_depth for t in ast_trees]) if ast_trees else 0
                ),
                "avg_branching_factor": (
                    statistics.mean([self._avg_branching(t) for t in ast_trees]) if ast_trees else 0
                ),
                "total_nodes": sum(t.subtree_size for t in ast_trees),
            },
            "pattern_fingerprints": [
                self._fingerprint_to_dict(fp) for fp in fingerprints[:5]
            ],  # Top 5 for inspection
        }

    def _extract_fingerprint(self, tree: ASTNode) -> PatternFingerprint:
        """Extract multi-dimensional fingerprint from AST"""
        # Calculate depth distribution
        depth_distribution = defaultdict(int)
        self._count_depth_distribution(tree, depth_distribution)

        # Calculate branching distribution
        branching_distribution = []
        self._collect_branching_factors(tree, branching_distribution)

        # Calculate node type distribution
        node_type_distribution = defaultdict(int)
        self._count_node_types(tree, node_type_distribution)

        # Pattern-specific features
        identity_count = self._count_identity_patterns(tree)
        nested_depth = self._max_nesting_depth(tree)
        operator_diversity = self._calculate_operator_diversity(tree)
        quantifier_nesting = self._count_quantifier_nesting(tree)

        # Structural metrics
        all_depths = []
        self._collect_all_depths(tree, all_depths)

        return PatternFingerprint(
            depth_distribution=list(depth_distribution.values()),
            branching_distribution=branching_distribution,
            node_type_distribution=dict(node_type_distribution),
            avg_depth=statistics.mean(all_depths) if all_depths else 0,
            max_depth=max(all_depths) if all_depths else 0,
            avg_branching_factor=(
                statistics.mean(branching_distribution) if branching_distribution else 0
            ),
            total_nodes=tree.subtree_size,
            identity_pattern_count=identity_count,
            nested_pattern_depth=nested_depth,
            operator_diversity=operator_diversity,
            quantifier_nesting=quantifier_nesting,
            structural_hash=self._hash_tree_structure(tree),
            canonical_form=self._canonicalize_tree(tree),
        )

    def _count_depth_distribution(self, node: ASTNode, distribution: Dict[int, int]) -> None:
        """Count nodes at each depth level"""
        distribution[node.depth] += 1
        for child in node.children:
            self._count_depth_distribution(child, distribution)

    def _collect_branching_factors(self, node: ASTNode, factors: List[int]) -> None:
        """Collect all branching factors in the tree"""
        if node.children:
            factors.append(len(node.children))
        for child in node.children:
            self._collect_branching_factors(child, factors)

    def _count_node_types(self, node: ASTNode, distribution: Dict[NodeType, int]) -> None:
        """Count occurrences of each node type"""
        distribution[node.node_type] += 1
        for child in node.children:
            self._count_node_types(child, distribution)

    def _count_identity_patterns(self, node: ASTNode) -> int:
        """Count identity-like patterns in the tree - improved version"""
        count = 0

        # Direct check for identity pattern nodes (from improved parser)
        # Use string comparison to handle different NodeType enums
        if hasattr(node.node_type, "value") and node.node_type.value == "identity_pattern":
            count += 1

        # Also check for operator patterns that match identity forms
        elif node.node_type == NodeType.OPERATOR:
            # Check for additive identity: n + 0 or 0 + n
            if node.value == "+":
                left = node.children[0] if len(node.children) > 0 else None
                right = node.children[1] if len(node.children) > 1 else None

                if (left and left.node_type == NodeType.LITERAL and left.value == "0") or (
                    right and right.node_type == NodeType.LITERAL and right.value == "0"
                ):
                    count += 1

            # Check for multiplicative identity: n * 1 or 1 * n
            elif node.value == "*":
                left = node.children[0] if len(node.children) > 0 else None
                right = node.children[1] if len(node.children) > 1 else None

                if (left and left.node_type == NodeType.LITERAL and left.value == "1") or (
                    right and right.node_type == NodeType.LITERAL and right.value == "1"
                ):
                    count += 1

            # Check for subtraction identity: n - 0
            elif node.value == "-":
                right = node.children[1] if len(node.children) > 1 else None
                if right and right.node_type == NodeType.LITERAL and right.value == "0":
                    count += 1

        # Check for list identity patterns
        elif hasattr(node.node_type, "value") and node.node_type.value == "list_pattern":
            if node.value == "++":
                left = node.children[0] if len(node.children) > 0 else None
                right = node.children[1] if len(node.children) > 1 else None

                if (left and left.node_type == NodeType.LITERAL and left.value == "[]") or (
                    right and right.node_type == NodeType.LITERAL and right.value == "[]"
                ):
                    count += 1

        # Recurse through children
        for child in node.children:
            count += self._count_identity_patterns(child)

        return count

    def _max_nesting_depth(self, node: ASTNode) -> int:
        """Calculate maximum nesting depth of similar node types"""

        def nesting_depth(node: ASTNode, target_type: NodeType, current_depth: int = 0) -> int:
            if node.node_type == target_type:
                current_depth += 1

            if not node.children:
                return current_depth

            return max(nesting_depth(child, target_type, current_depth) for child in node.children)

        # Check nesting for different node types
        max_depth = 0
        for node_type in [NodeType.QUANTIFIER, NodeType.OPERATOR, NodeType.APPLICATION]:
            depth = nesting_depth(node, node_type)
            max_depth = max(max_depth, depth)

        return max_depth

    def _calculate_operator_diversity(self, node: ASTNode) -> float:
        """Calculate diversity of operators used"""
        operators = set()
        total_operators = 0

        def collect_operators(n: ASTNode) -> None:
            nonlocal total_operators
            if n.node_type == NodeType.OPERATOR:
                operators.add(n.value)
                total_operators += 1
            for child in n.children:
                collect_operators(child)

        collect_operators(node)

        if total_operators == 0:
            return 0.0

        return len(operators) / total_operators

    def _count_quantifier_nesting(self, node: ASTNode) -> int:
        """Count maximum nesting depth of quantifiers"""
        if node.node_type == NodeType.QUANTIFIER:
            max_child_nesting = 0
            for child in node.children:
                child_nesting = self._count_quantifier_nesting(child)
                max_child_nesting = max(max_child_nesting, child_nesting)
            return 1 + max_child_nesting

        max_nesting = 0
        for child in node.children:
            child_nesting = self._count_quantifier_nesting(child)
            max_nesting = max(max_nesting, child_nesting)

        return max_nesting

    def _collect_all_depths(self, node: ASTNode, depths: List[int]) -> None:
        """Collect all node depths"""
        depths.append(node.depth)
        for child in node.children:
            self._collect_all_depths(child, depths)

    def _hash_tree_structure(self, node: ASTNode) -> str:
        """Generate structural hash for fast comparison"""
        # Use node type and children count for hashing
        structure_str = f"{node.node_type.value}:{len(node.children)}"
        for child in node.children:
            structure_str += f"[{self._hash_tree_structure(child)}]"

        # Return first 16 chars of hash for compactness
        import hashlib

        return hashlib.md5(structure_str.encode()).hexdigest()[:16]

    def _canonicalize_tree(self, node: ASTNode) -> str:
        """Generate canonical representation of tree"""
        # Sort children by node type and value for canonical form
        sorted_children = sorted(node.children, key=lambda c: (c.node_type.value, c.value))

        result = f"{node.node_type.value}"
        if sorted_children:
            child_reprs = [self._canonicalize_tree(child) for child in sorted_children]
            result += f"({','.join(child_reprs)})"

        return result

    def _avg_branching(self, tree: ASTNode) -> float:
        """Calculate average branching factor for a tree"""
        factors = []
        self._collect_branching_factors(tree, factors)
        return statistics.mean(factors) if factors else 0

    def _calculate_structural_complexity(
        self, trees: List[ASTNode], fingerprints: List[PatternFingerprint]
    ) -> StructuralComplexity:
        """Calculate comprehensive structural complexity metrics"""
        # Cyclomatic complexity (based on decision points)
        cyclomatic = sum(self._count_decision_points(tree) for tree in trees)

        # Halstead volume (information content)
        operators = []
        operands = []
        for tree in trees:
            self._collect_halstead_elements(tree, operators, operands)

        n1 = len(set(operators))  # Unique operators
        n2 = len(set(operands))  # Unique operands
        N1 = len(operators)  # Total operators
        N2 = len(operands)  # Total operands

        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0

        # Maintainability index
        loc = sum(tree.subtree_size for tree in trees)
        maintainability = max(
            0,
            min(
                100, 171 - 5.2 * math.log(volume + 1) - 0.23 * cyclomatic - 16.2 * math.log(loc + 1)
            ),
        )

        # Cognitive complexity
        cognitive = sum(self._calculate_cognitive_complexity(tree) for tree in trees)

        # Nesting complexity
        avg_nesting = (
            statistics.mean([fp.nested_pattern_depth for fp in fingerprints]) if fingerprints else 0
        )

        # Pattern diversity
        unique_patterns = len({fp.canonical_form for fp in fingerprints})
        total_patterns = len(fingerprints)
        diversity = unique_patterns / total_patterns if total_patterns > 0 else 0

        return StructuralComplexity(
            cyclomatic_complexity=cyclomatic,
            halstead_volume=volume,
            maintainability_index=maintainability,
            cognitive_complexity=cognitive,
            nesting_complexity=avg_nesting,
            pattern_diversity=diversity,
        )

    def _count_decision_points(self, node: ASTNode) -> int:
        """Count decision points for cyclomatic complexity"""
        count = 0

        # Count conditional operators and quantifiers as decision points
        if node.node_type in [NodeType.OPERATOR, NodeType.QUANTIFIER]:
            if node.value in ["∧", "∨", "→", "↔", "∀", "∃"]:
                count += 1

        for child in node.children:
            count += self._count_decision_points(child)

        return count

    def _collect_halstead_elements(
        self, node: ASTNode, operators: List[str], operands: List[str]
    ) -> None:
        """Collect operators and operands for Halstead metrics"""
        if node.node_type in [NodeType.OPERATOR, NodeType.QUANTIFIER]:
            operators.append(node.value)
        elif node.node_type in [NodeType.IDENTIFIER, NodeType.LITERAL]:
            operands.append(node.value)

        for child in node.children:
            self._collect_halstead_elements(child, operators, operands)

    def _calculate_cognitive_complexity(self, node: ASTNode, nesting_level: int = 0) -> int:
        """Calculate cognitive complexity based on nesting and structure"""
        complexity = 0

        # Increase complexity for nested structures
        if node.node_type in [NodeType.QUANTIFIER, NodeType.IMPLICATION]:
            complexity += 1 + nesting_level
            nesting_level += 1

        # Increase for complex operators
        if node.node_type == NodeType.OPERATOR and node.value in ["→", "↔"]:
            complexity += 2

        for child in node.children:
            complexity += self._calculate_cognitive_complexity(child, nesting_level)

        return complexity

    def _identify_dominant_patterns(
        self, fingerprints: List[PatternFingerprint]
    ) -> Dict[str, float]:
        """Identify and quantify dominant pattern types"""
        if not fingerprints:
            return {}

        # Aggregate pattern characteristics
        total_nodes = sum(fp.total_nodes for fp in fingerprints)

        # Count node type frequencies
        type_counts = defaultdict(int)
        for fp in fingerprints:
            for node_type, count in fp.node_type_distribution.items():
                type_counts[node_type] += count

        # Calculate pattern type percentages
        dominant = {}

        # Identity patterns
        identity_nodes = sum(fp.identity_pattern_count for fp in fingerprints)
        dominant["identity_patterns"] = (
            (identity_nodes / total_nodes * 100) if total_nodes > 0 else 0
        )

        # Operator-heavy patterns
        operator_nodes = type_counts.get(NodeType.OPERATOR, 0)
        dominant["operator_patterns"] = (
            (operator_nodes / total_nodes * 100) if total_nodes > 0 else 0
        )

        # Quantifier patterns
        quantifier_nodes = type_counts.get(NodeType.QUANTIFIER, 0)
        dominant["quantifier_patterns"] = (
            (quantifier_nodes / total_nodes * 100) if total_nodes > 0 else 0
        )

        # Tactic patterns
        tactic_nodes = type_counts.get(NodeType.TACTIC, 0)
        dominant["tactic_patterns"] = (tactic_nodes / total_nodes * 100) if total_nodes > 0 else 0

        # List patterns - check both enum and string value
        list_nodes = 0
        for node_type, count in type_counts.items():
            if (hasattr(node_type, "value") and node_type.value == "list_pattern") or (
                hasattr(node_type, "name") and node_type.name == "LIST_PATTERN"
            ):
                list_nodes += count
        dominant["list_patterns"] = (list_nodes / total_nodes * 100) if total_nodes > 0 else 0

        # Simple vs complex patterns
        avg_depth = statistics.mean([fp.avg_depth for fp in fingerprints]) if fingerprints else 0
        if avg_depth < 3:
            dominant["pattern_complexity"] = "simple"
        elif avg_depth < 5:
            dominant["pattern_complexity"] = "moderate"
        else:
            dominant["pattern_complexity"] = "complex"

        return dominant

    def _calculate_mixing_coefficient(self, fingerprints: List[PatternFingerprint]) -> float:
        """
        Calculate pattern mixing coefficient (0-1).
        0 = homogeneous patterns, 1 = highly mixed patterns
        """
        if len(fingerprints) < 2:
            return 0.0

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                sim = fingerprints[i].similarity_to(fingerprints[j])
                similarities.append(sim)

        if not similarities:
            return 0.0

        # Mixing coefficient is inverse of average similarity
        avg_similarity = statistics.mean(similarities)
        mixing = 1.0 - avg_similarity

        return mixing

    def _calculate_overall_complexity_score(self, complexity: StructuralComplexity) -> float:
        """Calculate overall pattern complexity score (0-100)"""
        # Normalize each metric to 0-100 scale
        cyclomatic_score = min(100, complexity.cyclomatic_complexity * 5)  # Scale factor
        volume_score = min(100, complexity.halstead_volume / 100)  # Scale factor
        cognitive_score = min(100, complexity.cognitive_complexity * 3)  # Scale factor
        nesting_score = min(100, complexity.nesting_complexity * 20)  # Scale factor
        diversity_penalty = (
            1 - complexity.pattern_diversity
        ) * 20  # High diversity = lower complexity

        # Weighted combination
        score = (
            self.complexity_weights["depth"] * nesting_score
            + self.complexity_weights["branching"] * cyclomatic_score
            + self.complexity_weights["diversity"] * diversity_penalty
            + self.complexity_weights["nesting"] * volume_score
            + self.complexity_weights["cognitive"] * cognitive_score
        )

        # Include maintainability as a moderating factor
        maintainability_factor = complexity.maintainability_index / 100
        final_score = score * (
            1.5 - maintainability_factor
        )  # Lower maintainability increases complexity

        return min(100, max(0, final_score))

    def _fingerprint_to_dict(self, fp: PatternFingerprint) -> Dict[str, Any]:
        """Convert fingerprint to dictionary for JSON serialization"""
        return {
            "avg_depth": fp.avg_depth,
            "max_depth": fp.max_depth,
            "avg_branching_factor": fp.avg_branching_factor,
            "total_nodes": fp.total_nodes,
            "identity_pattern_count": fp.identity_pattern_count,
            "nested_pattern_depth": fp.nested_pattern_depth,
            "operator_diversity": fp.operator_diversity,
            "quantifier_nesting": fp.quantifier_nesting,
            "structural_hash": fp.structural_hash,
            "node_types": {k.value: v for k, v in fp.node_type_distribution.items()},
        }

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis for files without parseable content"""
        return {
            "pattern_complexity_score": 0,
            "dominant_patterns": {},
            "structural_complexity": {
                "cyclomatic_complexity": 0,
                "halstead_volume": 0.0,
                "maintainability_index": 100.0,
                "cognitive_complexity": 0,
                "nesting_complexity": 0.0,
                "pattern_diversity": 0.0,
            },
            "pattern_mixing_coefficient": 0.0,
            "ast_metrics": {
                "total_theorems": 0,
                "avg_tree_depth": 0.0,
                "avg_branching_factor": 0.0,
                "total_nodes": 0,
            },
            "pattern_fingerprints": [],
        }

    def compare_patterns(self, pattern1: ASTNode, pattern2: ASTNode) -> Dict[str, float]:
        """Compare two patterns using multiple similarity metrics"""
        # Tree edit distance
        edit_distance = TreeEditDistance.normalized_distance(pattern1, pattern2)

        # Structural similarity
        fp1 = self._extract_fingerprint(pattern1)
        fp2 = self._extract_fingerprint(pattern2)
        structural_similarity = fp1.similarity_to(fp2)

        # Size similarity
        size_ratio = min(pattern1.subtree_size, pattern2.subtree_size) / max(
            pattern1.subtree_size, pattern2.subtree_size
        )

        return {
            "edit_distance": edit_distance,
            "structural_similarity": structural_similarity,
            "size_similarity": size_ratio,
            "overall_similarity": (structural_similarity + size_ratio + (1 - edit_distance)) / 3,
        }

    def cluster_similar_patterns(
        self, trees: List[ASTNode], threshold: float = 0.7
    ) -> List[List[ASTNode]]:
        """Cluster similar patterns based on similarity threshold"""
        clusters = []
        clustered = set()

        for i, tree in enumerate(trees):
            if i in clustered:
                continue

            cluster = [tree]
            clustered.add(i)

            for j, other_tree in enumerate(trees[i + 1 :], start=i + 1):
                if j in clustered:
                    continue

                similarity = self.compare_patterns(tree, other_tree)["overall_similarity"]
                if similarity >= threshold:
                    cluster.append(other_tree)
                    clustered.add(j)

            clusters.append(cluster)

        return clusters


# Example usage and testing
if __name__ == "__main__":
    import tempfile

    analyzer = SophisticatedPatternAnalyzer()

    # Test cases with different pattern complexities
    test_cases = [
        # Simple arithmetic pattern
        """
theorem simple_arithmetic : ∀ n : Nat, n + 0 = n := by simp
theorem simple_mul : ∀ n : Nat, n * 1 = n := by simp
""",
        # Complex nested pattern
        """
theorem complex_nested : ∀ (p q r : Prop), 
  (∃ x : Nat, ∀ y : Nat, (x < y → (p ∧ q)) ∨ (y < x → (q ∨ r))) 
  → (∀ z : Nat, ∃ w : Nat, z + w = w + z) := by
  intro h
  intro z
  use z
  simp [add_comm]
""",
        # Mixed patterns
        """
theorem mixed_pattern (xs : List Nat) : 
  xs ++ [] = xs ∧ 0 :: xs = 0 :: xs := by simp

def recursive_function : Nat → Nat
  | 0 => 1
  | n + 1 => n * recursive_function n
  
theorem about_recursive : ∀ n : Nat, 
  recursive_function n > 0 := by
  intro n
  induction n with
  | zero => simp [recursive_function]
  | succ n ih => simp [recursive_function, ih]
""",
    ]

    # Analyze each test case
    for i, content in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}")
        print(f"{'='*60}")

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            test_file = Path(f.name)

        # Analyze
        result = analyzer.analyze_file(test_file)

        # Print results
        print(f"Pattern Complexity Score: {result['pattern_complexity_score']:.1f}/100")
        print(f"\nDominant Patterns:")
        for pattern, percentage in result["dominant_patterns"].items():
            if isinstance(percentage, (int, float)):
                print(f"  - {pattern}: {percentage:.1f}%")
            else:
                print(f"  - {pattern}: {percentage}")

        print(f"\nStructural Complexity:")
        for metric, value in result["structural_complexity"].items():
            if isinstance(value, float):
                print(f"  - {metric}: {value:.2f}")
            else:
                print(f"  - {metric}: {value}")

        print(f"\nPattern Mixing Coefficient: {result['pattern_mixing_coefficient']:.2f}")

        print(f"\nAST Metrics:")
        for metric, value in result["ast_metrics"].items():
            if isinstance(value, float):
                print(f"  - {metric}: {value:.2f}")
            else:
                print(f"  - {metric}: {value}")

        # Clean up
        test_file.unlink()

    print("\n" + "=" * 60)
    print("Pattern analysis complete!")

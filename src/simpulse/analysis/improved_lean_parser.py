#!/usr/bin/env python3
"""
Improved Lean 4 Parser for Sophisticated Pattern Analysis

This parser correctly handles Lean 4 syntax to build accurate ASTs for pattern detection.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


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
    IDENTITY_PATTERN = "identity_pattern"  # Special marker for identity patterns
    LIST_PATTERN = "list_pattern"  # Special marker for list patterns
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
    metadata: Dict[str, any] = field(default_factory=dict)  # Additional metadata

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
            prefix + ("└── " if is_last else "├── ")
            next_prefix = prefix + ("    " if is_last else "│   ")
            result += child.to_string(next_prefix)
        return result


class ImprovedLeanParser:
    """Enhanced parser for Lean 4 syntax with better pattern recognition"""

    def __init__(self):
        # Enhanced patterns for Lean constructs
        self.patterns = {
            # Match theorems/lemmas with proposition and proof separated
            "theorem": re.compile(
                r"(?:theorem|lemma)\s+(\w+)(?:\s*\[[^\]]*\])?\s*:\s*(.*?)\s*:=\s*(?:by\s+)?(.+?)(?=\n(?:theorem|lemma|def|example)|$)",
                re.MULTILINE | re.DOTALL,
            ),
            # Definitions
            "definition": re.compile(
                r"def\s+(\w+)(?:\s*\[[^\]]*\])?\s*(?::\s*(.*?))?\s*:=\s*(.+?)(?=\n(?:theorem|lemma|def|example)|$)",
                re.MULTILINE | re.DOTALL,
            ),
            # Quantifiers with better variable capture
            "quantifier": re.compile(
                r"^([∀∃])\s*(?:\(([^)]+)\)|(\w+(?:\s+\w+)*)\s*:\s*([^,]+))\s*,\s*(.+)"
            ),
            # Binary operators (ordered by precedence)
            "binary_op": re.compile(r"^(.+?)\s*(=|≠|≤|≥|<|>|∈|∉|⊆|⊇)\s*(.+)$"),
            "arithmetic_op": re.compile(r"^(.+?)\s*(\+|\*|-|/)\s*(.+)$"),
            "logical_op": re.compile(r"^(.+?)\s*(∧|∨|→|↔)\s*(.+)$"),
            # List operations
            "list_op": re.compile(r"^(.+?)\s*(\+\+|::)\s*(.+)$"),
            "list_method": re.compile(r"^(.+?)\.(length|reverse|map|filter)(?:\s+(.+))?$"),
            # Identity patterns - critical for classification
            "identity_patterns": [
                re.compile(r"^(\w+)\s*\+\s*0$"),  # n + 0
                re.compile(r"^0\s*\+\s*(\w+)$"),  # 0 + n
                re.compile(r"^(\w+)\s*\*\s*1$"),  # n * 1
                re.compile(r"^1\s*\*\s*(\w+)$"),  # 1 * n
                re.compile(r"^(\w+)\s*-\s*0$"),  # n - 0
                re.compile(r"^(\w+)\s*\+\+\s*\[\]$"),  # xs ++ []
                re.compile(r"^\[\]\s*\+\+\s*(\w+)$"),  # [] ++ xs
            ],
            # Tactics
            "tactic": re.compile(
                r"^(simp|rfl|sorry|exact|apply|intro|intros|constructor|cases|use|obtain)(?:\s+(.+))?$"
            ),
            # Literals and identifiers
            "literal": re.compile(r"^(\d+|true|false|True|False|nil|\[\])$"),
            "identifier": re.compile(r"^[a-zA-Z_]\w*$"),
            # Function application
            "application": re.compile(r"^(\w+)\s+(.+)$"),
        }

    def parse_expression(self, expr: str, depth: int = 0) -> ASTNode:
        """Parse a Lean expression into an AST with enhanced pattern detection"""
        expr = expr.strip()
        if not expr:
            return ASTNode(NodeType.UNKNOWN, "", depth)

        # Remove outer parentheses if balanced
        if expr.startswith("(") and expr.endswith(")") and self._balanced_parens(expr[1:-1]):
            return self.parse_expression(expr[1:-1], depth)

        # Check for tactics first
        tactic_match = self.patterns["tactic"].match(expr)
        if tactic_match:
            return ASTNode(NodeType.TACTIC, tactic_match.group(1), depth)

        # Check for identity patterns - CRITICAL for classification
        for pattern in self.patterns["identity_patterns"]:
            match = pattern.match(expr)
            if match:
                # Create special identity pattern node
                node = ASTNode(NodeType.IDENTITY_PATTERN, expr, depth)
                node.metadata["identity_type"] = self._classify_identity(expr)

                # Still parse the structure
                if "+" in expr:
                    parts = expr.split("+", 1)
                    op_node = ASTNode(NodeType.OPERATOR, "+", depth + 1)
                elif "*" in expr:
                    parts = expr.split("*", 1)
                    op_node = ASTNode(NodeType.OPERATOR, "*", depth + 1)
                elif "-" in expr:
                    parts = expr.split("-", 1)
                    op_node = ASTNode(NodeType.OPERATOR, "-", depth + 1)
                elif "++" in expr:
                    parts = expr.split("++", 1)
                    op_node = ASTNode(NodeType.OPERATOR, "++", depth + 1)
                else:
                    return node

                left_node = self.parse_expression(parts[0].strip(), depth + 2)
                right_node = self.parse_expression(parts[1].strip(), depth + 2)

                op_node.children = [left_node, right_node]
                left_node.parent = op_node
                right_node.parent = op_node

                node.children.append(op_node)
                op_node.parent = node

                return node

        # Handle quantifiers
        quant_match = self.patterns["quantifier"].match(expr)
        if quant_match:
            quantifier = quant_match.group(1)  # ∀ or ∃

            node = ASTNode(NodeType.QUANTIFIER, quantifier, depth)

            # Parse variable declarations
            if quant_match.group(2):  # Parenthesized form
                quant_match.group(2)
                body = quant_match.group(5) if quant_match.group(5) else ""
            else:  # Non-parenthesized form
                var_names = quant_match.group(3)
                var_type = quant_match.group(4)
                body = quant_match.group(5)

                # Create variable nodes
                for var_name in var_names.split():
                    var_node = ASTNode(NodeType.IDENTIFIER, var_name, depth + 1)
                    type_node = ASTNode(NodeType.TYPE, var_type, depth + 2)
                    var_node.children.append(type_node)
                    type_node.parent = var_node
                    node.children.append(var_node)
                    var_node.parent = node

            # Parse body
            if body:
                body_node = self.parse_expression(body, depth + 1)
                body_node.parent = node
                node.children.append(body_node)

            return node

        # Handle binary operators in order of precedence
        for pattern_name in ["binary_op", "arithmetic_op", "logical_op", "list_op"]:
            match = self.patterns[pattern_name].match(expr)
            if match:
                left = match.group(1).strip()
                op = match.group(2).strip()
                right = match.group(3).strip()

                # Special handling for list patterns
                if pattern_name == "list_op":
                    node = ASTNode(NodeType.LIST_PATTERN, op, depth)
                else:
                    node = ASTNode(NodeType.OPERATOR, op, depth)

                left_node = self.parse_expression(left, depth + 1)
                right_node = self.parse_expression(right, depth + 1)

                left_node.parent = node
                right_node.parent = node
                left_node.position = 0
                right_node.position = 1

                node.children = [left_node, right_node]
                return node

        # Handle list methods
        list_method_match = self.patterns["list_method"].match(expr)
        if list_method_match:
            obj = list_method_match.group(1)
            method = list_method_match.group(2)
            args = list_method_match.group(3)

            node = ASTNode(NodeType.LIST_PATTERN, method, depth)
            obj_node = self.parse_expression(obj, depth + 1)
            obj_node.parent = node
            node.children.append(obj_node)

            if args:
                arg_node = self.parse_expression(args, depth + 1)
                arg_node.parent = node
                node.children.append(arg_node)

            return node

        # Handle literals
        if self.patterns["literal"].match(expr):
            return ASTNode(NodeType.LITERAL, expr, depth)

        # Handle identifiers
        if self.patterns["identifier"].match(expr):
            return ASTNode(NodeType.IDENTIFIER, expr, depth)

        # Handle function application (last resort)
        app_match = self.patterns["application"].match(expr)
        if app_match and app_match.group(1) not in ["by", "theorem", "lemma", "def"]:
            func = app_match.group(1)
            args = app_match.group(2)

            node = ASTNode(NodeType.APPLICATION, func, depth)
            arg_node = self.parse_expression(args, depth + 1)
            arg_node.parent = node
            node.children.append(arg_node)
            return node

        # Default to expression
        return ASTNode(NodeType.EXPRESSION, expr[:30] + "..." if len(expr) > 30 else expr, depth)

    def parse_file(self, content: str) -> List[ASTNode]:
        """Parse entire Lean file into AST forest with accurate pattern extraction"""
        trees = []

        # Parse theorems and lemmas
        for match in self.patterns["theorem"].finditer(content):
            name = match.group(1)
            prop = match.group(2).strip()  # The proposition
            proof = match.group(3).strip()  # The proof

            theorem_node = ASTNode(NodeType.THEOREM, name, 0)

            # Parse the proposition - this is where identity patterns appear
            if prop:
                prop_ast = self.parse_expression(prop, 1)
                prop_ast.parent = theorem_node
                theorem_node.children.append(prop_ast)

            # Parse the proof
            if proof in ["simp", "rfl", "sorry"]:
                proof_ast = ASTNode(NodeType.TACTIC, proof, 1)
            else:
                proof_ast = self.parse_expression(proof, 1)

            proof_ast.parent = theorem_node
            theorem_node.children.append(proof_ast)

            trees.append(theorem_node)

        # Parse definitions
        for match in self.patterns["definition"].finditer(content):
            name = match.group(1)
            type_sig = match.group(2)
            body = match.group(3).strip()

            def_node = ASTNode(NodeType.DEFINITION, name, 0)

            # Parse type signature if present
            if type_sig:
                type_ast = self.parse_expression(type_sig.strip(), 1)
                type_ast.parent = def_node
                def_node.children.append(type_ast)

            # Parse body
            if body:
                body_ast = self.parse_expression(body, 1)
                body_ast.parent = def_node
                def_node.children.append(body_ast)

            trees.append(def_node)

        return trees

    def _balanced_parens(self, s: str) -> bool:
        """Check if parentheses are balanced"""
        count = 0
        for c in s:
            if c == "(":
                count += 1
            elif c == ")":
                count -= 1
                if count < 0:
                    return False
        return count == 0

    def _classify_identity(self, expr: str) -> str:
        """Classify the type of identity pattern"""
        if "+ 0" in expr or "0 +" in expr:
            return "additive_identity"
        elif "* 1" in expr or "1 *" in expr:
            return "multiplicative_identity"
        elif "- 0" in expr:
            return "subtraction_identity"
        elif "++ []" in expr or "[] ++" in expr:
            return "list_append_identity"
        return "unknown_identity"

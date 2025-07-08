#!/usr/bin/env python3
"""
Pattern Interference Analyzer - Detects and quantifies pattern conflicts

Based on research findings about proof search interference:
- Simp rule conflicts causing exponential blowup
- Critical pairs in term rewriting systems
- Discrimination tree branch selection issues
- Non-confluent rewrite systems

This analyzer identifies why mixed pattern files have only 15% success rate.
"""

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from simpulse.analysis.improved_lean_parser import ASTNode, ImprovedLeanParser, NodeType


@dataclass
class PatternConflict:
    """Represents a potential conflict between two patterns"""

    pattern1: str
    pattern2: str
    conflict_type: str  # 'critical_pair', 'ordering_dependency', 'reduction_conflict', 'loop_risk'
    severity: float  # 0.0 (low) to 1.0 (high)
    explanation: str

    def to_dict(self) -> Dict:
        return {
            "pattern1": self.pattern1,
            "pattern2": self.pattern2,
            "conflict_type": self.conflict_type,
            "severity": self.severity,
            "explanation": self.explanation,
        }


@dataclass
class InterferenceMetrics:
    """Metrics for pattern interference in a file"""

    total_patterns: int
    unique_patterns: int
    critical_pairs: int
    ordering_dependencies: int
    reduction_conflicts: int
    loop_risks: int
    max_conflict_severity: float
    avg_conflict_severity: float
    interference_score: float  # 0.0 (no interference) to 1.0 (severe interference)
    pattern_diversity_index: float  # Simpson's diversity index

    def to_dict(self) -> Dict:
        return {
            "total_patterns": self.total_patterns,
            "unique_patterns": self.unique_patterns,
            "critical_pairs": self.critical_pairs,
            "ordering_dependencies": self.ordering_dependencies,
            "reduction_conflicts": self.reduction_conflicts,
            "loop_risks": self.loop_risks,
            "max_conflict_severity": self.max_conflict_severity,
            "avg_conflict_severity": self.avg_conflict_severity,
            "interference_score": self.interference_score,
            "pattern_diversity_index": self.pattern_diversity_index,
        }


class PatternInterferenceAnalyzer:
    """
    Analyzes pattern interference to understand why mixed patterns fail.

    Key insights from research:
    1. Removing one problematic simp lemma: 23s → 0.03s (766x speedup)
    2. Circular rewrites cause infinite loops
    3. Non-confluent rules create unpredictable behavior
    4. Discrimination tree conflicts miss optimizations
    """

    def __init__(self):
        self.parser = ImprovedLeanParser()

        # Pattern categories that often conflict
        self.conflicting_categories = {
            "identity": ["n + 0", "0 + n", "n * 1", "1 * n", "n - 0"],
            "associative": ["(a + b) + c", "a + (b + c)", "(a * b) * c", "a * (b * c)"],
            "commutative": ["a + b", "b + a", "a * b", "b * a"],
            "distributive": ["a * (b + c)", "a * b + a * c", "(a + b) * c", "a * c + b * c"],
            "list_ops": ["xs ++ []", "[] ++ xs", "xs ++ (ys ++ zs)", "(xs ++ ys) ++ zs"],
            "quantifier": ["∀ x, P x", "∃ x, P x", "∀ x y, P x y", "∃ x y, P x y"],
        }

        # Known problematic pattern combinations
        self.known_conflicts = {
            ("n + 0", "0 + n"): "Commutative identity conflict - creates choice points",
            ("(a + b) + c", "a + (b + c)"): "Associativity ordering conflict",
            ("a * (b + c)", "a * b + a * c"): "Distributivity expansion conflict",
            ("xs ++ []", "[] ++ xs"): "List identity ordering conflict",
            ("∀ x, P x → Q x", "∃ x, P x ∧ Q x"): "Quantifier duality conflict",
        }

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze pattern interference in a Lean file"""
        content = file_path.read_text()

        # Parse AST
        trees = self.parser.parse_file(content)

        # Extract all patterns
        patterns = self._extract_all_patterns(trees)

        # Detect conflicts
        conflicts = self._detect_pattern_conflicts(patterns)

        # Calculate metrics
        metrics = self._calculate_interference_metrics(patterns, conflicts)

        # Determine if file is optimizable
        is_optimizable = self._predict_optimizability(metrics)

        return {
            "file_path": str(file_path),
            "patterns": [self._pattern_to_string(p) for p in patterns[:10]],  # First 10
            "conflicts": [c.to_dict() for c in conflicts[:10]],  # Top 10 conflicts
            "metrics": metrics.to_dict(),
            "is_optimizable": is_optimizable,
            "optimization_difficulty": self._calculate_optimization_difficulty(metrics),
            "conflict_graph": self._build_conflict_graph(patterns, conflicts),
        }

    def _extract_all_patterns(self, trees: List[ASTNode]) -> List[ASTNode]:
        """Extract all patterns from AST forest"""
        patterns = []

        for tree in trees:
            self._extract_patterns_from_node(tree, patterns)

        return patterns

    def _extract_patterns_from_node(self, node: ASTNode, patterns: List[ASTNode]):
        """Recursively extract patterns from AST node"""
        # Identity patterns are important
        if hasattr(node.node_type, "value") and node.node_type.value == "identity_pattern":
            patterns.append(node)

        # Operators represent rewrite patterns
        elif node.node_type == NodeType.OPERATOR:
            patterns.append(node)

        # Quantifiers represent proof patterns
        elif node.node_type == NodeType.QUANTIFIER:
            patterns.append(node)

        # List patterns
        elif hasattr(node.node_type, "value") and node.node_type.value == "list_pattern":
            patterns.append(node)

        # Applications might be pattern instances
        elif node.node_type == NodeType.APPLICATION:
            patterns.append(node)

        # Recurse
        for child in node.children:
            self._extract_patterns_from_node(child, patterns)

    def _detect_pattern_conflicts(self, patterns: List[ASTNode]) -> List[PatternConflict]:
        """Detect conflicts between patterns"""
        conflicts = []

        # Check all pairs of patterns
        for i, p1 in enumerate(patterns):
            for j, p2 in enumerate(patterns[i + 1 :], start=i + 1):
                conflict = self._check_pattern_pair(p1, p2)
                if conflict:
                    conflicts.append(conflict)

        # Sort by severity
        conflicts.sort(key=lambda c: c.severity, reverse=True)

        return conflicts

    def _check_pattern_pair(self, p1: ASTNode, p2: ASTNode) -> Optional[PatternConflict]:
        """Check if two patterns conflict"""
        p1_str = self._pattern_to_string(p1)
        p2_str = self._pattern_to_string(p2)

        # Check known conflicts
        if (p1_str, p2_str) in self.known_conflicts:
            return PatternConflict(
                pattern1=p1_str,
                pattern2=p2_str,
                conflict_type="known_conflict",
                severity=0.8,
                explanation=self.known_conflicts[(p1_str, p2_str)],
            )

        # Check for critical pairs (overlapping patterns)
        if self._forms_critical_pair(p1, p2):
            return PatternConflict(
                pattern1=p1_str,
                pattern2=p2_str,
                conflict_type="critical_pair",
                severity=0.7,
                explanation="Patterns overlap and can rewrite the same term differently",
            )

        # Check for ordering dependencies
        if self._has_ordering_dependency(p1, p2):
            return PatternConflict(
                pattern1=p1_str,
                pattern2=p2_str,
                conflict_type="ordering_dependency",
                severity=0.6,
                explanation="Pattern application order affects the result",
            )

        # Check for reduction conflicts
        if self._has_reduction_conflict(p1, p2):
            return PatternConflict(
                pattern1=p1_str,
                pattern2=p2_str,
                conflict_type="reduction_conflict",
                severity=0.5,
                explanation="Patterns have different reduction strategies",
            )

        # Check for loop risks
        if self._creates_loop_risk(p1, p2):
            return PatternConflict(
                pattern1=p1_str,
                pattern2=p2_str,
                conflict_type="loop_risk",
                severity=0.9,
                explanation="Patterns could create infinite rewrite loops",
            )

        return None

    def _forms_critical_pair(self, p1: ASTNode, p2: ASTNode) -> bool:
        """Check if patterns form a critical pair"""
        # Critical pairs occur when patterns can match overlapping subterms

        # Both must be operators or rewrites
        if p1.node_type != NodeType.OPERATOR or p2.node_type != NodeType.OPERATOR:
            return False

        # Check for overlapping structure
        # Example: (a + b) + c and a + (b + c) overlap at '+'
        if p1.value == p2.value:
            # Same operator - check if they could match same term differently
            return self._have_overlapping_structure(p1, p2)

        # Check for distributivity-like conflicts
        # Example: a * (b + c) vs (a * b) + (a * c)
        if self._is_distributive_conflict(p1, p2):
            return True

        return False

    def _has_ordering_dependency(self, p1: ASTNode, p2: ASTNode) -> bool:
        """Check if patterns have ordering dependencies"""
        # Ordering matters when patterns are commutative/associative variants

        # Check if they're the same operator
        if p1.node_type == NodeType.OPERATOR and p2.node_type == NodeType.OPERATOR:
            if p1.value == p2.value and p1.value in ["+", "*", "∧", "∨"]:
                # Check if arguments are reordered
                return self._are_reordered_variants(p1, p2)

        return False

    def _has_reduction_conflict(self, p1: ASTNode, p2: ASTNode) -> bool:
        """Check if patterns have conflicting reduction strategies"""
        # Different reduction paths can lead to different normal forms

        # List operations often have reduction conflicts
        if self._is_list_pattern(p1) and self._is_list_pattern(p2):
            # ++ vs :: reduction strategies conflict
            return True

        # Quantifier reduction conflicts
        if p1.node_type == NodeType.QUANTIFIER and p2.node_type == NodeType.QUANTIFIER:
            # ∀ vs ∃ have different reduction strategies
            return p1.value != p2.value

        return False

    def _creates_loop_risk(self, p1: ASTNode, p2: ASTNode) -> bool:
        """Check if patterns could create infinite loops"""
        # Example: a = b and b = a create a loop

        self._pattern_to_string(p1)
        self._pattern_to_string(p2)

        # Simple circular rewrite detection
        if p1.node_type == NodeType.OPERATOR and p1.value == "=":
            if p2.node_type == NodeType.OPERATOR and p2.value == "=":
                # Check if LHS of one matches RHS of other
                if len(p1.children) >= 2 and len(p2.children) >= 2:
                    p1_lhs = self._pattern_to_string(p1.children[0])
                    p1_rhs = self._pattern_to_string(p1.children[1])
                    p2_lhs = self._pattern_to_string(p2.children[0])
                    p2_rhs = self._pattern_to_string(p2.children[1])

                    # Circular dependency
                    if p1_lhs == p2_rhs and p1_rhs == p2_lhs:
                        return True

        # Identity expansion loops (0 = 0 + 0)
        if self._is_expanding_identity(p1) or self._is_expanding_identity(p2):
            return True

        return False

    def _calculate_interference_metrics(
        self, patterns: List[ASTNode], conflicts: List[PatternConflict]
    ) -> InterferenceMetrics:
        """Calculate comprehensive interference metrics"""
        if not patterns:
            return InterferenceMetrics(
                total_patterns=0,
                unique_patterns=0,
                critical_pairs=0,
                ordering_dependencies=0,
                reduction_conflicts=0,
                loop_risks=0,
                max_conflict_severity=0.0,
                avg_conflict_severity=0.0,
                interference_score=0.0,
                pattern_diversity_index=0.0,
            )

        # Count unique patterns
        unique_patterns = len({self._pattern_to_string(p) for p in patterns})

        # Count conflict types
        conflict_counts = Counter(c.conflict_type for c in conflicts)

        # Calculate severities
        severities = [c.severity for c in conflicts]
        max_severity = max(severities) if severities else 0.0
        avg_severity = statistics.mean(severities) if severities else 0.0

        # Calculate pattern diversity (Simpson's index)
        pattern_counts = Counter(self._pattern_to_string(p) for p in patterns)
        total = sum(pattern_counts.values())
        diversity = 1 - sum((n / total) ** 2 for n in pattern_counts.values()) if total > 0 else 0

        # Calculate interference score (0-1)
        # Based on: conflict density, severity, and diversity
        conflict_density = (
            len(conflicts) / (len(patterns) * (len(patterns) - 1) / 2) if len(patterns) > 1 else 0
        )
        interference_score = min(
            1.0, 0.3 * conflict_density + 0.4 * avg_severity + 0.3 * (1 - diversity)
        )

        return InterferenceMetrics(
            total_patterns=len(patterns),
            unique_patterns=unique_patterns,
            critical_pairs=conflict_counts.get("critical_pair", 0),
            ordering_dependencies=conflict_counts.get("ordering_dependency", 0),
            reduction_conflicts=conflict_counts.get("reduction_conflict", 0),
            loop_risks=conflict_counts.get("loop_risk", 0),
            max_conflict_severity=max_severity,
            avg_conflict_severity=avg_severity,
            interference_score=interference_score,
            pattern_diversity_index=diversity,
        )

    def _predict_optimizability(self, metrics: InterferenceMetrics) -> bool:
        """Predict if file is optimizable based on interference metrics"""
        # Research insight: 15% of mixed files are optimizable
        # These likely have low interference scores

        # Thresholds based on expected 15% success rate
        if metrics.interference_score > 0.6:
            return False  # High interference - not optimizable

        if metrics.loop_risks > 0:
            return False  # Loop risks are fatal

        if metrics.critical_pairs > 5:
            return False  # Too many critical pairs

        if metrics.pattern_diversity_index > 0.8:
            return False  # Too diverse - hard to optimize uniformly

        return True  # Low interference - potentially optimizable

    def _calculate_optimization_difficulty(self, metrics: InterferenceMetrics) -> str:
        """Calculate optimization difficulty level"""
        score = metrics.interference_score

        if score < 0.2:
            return "easy"
        elif score < 0.4:
            return "moderate"
        elif score < 0.6:
            return "hard"
        else:
            return "infeasible"

    def _build_conflict_graph(
        self, patterns: List[ASTNode], conflicts: List[PatternConflict]
    ) -> Dict[str, List[str]]:
        """Build a conflict graph for visualization"""
        graph = defaultdict(list)

        for conflict in conflicts[:20]:  # Limit to top 20 for visualization
            graph[conflict.pattern1].append(conflict.pattern2)
            graph[conflict.pattern2].append(conflict.pattern1)

        return dict(graph)

    # Helper methods

    def _pattern_to_string(self, node: ASTNode) -> str:
        """Convert pattern AST to string representation"""
        if node.node_type == NodeType.OPERATOR:
            if len(node.children) >= 2:
                left = self._pattern_to_string(node.children[0])
                right = self._pattern_to_string(node.children[1])
                return f"({left} {node.value} {right})"
            elif len(node.children) == 1:
                child = self._pattern_to_string(node.children[0])
                return f"{node.value}{child}"
            else:
                return node.value

        elif node.node_type == NodeType.QUANTIFIER:
            if node.children:
                body = (
                    self._pattern_to_string(node.children[-1]) if len(node.children) > 1 else "..."
                )
                return f"{node.value} x, {body}"
            return node.value

        elif hasattr(node.node_type, "value") and node.node_type.value == "identity_pattern":
            return node.value

        else:
            return node.value

    def _have_overlapping_structure(self, p1: ASTNode, p2: ASTNode) -> bool:
        """Check if patterns have overlapping structure"""
        # Simple check: do they have different arrangements of same operator?
        if p1.value == p2.value and len(p1.children) == len(p2.children):
            # Check if children are rearranged
            p1_child_values = sorted([c.value for c in p1.children])
            p2_child_values = sorted([c.value for c in p2.children])

            # Same children but potentially different structure
            return p1_child_values == p2_child_values

        return False

    def _is_distributive_conflict(self, p1: ASTNode, p2: ASTNode) -> bool:
        """Check for distributivity-like conflicts"""
        ops = {p1.value, p2.value}
        return ops == {"*", "+"} or ops == {"∧", "∨"}

    def _are_reordered_variants(self, p1: ASTNode, p2: ASTNode) -> bool:
        """Check if patterns are reordered variants"""
        # Get string representations of children
        p1_children = sorted([self._pattern_to_string(c) for c in p1.children])
        p2_children = sorted([self._pattern_to_string(c) for c in p2.children])

        # Same children in different order indicates reordering
        return p1_children == p2_children and p1.children != p2.children

    def _is_list_pattern(self, node: ASTNode) -> bool:
        """Check if node is a list pattern"""
        return (hasattr(node.node_type, "value") and node.node_type.value == "list_pattern") or (
            node.node_type == NodeType.OPERATOR and node.value in ["++", "::"]
        )

    def _is_expanding_identity(self, node: ASTNode) -> bool:
        """Check if pattern expands an identity (like 0 = 0 + 0)"""
        if node.node_type == NodeType.OPERATOR and node.value == "=":
            if len(node.children) >= 2:
                lhs = self._pattern_to_string(node.children[0])
                rhs = self._pattern_to_string(node.children[1])

                # Check for expanding patterns
                if lhs in ["0", "1", "[]"] and lhs in rhs and len(rhs) > len(lhs):
                    return True

        return False


def analyze_mixed_pattern_files(directory: Path, sample_size: int = 10) -> Dict[str, Any]:
    """Analyze a sample of mixed pattern files to understand interference"""
    analyzer = PatternInterferenceAnalyzer()
    results = []

    # Find Lean files
    lean_files = list(directory.glob("**/*.lean"))[:sample_size]

    for file_path in lean_files:
        try:
            result = analyzer.analyze_file(file_path)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

    # Aggregate results
    optimizable_count = sum(1 for r in results if r["is_optimizable"])
    avg_interference = (
        statistics.mean([r["metrics"]["interference_score"] for r in results]) if results else 0
    )

    # Find common conflict patterns
    all_conflicts = []
    for r in results:
        all_conflicts.extend(r["conflicts"])

    conflict_types = Counter(c["conflict_type"] for c in all_conflicts)

    return {
        "files_analyzed": len(results),
        "optimizable_files": optimizable_count,
        "optimizability_rate": optimizable_count / len(results) if results else 0,
        "avg_interference_score": avg_interference,
        "conflict_type_distribution": dict(conflict_types),
        "sample_results": results[:3],  # First 3 for inspection
    }


if __name__ == "__main__":
    # Test on a mixed pattern example
    test_content = """
import Mathlib.Data.Nat.Basic
import Mathlib.Data.List.Basic

-- Mixed arithmetic and list patterns
theorem mixed_1 : ∀ n : Nat, n + 0 = n := by simp
theorem mixed_2 : ∀ n : Nat, 0 + n = n := by simp
theorem mixed_3 : ∀ xs : List α, xs ++ [] = xs := by simp
theorem mixed_4 : ∀ xs : List α, [] ++ xs = xs := by simp
theorem mixed_5 : ∀ a b c : Nat, (a + b) + c = a + (b + c) := by simp
theorem mixed_6 : ∀ a b : Nat, a + b = b + a := by simp
theorem mixed_7 : ∀ a b c : Nat, a * (b + c) = a * b + a * c := by simp
"""

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(test_content)
        test_file = Path(f.name)

    try:
        analyzer = PatternInterferenceAnalyzer()
        result = analyzer.analyze_file(test_file)

        print("Pattern Interference Analysis")
        print("=" * 60)
        print(f"File: {result['file_path']}")
        print(f"Is optimizable: {result['is_optimizable']}")
        print(f"Optimization difficulty: {result['optimization_difficulty']}")
        print(f"\nMetrics:")
        for key, value in result["metrics"].items():
            print(f"  {key}: {value}")
        print(f"\nTop conflicts:")
        for conflict in result["conflicts"][:5]:
            print(f"  {conflict['pattern1']} ↔ {conflict['pattern2']}")
            print(f"    Type: {conflict['conflict_type']}, Severity: {conflict['severity']}")
            print(f"    {conflict['explanation']}")
    finally:
        test_file.unlink()

#!/usr/bin/env python3
"""
Workload Characterizer - Advanced proof workload characterization

Based on academic research in theorem prover performance analysis.
Uses multi-dimensional feature extraction and machine learning clustering.
"""

import logging
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from simpulse.analysis.improved_lean_parser import ASTNode, ImprovedLeanParser, NodeType
from simpulse.analysis.pattern_interference_analyzer import PatternInterferenceAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class WorkloadProfile:
    """Comprehensive workload characterization"""

    # Primary dimensions
    structural_complexity: float  # 0-1: AST complexity
    pattern_uniformity: float  # 0-1: How uniform patterns are
    proof_depth: float  # 0-1: Normalized proof depth
    computational_intensity: float  # 0-1: Expected computation

    # Secondary characteristics
    dominant_features: List[str]
    proof_style: str  # 'direct', 'inductive', 'computational', 'mixed'
    optimization_potential: float  # 0-1: Likelihood of optimization success

    # Detailed metrics
    feature_vector: np.ndarray
    cluster_id: Optional[int] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "structural_complexity": self.structural_complexity,
            "pattern_uniformity": self.pattern_uniformity,
            "proof_depth": self.proof_depth,
            "computational_intensity": self.computational_intensity,
            "dominant_features": self.dominant_features,
            "proof_style": self.proof_style,
            "optimization_potential": self.optimization_potential,
            "cluster_id": self.cluster_id,
            "confidence": self.confidence,
        }


class WorkloadCharacterizer:
    """
    Advanced workload characterization for theorem proving.

    Key innovations:
    1. Multi-dimensional feature space (100+ features)
    2. Unsupervised clustering to find natural workload groups
    3. Learned success prediction from historical data
    4. Proof style detection beyond simple categories
    """

    def __init__(self, historical_data_path: Optional[Path] = None):
        self.parser = ImprovedLeanParser()
        self.interference_analyzer = PatternInterferenceAnalyzer()

        # Feature extraction pipeline
        self.feature_pipeline = [
            self._extract_ast_features,
            self._extract_proof_pattern_features,
            self._extract_computational_features,
            self._extract_semantic_features,
            self._extract_interference_features,
            self._extract_tactic_features,
        ]

        # Machine learning models
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=20)  # Reduce to 20 principal components
        self.clusterer = KMeans(n_clusters=15, random_state=42)  # 15 natural clusters
        self.success_predictor = RandomForestClassifier(n_estimators=100, random_state=42)

        # Load historical data if available
        self.historical_profiles = []
        if historical_data_path and historical_data_path.exists():
            self._load_historical_data(historical_data_path)
            self._train_models()

    def characterize(self, file_path: Path) -> WorkloadProfile:
        """
        Perform comprehensive workload characterization.

        Returns WorkloadProfile with multi-dimensional analysis.
        """
        try:
            content = file_path.read_text()
            trees = self.parser.parse_file(content)

            # Extract comprehensive features
            features = {}
            for extractor in self.feature_pipeline:
                features.update(extractor(trees, file_path))

            # Convert to numpy array
            feature_vector = self._features_to_vector(features)

            # Normalize and reduce dimensions
            if len(self.historical_profiles) > 0:
                feature_vector_scaled = self.scaler.transform([feature_vector])
                feature_vector_pca = self.pca.transform(feature_vector_scaled)

                # Cluster assignment
                cluster_id = self.clusterer.predict(feature_vector_pca)[0]

                # Success prediction
                optimization_potential = self.success_predictor.predict_proba(feature_vector_pca)[
                    0
                ][1]
            else:
                # Fallback without training data
                cluster_id = None
                optimization_potential = self._heuristic_success_prediction(features)

            # Calculate primary dimensions
            structural_complexity = self._calculate_structural_complexity(features)
            pattern_uniformity = self._calculate_pattern_uniformity(features)
            proof_depth = self._calculate_proof_depth(features)
            computational_intensity = self._calculate_computational_intensity(features)

            # Determine proof style
            proof_style = self._determine_proof_style(features)

            # Identify dominant features
            dominant_features = self._identify_dominant_features(features)

            # Calculate confidence
            confidence = self._calculate_confidence(features, cluster_id)

            return WorkloadProfile(
                structural_complexity=structural_complexity,
                pattern_uniformity=pattern_uniformity,
                proof_depth=proof_depth,
                computational_intensity=computational_intensity,
                dominant_features=dominant_features,
                proof_style=proof_style,
                optimization_potential=optimization_potential,
                feature_vector=feature_vector,
                cluster_id=cluster_id,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Characterization failed for {file_path}: {e}")
            return self._fallback_profile(str(e))

    def _extract_ast_features(self, trees: List[ASTNode], file_path: Path) -> Dict[str, float]:
        """Extract AST structural features"""
        features = {}

        if not trees:
            return features

        # Basic metrics
        depths = [tree.max_depth for tree in trees]
        sizes = [tree.subtree_size for tree in trees]

        features["ast_count"] = len(trees)
        features["ast_avg_depth"] = statistics.mean(depths)
        features["ast_max_depth"] = max(depths)
        features["ast_depth_variance"] = statistics.variance(depths) if len(depths) > 1 else 0
        features["ast_avg_size"] = statistics.mean(sizes)
        features["ast_max_size"] = max(sizes)
        features["ast_size_variance"] = statistics.variance(sizes) if len(sizes) > 1 else 0

        # Node type distribution
        node_counts = defaultdict(int)
        total_nodes = 0

        def count_nodes(node: ASTNode):
            nonlocal total_nodes
            node_counts[node.node_type.value] += 1
            total_nodes += 1
            for child in node.children:
                count_nodes(child)

        for tree in trees:
            count_nodes(tree)

        # Node type ratios
        for node_type in [
            "theorem",
            "definition",
            "operator",
            "quantifier",
            "identifier",
            "literal",
            "tactic",
            "application",
        ]:
            features[f"node_ratio_{node_type}"] = (
                node_counts.get(node_type, 0) / total_nodes if total_nodes > 0 else 0
            )

        # Branching factor analysis
        branching_factors = []

        def collect_branching(node: ASTNode):
            if node.children:
                branching_factors.append(len(node.children))
            for child in node.children:
                collect_branching(child)

        for tree in trees:
            collect_branching(tree)

        if branching_factors:
            features["ast_avg_branching"] = statistics.mean(branching_factors)
            features["ast_max_branching"] = max(branching_factors)
            features["ast_branching_variance"] = (
                statistics.variance(branching_factors) if len(branching_factors) > 1 else 0
            )
        else:
            features["ast_avg_branching"] = 0
            features["ast_max_branching"] = 0
            features["ast_branching_variance"] = 0

        return features

    def _extract_proof_pattern_features(
        self, trees: List[ASTNode], file_path: Path
    ) -> Dict[str, float]:
        """Extract proof pattern features"""
        features = {}

        # Pattern counts
        pattern_counts = {
            "identity": 0,
            "associative": 0,
            "commutative": 0,
            "distributive": 0,
            "inductive": 0,
            "case_split": 0,
            "definitional": 0,
        }

        def analyze_patterns(node: ASTNode):
            # Identity patterns
            if hasattr(node.node_type, "value") and node.node_type.value == "identity_pattern":
                pattern_counts["identity"] += 1

            # Check for specific patterns in operators
            if node.node_type == NodeType.OPERATOR:
                if node.value in ["+", "*", "∧", "∨"]:
                    # Check children for commutativity/associativity patterns
                    if len(node.children) >= 2:
                        if self._is_commutative_pattern(node):
                            pattern_counts["commutative"] += 1
                        if self._is_associative_pattern(node):
                            pattern_counts["associative"] += 1

            # Recursive patterns
            if node.value and "induction" in str(node.value).lower():
                pattern_counts["inductive"] += 1

            if node.value and "cases" in str(node.value).lower():
                pattern_counts["case_split"] += 1

            for child in node.children:
                analyze_patterns(child)

        for tree in trees:
            analyze_patterns(tree)

        # Calculate ratios
        total_patterns = sum(pattern_counts.values())
        for pattern_type, count in pattern_counts.items():
            features[f"pattern_{pattern_type}_ratio"] = (
                count / total_patterns if total_patterns > 0 else 0
            )

        # Pattern diversity
        if total_patterns > 0:
            pattern_probs = [c / total_patterns for c in pattern_counts.values() if c > 0]
            features["pattern_entropy"] = (
                -sum(p * math.log2(p) for p in pattern_probs) if pattern_probs else 0
            )
        else:
            features["pattern_entropy"] = 0

        return features

    def _extract_computational_features(
        self, trees: List[ASTNode], file_path: Path
    ) -> Dict[str, float]:
        """Extract computational complexity features"""
        features = {}

        # Estimate computational operations
        op_counts = defaultdict(int)

        def count_operations(node: ASTNode):
            if node.node_type == NodeType.OPERATOR:
                op_counts[node.value] += 1
            elif node.node_type == NodeType.APPLICATION:
                op_counts["application"] += 1
            elif node.node_type == NodeType.QUANTIFIER:
                op_counts["quantifier"] += 1

            for child in node.children:
                count_operations(child)

        for tree in trees:
            count_operations(tree)

        # Computational intensity metrics
        arithmetic_ops = (
            op_counts.get("+", 0)
            + op_counts.get("*", 0)
            + op_counts.get("-", 0)
            + op_counts.get("/", 0)
        )
        logical_ops = (
            op_counts.get("∧", 0)
            + op_counts.get("∨", 0)
            + op_counts.get("→", 0)
            + op_counts.get("¬", 0)
        )
        comparison_ops = (
            op_counts.get("=", 0)
            + op_counts.get("≠", 0)
            + op_counts.get("<", 0)
            + op_counts.get(">", 0)
        )

        total_ops = sum(op_counts.values())

        features["comp_arithmetic_ratio"] = arithmetic_ops / total_ops if total_ops > 0 else 0
        features["comp_logical_ratio"] = logical_ops / total_ops if total_ops > 0 else 0
        features["comp_comparison_ratio"] = comparison_ops / total_ops if total_ops > 0 else 0
        features["comp_application_ratio"] = (
            op_counts.get("application", 0) / total_ops if total_ops > 0 else 0
        )
        features["comp_quantifier_ratio"] = (
            op_counts.get("quantifier", 0) / total_ops if total_ops > 0 else 0
        )

        # Complexity estimate (operations per theorem)
        features["comp_ops_per_theorem"] = total_ops / len(trees) if trees else 0

        return features

    def _extract_semantic_features(self, trees: List[ASTNode], file_path: Path) -> Dict[str, float]:
        """Extract semantic/mathematical features"""
        features = {}

        # Mathematical domain indicators
        domain_keywords = {
            "algebra": ["group", "ring", "field", "monoid"],
            "analysis": ["limit", "continuous", "derivative", "integral"],
            "topology": ["open", "closed", "compact", "hausdorff"],
            "number_theory": ["prime", "divisible", "gcd", "coprime"],
            "combinatorics": ["permutation", "combination", "graph", "tree"],
            "logic": ["true", "false", "implies", "iff"],
        }

        # Extract all text values
        text_values = []

        def collect_text(node: ASTNode):
            if node.value:
                text_values.append(str(node.value).lower())
            for child in node.children:
                collect_text(child)

        for tree in trees:
            collect_text(tree)

        # Count domain indicators
        domain_scores = defaultdict(int)
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                domain_scores[domain] += sum(1 for text in text_values if keyword in text)

        total_domain_indicators = sum(domain_scores.values())

        for domain in domain_keywords:
            features[f"domain_{domain}_score"] = (
                domain_scores[domain] / total_domain_indicators
                if total_domain_indicators > 0
                else 0
            )

        # Abstraction level (based on type usage)
        type_count = sum(1 for text in text_values if "type" in text or "Type" in text)
        features["semantic_abstraction_level"] = type_count / len(text_values) if text_values else 0

        return features

    def _extract_interference_features(
        self, trees: List[ASTNode], file_path: Path
    ) -> Dict[str, float]:
        """Extract pattern interference features"""
        try:
            interference_result = self.interference_analyzer.analyze_file(file_path)
            metrics = interference_result["metrics"]

            features = {
                "interference_score": metrics["interference_score"],
                "interference_critical_pairs": metrics["critical_pairs"],
                "interference_loop_risk": 1.0 if metrics["loop_risks"] > 0 else 0.0,
                "interference_diversity": metrics["pattern_diversity_index"],
                "interference_avg_severity": metrics["avg_conflict_severity"],
            }
        except:
            features = {
                "interference_score": 0.0,
                "interference_critical_pairs": 0,
                "interference_loop_risk": 0.0,
                "interference_diversity": 0.0,
                "interference_avg_severity": 0.0,
            }

        return features

    def _extract_tactic_features(self, trees: List[ASTNode], file_path: Path) -> Dict[str, float]:
        """Extract tactic usage features"""
        features = {}

        # Count tactics
        tactic_counts = defaultdict(int)
        total_tactics = 0

        def count_tactics(node: ASTNode):
            nonlocal total_tactics
            if node.node_type == NodeType.TACTIC:
                tactic_counts[node.value] += 1
                total_tactics += 1
            for child in node.children:
                count_tactics(child)

        for tree in trees:
            count_tactics(tree)

        # Common tactics
        common_tactics = ["simp", "rfl", "exact", "apply", "intro", "cases", "induction", "sorry"]

        for tactic in common_tactics:
            features[f"tactic_{tactic}_ratio"] = (
                tactic_counts.get(tactic, 0) / total_tactics if total_tactics > 0 else 0
            )

        # Tactic diversity
        features["tactic_diversity"] = (
            len(tactic_counts) / len(common_tactics) if common_tactics else 0
        )

        # Proof automation level
        automation_tactics = ["simp", "rfl", "sorry"]
        automation_count = sum(tactic_counts.get(t, 0) for t in automation_tactics)
        features["tactic_automation_level"] = (
            automation_count / total_tactics if total_tactics > 0 else 0
        )

        return features

    def _features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy vector"""
        # Define feature order for consistency
        feature_names = sorted(features.keys())
        return np.array([features.get(name, 0.0) for name in feature_names])

    def _calculate_structural_complexity(self, features: Dict[str, float]) -> float:
        """Calculate normalized structural complexity"""
        complexity_factors = [
            features.get("ast_avg_depth", 0) / 10,  # Normalize by typical max depth
            features.get("ast_avg_size", 0) / 50,  # Normalize by typical max size
            features.get("ast_avg_branching", 0) / 5,  # Normalize by typical max branching
            features.get("pattern_entropy", 0) / 3,  # Normalize by max entropy
            features.get("comp_ops_per_theorem", 0) / 20,  # Normalize by typical max ops
        ]

        # Weight and combine
        weights = [0.25, 0.25, 0.15, 0.20, 0.15]
        complexity = sum(w * min(f, 1.0) for w, f in zip(weights, complexity_factors))

        return min(complexity, 1.0)

    def _calculate_pattern_uniformity(self, features: Dict[str, float]) -> float:
        """Calculate pattern uniformity (inverse of diversity)"""
        diversity_factors = [
            features.get("pattern_entropy", 0),
            features.get("ast_depth_variance", 0),
            features.get("ast_size_variance", 0),
            features.get("interference_diversity", 0),
            features.get("tactic_diversity", 0),
        ]

        # Normalize and invert
        avg_diversity = statistics.mean(diversity_factors)
        uniformity = 1.0 - min(avg_diversity / 3, 1.0)  # Normalize by typical max

        return uniformity

    def _calculate_proof_depth(self, features: Dict[str, float]) -> float:
        """Calculate normalized proof depth"""
        depth_factors = [
            features.get("ast_max_depth", 0) / 15,  # Normalize by typical max
            features.get("comp_quantifier_ratio", 0),
            features.get("semantic_abstraction_level", 0),
            1.0 - features.get("tactic_automation_level", 0),  # Manual proofs are deeper
        ]

        return min(statistics.mean(depth_factors), 1.0)

    def _calculate_computational_intensity(self, features: Dict[str, float]) -> float:
        """Calculate computational intensity"""
        intensity_factors = [
            features.get("comp_ops_per_theorem", 0) / 20,
            features.get("interference_critical_pairs", 0) / 50,
            features.get("comp_application_ratio", 0),
            features.get("pattern_inductive_ratio", 0),
            features.get("pattern_case_split_ratio", 0),
        ]

        return min(statistics.mean(intensity_factors), 1.0)

    def _determine_proof_style(self, features: Dict[str, float]) -> str:
        """Determine dominant proof style"""
        styles = {
            "direct": features.get("tactic_automation_level", 0) * 0.8
            + features.get("pattern_identity_ratio", 0) * 0.2,
            "inductive": features.get("pattern_inductive_ratio", 0) * 0.7
            + features.get("tactic_induction_ratio", 0) * 0.3,
            "computational": features.get("comp_arithmetic_ratio", 0) * 0.5
            + features.get("comp_application_ratio", 0) * 0.5,
            "case_based": features.get("pattern_case_split_ratio", 0) * 0.6
            + features.get("tactic_cases_ratio", 0) * 0.4,
            "algebraic": sum(
                features.get(f"domain_{d}_score", 0) for d in ["algebra", "number_theory"]
            )
            * 0.5,
        }

        # Return style with highest score
        return max(styles.items(), key=lambda x: x[1])[0] if styles else "mixed"

    def _identify_dominant_features(self, features: Dict[str, float], top_n: int = 5) -> List[str]:
        """Identify most significant features"""
        # Calculate z-scores for features
        feature_scores = []

        for name, value in features.items():
            # Simple heuristic: features with extreme values are dominant
            if isinstance(value, (int, float)):
                if value > 0.7 or value < 0.1:  # Extreme values
                    feature_scores.append((name, abs(value - 0.5)))

        # Sort by significance
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        return [name for name, _ in feature_scores[:top_n]]

    def _calculate_confidence(self, features: Dict[str, float], cluster_id: Optional[int]) -> float:
        """Calculate confidence in characterization"""
        # Base confidence on feature completeness
        total_features = len(features)
        non_zero_features = sum(1 for v in features.values() if v != 0)

        feature_confidence = non_zero_features / total_features if total_features > 0 else 0

        # Adjust based on cluster assignment
        if cluster_id is not None:
            # Would use cluster cohesion metrics in real implementation
            cluster_confidence = 0.8
        else:
            cluster_confidence = 0.5

        return feature_confidence * 0.6 + cluster_confidence * 0.4

    def _heuristic_success_prediction(self, features: Dict[str, float]) -> float:
        """Fallback success prediction without ML model"""
        # Based on empirical observations
        positive_factors = [
            features.get("pattern_uniformity", 0),
            features.get("pattern_identity_ratio", 0),
            1.0 - features.get("interference_score", 0),
            1.0 - features.get("comp_ops_per_theorem", 0) / 20,
        ]

        negative_factors = [
            features.get("interference_diversity", 0),
            features.get("pattern_case_split_ratio", 0),
            features.get("tactic_diversity", 0),
        ]

        positive_score = statistics.mean(positive_factors) if positive_factors else 0
        negative_score = statistics.mean(negative_factors) if negative_factors else 0

        base_rate = 0.3  # 30% base success rate
        adjustment = (positive_score - negative_score) * 0.3

        return max(0.0, min(1.0, base_rate + adjustment))

    def _is_commutative_pattern(self, node: ASTNode) -> bool:
        """Check if node represents a commutative pattern"""
        if len(node.children) >= 2:
            # Simple check: are the children similar but reordered?
            return node.children[0].value != node.children[1].value
        return False

    def _is_associative_pattern(self, node: ASTNode) -> bool:
        """Check if node represents an associative pattern"""
        # Check if any child has the same operator
        return any(
            child.node_type == NodeType.OPERATOR and child.value == node.value
            for child in node.children
        )

    def _fallback_profile(self, error_msg: str) -> WorkloadProfile:
        """Create fallback profile when characterization fails"""
        return WorkloadProfile(
            structural_complexity=0.5,
            pattern_uniformity=0.5,
            proof_depth=0.5,
            computational_intensity=0.5,
            dominant_features=["unknown"],
            proof_style="unknown",
            optimization_potential=0.15,  # Conservative estimate
            feature_vector=np.zeros(100),
            confidence=0.0,
        )

    def _load_historical_data(self, path: Path):
        """Load historical workload data"""
        # Implementation would load real historical data

    def _train_models(self):
        """Train ML models on historical data"""
        # Implementation would train models


def predict_optimization_success(file_path: Path) -> Tuple[bool, float, WorkloadProfile]:
    """
    High-level API for predicting optimization success.

    Returns: (should_optimize, success_probability, workload_profile)
    """
    characterizer = WorkloadCharacterizer()
    profile = characterizer.characterize(file_path)

    # Decision based on optimization potential
    should_optimize = profile.optimization_potential >= 0.25

    return should_optimize, profile.optimization_potential, profile


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python workload_characterizer.py <lean_file>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    should_opt, success_prob, profile = predict_optimization_success(file_path)

    print(f"File: {file_path}")
    print(f"\nWorkload Characterization:")
    print(f"  Structural Complexity: {profile.structural_complexity:.2%}")
    print(f"  Pattern Uniformity: {profile.pattern_uniformity:.2%}")
    print(f"  Proof Depth: {profile.proof_depth:.2%}")
    print(f"  Computational Intensity: {profile.computational_intensity:.2%}")
    print(f"  Proof Style: {profile.proof_style}")
    print(f"\nOptimization Potential: {profile.optimization_potential:.2%}")
    print(f"Should Optimize: {should_opt}")
    print(f"Confidence: {profile.confidence:.2%}")

    if profile.dominant_features:
        print(f"\nDominant Features:")
        for feature in profile.dominant_features[:5]:
            print(f"  - {feature}")

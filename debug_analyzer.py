#!/usr/bin/env python3
"""Debug the sophisticated pattern analyzer percentage calculations"""

import tempfile
from pathlib import Path

from simpulse.analysis.sophisticated_pattern_analyzer import SophisticatedPatternAnalyzer

# Create a test file with pure arithmetic patterns
test_content = """
import Mathlib.Data.Nat.Basic

theorem arith_0_1 : ∀ n : Nat, n + 0 = n := by simp
theorem arith_0_2 : ∀ n : Nat, 0 + n = n := by simp
theorem arith_0_3 : ∀ n : Nat, n * 1 = n := by simp
theorem arith_0_4 : ∀ n : Nat, 1 * n = n := by simp
theorem arith_0_5 : ∀ n m : Nat, (n + 0) * (m * 1) = n * m := by simp
theorem arith_0_6 : ∀ n : Nat, n - 0 = n := by simp
"""

# Create analyzer
analyzer = SophisticatedPatternAnalyzer()

# Create temporary file
with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
    f.write(test_content)
    test_file = Path(f.name)

try:
    # Analyze file
    print("Analyzing test file...")
    print("=" * 60)

    result = analyzer.analyze_file(test_file)

    print(f"File: {result['file_path']}")
    print(f"Pattern complexity score: {result['pattern_complexity_score']}")
    print(f"Pattern mixing coefficient: {result['pattern_mixing_coefficient']}")

    print("\nDominant patterns:")
    for pattern_type, percentage in result["dominant_patterns"].items():
        print(f"  {pattern_type}: {percentage}%")

    print("\nStructural complexity:")
    for metric, value in result["structural_complexity"].items():
        print(f"  {metric}: {value}")

    print("\nAST metrics:")
    for metric, value in result["ast_metrics"].items():
        print(f"  {metric}: {value}")

    # Debug the fingerprints
    print("\nPattern fingerprints (first 5):")
    for i, fp in enumerate(result["pattern_fingerprints"][:5]):
        print(f"\n  Fingerprint {i+1}:")
        print(f"    Total nodes: {fp['total_nodes']}")
        print(f"    Identity pattern count: {fp['identity_pattern_count']}")
        print(f"    Node type distribution: {fp['node_type_distribution']}")

    # Manually check identity pattern percentage calculation
    print("\n" + "=" * 60)
    print("Manual calculation check:")

    # Parse the file directly
    trees = analyzer.parser.parse_file(test_content)
    print(f"Number of theorems: {len(trees)}")

    # Extract fingerprints
    fingerprints = [analyzer._extract_fingerprint(tree) for tree in trees]

    # Calculate totals
    total_nodes = sum(fp.total_nodes for fp in fingerprints)
    total_identity_patterns = sum(fp.identity_pattern_count for fp in fingerprints)

    print(f"Total nodes across all theorems: {total_nodes}")
    print(f"Total identity patterns: {total_identity_patterns}")
    print(
        f"Manual identity percentage: {(total_identity_patterns / total_nodes * 100) if total_nodes > 0 else 0:.2f}%"
    )

    # Check individual fingerprints
    print("\nIndividual theorem analysis:")
    for i, (tree, fp) in enumerate(zip(trees, fingerprints)):
        print(f"\nTheorem {i+1}: {tree.value}")
        print(f"  Total nodes in theorem: {fp.total_nodes}")
        print(f"  Identity patterns: {fp.identity_pattern_count}")
        print(f"  Node types: {dict(fp.node_type_distribution)}")

finally:
    # Clean up
    test_file.unlink()

#!/usr/bin/env python3
"""Debug the analyzer to see what's in the AST"""

import tempfile
from pathlib import Path

from simpulse.analysis.sophisticated_pattern_analyzer import SophisticatedPatternAnalyzer

# Create a test file with pure arithmetic patterns
test_content = """
theorem arith_0_1 : ∀ n : Nat, n + 0 = n := by simp
"""

# Create analyzer
analyzer = SophisticatedPatternAnalyzer()

# Create temporary file
with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
    f.write(test_content)
    test_file = Path(f.name)

try:
    # Parse the file directly
    trees = analyzer.parser.parse_file(test_content)
    print(f"Number of theorems: {len(trees)}")

    # Look at the first theorem
    if trees:
        tree = trees[0]
        print(f"\nTheorem: {tree.value}")
        print("AST structure:")
        print(tree.to_string())

        # Check what NodeType values are available
        print("\nChecking node types in the tree:")

        def print_node_types(node, indent=""):
            print(f"{indent}Node: {node.node_type} (value: {node.value})")
            # Check if this is an identity pattern
            if hasattr(node.node_type, "value") and node.node_type.value == "identity_pattern":
                print(f"{indent}  ^^^ THIS IS AN IDENTITY PATTERN!")
            for child in node.children:
                print_node_types(child, indent + "  ")

        print_node_types(tree)

        # Test the _count_identity_patterns method
        print("\nTesting _count_identity_patterns:")
        identity_count = analyzer._count_identity_patterns(tree)
        print(f"Identity patterns found by analyzer: {identity_count}")

        # Extract fingerprint
        print("\nExtracting fingerprint:")
        fp = analyzer._extract_fingerprint(tree)
        print(f"Fingerprint identity_pattern_count: {fp.identity_pattern_count}")
        print(f"Fingerprint total_nodes: {fp.total_nodes}")
        print(f"Fingerprint node_type_distribution: {fp.node_type_distribution}")

finally:
    # Clean up
    test_file.unlink()

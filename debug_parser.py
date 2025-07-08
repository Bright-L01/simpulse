#!/usr/bin/env python3
"""Debug the improved parser to see what's happening"""

from simpulse.analysis.improved_lean_parser import ImprovedLeanParser, NodeType

# Test the parser on a simple arithmetic theorem
test_content = """
import Mathlib.Data.Nat.Basic

theorem arith_0_1 : ∀ n : Nat, n + 0 = n := by simp
theorem arith_0_2 : ∀ n : Nat, 0 + n = n := by simp
theorem arith_0_3 : ∀ n : Nat, n * 1 = n := by simp
theorem arith_0_4 : ∀ n : Nat, 1 * n = n := by simp
theorem arith_0_5 : ∀ n m : Nat, (n + 0) * (m * 1) = n * m := by simp
theorem arith_0_6 : ∀ n : Nat, n - 0 = n := by simp
"""

parser = ImprovedLeanParser()

# Parse the content
print("Testing parser on arithmetic content...")
print("=" * 60)

trees = parser.parse_file(test_content)
print(f"Found {len(trees)} top-level AST nodes")

# Analyze each theorem
for i, tree in enumerate(trees):
    print(f"\nTheorem {i+1}: {tree.value}")
    print(f"  Type: {tree.node_type}")
    print(f"  Children: {len(tree.children)}")

    # Look at the proposition
    if tree.children and len(tree.children) > 0:
        prop = tree.children[0]
        print(f"  Proposition AST:")
        print(prop.to_string("    "))

        # Count identity patterns in this theorem
        identity_count = count_identity_patterns(prop)
        print(f"  Identity patterns found: {identity_count}")


def count_identity_patterns(node):
    """Count identity patterns in an AST"""
    count = 0

    if node.node_type == NodeType.IDENTITY_PATTERN:
        count += 1

    elif node.node_type == NodeType.OPERATOR:
        # Check for identity patterns manually
        if node.value == "+" and len(node.children) >= 2:
            left = node.children[0]
            right = node.children[1]
            if (left.node_type == NodeType.LITERAL and left.value == "0") or (
                right.node_type == NodeType.LITERAL and right.value == "0"
            ):
                count += 1
                print(f"      Found identity: {left.value} + {right.value}")

        elif node.value == "*" and len(node.children) >= 2:
            left = node.children[0]
            right = node.children[1]
            if (left.node_type == NodeType.LITERAL and left.value == "1") or (
                right.node_type == NodeType.LITERAL and right.value == "1"
            ):
                count += 1
                print(f"      Found identity: {left.value} * {right.value}")

    # Recurse
    for child in node.children:
        count += count_identity_patterns(child)

    return count


# Test individual expression parsing
print("\n" + "=" * 60)
print("Testing individual expression parsing...")

test_expressions = [
    "n + 0 = n",
    "0 + n = n",
    "n * 1 = n",
    "1 * n = n",
    "n - 0 = n",
    "∀ n : Nat, n + 0 = n",
]

for expr in test_expressions:
    print(f"\nParsing: {expr}")
    ast = parser.parse_expression(expr)
    print(ast.to_string("  "))
    print(f"  Identity count: {count_identity_patterns(ast)}")

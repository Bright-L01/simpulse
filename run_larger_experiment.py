#!/usr/bin/env python3
"""
Run a larger experiment to get meaningful statistical data
"""

import tempfile
from pathlib import Path

from experiment_runner import ExperimentRunner


def create_synthetic_lean_files(count: int = 50) -> list[Path]:
    """Create synthetic Lean files with different patterns for testing"""
    files = []

    patterns = {
        "pure_identity": """
theorem id1 : ∀ n : Nat, n + 0 = n := by simp
theorem id2 : ∀ n : Nat, 0 + n = n := by simp  
theorem id3 : ∀ n : Nat, n * 1 = n := by simp
theorem id4 : ∀ n : Nat, 1 * n = n := by simp
""",
        "mixed_arithmetic": """
theorem mix1 : ∀ n : Nat, n + 0 = n := by simp
theorem mix2 : ∀ xs : List Nat, xs ++ [] = xs := by simp
theorem mix3 : ∀ x y : Nat, x * (y + 0) = x * y := by simp
theorem mix4 : ∀ P : Nat → Prop, (∀ n, P n) → P 0 := by intro P h; exact h 0
""",
        "complex_proof": """
inductive Tree (α : Type) where
  | leaf : α → Tree α
  | node : Tree α → Tree α → Tree α

def tree_size {α : Type} : Tree α → Nat
  | Tree.leaf _ => 1
  | Tree.node l r => tree_size l + tree_size r

theorem tree_size_pos {α : Type} : ∀ t : Tree α, tree_size t > 0 := by
  intro t
  cases t with
  | leaf _ => simp [tree_size]
  | node l r => simp [tree_size, tree_size_pos l, tree_size_pos r]
""",
        "computational": """
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem fact_pos : ∀ n : Nat, factorial n > 0 := by
  intro n
  induction n with
  | zero => simp [factorial]
  | succ k ih => simp [factorial, ih]
""",
        "list_operations": """
theorem append_nil {α : Type} : ∀ xs : List α, xs ++ [] = xs := by
  intro xs
  induction xs with
  | nil => simp
  | cons x xs ih => simp [ih]

theorem length_append {α : Type} : ∀ xs ys : List α, 
  (xs ++ ys).length = xs.length + ys.length := by
  intro xs ys
  induction xs with
  | nil => simp
  | cons x xs ih => simp [ih]
""",
    }

    # Create files with different patterns
    for i in range(count):
        pattern_name = list(patterns.keys())[i % len(patterns)]
        content = patterns[pattern_name]

        # Add some variation
        variation = f"""
-- File {i}: {pattern_name} variation
{content}
theorem extra_{i} : True := by trivial
"""

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(variation)
            files.append(Path(f.name))

    return files


def main():
    print("🧪 CREATING LARGER EXPERIMENT DATASET")
    print("=" * 50)

    # Create synthetic test files
    print("Creating 50 synthetic Lean files with diverse patterns...")
    lean_files = create_synthetic_lean_files(50)

    print(f"✓ Created {len(lean_files)} test files")

    # Run experiments
    runner = ExperimentRunner(output_dir=Path("larger_experiment"))

    print(
        f"\nRunning {len(lean_files)} × {len(runner.STRATEGIES)} = {len(lean_files) * len(runner.STRATEGIES)} experiments..."
    )

    try:
        runner.run_experiments(lean_files, max_workers=2)
        print("✅ Experiments complete!")

        # Clean up temp files
        for f in lean_files:
            f.unlink()

        print(f"\n📊 Results saved to: larger_experiment/")
        print("Now run: python empirical_analyzer.py --results-dir larger_experiment --export")

    except Exception as e:
        print(f"❌ Error: {e}")
        # Clean up on error
        for f in lean_files:
            try:
                f.unlink()
            except:
                pass


if __name__ == "__main__":
    main()

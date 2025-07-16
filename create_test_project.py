#!/usr/bin/env python3
"""
Create a test Lean 4 project for testing Simpulse 2.0.

This creates a realistic project structure with simp rules for testing.
"""

import os
from pathlib import Path

def create_test_project(project_path: Path):
    """Create a test Lean project with simp rules."""
    project_path.mkdir(exist_ok=True)
    
    # Create basic Lean project structure
    (project_path / "lakefile.lean").write_text("""
import Lake
open Lake DSL

package «test_project» where
  -- add any package configuration options here

lean_lib «TestProject» where
  -- add any library configuration options here
""")
    
    (project_path / "lean-toolchain").write_text("4.8.0")
    
    # Create source directory
    src_dir = project_path / "TestProject"
    src_dir.mkdir(exist_ok=True)
    
    # Create Main.lean with various simp rules
    (src_dir / "Main.lean").write_text("""
-- Test project for Simpulse optimization

-- Some basic simp rules with different usage patterns
@[simp]
theorem frequently_used_rule : ∀ n : Nat, n + 0 = n := by
  intro n
  rfl

@[simp]  
theorem occasionally_used_rule : ∀ l : List α, l ++ [] = l := by
  intro l
  simp

@[simp 1500]
theorem rarely_used_rule : ∀ x : Nat, x * 1 = x := by
  intro x
  simp

-- Some theorems that use the simp rules above
theorem test_proof_1 (n : Nat) : n + 0 + 0 = n := by
  simp [frequently_used_rule]

theorem test_proof_2 (l : List Nat) : l ++ [] ++ [] = l := by  
  simp [occasionally_used_rule]

theorem test_proof_3 (x y : Nat) : (x + 0) * 1 = x := by
  simp [frequently_used_rule, rarely_used_rule]

theorem test_proof_4 (a b : Nat) : a + 0 = a ∧ b + 0 = b := by
  simp [frequently_used_rule]

-- More complex proofs that would benefit from optimization
theorem complex_proof (l1 l2 : List Nat) (n m : Nat) : 
  (l1 ++ []) ++ (l2 ++ []) = l1 ++ l2 ∧ n + 0 = n ∧ m + 0 = m := by
  simp [occasionally_used_rule, frequently_used_rule]
""")
    
    # Create Basic.lean with more simp rules
    (src_dir / "Basic.lean").write_text("""
-- Additional simp rules for testing

@[simp]
theorem zero_add (n : Nat) : 0 + n = n := by
  rfl

@[simp]
theorem list_length_nil : List.length ([] : List α) = 0 := by
  rfl

@[simp]  
theorem bool_and_true (b : Bool) : b && true = b := by
  cases b <;> rfl

-- Proofs using these rules
theorem basic_test_1 (n : Nat) : 0 + (n + 0) = n := by
  simp [zero_add]

theorem basic_test_2 (l : List Nat) : List.length [] + List.length l = List.length l := by
  simp [list_length_nil, zero_add]

theorem basic_test_3 (b1 b2 : Bool) : (b1 && true) && (b2 && true) = b1 && b2 := by
  simp [bool_and_true]
""")
    
    print(f"✅ Created test project at: {project_path}")
    print(f"   Files created:")
    print(f"   - lakefile.lean")
    print(f"   - lean-toolchain") 
    print(f"   - TestProject/Main.lean (with 3 simp rules)")
    print(f"   - TestProject/Basic.lean (with 3 more simp rules)")
    print(f"\n🧪 To test Simpulse:")
    print(f"   cd {project_path}")
    print(f"   simpulse analyze .")
    print(f"   simpulse preview .")
    print(f"   simpulse optimize .")


if __name__ == "__main__":
    # Create in current directory
    test_project_path = Path.cwd() / "simpulse_test_project"
    create_test_project(test_project_path)
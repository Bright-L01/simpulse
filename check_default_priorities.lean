-- Check default simp priorities in core Lean

open Lean Meta Simp

-- This would show us actual priorities if we could access them
-- But Lean doesn't expose simp priorities directly in the syntax

-- Let's trace what happens
set_option trace.Meta.Tactic.simp true

-- Simple test to see lemma application order
example (n : Nat) : n + 0 = n := by
  simp
  
-- Another test
example (n : Nat) : (n + 0) * 1 = n := by
  simp

-- Test with explicit -simp to see if lemma is marked simp by default
attribute [-simp] Nat.add_zero

example (n : Nat) : n + 0 = n := by
  simp -- This should fail now
  sorry
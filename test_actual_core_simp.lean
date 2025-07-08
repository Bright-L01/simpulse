-- What ACTUALLY exists in core Lean 4

-- These DO work with simp in core:
example (n : Nat) : n + 0 = n := by simp
example (n : Nat) : 0 + n = n := by simp  
example (n : Nat) : n * 1 = n := by simp
example (n : Nat) : 1 * n = n := by simp

-- Test our attribute commands
attribute [simp 1200] Nat.add_zero
attribute [simp 1200] Nat.zero_add

-- This should fail because these don't exist
#check Nat.add_zero
#check @Nat.zero_add
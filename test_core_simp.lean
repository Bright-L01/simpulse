-- Test what simp lemmas exist in core Lean 4 (no Mathlib)

#check @add_zero
#check @zero_add  
#check @mul_one
#check @one_mul

-- Test if these are simp lemmas
example (n : Nat) : n + 0 = n := by simp
example (n : Nat) : 0 + n = n := by simp
example (n : Nat) : n * 1 = n := by simp
example (n : Nat) : 1 * n = n := by simp

-- List what simp can solve
example (n : Nat) : n + 0 = n ∧ 0 + n = n := by simp
example (b : Bool) : (if true then b else false) = b := by simp
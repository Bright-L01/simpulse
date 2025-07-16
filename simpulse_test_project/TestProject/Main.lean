
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

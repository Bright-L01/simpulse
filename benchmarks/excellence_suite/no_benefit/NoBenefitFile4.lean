-- NoBenefitFile4.lean
theorem no_simp_rule_4 (n : Nat) : n + 0 = n := Nat.add_zero n
example : 5 + 0 = 5 := by rw [no_simp_rule_4]

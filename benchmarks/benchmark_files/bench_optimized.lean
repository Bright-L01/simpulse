
-- Benchmark: With simp priority optimization
attribute [simp 1200] Nat.add_zero Nat.zero_add
attribute [simp 1199] Nat.mul_one Nat.one_mul

example (n : Nat) : n + 0 = n := by simp
example (n : Nat) : 0 + n = n := by simp
example (n : Nat) : n * 1 = n := by simp
example (n : Nat) : 1 * n = n := by simp

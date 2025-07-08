
-- Benchmark: Simple arithmetic
example (n : Nat) : n + 0 = n := by simp
example (n : Nat) : 0 + n = n := by simp
example (n : Nat) : n * 1 = n := by simp
example (n : Nat) : 1 * n = n := by simp

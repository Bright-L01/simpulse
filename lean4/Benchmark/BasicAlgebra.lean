-- BasicAlgebra.lean
-- Test simp performance on arithmetic simplifications

theorem add_zero (n : Nat) : n + 0 = n := by simp

theorem zero_add (n : Nat) : 0 + n = n := by simp

theorem add_comm (n m : Nat) : n + m = m + n := by simp

theorem add_assoc (n m k : Nat) : (n + m) + k = n + (m + k) := by simp

theorem mul_one (n : Nat) : n * 1 = n := by simp

theorem one_mul (n : Nat) : 1 * n = n := by simp

theorem mul_zero (n : Nat) : n * 0 = 0 := by simp

theorem zero_mul (n : Nat) : 0 * n = 0 := by simp

theorem mul_comm (n m : Nat) : n * m = m * n := by simp

theorem mul_add (n m k : Nat) : n * (m + k) = n * m + n * k := by simp

theorem add_mul (n m k : Nat) : (n + m) * k = n * k + m * k := by simp

theorem pow_zero (n : Nat) : n ^ 0 = 1 := by simp

theorem pow_succ (n m : Nat) : n ^ (m + 1) = n * n ^ m := by simp

theorem sub_self (n : Nat) : n - n = 0 := by simp

theorem add_sub_cancel (n m : Nat) : n + m - m = n := by simp

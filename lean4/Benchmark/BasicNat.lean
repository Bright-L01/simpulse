-- BasicNat.lean
-- Test simp performance on natural number operations

theorem succ_pred (n : Nat) (h : n > 0) : n.pred.succ = n := by simp [Nat.succ_pred h]

theorem pred_succ (n : Nat) : n.succ.pred = n := by simp

theorem add_succ (n m : Nat) : n + m.succ = (n + m).succ := by simp

theorem succ_add (n m : Nat) : n.succ + m = (n + m).succ := by simp

theorem add_one (n : Nat) : n + 1 = n.succ := by simp

theorem one_add (n : Nat) : 1 + n = n.succ := by simp

theorem mul_succ (n m : Nat) : n * m.succ = n * m + n := by simp

theorem succ_mul (n m : Nat) : n.succ * m = n * m + m := by simp

theorem zero_lt_succ (n : Nat) : 0 < n.succ := by simp

theorem succ_le_succ (n m : Nat) : n.succ ≤ m.succ ↔ n ≤ m := by simp

theorem lt_succ_self (n : Nat) : n < n.succ := by simp

theorem le_refl (n : Nat) : n ≤ n := by simp

theorem le_trans {n m k : Nat} (h1 : n ≤ m) (h2 : m ≤ k) : n ≤ k := by simp [Nat.le_trans h1 h2]

theorem min_self (n : Nat) : min n n = n := by simp

theorem max_self (n : Nat) : max n n = n := by simp

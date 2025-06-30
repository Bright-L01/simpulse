-- Test file for Simpulse optimization

-- Simple rules (should be high priority)
@[simp] theorem add_zero (n : Nat) : n + 0 = n := Nat.add_zero n
@[simp] theorem zero_add (n : Nat) : 0 + n = n := Nat.zero_add n
@[simp] theorem mul_one (n : Nat) : n * 1 = n := Nat.mul_one n

-- Test theorems using simp
theorem test1 (x : Nat) : (x + 0) * 1 = x := by simp
theorem test2 (a b : Nat) : 0 + a * 1 + 0 = a := by simp
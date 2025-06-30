-- Demo test to show simp optimization performance

-- Complex rules (should be low priority)
@[simp] theorem complex1 (a b c d : Nat) : (a + b) * (c + d) = a * c + a * d + b * c + b * d := by ring
@[simp] theorem complex2 (x y z : Nat) : x * (y + z) = x * y + x * z := by ring

-- Simple common rules (should be high priority)
@[simp] theorem simple_add_zero (n : Nat) : n + 0 = n := Nat.add_zero n
@[simp] theorem simple_zero_add (n : Nat) : 0 + n = n := Nat.zero_add n
@[simp] theorem simple_mul_one (n : Nat) : n * 1 = n := Nat.mul_one n
@[simp] theorem simple_one_mul (n : Nat) : 1 * n = n := Nat.one_mul n

-- Test cases
theorem test1 (x : Nat) : (x + 0) * 1 = x := by simp
theorem test2 (a b : Nat) : 0 + a * 1 + b + 0 = a + b := by simp
theorem test3 : ∀ n : Nat, n + 0 = n := by simp
theorem test4 : ∀ n : Nat, (n + 0) * 1 + 0 = n := by simp

-- Many tests to amplify performance difference
section ManyTests
variable (n m : Nat)

theorem t1 : (n + 0) * 1 = n := by simp
theorem t2 : 0 + n * 1 = n := by simp
theorem t3 : n * 1 + 0 = n := by simp
theorem t4 : (0 + n) * 1 = n := by simp
theorem t5 : 1 * n + 0 = n := by simp
theorem t6 : n + 0 + 0 = n := by simp
theorem t7 : 0 + 0 + n = n := by simp
theorem t8 : (n * 1) * 1 = n := by simp
theorem t9 : 1 * (1 * n) = n := by simp
theorem t10 : (n + 0) + 0 = n := by simp

end ManyTests
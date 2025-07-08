-- TestSimp.lean
-- A simple file to test simp performance measurement

-- Basic arithmetic lemmas that simp uses
theorem test_add_zero (n : Nat) : n + 0 = n := by simp

theorem test_zero_add (n : Nat) : 0 + n = n := by simp

theorem test_mul_one (n : Nat) : n * 1 = n := by simp

theorem test_one_mul (n : Nat) : 1 * n = n := by simp

theorem test_mul_zero (n : Nat) : n * 0 = 0 := by simp

theorem test_zero_mul (n : Nat) : 0 * n = 0 := by simp

-- Some simple custom simp lemmas
@[simp 110] theorem my_add_comm (a b : Nat) : a + b = b + a := Nat.add_comm a b

@[simp 120] theorem my_mul_comm (a b : Nat) : a * b = b * a := Nat.mul_comm a b

-- More complex theorems using simp
theorem complex_simp_1 (a b c : Nat) : (a + 0) * 1 + (b * 0 + c + 0) = a + c := by
  simp

theorem complex_simp_2 (a b c : Nat) : (a + b + 0) * (1 + 0) + (0 + c * 1) = a + b + c := by
  simp

theorem complex_simp_3 (x y z : Nat) : (x + y) * 0 + z * 1 + 0 = z := by
  simp

-- Many simple theorems to give simp more work
theorem many_simps_1 (a b c d : Nat) : a + 0 + b * 1 + c + 0 * d = a + b + c := by simp
theorem many_simps_2 (a b c d : Nat) : 0 + a + 1 * b + 0 + c + d * 0 = a + b + c := by simp
theorem many_simps_3 (a b c d : Nat) : a * 1 + 0 + b + c * 0 + d + 0 = a + b + d := by simp
theorem many_simps_4 (a b c d : Nat) : 1 * a + b + 0 * c + 0 + d + 0 = a + b + d := by simp
theorem many_simps_5 (a b c d : Nat) : a + b * 0 + c + d * 1 + 0 + 0 = a + c + d := by simp

-- Test with our custom lemmas
theorem use_custom_1 (a b : Nat) : (0 + a) + (b + 0) = a + b := by simp
theorem use_custom_2 (a b : Nat) : (1 * a) + (b * 1) = a + b := by simp
theorem use_custom_3 (x y : Nat) : x + y = y + x := by simp [my_add_comm]
theorem use_custom_4 (x y : Nat) : x * y = y * x := by simp [my_mul_comm]
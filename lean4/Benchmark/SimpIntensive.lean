-- SimpIntensive.lean
-- More simp-intensive tests to better measure performance differences

-- Custom simp lemmas with different complexities
@[simp] theorem custom_add_zero (n : Nat) : n + 0 = n := Nat.add_zero n
@[simp] theorem custom_zero_add (n : Nat) : 0 + n = n := Nat.zero_add n
@[simp] theorem custom_mul_one (n : Nat) : n * 1 = n := Nat.mul_one n
@[simp] theorem custom_one_mul (n : Nat) : 1 * n = n := Nat.one_mul n

-- More complex custom lemmas
@[simp 100] theorem distrib_1 (a b c : Nat) : a * (b + c) = a * b + a * c := Nat.mul_add a b c
@[simp] theorem distrib_2 (a b c : Nat) : (a + b) * c = a * c + b * c := Nat.add_mul a b c

-- Recursive simp applications
theorem simp_chain_1 (a b c d : Nat) : 
  (a + 0) * (b + 0) + (c * 1) * (d * 1) = a * b + c * d := by simp

theorem simp_chain_2 (a b c d e : Nat) :
  ((a + 0) + (b * 1)) * ((c + 0) + (d * 1)) + e * 0 = (a + b) * (c + d) := by simp

theorem simp_chain_3 (a b c : Nat) :
  (a + 0) * ((b + 0) * (c + 0)) + 0 * (a * b * c) = a * (b * c) := by simp

-- Nested simp applications
theorem nested_simp_1 (a b c d : Nat) :
  ((a + 0) * 1 + 0) * ((b * 1 + 0) + (c + 0 * d)) = a * (b + c) := by simp

theorem nested_simp_2 (x y z : Nat) :
  (((x + 0) * 1) + ((y * 1) + 0)) * (z + 0) = (x + y) * z := by simp

-- Heavy simp usage
theorem heavy_simp_1 (a b c d e f : Nat) :
  (a + 0) * (b * 1) + (c + 0) * (d * 1) + (e + 0) * (f * 1) = a * b + c * d + e * f := by simp

theorem heavy_simp_2 (a b c d : Nat) :
  ((a + b) + 0) * ((c + d) * 1) + 0 * (a + b + c + d) = (a + b) * (c + d) := by simp

-- Distribution tests
theorem distrib_test_1 (a b c d : Nat) :
  a * (b + c + d + 0) = a * b + a * c + a * d := by simp [distrib_1]

theorem distrib_test_2 (a b c d e : Nat) :
  (a + b + 0) * (c + d + e * 1) = (a + b) * (c + d + e) := by simp

-- Large expressions
theorem large_expr_1 (a b c d e f g h : Nat) :
  (a + 0) + (b * 1) + (c + 0) + (d * 1) + (e + 0) + (f * 1) + (g + 0) + (h * 1) =
  a + b + c + d + e + f + g + h := by simp

theorem large_expr_2 (a b c d : Nat) :
  (a * 1 + 0) * (b + 0 * c) + (c * 1 + 0) * (d + 0 * a) = a * b + c * d := by simp

-- Generate many similar theorems to increase simp workload
theorem work_1 (a b : Nat) : (a + 0) * (b * 1) = a * b := by simp
theorem work_2 (a b : Nat) : (a * 1) * (b + 0) = a * b := by simp
theorem work_3 (a b : Nat) : (a + 0 * b) * (b * 1 + 0) = a * b := by simp
theorem work_4 (a b : Nat) : (a * 1 + 0 * a) * (b + 0 * b) = a * b := by simp
theorem work_5 (a b : Nat) : ((a + 0) + 0) * ((b * 1) * 1) = a * b := by simp
theorem work_6 (a b : Nat) : (a * 1 * 1) * (b + 0 + 0) = a * b := by simp
theorem work_7 (a b c : Nat) : (a + 0) * (b * 1) + c * 0 = a * b := by simp
theorem work_8 (a b c : Nat) : a * 0 + (b + 0) * (c * 1) = b * c := by simp
theorem work_9 (a b c : Nat) : (a * 1 + b * 0) * (c + 0) = a * c := by simp
theorem work_10 (a b c : Nat) : (a + 0 * b) * (c * 1 + 0 * c) = a * c := by simp
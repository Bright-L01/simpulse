-- Test file with various @[simp] attribute patterns

-- Basic simp attribute
@[simp] theorem add_zero' (n : Nat) : n + 0 = n := Nat.add_zero n

-- Simp with priority
@[simp, priority := 500] theorem zero_add' (n : Nat) : 0 + n = n := Nat.zero_add n

-- Simp with high priority
@[simp, high_priority] theorem mul_one' (n : Nat) : n * 1 = n := Nat.mul_one n

-- Multi-line simp attribute
@[simp] 
theorem one_mul' (n : Nat) : 1 * n = n := Nat.one_mul n

-- Simp with custom priority on separate lines
@[simp, priority := 1500]
theorem complex_rule (a b c d : Nat) :
  (a + b) * (c + d) = a * c + a * d + b * c + b * d := by ring

-- Simp with low priority  
@[simp, priority := 100] theorem simple_eq (x : Nat) : x = x := rfl

-- Multiple attributes
@[simp, priority := 800]
@[inline]
theorem distributive (a b c : Nat) : a * (b + c) = a * b + a * c := Nat.mul_add a b c

-- Edge case: simp in comment (should be ignored)
-- @[simp] theorem commented_out : True := trivial

-- Normal theorem without simp
theorem not_simp (x y : Nat) : x + y = y + x := Nat.add_comm x y
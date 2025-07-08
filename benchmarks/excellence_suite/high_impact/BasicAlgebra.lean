-- BasicAlgebra.lean  
-- Basic algebraic operations with heavy simp usage
-- Expected: 2x+ speedup from frequent algebraic simplification rules

-- Ring axioms (very frequently used)
@[simp] theorem add_zero_ring {R : Type*} [Ring R] (a : R) : a + 0 = a := by
  rw [add_zero]

@[simp] theorem zero_add_ring {R : Type*} [Ring R] (a : R) : 0 + a = a := by
  rw [zero_add]

@[simp] theorem mul_one_ring {R : Type*} [Ring R] (a : R) : a * 1 = a := by
  rw [mul_one]

@[simp] theorem one_mul_ring {R : Type*} [Ring R] (a : R) : 1 * a = a := by
  rw [one_mul]

@[simp] theorem mul_zero_ring {R : Type*} [Ring R] (a : R) : a * 0 = 0 := by
  rw [mul_zero]

@[simp] theorem zero_mul_ring {R : Type*} [Ring R] (a : R) : 0 * a = 0 := by
  rw [zero_mul]

@[simp] theorem add_neg_cancel {R : Type*} [Ring R] (a : R) : a + (-a) = 0 := by
  rw [add_neg_cancel]

@[simp] theorem neg_add_cancel {R : Type*} [Ring R] (a : R) : (-a) + a = 0 := by
  rw [neg_add_cancel]

-- Commutativity (frequently used)
@[simp] theorem add_comm_ring {R : Type*} [CommRing R] (a b : R) : a + b = b + a := by
  rw [add_comm]

@[simp] theorem mul_comm_ring {R : Type*} [CommRing R] (a b : R) : a * b = b * a := by
  rw [mul_comm]

-- Associativity (moderately used)
@[simp] theorem add_assoc_ring {R : Type*} [Ring R] (a b c : R) : (a + b) + c = a + (b + c) := by
  rw [add_assoc]

@[simp] theorem mul_assoc_ring {R : Type*} [Ring R] (a b c : R) : (a * b) * c = a * (b * c) := by
  rw [mul_assoc]

-- Distribution laws (moderately used)
@[simp] theorem left_distrib_ring {R : Type*} [Ring R] (a b c : R) : a * (b + c) = a * b + a * c := by
  rw [left_distrib]

@[simp] theorem right_distrib_ring {R : Type*} [Ring R] (a b c : R) : (a + b) * c = a * c + b * c := by
  rw [right_distrib]

-- Heavy usage examples (simulates real algebraic manipulation)
variable {R : Type*} [CommRing R]

example (a b c : R) : a + 0 + b + 0 + c = a + b + c := by simp

example (a b : R) : 0 + a + 0 + b = a + b := by simp

example (a : R) : a * 1 * 1 = a := by simp

example (a b c d : R) : a + b + c + d = d + c + b + a := by simp [add_comm_ring, add_assoc_ring]

example (a b : R) : (a + 0) * (1 + b) = a * (1 + b) := by simp

example (a : R) : (a + 0) + (0 + a) = a + a := by simp

example (a b c : R) : a * (b + 0) + c * 1 = a * b + c := by simp

example (x y z : R) : (x + y + 0) * (z * 1) = (x + y) * z := by simp

example (m n : R) : m * 1 + n * 1 + 0 = m + n := by simp

example (p q r : R) : (p + 0) + (q + 0) + (r + 0) = p + q + r := by simp

-- Negation patterns (frequent in real algebra)
example (a b : R) : a + (-a) + b = b := by simp

example (a : R) : (-a) + a + 0 = 0 := by simp

example (a b : R) : a * 0 + b = b := by simp

example (a b : R) : 0 * a + 0 * b = 0 := by simp

-- Distribution patterns (common in expanding expressions)
example (a b c : R) : a * (b + c + 0) = a * b + a * c := by simp

example (a b c : R) : (a + b + 0) * c = a * c + b * c := by simp

example (a b c d : R) : (a + b) * (c + d) = a * c + a * d + b * c + b * d := by
  simp [left_distrib_ring, right_distrib_ring, add_assoc_ring]

-- Complex algebraic expressions
example (a b c d e : R) : 
  ((a + 0) + (b * 1)) + ((c + 0) * (d * 1)) + (e + 0) = 
  a + b + (c * d) + e := by simp

example (x y z w : R) :
  (x + 0) * (y + 0) + (z * 1) * (w * 1) = x * y + z * w := by simp

example (a : R) : 
  (a + 0) + (a * 1) + (0 + a) + (1 * a) = a + a + a + a := by simp

-- Nested operations (stress test for simp)
example (a b c : R) : 
  ((a + 0) + (b + 0)) + ((c + 0) + 0) = (a + b) + c := by simp

example (x y : R) :
  (x * 1) * (y * 1) * 1 = x * y := by simp

-- Ring homomorphism patterns
example (f : R →+* R) (a : R) : f (a + 0) = f a + 0 := by simp

example (f : R →+* R) (a : R) : f (a * 1) = f a * 1 := by simp

example (f : R →+* R) : f 0 = 0 := by simp

-- Power operations (if available)
example (a : R) : a^1 = a := by simp

example (a : R) : a^0 = 1 := by simp

-- Polynomial-like expressions
example (a b c x : R) : a * x + b * x + c * x = (a + b + c) * x := by
  simp [← right_distrib_ring, add_assoc_ring]

example (a b x y : R) : a * x + a * y + b * x + b * y = (a + b) * (x + y) := by
  simp [left_distrib_ring, right_distrib_ring, add_assoc_ring, add_comm_ring]

-- Very frequent micro-patterns  
example (a : R) : a + 0 = a := by simp
example (a : R) : 0 + a = a := by simp
example (a : R) : a * 1 = a := by simp
example (a : R) : 1 * a = a := by simp
example (a : R) : a * 0 = 0 := by simp
example (a : R) : 0 * a = 0 := by simp

-- Edge cases
example : (0 : R) + 0 = 0 := by simp
example : (1 : R) * 1 = 1 := by simp
example (a : R) : a + 0 + 0 + 0 = a := by simp
example (a : R) : a * 1 * 1 * 1 = a := by simp
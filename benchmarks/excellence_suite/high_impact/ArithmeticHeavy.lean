-- ArithmeticHeavy.lean
-- Heavy arithmetic with many frequently-used simp rules
-- Expected: 2x+ speedup from priority optimization

-- Core arithmetic rules (very frequently used)
@[simp] theorem add_zero_nat (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.add_succ, ih]

@[simp] theorem zero_add_nat (n : Nat) : 0 + n = n := by rfl

@[simp] theorem mul_one_nat (n : Nat) : n * 1 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.mul_succ, ih]

@[simp] theorem one_mul_nat (n : Nat) : 1 * n = n := by rfl

@[simp] theorem add_comm_nat (a b : Nat) : a + b = b + a := by
  induction a with
  | zero => simp
  | succ a ih => simp [Nat.add_succ, Nat.succ_add, ih]

-- Medium frequency rules  
@[simp] theorem add_assoc_nat (a b c : Nat) : (a + b) + c = a + (b + c) := by
  induction a with
  | zero => simp
  | succ a ih => simp [Nat.add_succ, ih]

@[simp] theorem mul_comm_nat (a b : Nat) : a * b = b * a := by
  induction a with
  | zero => simp [Nat.zero_mul]
  | succ a ih => simp [Nat.succ_mul, Nat.mul_succ, ih, add_comm_nat]

@[simp] theorem mul_assoc_nat (a b c : Nat) : (a * b) * c = a * (b * c) := by
  induction a with
  | zero => simp
  | succ a ih => simp [Nat.succ_mul, Nat.add_mul, ih]

-- Distribution laws (moderately used)
@[simp] theorem left_distrib_nat (a b c : Nat) : a * (b + c) = a * b + a * c := by
  induction a with
  | zero => simp
  | succ a ih => simp [Nat.succ_mul, ih, add_assoc_nat, add_comm_nat]

@[simp] theorem right_distrib_nat (a b c : Nat) : (a + b) * c = a * c + b * c := by
  simp [mul_comm_nat, left_distrib_nat]

-- Proofs that heavily use these rules (simulates real usage)
example (a b c : Nat) : a + 0 + b + 0 + c = a + b + c := by simp
example (a b : Nat) : 0 + a + 0 + b = a + b := by simp  
example (n : Nat) : n * 1 * 1 = n := by simp
example (a b c d : Nat) : a + b + c + d = d + c + b + a := by simp [add_comm_nat, add_assoc_nat]
example (a b : Nat) : (a + 0) * (1 + b) = a * (1 + b) := by simp
example (n : Nat) : (n + 0) + (0 + n) = n + n := by simp
example (a b c : Nat) : a * (b + 0) + c * 1 = a * b + c := by simp
example (x y z : Nat) : (x + y + 0) * (z * 1) = (x + y) * z := by simp
example (m n : Nat) : m * 1 + n * 1 + 0 = m + n := by simp
example (p q r : Nat) : (p + 0) + (q + 0) + (r + 0) = p + q + r := by simp

-- More complex patterns that heavily exercise simp
example (a b c d e : Nat) : 
  ((a + 0) + (b * 1)) + ((c + 0) * (d * 1)) + (e + 0) = 
  a + b + (c * d) + e := by simp

example (x y z w : Nat) :
  (x + 0) * (y + 0) + (z * 1) * (w * 1) = x * y + z * w := by simp

example (n : Nat) : 
  (n + 0) + (n * 1) + (0 + n) + (1 * n) = n + n + n + n := by simp

-- Nested arithmetic (stress test for simp)
example (a b c : Nat) : 
  ((a + 0) + (b + 0)) + ((c + 0) + 0) = (a + b) + c := by simp

example (x y : Nat) :
  (x * 1) * (y * 1) * 1 = x * y := by simp

-- Very frequent patterns in real code
example (i j k : Nat) : i + j + 0 + k = i + j + k := by simp
example (a b : Nat) : a * 1 + b * 1 = a + b := by simp
example (m n : Nat) : 0 + m + 0 + n + 0 = m + n := by simp
example (p q : Nat) : (p + 0) + (0 + q) = p + q := by simp
example (r s t : Nat) : r * 1 + s * 1 + t * 1 = r + s + t := by simp

-- Edge cases
example : (0 + 0) + (0 * 1) = 0 := by simp
example (n : Nat) : n + 0 + 0 + 0 + 0 = n := by simp
example (n : Nat) : n * 1 * 1 * 1 * 1 = n := by simp
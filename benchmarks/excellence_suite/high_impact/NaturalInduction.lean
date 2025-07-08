-- NaturalInduction.lean
-- Natural number operations with heavy simp usage in inductive proofs
-- Expected: 2x+ speedup from frequent Nat simplification rules

-- Core Nat rules (very frequently used in inductive proofs)
@[simp] theorem nat_zero_add (n : Nat) : 0 + n = n := by rfl

@[simp] theorem nat_add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.add_succ, ih]

@[simp] theorem nat_succ_add (m n : Nat) : Nat.succ m + n = Nat.succ (m + n) := by
  rfl

@[simp] theorem nat_add_succ (m n : Nat) : m + Nat.succ n = Nat.succ (m + n) := by
  induction m with
  | zero => simp
  | succ m ih => simp [Nat.succ_add, ih]

@[simp] theorem nat_zero_mul (n : Nat) : 0 * n = 0 := by rfl

@[simp] theorem nat_mul_zero (n : Nat) : n * 0 = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.succ_mul, ih]

@[simp] theorem nat_one_mul (n : Nat) : 1 * n = n := by
  simp [Nat.one_mul]

@[simp] theorem nat_mul_one (n : Nat) : n * 1 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.succ_mul, ih]

-- Inductive patterns (very frequently used)
@[simp] theorem nat_succ_mul (m n : Nat) : Nat.succ m * n = n + m * n := by
  rfl

@[simp] theorem nat_mul_succ (m n : Nat) : m * Nat.succ n = m + m * n := by
  induction m with
  | zero => simp
  | succ m ih => simp [Nat.succ_mul, ih, Nat.add_assoc, Nat.add_comm]

-- Heavy inductive proof patterns (simulates real Nat reasoning)
theorem add_comm_nat (m n : Nat) : m + n = n + m := by
  induction m with
  | zero => simp
  | succ m ih => simp [Nat.succ_add, Nat.add_succ, ih]

theorem add_assoc_nat (l m n : Nat) : (l + m) + n = l + (m + n) := by
  induction l with
  | zero => simp
  | succ l ih => simp [Nat.succ_add, ih]

theorem mul_comm_nat (m n : Nat) : m * n = n * m := by
  induction m with
  | zero => simp
  | succ m ih => simp [Nat.succ_mul, Nat.mul_succ, ih, add_comm_nat]

theorem mul_assoc_nat (l m n : Nat) : (l * m) * n = l * (m * n) := by
  induction l with
  | zero => simp
  | succ l ih => simp [Nat.succ_mul, Nat.add_mul, ih]

theorem left_distrib_nat (l m n : Nat) : l * (m + n) = l * m + l * n := by
  induction l with
  | zero => simp
  | succ l ih => simp [Nat.succ_mul, ih, add_assoc_nat, add_comm_nat]

-- Complex inductive proofs that heavily exercise simp
theorem add_mul_nat (l m n : Nat) : (l + m) * n = l * n + m * n := by
  induction l with
  | zero => simp
  | succ l ih => simp [Nat.succ_add, Nat.succ_mul, ih, add_assoc_nat]

theorem pow_add_nat (m : Nat) (a b : Nat) : m ^ (a + b) = m ^ a * m ^ b := by
  induction a with
  | zero => simp [Nat.pow_zero, Nat.one_mul]
  | succ a ih => simp [Nat.pow_succ, Nat.add_succ, ih, mul_assoc_nat]

theorem pow_mul_nat (m a b : Nat) : m ^ (a * b) = (m ^ a) ^ b := by
  induction b with
  | zero => simp [Nat.mul_zero, Nat.pow_zero]
  | succ b ih => simp [Nat.mul_succ, pow_add_nat, Nat.pow_succ, ih]

-- List length calculations (frequent pattern)
theorem length_repeat (n : Nat) (a : α) : (List.replicate n a).length = n := by
  induction n with
  | zero => simp [List.replicate]
  | succ n ih => simp [List.replicate, List.length_cons, ih]

theorem length_range (n : Nat) : (List.range n).length = n := by
  induction n with
  | zero => simp [List.range]
  | succ n ih => simp [List.range, List.length_cons, ih]

-- Heavy usage examples in inductive contexts
example (n : Nat) : n + 0 + 0 = n := by simp

example (m n : Nat) : m * 1 + n * 1 = m + n := by simp

example (a b c : Nat) : (a + b) + c + 0 = a + (b + c) := by simp [add_assoc_nat]

example (n : Nat) : (n + 0) * (1 + 0) = n := by simp

-- Induction stress tests
example (n : Nat) : 
  List.range n |>.map (· + 0) |>.length = n := by simp [length_range]

example (m n : Nat) :
  (List.replicate m 0).length + (List.replicate n 1).length = m + n := by
  simp [length_repeat]

-- Nested inductive patterns
theorem double_induction (f : Nat → Nat → Nat) (base_zero : ∀ n, f 0 n = n) 
    (base_succ : ∀ m n, f (m + 1) n = f m n + 1) :
    ∀ m n, f m n = m + n := by
  intro m n
  induction m with
  | zero => simp [base_zero]
  | succ m ih => simp [base_succ, ih, Nat.succ_add]

-- Performance stress patterns with heavy simp usage
example (n : Nat) : (n + 0) + (0 + n) + (n * 1) + (1 * n) = n + n + n + n := by simp

example (a b c d : Nat) :
  (a + 0) * (b + 0) + (c * 1) * (d * 1) = a * b + c * d := by simp

example (m n p : Nat) :
  ((m + 0) + (n + 0)) * (p * 1) = (m + n) * p := by simp

-- Recursive structure operations
theorem list_sum_replicate (n k : Nat) : 
  (List.replicate n k).sum = n * k := by
  induction n with
  | zero => simp [List.replicate, List.sum]
  | succ n ih => simp [List.replicate, List.sum, ih, Nat.succ_mul]

-- Very frequent micro-patterns in inductive proofs
example (n : Nat) : n + 0 = n := by simp
example (n : Nat) : 0 + n = n := by simp  
example (n : Nat) : n * 1 = n := by simp
example (n : Nat) : 1 * n = n := by simp
example (n : Nat) : n * 0 = 0 := by simp
example (n : Nat) : 0 * n = 0 := by simp

-- Edge cases that appear in base cases
example : 0 + 0 = 0 := by simp
example : 0 * 1 = 0 := by simp
example : 1 * 0 = 0 := by simp
example (n : Nat) : Nat.succ n + 0 = Nat.succ n := by simp
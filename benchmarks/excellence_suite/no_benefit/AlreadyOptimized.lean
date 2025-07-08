-- AlreadyOptimized.lean
-- All rules already have well-tuned priorities
-- Expected: No benefit (Simpulse should detect this and skip optimization)

-- All rules already have optimal priorities
@[simp, priority := 100] theorem opt_add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.add_succ, ih]

@[simp, priority := 110] theorem opt_zero_add (n : Nat) : 0 + n = n := by rfl

@[simp, priority := 120] theorem opt_mul_one (n : Nat) : n * 1 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.mul_succ, ih]

@[simp, priority := 130] theorem opt_one_mul (n : Nat) : 1 * n = n := by rfl

@[simp, priority := 200] theorem opt_list_append_nil (l : List α) : l ++ [] = l := by
  induction l with
  | nil => rfl
  | cons head tail ih => simp [List.cons_append, ih]

@[simp, priority := 210] theorem opt_list_nil_append (l : List α) : [] ++ l = l := by rfl

@[simp, priority := 300] theorem opt_string_append_empty (s : String) : s ++ "" = s := by rfl

@[simp, priority := 310] theorem opt_empty_append_string (s : String) : "" ++ s = s := by rfl

@[simp, priority := 400] theorem opt_option_map_none (f : α → β) : 
  (none : Option α).map f = none := by rfl

@[simp, priority := 410] theorem opt_option_map_some (f : α → β) (a : α) : 
  (some a).map f = some (f a) := by rfl

-- All patterns use the already-optimized rules
example (n m : Nat) : n + 0 + m * 1 = n + m := by simp

example (l1 l2 : List String) : l1 ++ [] ++ l2 = l1 ++ l2 := by simp

example (s : String) : s ++ "" ++ "" = s := by simp

example (opt : Option Nat) : opt.map (· + 0) = opt := by cases opt <;> simp

-- No optimization opportunities remain
example (data : List Nat) : data ++ [] = data := by simp [opt_list_append_nil]

example (x y : Nat) : x + 0 + y + 0 = x + y := by simp [opt_add_zero]

-- All priorities are already well-balanced for usage patterns
example (a b c d : Nat) : 
  (a + 0) + (b * 1) + (c + 0) + (d * 1) = a + b + c + d := by simp

-- Simpulse should detect this file is already optimized
example : (0 : Nat) + 0 = 0 := by simp [opt_add_zero]
example : ("" : String) ++ "" = "" := by simp [opt_string_append_empty]
example : (none : Option Nat).map (· + 1) = none := by simp [opt_option_map_none]
-- MixedOperations.lean
-- Mixed operations with some simp optimization opportunities
-- Expected: Modest improvement (20-50% speedup from selective optimization)

-- Some frequently used rules (moderate optimization potential)
@[simp] theorem mixed_add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.add_succ, ih]

@[simp] theorem mixed_mul_one (n : Nat) : n * 1 = n := by
  induction n with  
  | zero => rfl
  | succ n ih => simp [Nat.mul_succ, ih]

-- Some moderately used rules  
@[simp] theorem mixed_list_append_nil (l : List α) : l ++ [] = l := by
  induction l with
  | nil => rfl
  | cons head tail ih => simp [List.cons_append, ih]

@[simp] theorem mixed_option_map_some (f : α → β) (a : α) : 
  (some a).map f = some (f a) := by rfl

-- Some rarely used rules (low optimization potential)
@[simp] theorem mixed_string_singleton_append (c : Char) (s : String) :
  String.singleton c ++ s = String.push s c := by
  sorry -- Complex implementation

@[simp] theorem mixed_complex_pattern (a b c d : Nat) :
  (a + b) * (c + d) = a * c + a * d + b * c + b * d := by
  ring

-- Moderate usage patterns (some benefit but not dramatic)
example (n m : Nat) : n + 0 + m * 1 = n + m := by simp

example (l1 l2 : List String) : l1 ++ [] ++ l2 = l1 ++ l2 := by simp

example (opt : Option Nat) (f : Nat → String) :
  opt.map f |>.map String.length = opt.map (fun n => (f n).length) := by
  simp [Option.map_map]

-- Mixed complexity proofs
example (a b : Nat) (l : List Nat) :
  (l ++ []).map (· + 0) |>.filter (· < a * 1 + b) = 
  l.map id |>.filter (· < a + b) := by simp

example (s : String) (opt : Option String) :
  (opt.map (· ++ "") |>.map (s ++ ·)).isSome = opt.isSome := by
  cases opt <;> simp

-- Some manual optimizations mixed in (reduces optimization potential)
@[simp, priority := 100] theorem mixed_high_priority_rule (n : Nat) : 
  n + 0 + 0 = n := by simp

@[simp, priority := 500] theorem mixed_medium_priority_rule (l : List α) :
  l ++ [] ++ [] = l := by simp

-- Standard patterns with room for improvement
example (data : List Nat) : 
  data.map (· + 0) |>.map (· * 1) = data := by simp

example (x y z : Nat) :
  (x + 0) + (y * 1) + (z + 0) = x + y + z := by simp

-- Complex expressions with mixed optimization potential
example (f : Nat → Option String) (g : String → Bool) (n : Nat) :
  (f n).map (· ++ "") |>.bind (fun s => if g s then some s.length else none) =
  (f n).bind (fun s => if g s then some s.length else none) := by
  cases f n <;> simp
  split <;> simp

-- Nested operations with partial optimization opportunities
example (nested : List (List (Option Nat))) :
  nested.map (·.map (·.map (· + 0))) = nested := by simp

example (pairs : List (Nat × String)) :
  pairs.map (fun (n, s) => (n + 0, s ++ "")) = pairs := by simp

-- Some patterns that won't benefit much from optimization
theorem mixed_complex_theorem (P Q R : Prop) (h1 : P → Q) (h2 : Q → R) : P → R := by
  intro hp
  apply h2
  apply h1
  exact hp

example (n : Nat) (h : n > 0) : n.pred.succ = n := by
  cases n with
  | zero => contradiction
  | succ n => simp [Nat.pred_succ]

-- Moderate frequency patterns
example (l : List Nat) : (l ++ []).length + 0 = l.length := by simp
example (s : String) : (s ++ "").length * 1 = s.length := by simp
example (opt : Option Nat) : opt.map (· + 0) |>.isSome = opt.isSome := by cases opt <;> simp

-- Some edge cases with limited optimization potential
example : ([] : List Nat) ++ [] = [] := by simp
example : ("" : String) ++ "" = "" := by simp
example : (none : Option Nat).map (· + 0) = none := by simp
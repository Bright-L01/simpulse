-- PartialOptimization.lean
-- Already has some priorities set, moderate optimization potential
-- Expected: Modest improvement (15-30% speedup from remaining opportunities)

-- Some rules already optimized (won't change)
@[simp, priority := 100] theorem partial_add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl  
  | succ n ih => simp [Nat.add_succ, ih]

@[simp, priority := 110] theorem partial_zero_add (n : Nat) : 0 + n = n := by rfl

-- Some rules not optimized (optimization potential)
@[simp] theorem partial_mul_one (n : Nat) : n * 1 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.mul_succ, ih]

@[simp] theorem partial_one_mul (n : Nat) : 1 * n = n := by rfl

@[simp] theorem partial_list_nil_append (l : List α) : [] ++ l = l := by rfl

-- Mixed priority levels (some room for improvement)
@[simp, priority := 200] theorem partial_list_append_nil (l : List α) : l ++ [] = l := by
  induction l with
  | nil => rfl
  | cons head tail ih => simp [List.cons_append, ih]

@[simp] theorem partial_list_length_nil : [].length = 0 := by rfl

@[simp] theorem partial_list_length_cons (head : α) (tail : List α) :
  (head :: tail).length = tail.length + 1 := by rfl

-- String operations with partial optimization
@[simp, priority := 150] theorem partial_string_append_empty (s : String) : s ++ "" = s := by rfl

@[simp] theorem partial_empty_append_string (s : String) : "" ++ s = s := by rfl

-- Option operations (mixed optimization state)
@[simp, priority := 180] theorem partial_option_map_none (f : α → β) : 
  (none : Option α).map f = none := by rfl

@[simp] theorem partial_option_map_some (f : α → β) (a : α) : 
  (some a).map f = some (f a) := by rfl

@[simp] theorem partial_option_bind_none (f : α → Option β) : 
  (none : Option α).bind f = none := by rfl

-- Usage patterns (moderate optimization benefit)
example (n m : Nat) : n + 0 + m * 1 = n + m := by simp

example (l1 l2 : List String) : [] ++ l1 ++ l2 ++ [] = l1 ++ l2 := by simp

example (s : String) : s ++ "" ++ "" = s := by simp

example (opt : Option Nat) : opt.map (· + 0) |>.bind (fun n => some (n * 1)) = opt := by
  cases opt <;> simp

-- Partially optimized chains
example (a b c : Nat) : (a + 0) * 1 + (b + 0) + (c * 1) = a + b + c := by simp

example (data : List Nat) : 
  ([] ++ data ++ []).map (· * 1) = data := by simp

-- Mixed complexity with partial optimization
example (f : Nat → String) (g : String → Option Bool) (x : Option Nat) :
  x.map f |>.bind g |>.map not = x.bind (fun n => (g (f n)).map not) := by
  cases x <;> simp [Option.bind_assoc]

-- Some already well-optimized patterns (limited improvement)
example (l : List α) : [] ++ l = l := by simp [partial_list_nil_append]
example (n : Nat) : n + 0 = n := by simp [partial_add_zero]

-- Some patterns with optimization potential
example (m n : Nat) : m * 1 + n * 1 = m + n := by simp
example (s1 s2 : String) : s1 ++ "" ++ s2 = s1 ++ s2 := by simp

-- Nested operations with mixed optimization state
example (nested : List (Option String)) :
  nested.map (·.map (· ++ "")) = nested := by simp

example (pairs : Option (Nat × String)) :
  pairs.map (fun (n, s) => (n + 0, s ++ "")) = pairs := by
  cases pairs <;> simp

-- Performance patterns with partial optimization
example (xs : List Nat) (f : Nat → String) :
  ([] ++ xs).map f |>.map (· ++ "") = xs.map f := by simp

example (opt1 opt2 : Option Nat) :
  opt1.bind (fun _ => opt2.map (· * 1)) = 
  if opt1.isSome then opt2 else none := by
  cases opt1 <;> cases opt2 <;> simp

-- Edge cases with mixed optimization potential
example : (0 : Nat) + 0 = 0 := by simp [partial_add_zero]
example : (1 : Nat) * 1 = 1 := by simp
example : ("" : String) ++ "" = "" := by simp [partial_string_append_empty]
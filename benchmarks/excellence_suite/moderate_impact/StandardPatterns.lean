-- StandardPatterns.lean
-- Common proof patterns with moderate simp optimization potential
-- Expected: Modest improvement (25-40% speedup from standard optimizations)

-- Standard arithmetic patterns (moderate frequency)
@[simp] theorem standard_add_zero (n : Nat) : n + 0 = n := Nat.add_zero n

@[simp] theorem standard_zero_add (n : Nat) : 0 + n = n := Nat.zero_add n

@[simp] theorem standard_mul_one (n : Nat) : n * 1 = n := Nat.mul_one n

@[simp] theorem standard_one_mul (n : Nat) : 1 * n = n := Nat.one_mul n

-- Standard list patterns (moderate usage)
@[simp] theorem standard_list_append_nil (l : List α) : l ++ [] = l := List.append_nil l

@[simp] theorem standard_list_nil_append (l : List α) : [] ++ l = l := List.nil_append l

@[simp] theorem standard_list_length_nil : [].length = 0 := List.length_nil

@[simp] theorem standard_list_map_nil (f : α → β) : [].map f = [] := List.map_nil f

@[simp] theorem standard_list_map_cons (f : α → β) (head : α) (tail : List α) :
  (head :: tail).map f = f head :: tail.map f := List.map_cons f head tail

-- Standard option patterns (moderate usage) 
@[simp] theorem standard_option_map_none (f : α → β) : (none : Option α).map f = none := Option.map_none f

@[simp] theorem standard_option_map_some (f : α → β) (a : α) : (some a).map f = some (f a) := Option.map_some f a

@[simp] theorem standard_option_bind_none (f : α → Option β) : (none : Option α).bind f = none := Option.bind_none f

@[simp] theorem standard_option_bind_some (f : α → Option β) (a : α) : (some a).bind f = f a := Option.bind_some f a

-- Typical proof patterns with moderate simp usage
example (n m : Nat) : n + 0 + m = n + m := by simp

example (l1 l2 l3 : List α) : l1 ++ [] ++ l2 ++ l3 = l1 ++ l2 ++ l3 := by simp [List.append_assoc]

example (f : α → β) (l : List α) : (l ++ []).map f = l.map f := by simp

example (opt : Option Nat) : opt.map (· + 0) = opt := by cases opt <;> simp

-- Standard computational patterns
example (data : List Nat) : data.map (· * 1) |>.length = data.length := by simp

example (s : String) : (s ++ "").length = s.length := by simp

example (opt1 opt2 : Option Nat) : 
  opt1.bind (fun _ => opt2.map (· + 0)) = opt1.bind (fun _ => opt2) := by simp

-- Conditional patterns with moderate optimization
example (n : Nat) (h : n > 0) : n + 0 = n := by simp

example (l : List α) (h : l ≠ []) : (l ++ []).length > 0 := by simp [List.length_pos_of_ne_nil h]

-- Standard functional patterns
example (f g : α → α) (x : α) : (f ∘ g) x = f (g x) := by simp [Function.comp_apply]

example (l : List α) : l.map id = l := by simp [List.map_id]

example (opt : Option α) : opt.bind some = opt := by simp [Option.bind_some_eq]

-- Moderately complex expressions
example (f : Nat → String) (g : String → Bool) (data : List Nat) :
  data.map f |>.map (· ++ "") |>.filter g = (data.map f).filter g := by simp

example (nested : List (Option Nat)) :
  nested.map (·.map (· + 0)) |>.map (·.isSome) = nested.map (·.isSome) := by simp

-- Standard algebraic manipulations
example (a b c : Nat) : (a + 0) + (b * 1) + c = a + b + c := by simp

example (x y : Nat) : (x * 1) + (y + 0) = x + y := by simp [Nat.add_comm]

-- Error handling patterns
example (opt : Option String) : opt.map (· ++ "") |>.getD "default" = opt.getD "default" := by
  cases opt <;> simp [Option.getD]

-- Standard inductive patterns (moderate simp usage)
theorem standard_list_sum_cons (head : Nat) (tail : List Nat) :
  (head :: tail).sum = head + tail.sum := by simp [List.sum_cons]

theorem standard_list_length_append (l1 l2 : List α) :
  (l1 ++ l2).length = l1.length + l2.length := by simp [List.length_append]

-- Typical algorithm patterns
example (pred : α → Bool) (l : List α) : 
  (l ++ []).filter pred = l.filter pred := by simp

example (f : α → β) (g : β → γ) (l : List α) :
  (l.map f).map g = l.map (g ∘ f) := by simp [List.map_map]

-- Standard edge cases
example : (0 : Nat) + 0 = 0 := by simp
example : ([] : List Nat) ++ [] = [] := by simp
example : (none : Option Nat).map (· + 0) = none := by simp
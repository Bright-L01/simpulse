-- OptionMonad.lean
-- Option/Maybe monad operations with heavy simp usage
-- Expected: 2x+ speedup from frequent Option simplification rules

-- Core Option rules (very frequently used)
@[simp] theorem option_some_get (a : α) : (some a).get? = some a := by rfl

@[simp] theorem option_none_get : (none : Option α).get? = none := by rfl

@[simp] theorem option_map_none (f : α → β) : (none : Option α).map f = none := by rfl

@[simp] theorem option_map_some (f : α → β) (a : α) : (some a).map f = some (f a) := by rfl

@[simp] theorem option_bind_none (f : α → Option β) : (none : Option α).bind f = none := by rfl

@[simp] theorem option_bind_some (f : α → Option β) (a : α) : (some a).bind f = f a := by rfl

-- Frequently used Option operations
@[simp] theorem option_is_some_none : (none : Option α).isSome = false := by rfl

@[simp] theorem option_is_some_some (a : α) : (some a).isSome = true := by rfl

@[simp] theorem option_is_none_none : (none : Option α).isNone = true := by rfl

@[simp] theorem option_is_none_some (a : α) : (some a).isNone = false := by rfl

@[simp] theorem option_or_else_some (a : α) (b : Option α) : (some a).orElse b = some a := by rfl

@[simp] theorem option_or_else_none (b : Option α) : (none : Option α).orElse b = b := by rfl

-- Filter operations (moderately used)
@[simp] theorem option_filter_none (p : α → Bool) : (none : Option α).filter p = none := by rfl

@[simp] theorem option_filter_some_true (a : α) (p : α → Bool) (h : p a = true) : 
  (some a).filter p = some a := by simp [Option.filter, h]

@[simp] theorem option_filter_some_false (a : α) (p : α → Bool) (h : p a = false) : 
  (some a).filter p = none := by simp [Option.filter, h]

-- Heavy usage patterns (simulates real monadic code)
example (f : Nat → String) (x : Option Nat) :
  x.map f |>.map String.length = x.map (fun n => (f n).length) := by simp [Option.map_map]

example (x : Option Nat) :
  x.bind (fun n => some (n + 1)) = x.map (· + 1) := by
  cases x <;> simp

example (x y : Option Nat) :
  x.bind (fun _ => y) = if x.isSome then y else none := by
  cases x <;> simp

-- Chained operations (very common)
example (f : Nat → Option String) (g : String → Option Bool) (x : Option Nat) :
  x.bind f |>.bind g = x.bind (fun n => (f n).bind g) := by
  cases x <;> simp [Option.bind_assoc]

example (x : Option Nat) :
  x.bind (fun n => some n) = x := by
  cases x <;> simp

example (f : Nat → String) (x : Option Nat) :
  x.bind (fun n => some (f n)) = x.map f := by
  cases x <;> simp

-- OrElse chains (frequent in error handling)
example (x y z : Option Nat) :
  (x.orElse y).orElse z = x.orElse (y.orElse z) := by
  cases x <;> simp [Option.orElse_assoc]

example (x : Option Nat) :
  x.orElse none = x := by
  cases x <;> simp

example (x : Option Nat) :
  none.orElse x = x := by simp

-- Filter and map combinations
example (p : Nat → Bool) (f : Nat → String) (x : Option Nat) :
  x.filter p |>.map f = x.bind (fun n => if p n then some (f n) else none) := by
  cases x <;> simp [Option.filter]
  split <;> simp

-- IsSome/isNone patterns (very frequent in conditional logic)
example (x y : Option Nat) :
  (x.orElse y).isSome = x.isSome || y.isSome := by
  cases x <;> cases y <;> simp

example (f : Nat → Option String) (x : Option Nat) :
  (x.bind f).isSome = x.isSome && (x.bind f).isSome := by
  cases x <;> simp

-- Get operations with default values
example (x : Option Nat) (default : Nat) :
  x.getD default = if x.isSome then x.get! else default := by
  cases x <;> simp [Option.getD]

-- Option sequence operations
example (xs : List (Option Nat)) :
  xs.filterMap id = xs.bind (fun opt => opt.toList) := by
  induction xs <;> simp [*]
  cases head <;> simp

-- Nested Option operations (common in parser combinators)
example (x : Option (Option Nat)) :
  x.bind id = x.join := by
  cases x <;> simp [Option.join]

example (f : Nat → Option String) (x : Option (Option Nat)) :
  x.bind (fun opt => opt.bind f) = x.join.bind f := by
  cases x <;> simp [Option.join]

-- Error handling patterns
example (x : Option Nat) (error : String) :
  (x.map toString).getD error = if x.isSome then toString x.get! else error := by
  cases x <;> simp

-- Applicative patterns
example (f : Option (Nat → String)) (x : Option Nat) :
  f.bind (fun g => x.map g) = 
  match f, x with
  | some g, some n => some (g n)
  | _, _ => none := by
  cases f <;> cases x <;> simp

-- Very frequent micro-patterns in real code
example (x : Option Nat) : x.map id = x := by cases x <;> simp
example (x : Option Nat) : x.bind some = x := by cases x <;> simp
example (f : Nat → String) : (none : Option Nat).map f = none := by simp
example (x : Option Nat) : x.orElse (some 0) = x.orElse (some 0) := by rfl

-- Performance stress patterns
example (f g h : Nat → Nat) (x : Option Nat) :
  x.map f |>.map g |>.map h = x.map (h ∘ g ∘ f) := by
  cases x <;> simp [Function.comp]

example (x y z w : Option Nat) :
  ((x.orElse y).orElse z).orElse w = x.orElse (y.orElse (z.orElse w)) := by
  cases x <;> cases y <;> cases z <;> simp

-- Edge cases that exercise simp heavily
example : (none : Option Nat).map (· + 1) = none := by simp
example : (some 5).bind (fun _ => none) = none := by simp
example : (none : Option Nat).orElse none = none := by simp
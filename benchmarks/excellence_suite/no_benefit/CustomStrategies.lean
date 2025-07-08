-- CustomStrategies.lean
-- Uses custom simp strategies that Simpulse shouldn't interfere with
-- Expected: No benefit (Simpulse should avoid files with custom simp usage)

-- Some basic rules (but they're used with custom strategies)
@[simp] theorem custom_add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.add_succ, ih]

@[simp] theorem custom_list_append_nil (l : List α) : l ++ [] = l := by
  induction l with
  | nil => rfl
  | cons head tail ih => simp [List.cons_append, ih]

@[simp] theorem custom_string_append_empty (s : String) : s ++ "" = s := by rfl

-- Critical: Uses simp_rw instead of simp (custom strategy)
example (n m : Nat) : n + 0 + m + 0 = n + m := by
  simp_rw [custom_add_zero]

example (l1 l2 l3 : List α) : l1 ++ [] ++ l2 ++ [] ++ l3 = l1 ++ l2 ++ l3 := by
  simp_rw [custom_list_append_nil]

-- Uses simp only (selective simp application)
example (s1 s2 : String) : s1 ++ "" ++ s2 ++ "" = s1 ++ s2 := by
  simp only [custom_string_append_empty]

example (data : List Nat) : data ++ [] ++ [] = data := by
  simp only [custom_list_append_nil]

-- Custom simp configuration with specific rules
example (a b c : Nat) : (a + 0) + (b + 0) + c = a + b + c := by
  simp only [custom_add_zero, Nat.add_assoc]

-- Uses simp_all (global context modification)
example (n m : Nat) (h1 : n = m + 0) : n = m := by
  simp_all only [custom_add_zero]

-- Conditional simp_rw usage
example (l : List Nat) (h : l.length > 0) : (l ++ []).length > 0 := by
  simp_rw [custom_list_append_nil]
  exact h

-- Custom simp attribute sets
@[simp, my_custom_simp] theorem custom_special_rule (n : Nat) : n * 1 = n := by
  induction n with
  | zero => rfl  
  | succ n ih => simp [Nat.mul_succ, ih]

example (x y : Nat) : x * 1 + y * 1 = x + y := by
  simp only [custom_special_rule]

-- Selective simp with disabled rules  
example (n : Nat) : n + 0 + 0 = n := by
  simp only [custom_add_zero] at *

-- Custom rewrite sequences that depend on rule order
example (s : String) : (s ++ "" ++ "").length = s.length := by
  simp_rw [custom_string_append_empty, String.length_append]

-- Complex custom simp patterns that would break with priority changes
example (nested : List (List String)) :
  nested.map (·.map (· ++ "")) = nested := by
  simp only [List.map_map, custom_string_append_empty, List.map_id]

-- Conditional simp applications
example (opt : Option String) : opt.map (· ++ "") = opt := by
  cases opt with
  | none => simp only []
  | some s => simp only [Option.map_some, custom_string_append_empty]

-- Simp with arithmetic normalization (order-dependent)
example (a b c d : Nat) : a + 0 + b + 0 + c + 0 + d = a + b + c + d := by
  simp only [custom_add_zero]
  abel

-- Custom simp contexts that rely on current priorities
local attribute [simp] Nat.add_comm in
example (n m : Nat) : n + 0 + m = m + n := by
  simp only [custom_add_zero]

-- Simpulse should avoid this file due to custom simp usage patterns
example : (0 : Nat) + 0 = 0 := by simp_rw [custom_add_zero]
example : ("" : String) ++ "" = "" := by simp only [custom_string_append_empty]
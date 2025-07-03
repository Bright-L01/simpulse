-- Sample Lean 4 file with simp rules to test frequency counting

@[simp] theorem list_append_nil (l : List α) : l ++ [] = l := by
  induction l with
  | nil => rfl
  | cons h t ih => simp [List.cons_append, ih]

@[simp] theorem list_nil_append (l : List α) : [] ++ l = l := rfl

@[simp, priority := 1100] theorem add_zero (n : Nat) : n + 0 = n := by
  simp [Nat.add_zero]

@[simp] theorem zero_add (n : Nat) : 0 + n = n := by 
  simp [Nat.zero_add]

theorem example1 (l : List α) : (l ++ []) ++ [] = l := by
  simp [list_append_nil]  -- Explicit use of list_append_nil

theorem example2 (l m : List α) : [] ++ (l ++ m) = l ++ m := by
  simp  -- Implicit use of list_nil_append

theorem example3 (l : List α) (n : Nat) : (l ++ []).length + 0 = l.length := by
  simp [list_append_nil, add_zero]  -- Explicit use of both rules

theorem example4 (a b c : Nat) : (a + 0) + (0 + b) = a + b := by
  simp [add_zero, zero_add]  -- Explicit use of arithmetic rules

theorem example5 (l : List α) : l ++ [] ++ [] = l := by
  simp  -- Implicit use of list_append_nil (applied twice)

-- More complex example with nested simp calls
theorem example6 (l m : List α) (n : Nat) : 
    ((l ++ []) ++ ([] ++ m)).length + 0 = (l ++ m).length := by
  have h1 : (l ++ []) ++ ([] ++ m) = l ++ m := by simp
  simp [h1]  -- Uses add_zero implicitly

-- Test various simp patterns
theorem example7 (x y : Nat) : x + 0 + (0 + y) + 0 = x + y := by
  simp only [add_zero, zero_add]  -- simp only pattern

theorem example8 (l : List α) : l = l ++ [] := by
  simp [list_append_nil]  -- Another explicit use
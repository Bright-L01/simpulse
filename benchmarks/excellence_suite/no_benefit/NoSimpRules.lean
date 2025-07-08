-- NoSimpRules.lean
-- No @[simp] annotations at all
-- Expected: No benefit (Simpulse should detect no simp rules and skip)

-- Regular theorems without @[simp] annotations
theorem manual_add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Nat.add_succ, ih]

theorem manual_zero_add (n : Nat) : 0 + n = n := by rfl

theorem manual_list_append_nil (l : List α) : l ++ [] = l := by
  induction l with
  | nil => rfl  
  | cons head tail ih => rw [List.cons_append, ih]

theorem manual_string_append_empty (s : String) : s ++ "" = s := by rfl

-- Proofs that use explicit tactics instead of simp
example (n m : Nat) : n + 0 + m = n + m := by
  rw [manual_add_zero]

example (l1 l2 : List α) : l1 ++ [] ++ l2 = l1 ++ l2 := by
  rw [manual_list_append_nil]

example (s : String) : s ++ "" = s := by
  exact manual_string_append_empty s

-- Manual proof style (no simp usage)
example (a b c : Nat) : (a + 0) + (b + 0) + c = a + b + c := by
  rw [manual_add_zero, manual_add_zero]

example (data : List Nat) : data ++ [] = data := by
  apply manual_list_append_nil

-- Complex proofs without simp
theorem manual_list_length_append (l1 l2 : List α) : 
  (l1 ++ l2).length = l1.length + l2.length := by
  induction l1 with
  | nil => rfl
  | cons head tail ih => 
    rw [List.cons_append, List.length_cons, List.length_cons, ih]
    rw [Nat.add_assoc]

-- Definitional equalities (no simp needed)
example : (fun x => x) = id := by rfl

example (f : α → β) : f ∘ id = f := by rfl

-- Tactic-based proofs
example (P Q : Prop) (h1 : P) (h2 : P → Q) : Q := by
  apply h2
  exact h1

example (n : Nat) (h : n > 0) : ∃ m, n = m + 1 := by
  use n.pred
  exact Nat.succ_pred_eq_of_pos h

-- Manual rewriting chains
example (a b c d : Nat) : a + b + c + d = d + c + b + a := by
  rw [Nat.add_assoc, Nat.add_assoc]
  rw [Nat.add_comm (a + b), Nat.add_comm a]
  rw [← Nat.add_assoc, ← Nat.add_assoc, ← Nat.add_assoc]

-- No simp rules to optimize - Simpulse should skip this file entirely
example (l : List Nat) : l.reverse.reverse = l := by
  exact List.reverse_reverse l

example (opt : Option α) : opt.isSome = !opt.isNone := by
  cases opt <;> rfl
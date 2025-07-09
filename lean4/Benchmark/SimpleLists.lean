-- SimpleLists.lean
-- Test simp performance on basic list operations

theorem list_append_nil (l : List α) : l ++ [] = l := by simp

theorem list_nil_append (l : List α) : [] ++ l = l := by simp

theorem list_append_assoc (l1 l2 l3 : List α) : 
  (l1 ++ l2) ++ l3 = l1 ++ (l2 ++ l3) := by simp

theorem list_length_append (l1 l2 : List α) : 
  (l1 ++ l2).length = l1.length + l2.length := by simp

theorem list_reverse_append (l1 l2 : List α) : 
  (l1 ++ l2).reverse = l2.reverse ++ l1.reverse := by simp

theorem list_map_append (f : α → β) (l1 l2 : List α) :
  (l1 ++ l2).map f = l1.map f ++ l2.map f := by simp

theorem list_filter_append (p : α → Bool) (l1 l2 : List α) :
  (l1 ++ l2).filter p = l1.filter p ++ l2.filter p := by simp

theorem list_take_append (n : Nat) (l1 l2 : List α) :
  (l1 ++ l2).take n = if n ≤ l1.length then l1.take n else l1 ++ l2.take (n - l1.length) := by
  split <;> simp [*]

theorem list_drop_append (n : Nat) (l1 l2 : List α) :
  (l1 ++ l2).drop n = if n ≤ l1.length then l1.drop n ++ l2 else l2.drop (n - l1.length) := by
  split <;> simp [*]

theorem list_concat_eq_append (l : List α) (a : α) :
  l.concat a = l ++ [a] := by simp

-- ListIntensive.lean
-- Intensive list operations with frequent simp usage
-- Expected: 2x+ speedup from optimizing frequent list rules

-- Core list rules (very frequently used)
@[simp] theorem list_append_nil (l : List α) : l ++ [] = l := by
  induction l with
  | nil => rfl
  | cons head tail ih => simp [List.cons_append, ih]

@[simp] theorem list_nil_append (l : List α) : [] ++ l = l := by rfl

@[simp] theorem list_length_nil : [].length = 0 := by rfl

@[simp] theorem list_length_cons (head : α) (tail : List α) : 
  (head :: tail).length = tail.length + 1 := by rfl

@[simp] theorem list_length_append (l1 l2 : List α) : 
  (l1 ++ l2).length = l1.length + l2.length := by
  induction l1 with
  | nil => simp
  | cons head tail ih => simp [List.cons_append, ih, Nat.add_assoc]

-- Map operations (frequently used)
@[simp] theorem list_map_nil (f : α → β) : [].map f = [] := by rfl

@[simp] theorem list_map_cons (f : α → β) (head : α) (tail : List α) :
  (head :: tail).map f = f head :: tail.map f := by rfl

@[simp] theorem list_map_append (f : α → β) (l1 l2 : List α) :
  (l1 ++ l2).map f = l1.map f ++ l2.map f := by
  induction l1 with
  | nil => simp
  | cons head tail ih => simp [List.cons_append, ih]

-- Filter operations (moderately used)
@[simp] theorem list_filter_nil (p : α → Bool) : [].filter p = [] := by rfl

@[simp] theorem list_filter_append (p : α → Bool) (l1 l2 : List α) :
  (l1 ++ l2).filter p = l1.filter p ++ l2.filter p := by
  induction l1 with
  | nil => simp
  | cons head tail ih => 
    simp [List.filter_cons, ih]
    split <;> simp [List.cons_append]

-- Reverse operations (moderately used)  
@[simp] theorem list_reverse_nil : [].reverse = [] := by rfl

@[simp] theorem list_reverse_append (l1 l2 : List α) : 
  (l1 ++ l2).reverse = l2.reverse ++ l1.reverse := by
  induction l1 with
  | nil => simp
  | cons head tail ih => simp [List.cons_append, List.reverse_cons, ih, List.append_assoc]

-- Heavy simp usage examples (simulates real code patterns)
example (l1 l2 l3 : List Nat) : 
  (l1 ++ l2) ++ l3 ++ [] = l1 ++ (l2 ++ l3) := by simp [List.append_assoc]

example (l : List String) : 
  l ++ [] ++ [] = l := by simp

example (f : Nat → String) (l1 l2 : List Nat) :
  (l1 ++ l2 ++ []).map f = l1.map f ++ l2.map f := by simp

example (l1 l2 l3 l4 : List α) :
  ((l1 ++ l2) ++ []) ++ ((l3 ++ []) ++ l4) = l1 ++ l2 ++ l3 ++ l4 := by simp [List.append_assoc]

example (p : Nat → Bool) (l1 l2 : List Nat) :
  (l1 ++ l2 ++ []).filter p = l1.filter p ++ l2.filter p := by simp

-- Length calculations (very common in real code)
example (l1 l2 l3 : List α) :
  (l1 ++ l2 ++ l3).length = l1.length + l2.length + l3.length := by simp [Nat.add_assoc]

example (l : List Nat) :
  (l ++ [] ++ []).length = l.length := by simp

example (head : α) (tail : List α) :
  (head :: tail ++ []).length = tail.length + 1 := by simp

example (f : String → Nat) (l1 l2 : List String) :
  (l1 ++ l2).map f |>.length = l1.length + l2.length := by simp

-- Map composition patterns
example (f : Nat → String) (g : String → Bool) (l1 l2 : List Nat) :
  ((l1 ++ l2).map f).map g = (l1.map f).map g ++ (l2.map f).map g := by simp

example (f : α → β) (l : List α) :
  (l ++ [] ++ []).map f = l.map f := by simp

example (f : Nat → Nat) (l1 l2 l3 : List Nat) :
  ((l1 ++ l2) ++ l3).map f = l1.map f ++ l2.map f ++ l3.map f := by simp [List.append_assoc]

-- Filter and append combinations
example (p : α → Bool) (l1 l2 l3 : List α) :
  ((l1 ++ l2) ++ l3).filter p = l1.filter p ++ l2.filter p ++ l3.filter p := by simp [List.append_assoc]

example (p : Nat → Bool) (l : List Nat) :
  (l ++ []).filter p = l.filter p := by simp

-- Reverse patterns
example (l1 l2 : List α) :
  (l1 ++ l2 ++ []).reverse = l2.reverse ++ l1.reverse := by simp

example (l : List String) :
  (l ++ []).reverse = l.reverse := by simp

-- Complex nested operations (stress test)
example (f : Nat → String) (l1 l2 l3 : List Nat) :
  ((l1 ++ l2) ++ l3 ++ []).map f |>.length = 
  l1.length + l2.length + l3.length := by simp [Nat.add_assoc]

example (p : α → Bool) (l1 l2 : List α) :
  ((l1 ++ []) ++ (l2 ++ [])).filter p = 
  l1.filter p ++ l2.filter p := by simp

-- Very frequent patterns in real functional code
example (l : List Nat) : (l ++ []).length = l.length := by simp
example (f : α → β) (l : List α) : ([] ++ l).map f = l.map f := by simp
example (l1 l2 : List String) : (l1 ++ []) ++ l2 = l1 ++ l2 := by simp
example (p : Nat → Bool) (l : List Nat) : ([] ++ l).filter p = l.filter p := by simp

-- Edge cases that exercise simp heavily
example : ([] : List Nat) ++ [] = [] := by simp
example (f : α → β) : ([] : List α).map f ++ [].map f = [] := by simp
example (l : List α) : l ++ [] ++ [] ++ [] = l := by simp
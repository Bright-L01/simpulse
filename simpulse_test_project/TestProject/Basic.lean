
-- Additional simp rules for testing

@[simp]
theorem zero_add (n : Nat) : 0 + n = n := by
  rfl

@[simp]
theorem list_length_nil : List.length ([] : List α) = 0 := by
  rfl

@[simp]  
theorem bool_and_true (b : Bool) : b && true = b := by
  cases b <;> rfl

-- Proofs using these rules
theorem basic_test_1 (n : Nat) : 0 + (n + 0) = n := by
  simp [zero_add]

theorem basic_test_2 (l : List Nat) : List.length [] + List.length l = List.length l := by
  simp [list_length_nil, zero_add]

theorem basic_test_3 (b1 b2 : Bool) : (b1 && true) && (b2 && true) = b1 && b2 := by
  simp [bool_and_true]

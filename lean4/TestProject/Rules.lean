-- Test file with simp rules

namespace TestProject

-- Basic list rules
@[simp] theorem list_append_nil {α} (l : List α) : l ++ [] = l := by
  induction l <;> simp_all

@[simp] theorem list_nil_append {α} (l : List α) : [] ++ l = l := rfl

-- Arithmetic rules  
@[simp] theorem nat_add_zero (n : Nat) : n + 0 = n := Nat.add_zero n

@[simp] theorem nat_zero_add (n : Nat) : 0 + n = n := Nat.zero_add n

@[simp] theorem nat_mul_one (n : Nat) : n * 1 = n := Nat.mul_one n

@[simp] theorem nat_one_mul (n : Nat) : 1 * n = n := Nat.one_mul n

-- Option rules
@[simp] theorem option_some_eq (a b : α) : some a = some b ↔ a = b := by simp

@[simp] theorem option_none_eq_none : (none : Option α) = none ↔ True := by simp

end TestProject
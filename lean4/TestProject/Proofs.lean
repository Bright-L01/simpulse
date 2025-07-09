-- Test file with proofs that use simp rules

import TestProject.Rules

namespace TestProject

-- Examples using list rules heavily
theorem list_example1 {α} (l : List α) : l ++ [] ++ [] = l := by
  simp [list_append_nil]

theorem list_example2 {α} (l m : List α) : [] ++ l ++ [] = l := by
  simp [list_nil_append, list_append_nil]

theorem list_example3 {α} (l : List α) : (l ++ []) ++ [] = l := by
  simp -- Uses list_append_nil implicitly

theorem list_example4 {α} (a b c : List α) : (a ++ []) ++ ([] ++ b) ++ [] = a ++ b := by
  simp [list_append_nil, list_nil_append]

-- Examples using arithmetic rules
theorem nat_example1 (n : Nat) : n + 0 + 0 = n := by
  simp [nat_add_zero]

theorem nat_example2 (n m : Nat) : (n + 0) * 1 + 0 = n := by
  simp [nat_add_zero, nat_mul_one]

theorem nat_example3 (n : Nat) : 0 + n + 0 = n := by
  simp -- Uses nat_zero_add and nat_add_zero implicitly

theorem nat_example4 (a b : Nat) : (a + 0) * 1 + (0 + b) * 1 = a + b := by
  simp [nat_add_zero, nat_zero_add, nat_mul_one]

-- Examples rarely using option rules
theorem option_example1 (x : Option Nat) : x = x := by rfl

theorem option_example2 : some 5 = some 5 := by
  simp [option_some_eq] -- Explicit but rare use

-- More list usage
theorem list_heavy_use {α} (l m n : List α) : 
    ((l ++ []) ++ ([] ++ m)) ++ (n ++ []) = l ++ m ++ n := by
  simp [list_append_nil, list_nil_append]

theorem list_very_heavy {α} (a b c d : List α) :
    (a ++ []) ++ ([] ++ b) ++ (c ++ []) ++ ([] ++ d) = a ++ b ++ c ++ d := by
  simp [list_append_nil, list_nil_append, list_append_nil]

-- Use arithmetic rules more
theorem nat_heavy (x y z : Nat) : 
    (x + 0) * 1 + (0 + y) * 1 + (z + 0) * 1 = x + y + z := by
  simp [nat_add_zero, nat_zero_add, nat_mul_one]

end TestProject
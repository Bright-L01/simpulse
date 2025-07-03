-- Test file with various simp rules for pattern analysis demonstration

namespace TestSimp

-- Basic arithmetic simplifications
@[simp] theorem add_zero (n : Nat) : n + 0 = n := by simp

@[simp] theorem zero_add (n : Nat) : 0 + n = n := by simp

@[simp] theorem mul_one (n : Nat) : n * 1 = n := by simp

@[simp] theorem one_mul (n : Nat) : 1 * n = n := by simp

-- List simplifications
@[simp] theorem list_append_nil (l : List α) : l ++ [] = l := by simp

@[simp] theorem list_nil_append (l : List α) : [] ++ l = l := by simp

@[simp] theorem list_length_nil : [].length = 0 := by simp

-- Boolean simplifications
@[simp] theorem true_and (p : Prop) [Decidable p] : (True ∧ p) = p := by simp

@[simp] theorem and_true (p : Prop) [Decidable p] : (p ∧ True) = p := by simp

@[simp] theorem false_or (p : Prop) [Decidable p] : (False ∨ p) = p := by simp

-- Custom priority examples
@[simp, priority := 500] theorem high_priority_rule (x : Nat) : x + x = 2 * x := by ring

@[simp, priority := 100] theorem low_priority_rule (x : Nat) : 2 * x = x + x := by ring

-- Context-specific rules
namespace Algebra

@[simp] theorem ring_add_assoc (a b c : Nat) : (a + b) + c = a + (b + c) := by simp

@[simp] theorem ring_mul_comm (a b : Nat) : a * b = b * a := by simp

end Algebra

namespace Logic

@[simp] theorem imp_self (p : Prop) : (p → p) = True := by simp

@[simp] theorem not_not_not (p : Prop) [Decidable p] : ¬¬¬p = ¬p := by simp

end Logic

-- Complex rules that might benefit from optimization
@[simp] theorem complex_arithmetic (x y z : Nat) (h : x > 0) : 
  (x + y) * z + x * z = (x + y + x) * z := by ring

@[simp] theorem list_map_append (f : α → β) (l₁ l₂ : List α) :
  (l₁ ++ l₂).map f = l₁.map f ++ l₂.map f := by simp

-- Rules that work well together
@[simp] theorem step1 (x : Nat) : x + 2 * x = 3 * x := by ring

@[simp] theorem step2 (x : Nat) : 3 * x + x = 4 * x := by ring

@[simp] theorem step3 (x : Nat) : 4 * x - x = 3 * x := by ring

end TestSimp
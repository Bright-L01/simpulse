-- Comprehensive simp rule test file for optimization analysis

namespace ComprehensiveSimp

-- High-frequency basic rules (should get high priority)
@[simp] theorem nat_add_zero (n : Nat) : n + 0 = n := Nat.add_zero n
@[simp] theorem nat_zero_add (n : Nat) : 0 + n = n := Nat.zero_add n
@[simp] theorem nat_mul_one (n : Nat) : n * 1 = n := Nat.mul_one n
@[simp] theorem nat_one_mul (n : Nat) : 1 * n = n := Nat.one_mul n

-- List operations (context: data structures)
@[simp] theorem list_append_nil (l : List α) : l ++ [] = l := List.append_nil l
@[simp] theorem list_nil_append (l : List α) : [] ++ l = l := List.nil_append l
@[simp] theorem list_length_nil : List.length ([] : List α) = 0 := rfl
@[simp] theorem list_length_cons (a : α) (l : List α) : 
  List.length (a :: l) = List.length l + 1 := rfl

-- Boolean logic (context: logic)
@[simp] theorem true_and (p : Bool) : True && p = p := Bool.true_and p
@[simp] theorem and_true (p : Bool) : p && True = p := Bool.and_true p
@[simp] theorem false_or (p : Bool) : False || p = p := Bool.false_or p
@[simp] theorem or_false (p : Bool) : p || False = p := Bool.or_false p

-- Already optimized rules (should not change)
@[simp, priority := 100] theorem already_high_priority (x : Nat) : x + x = 2 * x := by ring
@[simp, priority := 900] theorem already_low_priority (x : Nat) : 2 * x = x + x := by ring

-- Complex rules (should get lower priority due to complexity)
@[simp] theorem complex_distributive (a b c d : Nat) : 
  (a + b) * (c + d) = a * c + a * d + b * c + b * d := by ring

@[simp] theorem complex_list_operation (l1 l2 l3 : List α) :
  (l1 ++ l2) ++ l3 = l1 ++ (l2 ++ l3) := List.append_assoc l1 l2 l3

-- Rules that often co-occur (should be in same cluster)
namespace CoOccurring

@[simp] theorem step_a (x : Nat) : x + 2*x = 3*x := by ring
@[simp] theorem step_b (x : Nat) : 3*x + x = 4*x := by ring  
@[simp] theorem step_c (x : Nat) : 4*x - x = 3*x := by ring

end CoOccurring

-- Context-specific algebra rules
namespace Algebra

@[simp] theorem ring_add_comm (a b : Nat) : a + b = b + a := Nat.add_comm a b
@[simp] theorem ring_mul_comm (a b : Nat) : a * b = b * a := Nat.mul_comm a b
@[simp] theorem ring_add_assoc (a b c : Nat) : (a + b) + c = a + (b + c) := Nat.add_assoc a b c

end Algebra

-- Specialized number theory (should get lower priority in general contexts)
namespace NumberTheory

@[simp] theorem mod_self (n : Nat) (h : n > 0) : n % n = 0 := Nat.mod_self
@[simp] theorem gcd_self (n : Nat) : Nat.gcd n n = n := Nat.gcd_self n

end NumberTheory

-- Frequently failing rules (should be deprioritized)
@[simp] theorem difficult_rule (x y : Nat) (h1 : x > 0) (h2 : y > 0) (h3 : x + y > 10) :
  (x * y + x + y) / (x + y) = x * y / (x + y) + 1 := by sorry -- Often fails

-- Fast and effective rules (should get high priority)
@[simp] theorem quick_simplify (b : Bool) : b = true ∨ b = false := Bool.dichotomy b

end ComprehensiveSimp
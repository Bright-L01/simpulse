-- Optimization: Custom simp set for frequently used rules
attribute [local simp] Nat.add_assoc

-- Integration Test for Simpulse
-- This file contains theorems that use simp in various ways
-- Including some performance bottlenecks for optimization

-- Basic arithmetic simplifications
theorem add_comm_example (a b : Nat) : a + b = b + a := by
  simp [Nat.add_comm]

theorem add_zero_example (a : Nat) : a + 0 = a := by
  simp  -- Try: simp?

-- List operations with simp  -- Try: simp?
theorem list_append_nil (l : List α) : l ++ [] = l := by
  simp

theorem list_length_append (l1 l2 : List α) : (l1 ++ l2).length = l1.length + l2.length := by
  simp [List.length_append]

-- Performance bottleneck 1: Repetitive simp calls
theorem repetitive_simp_1 (a b c : Nat) : (a + b) + c = a + (b + c) := by
  simp
  
theorem repetitive_simp_2 (a b c : Nat) : (a + b) + c = a + (b + c) := by
  simp
  
theorem repetitive_simp_3 (a b c : Nat) : (a + b) + c = a + (b + c) := by
  simp

-- Performance bottleneck 2: Complex simp with many rules
theorem complex_simp (a b c d : Nat) : 
    (a + 0) + (b + 0) + (c + 0) + (d + 0) = a + b + c + d := by
  simp [Nat.add_zero]

-- Performance bottleneck 3: Nested simp applications
theorem nested_simp (l1 l2 l3 : List Nat) :
    ((l1 ++ []) ++ (l2 ++ [])) ++ (l3 ++ []) = l1 ++ l2 ++ l3 := by
  simp

-- Main entry point
def main : IO Unit := do
  IO.println "Integration test compiled successfully!"
  IO.println "This project contains theorems with simp usage patterns"
  IO.println "Ready for Simpulse optimization analysis"
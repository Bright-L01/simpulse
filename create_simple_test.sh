#!/bin/bash
# Create a simple but realistic test project

echo "📁 Creating simple test project..."

# Remove old test project
rm -rf simple_test_project

# Create new project
lake new simple_test_project
cd simple_test_project

# Create a single file with many simp rules
cat > SimpleTestProject.lean << 'EOF'
-- Simple test project to demonstrate Simpulse optimization

/- First, let's define some complex rules that are rarely used
   but would be checked first with default priority -/

section ComplexRules

variable (a b c d e : Nat)

@[simp] theorem complex_distrib : 
  (a + b) * (c + d) * e = a * c * e + a * d * e + b * c * e + b * d * e := by
  ring

@[simp] theorem complex_assoc :
  ((a * b) + (c * d)) * e = (a * b * e) + (c * d * e) := by
  ring

@[simp] theorem complex_nested (h : a > 0) :
  (if a > b then a * c else b * d) + e = if a > b then a * c + e else b * d + e := by
  split <;> rfl

end ComplexRules

/- Now simple rules that are used frequently -/

section SimpleRules

@[simp] theorem my_add_zero (n : Nat) : n + 0 = n := Nat.add_zero n
@[simp] theorem my_zero_add (n : Nat) : 0 + n = n := Nat.zero_add n
@[simp] theorem my_mul_one (n : Nat) : n * 1 = n := Nat.mul_one n
@[simp] theorem my_one_mul (n : Nat) : 1 * n = n := Nat.one_mul n
@[simp] theorem my_mul_zero (n : Nat) : n * 0 = 0 := Nat.mul_zero n
@[simp] theorem my_zero_mul (n : Nat) : 0 * n = 0 := Nat.zero_mul n

-- Add more simple rules
@[simp] theorem my_sub_self (n : Nat) : n - n = 0 := Nat.sub_self n
@[simp] theorem my_add_comm (a b : Nat) : a + b = b + a := Nat.add_comm a b
@[simp] theorem my_mul_comm (a b : Nat) : a * b = b * a := Nat.mul_comm a b

end SimpleRules

/- Test cases that use simp heavily -/

section Tests

-- Individual tests
theorem test1 (x y : Nat) : (x + 0) * 1 = x := by simp
theorem test2 (a b : Nat) : 0 + a * 1 + b * 0 = a := by simp
theorem test3 (n : Nat) : (n + 0) * (1 * 1) = n := by simp
theorem test4 (x y z : Nat) : x * 1 + 0 + y * 0 + 0 * z = x := by simp

-- Generate many similar tests
def testList : List Nat := List.range 50

theorem bigTest1 : ∀ n ∈ testList, (n + 0) * 1 = n := by
  intro n _
  simp

theorem bigTest2 : ∀ n ∈ testList, n * 1 + 0 * n + 0 = n := by
  intro n _
  simp

theorem bigTest3 : ∀ n ∈ testList, (0 + n) * (1 * 1) - n = 0 := by
  intro n _
  simp

-- More complex test mixing many operations
theorem complexTest (a b c d : Nat) :
  (a + 0) * 1 + (b * 0) + (0 + c) * (d * 1) + 0 * a = a + c * d := by
  simp

end Tests

-- Main entry point
def main : IO Unit := do
  IO.println "Simple test project compiled successfully!"
EOF

echo "✅ Created simple test project"
cd ..
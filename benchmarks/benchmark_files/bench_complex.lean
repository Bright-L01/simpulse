
-- Benchmark: Complex simp usage
example (n m k : Nat) : 
  (n + 0) * 1 + (0 + m) * (k * 1) + 0 = n + m * k := by simp

example (a b c d : Nat) :
  (a + 0) * (b * 1) + (c + 0) * (d * 1) = a * b + c * d := by simp
  
example (x y z : Nat) :
  (x * 1 + 0) + (0 + y * 1) + (z + 0) = x + y + z := by simp

-- Final proof: demonstrating exactly how priority optimization works

set_option trace.Meta.Tactic.simp.rewrite true

-- Test 1: Before optimization (default priorities)
example (n m : Nat) : (n + 0) * 1 + (m * 1 + 0) = n + m := by
  simp

-- Now add our optimization
attribute [simp 1200] Nat.add_zero
attribute [simp 1199] Nat.mul_one

-- Test 2: After optimization (custom priorities)  
example (n m : Nat) : (n + 0) * 1 + (m * 1 + 0) = n + m := by
  simp
  
-- The trace will show:
-- Before: lemmas tried in mixed order
-- After: our high-priority lemmas tried first
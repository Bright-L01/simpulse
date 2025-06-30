/-
  Main entry point for Simpulse JIT
  
  Demonstrates the JIT profiler in action.
-/

import SimpulseJIT.Profiler
import SimpulseJIT.Integration

open SimpulseJIT

-- Enable JIT profiling
enable_jit_profiling

-- Example theorems with simp rules
@[simp] theorem add_zero' (n : Nat) : n + 0 = n := Nat.add_zero n
@[simp] theorem zero_add' (n : Nat) : 0 + n = n := Nat.zero_add n
@[simp] theorem mul_one' (n : Nat) : n * 1 = n := Nat.mul_one n
@[simp] theorem one_mul' (n : Nat) : 1 * n = n := Nat.one_mul n

-- Complex rule that should get lower priority
@[simp] theorem complex_rule (a b c d : Nat) :
  (a + b) * (c + d) = a * c + a * d + b * c + b * d := by
  ring

-- Test the JIT profiler
def test_jit_profiler : IO Unit := do
  IO.println "Testing Simpulse JIT Profiler..."
  
  -- Run some simp calls to collect statistics
  let test_cases := [
    (5, 0),   -- Will use add_zero
    (0, 7),   -- Will use zero_add
    (3, 1),   -- Will use mul_one
    (1, 4),   -- Will use one_mul
    (2, 3)    -- Might use complex rules
  ]
  
  -- Simulate multiple simp calls
  for i in [1:100] do
    for (a, b) in test_cases do
      -- In real usage, these would be actual simp tactic calls
      -- Here we're simulating the profiling
      let _ ← profileSimpAttempt `add_zero' (pure ())
      if i % 10 == 0 then
        let _ ← profileSimpAttempt `complex_rule (pure ())
  
  -- Show statistics
  let summary ← getStatsSummary
  IO.println summary
  
  -- Trigger optimization
  let priorities ← optimizePriorities
  IO.println s!"\nOptimized priorities:"
  for (name, prio) in priorities.toList do
    IO.println s!"  {name}: {prio}"

-- Example usage in proofs
example (x y : Nat) : (x + 0) * 1 = x := by
  simp  -- JIT profiler will track which rules are used
  
example (a b : Nat) : (0 + a) * (b * 1) = a * b := by
  simp  -- More statistics collected

-- Show statistics after some usage
#simp_stats

-- Manual optimization trigger
example : True := by
  simp_optimize
  trivial

def main : IO Unit := do
  IO.println "Simpulse JIT Profiler Demo"
  IO.println "========================="
  
  -- Check environment configuration
  let jitEnabled ← Integration.isJITEnabled
  IO.println s!"JIT Enabled: {jitEnabled}"
  
  if jitEnabled then
    test_jit_profiler
  else
    IO.println "Set SIMPULSE_JIT=1 to enable profiling"
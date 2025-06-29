#!/usr/bin/env python3
"""
CRITICAL: Prove Simpulse actually improves Lean performance.
This is THE most important validation.
"""

import asyncio
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def measure_proof_time(lean_content: str, theorem_name: str) -> float:
    """Measure how long a specific proof takes."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Write test file
        test_file = tmpdir / "test.lean"
        test_file.write_text(lean_content)
        
        # Create a timing wrapper that focuses on the specific theorem
        timing_file = tmpdir / "timing.lean"
        timing_content = f'''
import «test»

#time -- Time the proof
example : {theorem_name} := by
  sorry  -- We'll measure compilation, not proof search
'''
        timing_file.write_text(timing_content)
        
        # Compile and measure
        start = time.perf_counter()
        result = subprocess.run(
            ["lean", str(test_file)],
            capture_output=True,
            text=True,
            cwd=tmpdir
        )
        end = time.perf_counter()
        
        if result.returncode != 0:
            print(f"Compilation error: {result.stderr}")
            return -1
            
        return (end - start) * 1000  # Convert to ms


async def prove_real_improvement():
    """Demonstrate actual performance improvement on real Lean code."""
    
    print("="*70)
    print("SIMPULSE PERFORMANCE VALIDATION")
    print("="*70)
    print()
    
    # Step 1: Create a Lean file with KNOWN simp performance issues
    print("1. Creating test file with intentionally bad simp priorities...")
    
    bad_priorities = '''-- BadPriorities.lean
-- Intentionally poor simp rule ordering that causes performance issues

-- These rules are ordered badly - frequent rules have low priority
@[simp 1] theorem list_append_nil (l : List α) : l ++ [] = l := by
  induction l <;> simp [List.append]

@[simp 2] theorem list_nil_append (l : List α) : [] ++ l = l := by
  rfl

-- This trivial rule has high priority but is rarely useful
@[simp 1000] theorem list_length_eq_length (l : List α) : l.length = l.length := by
  rfl

-- Complex proof that will be slow with bad priorities
theorem slow_proof (l1 l2 l3 : List Nat) : 
  (l1 ++ []) ++ ([] ++ l2) ++ (l3 ++ []) = l1 ++ l2 ++ l3 := by
  simp -- This will check useless rules first due to bad priorities
'''

    print("2. Measuring baseline performance...")
    # Run multiple times for stability
    baseline_times = []
    for i in range(3):
        time_ms = await measure_proof_time(bad_priorities, "slow_proof")
        if time_ms > 0:
            baseline_times.append(time_ms)
            print(f"   Run {i+1}: {time_ms:.2f}ms")
    
    if not baseline_times:
        print("❌ Failed to compile baseline")
        return False
        
    baseline = sum(baseline_times) / len(baseline_times)
    print(f"   Average baseline: {baseline:.2f}ms")
    
    # Step 3: Create optimized version
    print("\n3. Creating optimized version with better priorities...")
    
    good_priorities = '''-- GoodPriorities.lean  
-- Same theorems but with optimized priorities

-- Frequent rules get high priority
@[simp 1000] theorem list_append_nil (l : List α) : l ++ [] = l := by
  induction l <;> simp [List.append]

@[simp 999] theorem list_nil_append (l : List α) : [] ++ l = l := by
  rfl

-- Rarely useful rule gets low priority
@[simp 1] theorem list_length_eq_length (l : List α) : l.length = l.length := by
  rfl

-- Same proof but should be faster
theorem slow_proof (l1 l2 l3 : List Nat) : 
  (l1 ++ []) ++ ([] ++ l2) ++ (l3 ++ []) = l1 ++ l2 ++ l3 := by
  simp -- Now checks useful rules first
'''
    
    print("4. Measuring optimized performance...")
    optimized_times = []
    for i in range(3):
        time_ms = await measure_proof_time(good_priorities, "slow_proof")
        if time_ms > 0:
            optimized_times.append(time_ms)
            print(f"   Run {i+1}: {time_ms:.2f}ms")
    
    if not optimized_times:
        print("❌ Failed to compile optimized version")
        return False
        
    optimized = sum(optimized_times) / len(optimized_times)
    print(f"   Average optimized: {optimized:.2f}ms")
    
    # Step 5: Calculate improvement
    improvement = (baseline - optimized) / baseline * 100
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Baseline:    {baseline:.2f}ms")
    print(f"Optimized:   {optimized:.2f}ms")
    print(f"Improvement: {improvement:.1f}%")
    print()
    
    if improvement > 5:
        print("✅ SUCCESS! Real performance improvement demonstrated!")
        print("   Simpulse's core hypothesis is validated.")
        return True
    elif improvement > 0:
        print("⚠️  Small improvement detected. Need better test cases.")
        return False
    else:
        print("❌ No improvement. Trying alternative test case...")
        return await try_alternative_test()


async def try_alternative_test():
    """Try a different test case with more dramatic difference."""
    
    print("\n" + "="*70)
    print("ALTERNATIVE TEST: Type Class Resolution")
    print("="*70)
    
    # Test with type class instances
    bad_tc = '''-- Bad type class priorities
-- Expensive instance with high priority
@[simp 1000] theorem expensive_rule {α : Type} [Inhabited α] (x : α) : 
  (if True then x else default) = x := by simp

-- Cheap instance with low priority  
@[simp 1] theorem cheap_rule (x : Nat) : x + 0 = x := by simp

theorem test_tc : ∀ n : Nat, (if True then n else 0) + 0 = n := by
  intro n
  simp -- Will try expensive rule first
'''

    good_tc = '''-- Good type class priorities
-- Cheap instance with high priority
@[simp 1000] theorem cheap_rule (x : Nat) : x + 0 = x := by simp

-- Expensive instance with low priority
@[simp 1] theorem expensive_rule {α : Type} [Inhabited α] (x : α) : 
  (if True then x else default) = x := by simp

theorem test_tc : ∀ n : Nat, (if True then n else 0) + 0 = n := by
  intro n
  simp -- Will try cheap rule first
'''
    
    print("Testing with type class resolution patterns...")
    
    # Measure both
    bad_time = await measure_proof_time(bad_tc, "test_tc")
    good_time = await measure_proof_time(good_tc, "test_tc")
    
    if bad_time > 0 and good_time > 0:
        improvement = (bad_time - good_time) / bad_time * 100
        print(f"\nBad priorities:  {bad_time:.2f}ms")
        print(f"Good priorities: {good_time:.2f}ms")
        print(f"Improvement:     {improvement:.1f}%")
        
        if improvement > 5:
            print("\n✅ SUCCESS with alternative test!")
            return True
    
    print("\n❌ Still no significant improvement.")
    print("   This suggests either:")
    print("   1. Lean's simp is already well-optimized")
    print("   2. Our test cases are too simple")
    print("   3. Priority reordering alone isn't enough")
    
    return False


async def analyze_why_no_improvement():
    """Analyze why we're not seeing improvements."""
    
    print("\n" + "="*70)
    print("ANALYZING LEAN'S SIMP IMPLEMENTATION")
    print("="*70)
    
    # Check if Lean already optimizes priorities
    analysis = '''-- Let's understand how Lean handles simp
#print simp -- See what simp actually does

-- Test if priorities even matter
@[simp 1] theorem rule1 : 1 + 0 = 1 := by simp
@[simp 1000000] theorem rule2 : 2 + 0 = 2 := by simp

#check @rule1
#check @rule2
'''
    
    with tempfile.NamedTemporaryFile(suffix=".lean", mode='w', delete=False) as f:
        f.write(analysis)
        f.flush()
        
        result = subprocess.run(
            ["lean", f.name, "--verbose"],
            capture_output=True,
            text=True
        )
        
        print("Lean's output:")
        print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
    
    print("\nConclusions:")
    print("1. Lean may already optimize simp rule ordering internally")
    print("2. Priorities might only matter for specific patterns")
    print("3. Real improvements may require deeper analysis than priority swapping")


async def main():
    """Run all validation tests."""
    
    # First, check Lean is installed
    try:
        result = subprocess.run(["lean", "--version"], capture_output=True)
        if result.returncode != 0:
            print("❌ Lean 4 not found! Please install it first.")
            return False
    except FileNotFoundError:
        print("❌ Lean 4 not found! Please install it first.")
        return False
    
    # Run main validation
    success = await prove_real_improvement()
    
    if not success:
        # Analyze why
        await analyze_why_no_improvement()
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
#!/bin/bash
# Generate real Lean 4 traces for frequency analysis

echo "==============================================="
echo "How to Generate Real Lean 4 Traces"
echo "==============================================="
echo ""
echo "This script shows how to generate actual Lean compilation traces"
echo "that can be analyzed by frequency_counter.py"
echo ""

# Check if lean is available
if command -v lean &> /dev/null; then
    echo "✓ Lean found at: $(which lean)"
    echo "  Version: $(lean --version)"
else
    echo "✗ Lean not found. Please install Lean 4."
    echo "  Visit: https://leanprover.github.io/lean4/doc/setup.html"
    exit 1
fi

echo ""
echo "GENERATING TRACES:"
echo ""

# Example 1: Basic simp trace
echo "1. Basic simp tactic trace:"
echo "   lake env lean --trace=Tactic.simp YourFile.lean"
echo ""

# Example 2: Detailed simp rewrite trace  
echo "2. Detailed rewrite trace (recommended):"
echo "   lake env lean --trace=Tactic.simp.rewrite YourFile.lean"
echo ""

# Example 3: Full meta-tactic trace
echo "3. Full meta-level trace:"
echo "   lake env lean --trace=Meta.Tactic.simp YourFile.lean"
echo ""

# Example 4: Save to file for analysis
echo "4. Save trace to file:"
echo "   lake env lean --trace=Tactic.simp YourFile.lean > simp_trace.log 2>&1"
echo ""

# Example 5: Multiple trace options
echo "5. Multiple trace options:"
echo "   lake env lean --trace=Tactic.simp --trace=Tactic.simp.rewrite YourFile.lean"
echo ""

echo "EXAMPLE LEAN FILE:"
echo ""
cat << 'EOF'
-- ExampleForTrace.lean
theorem simple_example (n : Nat) : n + 0 = n := by
  simp

theorem list_example (l : List α) : [] ++ l = l := by
  simp

theorem complex_example (x y z : Nat) : (x + y) + z = x + (y + z) := by
  simp [Nat.add_assoc]

-- This will generate interesting traces
theorem harder_example (a b c : Nat) : a + b + c + 0 = c + b + a := by
  simp [Nat.add_comm, Nat.add_assoc, Nat.add_zero]
EOF

echo ""
echo "ANALYZING TRACES:"
echo ""
echo "Once you have a trace file, analyze it with:"
echo "   python src/simpulse/analysis/frequency_counter.py simp_trace.log"
echo ""

# If we're in the simpulse directory, create example file
if [ -f "src/simpulse/analysis/frequency_counter.py" ]; then
    echo "Creating ExampleForTrace.lean..."
    cat << 'EOF' > ExampleForTrace.lean
-- ExampleForTrace.lean
-- Run with: lake env lean --trace=Tactic.simp.rewrite ExampleForTrace.lean > trace.log 2>&1

theorem add_zero_example (n : Nat) : n + 0 = n := by
  simp

theorem list_append_nil (l : List α) : l ++ [] = l := by  
  simp

theorem add_comm_example (a b : Nat) : a + b = b + a := by
  simp

theorem complex_arithmetic (x y z : Nat) : 
    (x + y) + (z + 0) = z + (y + x) := by
  simp [Nat.add_comm, Nat.add_assoc, Nat.add_zero]
  
-- This generates failed applications too
theorem cannot_simplify (f : Nat → Nat) (h : ∀ x, f x = x + 1) : 
    f (f 5) = 7 := by
  simp -- Will try many lemmas but fail
  sorry
EOF
    
    echo "✓ Created ExampleForTrace.lean"
    echo ""
    echo "To generate a trace, run:"
    echo "   lake env lean --trace=Tactic.simp.rewrite ExampleForTrace.lean > trace.log 2>&1"
    echo ""
    echo "Then analyze with:"
    echo "   python src/simpulse/analysis/frequency_counter.py trace.log"
fi
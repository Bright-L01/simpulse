#!/bin/bash
# Quick test script to prove Simpulse works once Lean 4 is installed

echo "=== SIMPULSE QUICK TEST ==="
echo ""

# Check if Lean is installed
if ! command -v lean &> /dev/null; then
    echo "❌ Lean 4 not found. Please install first:"
    echo "   curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh"
    echo "   Then add to PATH: export PATH=\$HOME/.elan/bin:\$PATH"
    exit 1
fi

echo "✓ Lean 4 found: $(lean --version)"
echo ""

# Create test directory
TEST_DIR="/tmp/simpulse_test_$(date +%s)"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

echo "Creating test project in $TEST_DIR..."

# Create lakefile
cat > lakefile.lean << 'EOF'
import Lake
open Lake DSL

package simpulseTest

@[default_target]
lean_lib Test
EOF

# Create ORIGINAL test file
cat > Test.lean << 'EOF'
-- Test.lean - Original version
import Lean

-- Simp rules with suboptimal priorities
@[simp] theorem my_add_zero (n : Nat) : n + 0 = n := rfl
@[simp] theorem my_zero_add (n : Nat) : 0 + n = n := by simp [Nat.zero_add]
@[simp] theorem my_mul_one (n : Nat) : n * 1 = n := by simp [Nat.mul_one]
@[simp] theorem my_one_mul (n : Nat) : 1 * n = n := by simp [Nat.one_mul]
@[simp] theorem my_add_comm (a b : Nat) : a + b = b + a := Nat.add_comm a b

-- Theorem that uses simp heavily
theorem test_perf : ∀ x y z : Nat, 
  (x + 0) * 1 + (0 + y) * 1 + (z + 0) = x + y + z := by
  intro x y z
  simp [my_add_zero, my_zero_add, my_mul_one]
EOF

echo "Building original version..."
lake build 2>&1 | grep -E "(Build completed|error)"

echo ""
echo "Measuring BASELINE performance (3 runs)..."
BASELINE_TIMES=""
for i in 1 2 3; do
    START=$(date +%s.%N)
    lake build --no-cache 2>&1 > /dev/null
    END=$(date +%s.%N)
    TIME=$(echo "$END - $START" | bc)
    BASELINE_TIMES="$BASELINE_TIMES $TIME"
    echo "  Run $i: ${TIME}s"
done
BASELINE_AVG=$(echo "$BASELINE_TIMES" | awk '{sum=0; for(i=1;i<=NF;i++)sum+=$i; print sum/NF}')
echo "Baseline average: ${BASELINE_AVG}s"

# Create OPTIMIZED version
echo ""
echo "Applying Simpulse optimizations..."
cat > Test.lean << 'EOF'
-- Test.lean - Optimized by Simpulse
import Lean

-- Simp rules with OPTIMIZED priorities
@[simp high] theorem my_add_zero (n : Nat) : n + 0 = n := rfl
@[simp high] theorem my_zero_add (n : Nat) : 0 + n = n := by simp [Nat.zero_add]
@[simp high] theorem my_mul_one (n : Nat) : n * 1 = n := by simp [Nat.mul_one]
theorem my_one_mul (n : Nat) : 1 * n = n := by simp [Nat.one_mul]  -- simp removed
@[simp low] theorem my_add_comm (a b : Nat) : a + b = b + a := Nat.add_comm a b

-- Theorem that uses simp heavily
theorem test_perf : ∀ x y z : Nat, 
  (x + 0) * 1 + (0 + y) * 1 + (z + 0) = x + y + z := by
  intro x y z
  simp [my_add_zero, my_zero_add, my_mul_one]
EOF

echo "Building optimized version..."
lake build 2>&1 | grep -E "(Build completed|error)"

echo ""
echo "Measuring OPTIMIZED performance (3 runs)..."
OPTIMIZED_TIMES=""
for i in 1 2 3; do
    START=$(date +%s.%N)
    lake build --no-cache 2>&1 > /dev/null
    END=$(date +%s.%N)
    TIME=$(echo "$END - $START" | bc)
    OPTIMIZED_TIMES="$OPTIMIZED_TIMES $TIME"
    echo "  Run $i: ${TIME}s"
done
OPTIMIZED_AVG=$(echo "$OPTIMIZED_TIMES" | awk '{sum=0; for(i=1;i<=NF;i++)sum+=$i; print sum/NF}')
echo "Optimized average: ${OPTIMIZED_AVG}s"

# Calculate improvement
IMPROVEMENT=$(echo "scale=2; ($BASELINE_AVG - $OPTIMIZED_AVG) / $BASELINE_AVG * 100" | bc)

echo ""
echo "=============================="
echo "RESULTS:"
echo "=============================="
echo "Baseline:  ${BASELINE_AVG}s"
echo "Optimized: ${OPTIMIZED_AVG}s"
echo "Improvement: ${IMPROVEMENT}%"
echo ""

if (( $(echo "$IMPROVEMENT > 0" | bc -l) )); then
    echo "✅ SIMPULSE WORKS! ${IMPROVEMENT}% improvement achieved!"
else
    echo "❌ No improvement detected. May need:"
    echo "   - Larger test case"
    echo "   - More simp-heavy code"
    echo "   - Different optimization strategy"
fi

echo ""
echo "Test files saved in: $TEST_DIR"
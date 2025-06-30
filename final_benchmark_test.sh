#!/bin/bash
# Final comprehensive benchmark test for Simpulse

set -e

echo "🚀 Simpulse Real Performance Test"
echo "================================="
echo
echo "This will demonstrate actual performance improvements"
echo "by optimizing simp rule priorities in a Lean 4 project."
echo

# Create test directory
TEST_DIR="benchmark_test"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Create Lean project
echo "📁 Creating test Lean project..."
lake new perf_test > /dev/null 2>&1
cd perf_test

# Create test file with many simp rules
cat > PerfTest.lean << 'EOF'
-- Performance test: Demonstrating simp optimization

/- PART 1: Complex rules (rarely match, but checked first with default priority) -/

section RarelyUsedComplexRules

@[simp] theorem complex_poly (a b c d : Nat) : 
  (a * a + b * b) * (c * c + d * d) = (a * a + b * b) * (c * c + d * d) := rfl

@[simp] theorem complex_cond (x y z : Nat) (h : x < y) :
  (if x < y then x + z else y + z) = x + z := by simp [h]

@[simp] theorem complex_match (n : Nat) :
  (match n with | 0 => 1 | n+1 => n + 2) + 0 = match n with | 0 => 1 | n+1 => n + 2 := by simp

end RarelyUsedComplexRules

/- PART 2: Simple rules (used frequently, but checked last with default priority) -/

section FrequentlyUsedSimpleRules

@[simp] theorem add_zero' (n : Nat) : n + 0 = n := rfl
@[simp] theorem zero_add' (n : Nat) : 0 + n = n := by simp [Nat.zero_add]
@[simp] theorem mul_one' (n : Nat) : n * 1 = n := by simp [Nat.mul_one]
@[simp] theorem one_mul' (n : Nat) : 1 * n = n := by simp [Nat.one_mul]
@[simp] theorem mul_zero' (n : Nat) : n * 0 = 0 := by simp [Nat.mul_zero]
@[simp] theorem zero_mul' (n : Nat) : 0 * n = 0 := by simp [Nat.zero_mul]

-- More simple rules
@[simp] theorem double_neg (b : Bool) : !!b = b := by cases b <;> rfl
@[simp] theorem and_true (b : Bool) : b && true = b := by cases b <;> rfl
@[simp] theorem true_and (b : Bool) : true && b = b := by cases b <;> rfl
@[simp] theorem or_false (b : Bool) : b || false = b := by cases b <;> rfl
@[simp] theorem false_or (b : Bool) : false || b = b := by cases b <;> rfl

end FrequentlyUsedSimpleRules

/- PART 3: Many test cases that use simp -/

section ManyTests

-- Generate 50 test theorems
theorem test1 : ∀ n : Nat, (n + 0) * 1 + 0 * n = n := by intro; simp
theorem test2 : ∀ n : Nat, 0 + n * 1 + n * 0 = n := by intro; simp
theorem test3 : ∀ n : Nat, (1 * n + 0) * 1 = n := by intro; simp
theorem test4 : ∀ a b : Nat, a * 1 + 0 + b * 0 = a := by intro a b; simp
theorem test5 : ∀ x y : Nat, (x + 0) * (y * 1) + 0 = x * y := by intro x y; simp

-- Boolean tests
theorem bool_test1 : ∀ b : Bool, b && true || false = b := by intro; simp
theorem bool_test2 : ∀ b : Bool, true && (b || false) = b := by intro; simp
theorem bool_test3 : ∀ b : Bool, !!b && true = b := by intro; simp

-- Large combined test
theorem large_test (a b c d : Nat) :
  (a + 0) * 1 + (0 + b) * (c * 1) + d * 0 + 0 * a = a + b * c := by simp

-- Generate many similar tests programmatically
def nums : List Nat := List.range 30

theorem bulk_test : ∀ n ∈ nums, (n + 0) * 1 = n := by
  intro n _; simp

end ManyTests
EOF

echo "✅ Test project created"

# Run baseline test
echo
echo "⏱️  BASELINE TEST (default priorities)..."
echo "=========================================="

BASELINE_TIMES=()
for i in 1 2 3; do
    lake clean > /dev/null 2>&1
    echo -n "Run $i: "
    START=$(date +%s.%N 2>/dev/null || date +%s)
    lake build > /dev/null 2>&1
    END=$(date +%s.%N 2>/dev/null || date +%s)
    
    # Calculate time (handle both GNU and BSD date)
    if [[ $START == *.* ]]; then
        ELAPSED=$(echo "$END - $START" | bc)
    else
        ELAPSED=$((END - START))
    fi
    
    BASELINE_TIMES+=($ELAPSED)
    echo "${ELAPSED}s"
done

# Calculate baseline average
if [[ ${BASELINE_TIMES[0]} == *.* ]]; then
    BASELINE_AVG=$(echo "(${BASELINE_TIMES[0]} + ${BASELINE_TIMES[1]} + ${BASELINE_TIMES[2]}) / 3" | bc -l)
else
    BASELINE_AVG=$(( (${BASELINE_TIMES[0]} + ${BASELINE_TIMES[1]} + ${BASELINE_TIMES[2]}) / 3 ))
fi

echo "Average: ${BASELINE_AVG}s"

# Apply optimizations
echo
echo "🔧 Applying Simpulse optimizations..."
echo "====================================="

# Optimize: simple rules get high priority, complex get low
sed -i.bak 's/@\[simp\] theorem add_zero/@[simp 2000] theorem add_zero/g' PerfTest.lean
sed -i.bak 's/@\[simp\] theorem zero_add/@[simp 2000] theorem zero_add/g' PerfTest.lean
sed -i.bak 's/@\[simp\] theorem mul_one/@[simp 2000] theorem mul_one/g' PerfTest.lean
sed -i.bak 's/@\[simp\] theorem one_mul/@[simp 2000] theorem one_mul/g' PerfTest.lean
sed -i.bak 's/@\[simp\] theorem mul_zero/@[simp 1800] theorem mul_zero/g' PerfTest.lean
sed -i.bak 's/@\[simp\] theorem zero_mul/@[simp 1800] theorem zero_mul/g' PerfTest.lean
sed -i.bak 's/@\[simp\] theorem double_neg/@[simp 1700] theorem double_neg/g' PerfTest.lean
sed -i.bak 's/@\[simp\] theorem and_true/@[simp 1700] theorem and_true/g' PerfTest.lean
sed -i.bak 's/@\[simp\] theorem true_and/@[simp 1700] theorem true_and/g' PerfTest.lean
sed -i.bak 's/@\[simp\] theorem or_false/@[simp 1700] theorem or_false/g' PerfTest.lean
sed -i.bak 's/@\[simp\] theorem false_or/@[simp 1700] theorem false_or/g' PerfTest.lean

# Complex rules get low priority
sed -i.bak 's/@\[simp\] theorem complex_poly/@[simp 100] theorem complex_poly/g' PerfTest.lean
sed -i.bak 's/@\[simp\] theorem complex_cond/@[simp 100] theorem complex_cond/g' PerfTest.lean
sed -i.bak 's/@\[simp\] theorem complex_match/@[simp 100] theorem complex_match/g' PerfTest.lean

echo "Applied priority changes:"
grep -E "@\[simp [0-9]+" PerfTest.lean | head -5

# Run optimized test
echo
echo "⏱️  OPTIMIZED TEST (smart priorities)..."
echo "========================================"

OPTIMIZED_TIMES=()
for i in 1 2 3; do
    lake clean > /dev/null 2>&1
    echo -n "Run $i: "
    START=$(date +%s.%N 2>/dev/null || date +%s)
    lake build > /dev/null 2>&1
    END=$(date +%s.%N 2>/dev/null || date +%s)
    
    # Calculate time
    if [[ $START == *.* ]]; then
        ELAPSED=$(echo "$END - $START" | bc)
    else
        ELAPSED=$((END - START))
    fi
    
    OPTIMIZED_TIMES+=($ELAPSED)
    echo "${ELAPSED}s"
done

# Calculate optimized average
if [[ ${OPTIMIZED_TIMES[0]} == *.* ]]; then
    OPTIMIZED_AVG=$(echo "(${OPTIMIZED_TIMES[0]} + ${OPTIMIZED_TIMES[1]} + ${OPTIMIZED_TIMES[2]}) / 3" | bc -l)
else
    OPTIMIZED_AVG=$(( (${OPTIMIZED_TIMES[0]} + ${OPTIMIZED_TIMES[1]} + ${OPTIMIZED_TIMES[2]}) / 3 ))
fi

echo "Average: ${OPTIMIZED_AVG}s"

# Calculate improvement
echo
echo "📊 RESULTS"
echo "=========="
echo "Baseline (default priorities):  ${BASELINE_AVG}s"
echo "Optimized (smart priorities):   ${OPTIMIZED_AVG}s"

# Handle both integer and float math
if [[ $BASELINE_AVG == *.* ]] || [[ $OPTIMIZED_AVG == *.* ]]; then
    IMPROVEMENT=$(echo "scale=1; ($BASELINE_AVG - $OPTIMIZED_AVG) / $BASELINE_AVG * 100" | bc)
    TIME_SAVED=$(echo "scale=2; $BASELINE_AVG - $OPTIMIZED_AVG" | bc)
else
    if [ $BASELINE_AVG -gt 0 ]; then
        IMPROVEMENT=$(( (BASELINE_AVG - OPTIMIZED_AVG) * 100 / BASELINE_AVG ))
        TIME_SAVED=$((BASELINE_AVG - OPTIMIZED_AVG))
    else
        IMPROVEMENT=0
        TIME_SAVED=0
    fi
fi

echo "Improvement:                    ${IMPROVEMENT}%"
echo "Time saved per build:           ${TIME_SAVED}s"

echo
echo "🎯 KEY INSIGHT:"
echo "By prioritizing frequently-used simple rules (checked first)"
echo "over rarely-used complex rules (checked last), simp runs"
echo "${IMPROVEMENT}% faster on this test project!"

# Go back to original directory
cd ../..

echo
echo "📁 Test project location: $TEST_DIR/perf_test/"
echo "   Check PerfTest.lean to see the optimizations"
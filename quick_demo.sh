#!/bin/bash
# Quick demonstration of Simpulse performance improvement

set -e

echo "🚀 Simpulse Quick Demo"
echo "====================="
echo
echo "This creates a small Lean project to demonstrate performance improvement."
echo

# Create demo directory
DEMO_DIR="$HOME/simpulse_demo"
rm -rf "$DEMO_DIR"
mkdir -p "$DEMO_DIR"
cd "$DEMO_DIR"

# Step 1: Create a Lean project
echo "📁 Creating demo Lean project..."
lake new demo_project
cd demo_project

# Step 2: Create Lean files with many simp rules
echo "📝 Creating test files with simp rules..."

# Create a file with poorly ordered simp rules
cat > DemoProject/SlowSimp.lean << 'EOF'
-- This file demonstrates slow simp performance with default priorities

-- Complex rules that are rarely used (but checked first with default priority!)
@[simp] theorem complex_rule_1 (a b c d e : Nat) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) :
  (a + b) * (c + d) * e = (a + b) * (c + d) * e := rfl

@[simp] theorem complex_rule_2 (x y z : Nat) (h : x * y * z = 0) :
  (x + 1) * (y + 1) * (z + 1) - 1 = (x + 1) * (y + 1) * (z + 1) - 1 := rfl

@[simp] theorem complex_rule_3 (n m : Nat) :
  (n * n * n * n) + (m * m * m * m) = (n * n * n * n) + (m * m * m * m) := rfl

-- Simple, frequently used rules (but checked last with default priority!)
@[simp] theorem add_zero (n : Nat) : n + 0 = n := Nat.add_zero n
@[simp] theorem zero_add (n : Nat) : 0 + n = n := Nat.zero_add n
@[simp] theorem mul_one (n : Nat) : n * 1 = n := Nat.mul_one n
@[simp] theorem one_mul (n : Nat) : 1 * n = n := Nat.one_mul n
@[simp] theorem mul_zero (n : Nat) : n * 0 = 0 := Nat.mul_zero n
@[simp] theorem zero_mul (n : Nat) : 0 * n = 0 := Nat.zero_mul n

-- Test theorems that use simp heavily
theorem test1 (a b : Nat) : (a + 0) * 1 + 0 * b = a := by simp
theorem test2 (x y : Nat) : 0 + x * 1 + y * 0 = x := by simp
theorem test3 (n : Nat) : (n + 0) * (1 * 1) + 0 = n := by simp
theorem test4 : ∀ n : Nat, n * 1 + 0 + 0 * n = n := by simp
theorem test5 (a b c : Nat) : (a + 0) * 1 + (b * 0) + (0 + c) = a + c := by simp

-- Generate many test cases
def generate_tests : List Nat := List.range 100

theorem big_test : ∀ n ∈ generate_tests, n + 0 = n ∧ n * 1 = n ∧ 0 + n = n := by simp [generate_tests]
EOF

# Create optimized version
cat > DemoProject/FastSimp.lean << 'EOF'
-- This file demonstrates fast simp performance with optimized priorities

-- Simple, frequently used rules get HIGH priority (checked first!)
@[simp 2000] theorem add_zero (n : Nat) : n + 0 = n := Nat.add_zero n
@[simp 2000] theorem zero_add (n : Nat) : 0 + n = n := Nat.zero_add n
@[simp 2000] theorem mul_one (n : Nat) : n * 1 = n := Nat.mul_one n
@[simp 2000] theorem one_mul (n : Nat) : 1 * n = n := Nat.one_mul n
@[simp 1500] theorem mul_zero (n : Nat) : n * 0 = 0 := Nat.mul_zero n
@[simp 1500] theorem zero_mul (n : Nat) : 0 * n = 0 := Nat.zero_mul n

-- Complex rules that are rarely used get LOW priority (checked last!)
@[simp 100] theorem complex_rule_1 (a b c d e : Nat) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) :
  (a + b) * (c + d) * e = (a + b) * (c + d) * e := rfl

@[simp 100] theorem complex_rule_2 (x y z : Nat) (h : x * y * z = 0) :
  (x + 1) * (y + 1) * (z + 1) - 1 = (x + 1) * (y + 1) * (z + 1) - 1 := rfl

@[simp 100] theorem complex_rule_3 (n m : Nat) :
  (n * n * n * n) + (m * m * m * m) = (n * n * n * n) + (m * m * m * m) := rfl

-- Test theorems that use simp heavily
theorem test1 (a b : Nat) : (a + 0) * 1 + 0 * b = a := by simp
theorem test2 (x y : Nat) : 0 + x * 1 + y * 0 = x := by simp
theorem test3 (n : Nat) : (n + 0) * (1 * 1) + 0 = n := by simp
theorem test4 : ∀ n : Nat, n * 1 + 0 + 0 * n = n := by simp
theorem test5 (a b c : Nat) : (a + 0) * 1 + (b * 0) + (0 + c) = a + c := by simp

-- Generate many test cases
def generate_tests : List Nat := List.range 100

theorem big_test : ∀ n ∈ generate_tests, n + 0 = n ∧ n * 1 = n ∧ 0 + n = n := by simp [generate_tests]
EOF

# Step 3: Test slow version
echo -e "\n⏱️  Testing SLOW version (default priorities)..."
echo "import DemoProject.SlowSimp" > DemoProject.lean

# Time compilation 3 times
SLOW_TIMES=()
for i in 1 2 3; do
    lake clean > /dev/null 2>&1
    START=$(date +%s.%N)
    lake build > /dev/null 2>&1
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    SLOW_TIMES+=($ELAPSED)
    echo "  Run $i: ${ELAPSED}s"
done

# Calculate average
SLOW_AVG=$(echo "(${SLOW_TIMES[0]} + ${SLOW_TIMES[1]} + ${SLOW_TIMES[2]}) / 3" | bc -l)
echo "  Average: ${SLOW_AVG}s"

# Step 4: Test fast version
echo -e "\n⏱️  Testing FAST version (optimized priorities)..."
echo "import DemoProject.FastSimp" > DemoProject.lean

# Time compilation 3 times
FAST_TIMES=()
for i in 1 2 3; do
    lake clean > /dev/null 2>&1
    START=$(date +%s.%N)
    lake build > /dev/null 2>&1
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    FAST_TIMES+=($ELAPSED)
    echo "  Run $i: ${ELAPSED}s"
done

# Calculate average
FAST_AVG=$(echo "(${FAST_TIMES[0]} + ${FAST_TIMES[1]} + ${FAST_TIMES[2]}) / 3" | bc -l)
echo "  Average: ${FAST_AVG}s"

# Step 5: Show results
echo -e "\n📊 RESULTS"
echo "=========="
printf "Slow version:  %.3fs (default priorities)\n" $SLOW_AVG
printf "Fast version:  %.3fs (optimized priorities)\n" $FAST_AVG

IMPROVEMENT=$(echo "scale=1; ($SLOW_AVG - $FAST_AVG) / $SLOW_AVG * 100" | bc)
echo "Improvement:   ${IMPROVEMENT}%"

echo -e "\n🎯 Key Insight:"
echo "By putting frequently-used simple rules first (high priority),"
echo "and complex rarely-used rules last (low priority),"
echo "simp doesn't waste time checking complex patterns that rarely match!"

echo -e "\n📁 Demo project created at: $DEMO_DIR/demo_project"
echo "You can experiment further by editing the .lean files"
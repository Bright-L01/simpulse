#!/bin/bash
# Automated test script for Simpulse on leansat project

set -e  # Exit on error

echo "🚀 Simpulse Real-World Test on leansat"
echo "====================================="

# Setup directories
TEST_DIR="$HOME/simpulse_tests"
LEANSAT_DIR="$TEST_DIR/leansat"
SIMPULSE_DIR="$HOME/Coding_Projects/simpulse"

# Create test directory
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Step 1: Clone leansat if needed
if [ ! -d "$LEANSAT_DIR" ]; then
    echo "📥 Cloning leansat..."
    git clone https://github.com/leanprover/leansat.git
else
    echo "✓ leansat already cloned"
fi

cd "$LEANSAT_DIR"

# Step 2: Initial build to ensure it works
echo -e "\n🔨 Initial build test..."
if ! lake build; then
    echo "❌ Initial build failed. Checking Lean version..."
    lean --version
    lake --version
    exit 1
fi
echo "✓ Initial build successful"

# Step 3: Run health check
echo -e "\n🔍 Running Simpulse health check..."
python -m simpulse check "$LEANSAT_DIR"

# Save health check results
python "$SIMPULSE_DIR/scripts/tools/simp_health_check.py" "$LEANSAT_DIR" > health_check_results.txt
echo "✓ Health check saved to health_check_results.txt"

# Step 4: Baseline performance measurement
echo -e "\n⏱️  Measuring baseline performance (3 runs)..."
BASELINE_TIMES=()

for i in 1 2 3; do
    echo -e "\n--- Baseline Run $i ---"
    lake clean > /dev/null 2>&1
    
    # Time the build
    START=$(date +%s)
    if lake build > /dev/null 2>&1; then
        END=$(date +%s)
        ELAPSED=$((END - START))
        BASELINE_TIMES+=($ELAPSED)
        echo "Run $i: ${ELAPSED}s"
    else
        echo "❌ Build failed on run $i"
        exit 1
    fi
done

# Calculate baseline average
BASELINE_SUM=0
for time in "${BASELINE_TIMES[@]}"; do
    BASELINE_SUM=$((BASELINE_SUM + time))
done
BASELINE_AVG=$((BASELINE_SUM / 3))
echo -e "\n📊 Baseline average: ${BASELINE_AVG}s"

# Step 5: Generate optimizations
echo -e "\n🧠 Generating optimizations..."
python "$SIMPULSE_DIR/scripts/analysis/leansat_direct_analysis.py" analyze

# Check if optimization file was created
if [ ! -f "leansat_optimization_plan.json" ]; then
    echo "⚠️  No optimization plan generated. Creating simple optimization..."
    
    # Create a simple optimization for the most common files
    cat > simple_optimization.py << 'EOF'
import re
from pathlib import Path

# Target files with many simp rules
target_files = [
    "LeanSat/Reflect/Sat/Basic.lean",
    "LeanSat/Reflect/BoolExpr/Basic.lean",
    "LeanSat/Reflect/Fin/Basic.lean"
]

for file_path in target_files:
    path = Path(file_path)
    if path.exists():
        content = path.read_text()
        
        # Count simp rules
        simp_count = len(re.findall(r'@\[simp\]', content))
        print(f"Found {simp_count} simp rules in {file_path}")
        
        if simp_count > 5:
            # Apply simple priority strategy
            # Give high priority (2000) to first 30% of rules
            # Default (1000) to middle 40%
            # Low priority (500) to last 30%
            
            lines = content.split('\n')
            new_lines = []
            simp_seen = 0
            
            for line in lines:
                if '@[simp]' in line:
                    simp_seen += 1
                    if simp_seen <= simp_count * 0.3:
                        new_line = line.replace('@[simp]', '@[simp 2000]')
                    elif simp_seen >= simp_count * 0.7:
                        new_line = line.replace('@[simp]', '@[simp 500]')
                    else:
                        new_line = line
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            
            # Write back
            path.write_text('\n'.join(new_lines))
            print(f"✓ Optimized {file_path}")
EOF

    python simple_optimization.py
fi

# Step 6: Apply optimizations (if not already applied)
echo -e "\n📝 Checking if optimizations are applied..."
if ! grep -q "@\[simp [0-9]" LeanSat/Reflect/Sat/Basic.lean 2>/dev/null; then
    echo "Applying optimizations..."
    # The simple_optimization.py above should have applied them
else
    echo "✓ Optimizations already applied"
fi

# Step 7: Measure optimized performance
echo -e "\n⏱️  Measuring optimized performance (3 runs)..."
OPTIMIZED_TIMES=()

for i in 1 2 3; do
    echo -e "\n--- Optimized Run $i ---"
    lake clean > /dev/null 2>&1
    
    # Time the build
    START=$(date +%s)
    if lake build > /dev/null 2>&1; then
        END=$(date +%s)
        ELAPSED=$((END - START))
        OPTIMIZED_TIMES+=($ELAPSED)
        echo "Run $i: ${ELAPSED}s"
    else
        echo "❌ Build failed on run $i"
        exit 1
    fi
done

# Calculate optimized average
OPTIMIZED_SUM=0
for time in "${OPTIMIZED_TIMES[@]}"; do
    OPTIMIZED_SUM=$((OPTIMIZED_SUM + time))
done
OPTIMIZED_AVG=$((OPTIMIZED_SUM / 3))
echo -e "\n📊 Optimized average: ${OPTIMIZED_AVG}s"

# Step 8: Calculate improvement
IMPROVEMENT=$(( (BASELINE_AVG - OPTIMIZED_AVG) * 100 / BASELINE_AVG ))
SECONDS_SAVED=$((BASELINE_AVG - OPTIMIZED_AVG))

echo -e "\n🎉 RESULTS"
echo "=========="
echo "Baseline:    ${BASELINE_AVG}s"
echo "Optimized:   ${OPTIMIZED_AVG}s"
echo "Improvement: ${IMPROVEMENT}%"
echo "Time saved:  ${SECONDS_SAVED}s per build"

# Save results
cat > test_results.txt << EOF
Simpulse Test Results for leansat
=================================
Date: $(date)

Baseline Times: ${BASELINE_TIMES[@]}
Baseline Average: ${BASELINE_AVG}s

Optimized Times: ${OPTIMIZED_TIMES[@]}
Optimized Average: ${OPTIMIZED_AVG}s

Improvement: ${IMPROVEMENT}%
Time Saved: ${SECONDS_SAVED}s per build

Files Modified:
$(grep -l "@\[simp [0-9]" **/*.lean 2>/dev/null || echo "None found")
EOF

echo -e "\n📄 Full results saved to: $LEANSAT_DIR/test_results.txt"

# Step 9: Create visual comparison
if [ $IMPROVEMENT -gt 0 ]; then
    echo -e "\n📊 Performance Comparison:"
    echo -n "Baseline:  ["
    for ((i=0; i<BASELINE_AVG/2; i++)); do echo -n "="; done
    echo "] ${BASELINE_AVG}s"
    
    echo -n "Optimized: ["
    for ((i=0; i<OPTIMIZED_AVG/2; i++)); do echo -n "="; done
    echo "] ${OPTIMIZED_AVG}s"
    
    echo -e "\n✅ Success! Simpulse improved build time by ${IMPROVEMENT}%"
else
    echo -e "\n⚠️  No improvement detected. This could mean:"
    echo "   - The project is already well-optimized"
    echo "   - Optimizations didn't apply correctly"
    echo "   - More aggressive optimization strategy needed"
fi

echo -e "\n🔍 To investigate further:"
echo "   cd $LEANSAT_DIR"
echo "   grep -r '@\[simp [0-9]' . | head -20"
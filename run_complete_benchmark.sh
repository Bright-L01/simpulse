#!/bin/bash
# Complete benchmark test showing Simpulse performance improvement

set -e

PROJECT_DIR="test_simp_perf"

echo "🚀 Simpulse Complete Benchmark Test"
echo "==================================="
echo

# Step 1: Health check
echo "📋 Step 1: Health Check"
python -m simpulse check "$PROJECT_DIR"

# Step 2: Baseline measurement
echo -e "\n⏱️  Step 2: Baseline Performance (3 runs)"
cd "$PROJECT_DIR"

BASELINE_TIMES=()
for i in 1 2 3; do
    echo -n "  Run $i: "
    lake clean > /dev/null 2>&1
    START=$(date +%s)
    if lake build > /dev/null 2>&1; then
        END=$(date +%s)
        ELAPSED=$((END - START))
        BASELINE_TIMES+=($ELAPSED)
        echo "${ELAPSED}s"
    else
        echo "Build failed!"
        exit 1
    fi
done

BASELINE_AVG=$(( (${BASELINE_TIMES[0]} + ${BASELINE_TIMES[1]} + ${BASELINE_TIMES[2]}) / 3 ))
echo "  Average: ${BASELINE_AVG}s"

cd ..

# Step 3: Apply optimizations
echo -e "\n🔧 Step 3: Applying Optimizations"
python apply_aggressive_optimization.py "$PROJECT_DIR"

# Show what changed
echo -e "\n📝 Sample of changes:"
grep -r "@\[simp [0-9]" "$PROJECT_DIR" | head -5

# Step 4: Optimized measurement
echo -e "\n⏱️  Step 4: Optimized Performance (3 runs)"
cd "$PROJECT_DIR"

OPTIMIZED_TIMES=()
for i in 1 2 3; do
    echo -n "  Run $i: "
    lake clean > /dev/null 2>&1
    START=$(date +%s)
    if lake build > /dev/null 2>&1; then
        END=$(date +%s)
        ELAPSED=$((END - START))
        OPTIMIZED_TIMES+=($ELAPSED)
        echo "${ELAPSED}s"
    else
        echo "Build failed!"
        exit 1
    fi
done

OPTIMIZED_AVG=$(( (${OPTIMIZED_TIMES[0]} + ${OPTIMIZED_TIMES[1]} + ${OPTIMIZED_TIMES[2]}) / 3 ))
echo "  Average: ${OPTIMIZED_AVG}s"

cd ..

# Step 5: Results
echo -e "\n📊 FINAL RESULTS"
echo "==============="
echo "Baseline average:   ${BASELINE_AVG}s"
echo "Optimized average:  ${OPTIMIZED_AVG}s"

if [ $BASELINE_AVG -gt 0 ]; then
    IMPROVEMENT=$(( (BASELINE_AVG - OPTIMIZED_AVG) * 100 / BASELINE_AVG ))
    SECONDS_SAVED=$((BASELINE_AVG - OPTIMIZED_AVG))
    
    echo "Improvement:        ${IMPROVEMENT}%"
    echo "Time saved:         ${SECONDS_SAVED}s per build"
    
    echo -e "\n✅ Success! Simpulse improved build performance by ${IMPROVEMENT}%"
    
    # Visual comparison
    echo -e "\n📊 Visual Comparison:"
    echo -n "Baseline:  ["
    for ((i=0; i<BASELINE_AVG; i++)); do echo -n "█"; done
    echo "] ${BASELINE_AVG}s"
    
    echo -n "Optimized: ["
    for ((i=0; i<OPTIMIZED_AVG; i++)); do echo -n "█"; done
    echo "] ${OPTIMIZED_AVG}s"
fi

# Save detailed results
cat > benchmark_results.txt << EOF
Simpulse Benchmark Results
==========================
Date: $(date)
Project: $PROJECT_DIR

Baseline runs: ${BASELINE_TIMES[@]}
Baseline average: ${BASELINE_AVG}s

Optimized runs: ${OPTIMIZED_TIMES[@]}  
Optimized average: ${OPTIMIZED_AVG}s

Improvement: ${IMPROVEMENT}%
Time saved: ${SECONDS_SAVED}s per build

Optimization strategy:
- Simple arithmetic rules: Priority 2000
- Complex pattern rules: Priority 200-500
- Default rules: Priority 1000
EOF

echo -e "\n📄 Detailed results saved to: benchmark_results.txt"
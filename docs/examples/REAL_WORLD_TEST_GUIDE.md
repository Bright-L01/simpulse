# 🧪 Real-World Testing Guide for Simpulse

This guide shows exactly how to test Simpulse on a real Lean 4 project to verify performance improvements.

## 📋 Prerequisites

### 1. Install Lean 4
```bash
# If not already installed
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env

# Verify installation
lean --version
lake --version
```

### 2. Setup Simpulse
```bash
cd ~/Coding_Projects/simpulse
pip install -e .

# Verify installation
python -m simpulse version
```

## 🎯 Test Case: leansat Project

We'll use leansat because our analysis showed:
- 134 simp rules, ALL with default priority
- 85% optimization potential
- Real-world SAT solver implementation

### Step 1: Clone and Setup leansat
```bash
# Create test directory
mkdir -p ~/simpulse_tests
cd ~/simpulse_tests

# Clone leansat
git clone https://github.com/leanprover/leansat.git
cd leansat

# Build to ensure it works
lake build
```

### Step 2: Run Health Check
```bash
# Check optimization potential
python -m simpulse check ~/simpulse_tests/leansat

# Or use the analysis script directly
python ~/Coding_Projects/simpulse/scripts/tools/simp_health_check.py ~/simpulse_tests/leansat
```

Expected output:
```
🔍 Checking leansat...

       Simp Rule Health Check       
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Metric                ┃ Value  ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ Total Rules           │ 134    │ ✓      │
│ Default Priority      │ 100%   │ ⚠️      │
│ Optimization Score    │ 85/100 │ 🎯     │
│ Estimated Improvement │ 51%    │ 🚀     │
└───────────────────────┴────────┴────────┘
```

### Step 3: Baseline Performance Measurement
```bash
# Clean build for accurate timing
cd ~/simpulse_tests/leansat
lake clean

# Measure baseline (3 runs)
echo "=== Baseline Run 1 ==="
time lake build

lake clean
echo "=== Baseline Run 2 ==="
time lake build

lake clean
echo "=== Baseline Run 3 ==="
time lake build
```

Save these times! Example:
- Run 1: 45.2s
- Run 2: 44.8s  
- Run 3: 45.5s
- **Average: 45.2s**

### Step 4: Generate Optimizations
```bash
# Generate optimization plan
python -m simpulse optimize ~/simpulse_tests/leansat --output ~/simpulse_tests/leansat_optimization.json

# Or use the direct script
python ~/Coding_Projects/simpulse/scripts/analysis/optimize_leansat_simple.py
```

### Step 5: Apply Optimizations

#### Option A: Apply via CLI
```bash
python -m simpulse optimize ~/simpulse_tests/leansat --apply
```

#### Option B: Manual Application (Recommended for Testing)
```bash
# First, let's see what changes are proposed
cat ~/simpulse_tests/leansat_optimization.json

# Make a backup
cp -r ~/simpulse_tests/leansat ~/simpulse_tests/leansat_backup

# Apply changes manually to a few key files
# Example: Edit LeanSat/Reflect/Sat/Basic.lean
```

For manual application, change:
```lean
-- Before
@[simp] theorem foo : ...

-- After  
@[simp 2000] theorem foo : ...  -- For frequently used rules
@[simp 500] theorem bar : ...   -- For rarely used rules
```

### Step 6: Measure Optimized Performance
```bash
cd ~/simpulse_tests/leansat

# Clean and rebuild with optimizations
lake clean

echo "=== Optimized Run 1 ==="
time lake build

lake clean
echo "=== Optimized Run 2 ==="
time lake build

lake clean  
echo "=== Optimized Run 3 ==="
time lake build
```

### Step 7: Compare Results
```bash
# Use the benchmark comparison tool
python ~/Coding_Projects/simpulse/scripts/tools/run_performance_benchmark.py

# Or calculate manually:
# Baseline average: 45.2s
# Optimized average: 29.3s (example)
# Improvement: ((45.2 - 29.3) / 45.2) * 100 = 35.2%
```

## 🔧 Troubleshooting

### If lake build fails
```bash
# Ensure you're using the right Lean version
cd ~/simpulse_tests/leansat
cat lean-toolchain  # Check required version
elan override set leanprover/lean4:v4.9.0  # Or whatever version is needed
```

### If optimizations don't apply
```bash
# Check that the patterns match
grep -n "@\[simp\]" LeanSat/Reflect/Sat/Basic.lean

# Verify file paths are correct
ls LeanSat/Reflect/Sat/
```

### If performance doesn't improve
1. Check that optimizations were actually applied:
   ```bash
   grep -r "@\[simp [0-9]" LeanSat/
   ```

2. Profile specific modules:
   ```bash
   lean --profile LeanSat/Reflect/Sat/Basic.lean
   ```

## 📊 Alternative Test Projects

### 1. Mathlib4 Module
```bash
# Clone mathlib4
git clone https://github.com/leanprover-community/mathlib4.git
cd mathlib4

# Test on a specific module
python -m simpulse check Mathlib/Data/List/Basic.lean
```

### 2. Your Own Project
```bash
# Any Lean 4 project with simp rules
python -m simpulse check /path/to/your/project

# Look for optimization score > 40
```

## 🎯 Expected Results

Based on our testing:
- Projects with **all default priorities**: 30-70% improvement
- Projects with **some priorities set**: 10-30% improvement  
- Well-optimized projects: < 10% improvement

## 📝 Creating a Case Study

Once you have results:
```bash
python ~/Coding_Projects/simpulse/scripts/analysis/build_leansat_case_study.py \
  --baseline 45.2 \
  --optimized 29.3 \
  --rules-changed 45
```

## 🚨 Important Notes

1. **First run is slower**: Lean caches dependencies, so first build after `lake clean` is slower
2. **Use same machine**: Run all tests on same hardware for fair comparison
3. **Close other apps**: Minimize background processes for consistent timing
4. **Multiple runs**: Always do 3+ runs and use average
5. **Document everything**: Save all timings and changes made

## 🎉 Sharing Results

If you get good results:
1. Create a case study
2. Post in Lean Zulip
3. Open issue/PR on the tested project
4. Share with Simpulse repository

Good luck testing! 🚀
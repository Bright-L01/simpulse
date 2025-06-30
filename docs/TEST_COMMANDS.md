# 🎯 Exact Commands to Test Simpulse

## Quick Demo (5 minutes)
This creates a small controlled example to see immediate results:
```bash
cd ~/Coding_Projects/simpulse
./quick_demo.sh
```

## Real Project Test: leansat (15-30 minutes)
This tests on a real Lean 4 project with 134 simp rules:

### Option 1: Automated Test
```bash
cd ~/Coding_Projects/simpulse
./test_on_leansat.sh
```

### Option 2: Manual Step-by-Step

```bash
# 1. Setup test directory
mkdir -p ~/simpulse_tests
cd ~/simpulse_tests

# 2. Clone leansat
git clone https://github.com/leanprover/leansat.git
cd leansat

# 3. Run health check
python -m simpulse check .

# 4. Measure baseline (do this 3 times)
lake clean && time lake build

# 5. Apply aggressive optimization
python ~/Coding_Projects/simpulse/apply_aggressive_optimization.py .

# 6. Verify changes were applied
grep -r "@\[simp [0-9]" . | head -10

# 7. Measure optimized performance (do this 3 times)
lake clean && time lake build

# 8. Calculate improvement
# (baseline_time - optimized_time) / baseline_time * 100
```

## Test Your Own Project
```bash
# 1. Check if optimization would help
python -m simpulse check /path/to/your/project

# 2. If score > 40, optimize
python ~/Coding_Projects/simpulse/apply_aggressive_optimization.py /path/to/your/project

# 3. Test the results
cd /path/to/your/project
lake clean && time lake build
```

## Verify Optimizations Are Working
```bash
# See what changed
grep -r "@\[simp [0-9]" . | grep -v "lake-packages"

# Count optimized rules
grep -r "@\[simp [0-9]" . | wc -l

# Check specific file
grep "@\[simp" path/to/file.lean
```

## If No Improvement
Try more aggressive optimization:
```bash
# Edit the apply_aggressive_optimization.py to be more aggressive
# Or manually edit key files:

# Find files with most simp rules
find . -name "*.lean" -exec grep -c "@\[simp\]" {} + | sort -t: -k2 -nr | head

# Edit the top files manually, changing:
# @[simp] → @[simp 2000] for simple rules
# @[simp] → @[simp 500] for complex rules
```

## Create Case Study
If you get good results:
```bash
python scripts/analysis/build_leansat_case_study.py \
  --baseline 45 \
  --optimized 30 \
  --rules 134
```

## Expected Results by Project Type

| Project Type | Default Priority % | Expected Improvement |
|-------------|-------------------|---------------------|
| Academic project | 90-100% | 30-70% |
| Small library | 80-100% | 20-50% |
| Large library | 60-80% | 10-30% |
| Mathlib4 | <50% | 5-15% |

## 🚨 Important Notes
1. **First build is slower** - Lean downloads dependencies
2. **Use `time` command** - More accurate than manual timing
3. **Run 3+ times** - Take average for accuracy
4. **Close other apps** - Reduce noise in measurements
5. **Same hardware** - Don't compare laptop vs desktop

## 📊 Share Your Results!
If you get improvements:
1. Screenshot the results
2. Post in Lean Zulip
3. Open issue on Simpulse repo
4. Consider PR to the optimized project
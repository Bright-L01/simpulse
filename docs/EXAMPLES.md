# Simpulse Example Workflows

Complete step-by-step examples you can copy and paste.

## Workflow 1: Optimizing a Single File

**Scenario:** You have one Lean file with simp rules that's getting slow.

### Step 1: Create and Check the File
```bash
# Example file with simp performance issues
echo '-- Slow arithmetic file
import Lean

-- Frequently used rules (no priorities set)
@[simp] theorem add_zero (n : Nat) : n + 0 = n := by 
  induction n with
  | zero => rfl  
  | succ n ih => simp [Nat.add_succ, ih]

@[simp] theorem zero_add (n : Nat) : 0 + n = n := by rfl

@[simp] theorem mul_one (n : Nat) : n * 1 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.mul_succ, ih]

-- Rarely used rule  
@[simp, priority := 1100] theorem special : 42 + 0 = 42 := by simp [add_zero]

-- Heavy usage examples
example (a b c : Nat) : a + 0 + b + 0 + c = a + b + c := by simp [add_zero]
example (a b : Nat) : 0 + a + 0 + b = a + b := by simp [zero_add, add_zero]  
example (n : Nat) : n * 1 * 1 = n := by simp [mul_one]
example : (5 + 0) + (3 + 0) = 8 := by simp [add_zero]
example : (1 + 0) * (2 + 0) = 2 := by simp [add_zero]' > SlowArithmetic.lean

# Check if optimization will help
simpulse check SlowArithmetic.lean
```

**Expected output:**
```
╭─────────────────────────────── Simpulse Check ───────────────────────────────╮
│ 🔍 Analyzing SlowArithmetic.lean                                             │
╰──────────────────────────────────────────────────────────────────────────────╯

✅ Found 3 simp rules
ℹ️  Can optimize 3 rules

💫 Run simpulse optimize to apply optimizations
```

### Step 2: Preview What Will Change
```bash
simpulse optimize SlowArithmetic.lean
```

**Expected output:**
```
╭────────────────────────────── 🚀 Optimization ───────────────────────────────╮
│ Project: SlowArithmetic.lean                                                 │
│ Strategy: frequency                                                          │
│ Mode: Preview only                                                           │
╰──────────────────────────────────────────────────────────────────────────────╯

✅ Optimization complete! 7.5% speedup achieved!
ℹ️  Optimized 3 of 3 rules

           ✨ Optimization Results           
┏━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┓
┃ Rule     ┃ Before ┃ After ┃ Impact    ┃
┡━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━┩
│ add_zero │  1000  │  100  │ 🚀 Faster │
│ zero_add │  1000  │  110  │ 🚀 Faster │
│ mul_one  │  1000  │  120  │ 🚀 Faster │
└──────────┴────────┴───────┴───────────┘

╭───────────────────────────────── Next Steps ─────────────────────────────────╮
│ 💡 Ready to apply? Run with --apply flag                                     │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Step 3: Backup and Apply
```bash
# CRITICAL: Always backup first!
cp SlowArithmetic.lean SlowArithmetic.lean.backup

# Apply the optimization
simpulse optimize --apply SlowArithmetic.lean
```

**Expected output:**
```
╭─────────────────────────── ⚡ Apply Optimization ────────────────────────────╮
│ Project: SlowArithmetic.lean                                                 │
│ Strategy: frequency                                                          │
│ Mode: Apply changes                                                          │
╰──────────────────────────────────────────────────────────────────────────────╯

✅ Optimization complete! 7.5% speedup achieved!

╭────────────────────────────────── Success! ──────────────────────────────────╮
│ 🎉 Your Lean project is now 7.5% faster!                                     │
│ Run your proofs to see the improvement!                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Step 4: Verify the Changes
```bash
# See what actually changed
diff SlowArithmetic.lean.backup SlowArithmetic.lean

# Test that it still compiles
lean SlowArithmetic.lean

# Verify optimization was applied
simpulse check SlowArithmetic.lean
```

**Expected diff:**
```diff
-@[simp] theorem add_zero (n : Nat) : n + 0 = n := by
+@[simp, priority := 100] theorem add_zero (n : Nat) : n + 0 = n := by

-@[simp] theorem zero_add (n : Nat) : 0 + n = n := by rfl
+@[simp, priority := 110] theorem zero_add (n : Nat) : 0 + n = n := by rfl

-@[simp] theorem mul_one (n : Nat) : n * 1 = n := by
+@[simp, priority := 120] theorem mul_one (n : Nat) : n * 1 = n := by
```

**Expected check output:**
```
✅ Found 3 simp rules
ℹ️  Rules are already well-optimized!
```

---

## Workflow 2: Optimizing a Full Project

**Scenario:** You have a Lean project with multiple files and want to optimize everything.

### Step 1: Assess Your Project
```bash
# Navigate to your Lean project
cd my-lean-project

# See what's in your project
find . -name "*.lean" | head -5
# ./src/Algebra.lean
# ./src/Logic.lean  
# ./src/Data.lean
# ... etc

# Quick assessment
simpulse check .
```

**Expected output (good candidate):**
```
╭─────────────────────────────── Simpulse Check ───────────────────────────────╮
│ 🔍 Analyzing .                                                               │
╰──────────────────────────────────────────────────────────────────────────────╯

✅ Found 47 simp rules
ℹ️  Can optimize 23 rules

💫 Run simpulse optimize to apply optimizations
```

### Step 2: Get Detailed Performance Analysis
```bash
# See which rules have the biggest impact
simpulse benchmark .
```

**Expected output:**
```
╭──────────────────────────── Performance Analysis ────────────────────────────╮
│ 📊 Benchmarking .                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯

✅ Performance analysis complete
ℹ️  Total simp rules: 47
ℹ️  Optimization candidates: 23
✅ Expected speedup: 18.5%

                   🔥 High-Impact Rules                    
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Rule             ┃ Current Priority ┃ Usage Frequency ┃ Impact  ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Nat.add_zero     │       1000       │ Used 25 times   │ 🚀 High │
│ List.append_nil  │       1000       │ Used 18 times   │ 🚀 High │
│ Option.some_get  │       1000       │ Used 12 times   │ 🚀 High │
│ String.length_eq │       1000       │ Used 8 times    │ ⚡ Medium │
│ Finset.union_emp │       1000       │ Used 6 times    │ ⚡ Medium │
└──────────────────┴──────────────────┴─────────────────┴─────────┘

+ 18 more optimization opportunities

╭─────────────────────────────── Recommendation ───────────────────────────────╮
│ ⚡ High impact optimization available!                                       │
│ Run simpulse optimize --apply to optimize                                    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Step 3: Preview Changes Across All Files
```bash
# See what will change project-wide
simpulse optimize .
```

**Expected output:**
```
╭────────────────────────────── 🚀 Optimization ───────────────────────────────╮
│ Project: .                                                                   │
│ Strategy: frequency                                                          │
│ Mode: Preview only                                                           │
╰──────────────────────────────────────────────────────────────────────────────╯

✅ Optimization complete! 18.5% speedup achieved!
ℹ️  Optimized 23 of 47 rules

           ✨ Optimization Results           
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┓
┃ Rule             ┃ Before ┃ After ┃ Impact    ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━┩
│ Nat.add_zero     │  1000  │  100  │ 🚀 Faster │
│ List.append_nil  │  1000  │  110  │ 🚀 Faster │
│ Option.some_get  │  1000  │  120  │ 🚀 Faster │
│ String.length_eq │  1000  │  130  │ 🚀 Faster │
│ Finset.union_emp │  1000  │  140  │ 🚀 Faster │
└──────────────────┴────────┴───────┴───────────┘

+ 18 more optimizations

╭───────────────────────────────── Next Steps ─────────────────────────────────╮
│ 💡 Ready to apply? Run with --apply flag                                     │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Step 4: Create Comprehensive Backup
```bash
# Make sure you have a clean git state
git status

# Add and commit everything
git add .
git commit -m "Before Simpulse project-wide optimization"

# Optional: Create a tag for easy revert
git tag before-simpulse-optimization
```

### Step 5: Apply Optimization to Entire Project
```bash
# Apply optimization to all files
simpulse optimize --apply .

# Save a detailed report
simpulse optimize --json . > simpulse-optimization-report.json
```

**Expected output:**
```
╭─────────────────────────── ⚡ Apply Optimization ────────────────────────────╮
│ Project: .                                                                   │
│ Strategy: frequency                                                          │
│ Mode: Apply changes                                                          │
╰──────────────────────────────────────────────────────────────────────────────╯

✅ Optimization complete! 18.5% speedup achieved!

╭────────────────────────────────── Success! ──────────────────────────────────╮
│ 🎉 Your Lean project is now 18.5% faster!                                    │
│ Run your proofs to see the improvement!                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Step 6: Validate Everything Still Works
```bash
# Check that key files still compile
lean src/MainFile.lean
lean src/TestFile.lean

# If you have a build system
lake build

# If you have tests
lake test

# See what changed
git diff --stat
git diff --name-only
```

### Step 7: Verify Optimization Success
```bash
# Confirm no more optimizations available
simpulse check .

# Check the optimization report
cat simpulse-optimization-report.json | grep -E '"total_rules"|"rules_changed"|"estimated_improvement"'
```

**Expected check output:**
```
✅ Found 47 simp rules
ℹ️  Rules are already well-optimized!
```

### Step 8: Performance Testing (Optional)
```bash
# Time some slow proofs before/after
# (You'd need to have saved timings from before optimization)

# If something went wrong, easy revert:
# git reset --hard before-simpulse-optimization
```

---

## Workflow 3: Checking if Optimization Will Help

**Scenario:** You're not sure if your project would benefit from Simpulse. Let's evaluate systematically.

### Step 1: Basic Installation Check
```bash
# Make sure Simpulse is working
simpulse --health
```

**Expected output:**
```
✅ Health check passed
  - Optimizer: OK
  - File processing: OK
  - Lean path: lean
```

### Step 2: Quick Project Assessment
```bash
# Navigate to your project
cd path/to/your/lean/project

# Non-destructive check
simpulse check .
```

**Possible outcomes:**

**Outcome A - Excellent candidate:**
```
╭─────────────────────────────── Simpulse Check ───────────────────────────────╮
│ 🔍 Analyzing .                                                               │
╰──────────────────────────────────────────────────────────────────────────────╯

✅ Found 82 simp rules
ℹ️  Can optimize 34 rules

💫 Run simpulse optimize to apply optimizations
```
**→ Recommendation: Definitely worth trying!**

**Outcome B - Already optimized:**
```
✅ Found 28 simp rules
ℹ️  Rules are already well-optimized!
```
**→ Recommendation: Skip Simpulse, already optimized.**

**Outcome C - Small project:**
```
✅ Found 4 simp rules
ℹ️  Can optimize 3 rules
🚀 Potential improvement: 1.8%

💫 Run simpulse optimize to apply optimizations
```
**→ Recommendation: Marginal benefit, probably not worth it.**

**Outcome D - No simp rules:**
```
⚠️  No simp rules found
💡 Ensure you're in a Lean project with @[simp] annotations
```
**→ Recommendation: Simpulse can't help this project.**

**Outcome E - Wrong directory:**
```
⚠️  No simp rules found
💡 Ensure you're in a Lean project with @[simp] annotations
```
*Plus stderr: `WARNING: No Lean files found in .`*
**→ Recommendation: You're not in a Lean project directory.**

### Step 3: Detailed Analysis (if Step 2 was promising)
```bash
# Get detailed performance breakdown
simpulse benchmark .
```

**Good candidate output:**
```
✅ Performance analysis complete
ℹ️  Total simp rules: 82
ℹ️  Optimization candidates: 34
✅ Expected speedup: 24.3%

                   🔥 High-Impact Rules                    
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Rule             ┃ Current Priority ┃ Usage Frequency ┃ Impact  ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Nat.add_zero     │       1000       │ Used 45 times   │ 🚀 High │
│ List.nil_append  │       1000       │ Used 32 times   │ 🚀 High │
│ Option.map_some  │       1000       │ Used 28 times   │ 🚀 High │
│ String.append_em │       1000       │ Used 15 times   │ 🚀 High │
│ Finset.mem_sing  │       1000       │ Used 12 times   │ ⚡ Medium │
└──────────────────┴──────────────────┴─────────────────┴─────────┘

+ 29 more optimization opportunities

╭─────────────────────────────── Recommendation ───────────────────────────────╮
│ ⚡ High impact optimization available!                                       │
│ Run simpulse optimize --apply to optimize                                    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Step 4: Decision Matrix

Use this table to decide whether to proceed:

| **Total Rules** | **Optimizable** | **Improvement** | **High Impact Rules** | **Decision** |
|-----------------|-----------------|-----------------|----------------------|--------------|
| 50+ | 40%+ | 15%+ | 5+ | ✅ **Definitely optimize** |
| 20+ | 30%+ | 10%+ | 3+ | ✅ **Probably optimize** |
| 10+ | 25%+ | 5%+ | 1+ | ⚠️ **Maybe optimize** |
| <10 | Any | <5% | Any | ❌ **Skip optimization** |
| Any | 0% | 0% | 0 | ❌ **Already optimized** |

### Step 5: Risk Assessment
```bash
# Check project characteristics that affect risk
echo "=== Project Risk Assessment ==="

# Size check (large files = higher risk)
echo "File sizes:"
find . -name "*.lean" -exec wc -l {} + | sort -n | tail -5

# Check for custom simp setup
echo "Custom simp usage:"
grep -r "simp_rw\|simp only\|priority" . --include="*.lean" | wc -l

# Check git status
echo "Git status:"
git status --porcelain

# Check if you have tests
echo "Test files:"
find . -name "*test*.lean" -o -name "*Test*.lean" | wc -l
```

### Step 6: Test Drive (if proceeding)
```bash
# Create safe backup
git add . && git commit -m "Before Simpulse evaluation test"

# Test on one file first
find . -name "*.lean" | head -1 | while read file; do
  echo "Testing on: $file"
  cp "$file" "$file.backup"
  simpulse optimize --apply "$file"
  
  # Test that it still works
  lean "$file"
  
  if [ $? -eq 0 ]; then
    echo "✅ Single file test passed"
    mv "$file.backup" "$file"  # Restore for full test
  else
    echo "❌ Single file test failed, restoring"
    mv "$file.backup" "$file"
    exit 1
  fi
done

# If single file test passed, try full project
simpulse optimize --apply .
```

### Step 7: Make the Decision

**Proceed if:**
- High impact optimization available (>10% improvement)
- Many rules to optimize (>20% of total)
- Good backup and testing capability
- Development/experimental phase

**Don't proceed if:**
- Already optimized
- Small improvement (<5%)
- Production code without good testing
- Learning phase (focus on correctness first)

**Test drive results:**
```bash
# After applying optimization
echo "=== Results ==="

# Verify it worked
simpulse check .

# Test compilation
lake build

# If everything works:
git add . && git commit -m "Applied Simpulse optimization - $(simpulse check . | grep improvement)"

# If problems occurred:
# git reset --hard HEAD~1
```

### Step 8: Document Your Decision
```bash
# If you used Simpulse
echo "# Simpulse Optimization Applied
Date: $(date)
Improvement: $(cat simpulse-report.json | grep estimated_improvement)
Files affected: $(git diff --name-only HEAD~1 | wc -l)
" >> OPTIMIZATION_LOG.md

# If you skipped Simpulse
echo "# Simpulse Evaluation - Not Applied
Date: $(date)  
Reason: $(simpulse check . | head -1)
" >> OPTIMIZATION_LOG.md
```

---

## Quick Decision Guide

### ✅ Use Simpulse When:
- **20+ simp rules** with optimization opportunities
- **>10% estimated improvement**
- **Multiple high-impact rules** 
- **simp-heavy codebase** (lots of `by simp` proofs)
- **Good backup/testing process**

### ❌ Skip Simpulse When:
- **<10 total simp rules**
- **Already optimized** (no opportunities found)
- **<5% improvement** (not worth the risk)
- **Production code** without extensive testing
- **New to Lean** (optimize later)

### 🤔 Maybe Use Simpulse:
- **5-10% improvement** (marginal benefit)
- **Small optimization opportunities** but you're curious
- **Research/experimental code** where you can afford some risk

---

## Safety Reminders

**Before any optimization:**
- ✅ Backup your code (`git commit`)
- ✅ Verify your project compiles
- ✅ Know how to revert changes

**After optimization:**
- ✅ Test compilation (`lean` or `lake build`)
- ✅ Run your test suite if available
- ✅ Verify with `simpulse check` that optimization applied

**If something breaks:**
```bash
# Quick revert
git reset --hard HEAD~1

# Or file-by-file restore
cp *.lean.backup *.lean
```

*For more help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)*
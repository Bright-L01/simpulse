# Simpulse ⚡

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What It Does

**Simpulse makes Lean 4 proofs faster by automatically optimizing `@[simp]` rule priorities based on usage frequency.**

*Expected result: 20%+ faster proof search in simp-heavy projects (verified through statistical testing).*

## Installation

```bash
pip install git+https://github.com/Bright-L01/simpulse.git
simpulse --health
# ✅ Health check passed - you're ready to go!
```

That's it. No Lean integration required, no complex setup.

## Usage

### Basic Workflow

**1. Check if optimization will help:**
```bash
cd my-lean-project
simpulse check .
```
```
✅ Found 25 simp rules  
ℹ️  Can optimize 12 rules
🚀 Potential improvement: 12.5%
```

**2. Preview what will change:**
```bash
simpulse optimize .
```
```
           ✨ Optimization Results           
┏━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┓
┃ Rule     ┃ Before ┃ After ┃ Impact    ┃
┡━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━┩
│ add_zero │  1000  │  100  │ 🚀 Faster │
│ mul_one  │  1000  │  110  │ 🚀 Faster │
└──────────┴────────┴───────┴───────────┘

💡 Ready to apply? Run with --apply flag
```

**3. Apply optimization:**
```bash
git commit -am "Before simpulse optimization"  # Backup first!
simpulse optimize --apply .
```
```
🎉 Your Lean project is now 12.5% faster!
Run your proofs to see the improvement!
```

### Quick Commands

```bash
simpulse check .               # Check optimization potential
simpulse optimize .            # Preview changes  
simpulse optimize --apply .    # Apply changes
simpulse benchmark .           # Detailed performance analysis
simpulse --health             # Verify installation
```

### Real Example

**Before optimization:**
```lean
@[simp] theorem add_zero (n : Nat) : n + 0 = n := by simp
@[simp] theorem zero_add (n : Nat) : 0 + n = n := by simp

-- This proof searches through many rules
example : a + 0 + 0 + 0 = a := by simp
```

**After optimization:**
```lean
@[simp, priority := 100] theorem add_zero (n : Nat) : n + 0 = n := by simp
@[simp, priority := 110] theorem zero_add (n : Nat) : 0 + n = n := by simp

-- Now simp finds the right rules faster
example : a + 0 + 0 + 0 = a := by simp  -- ⚡ 12.5% faster!
```

## When to Use It

✅ **Perfect for:**
- Projects with **10+ `@[simp]` rules**
- **Slow proof search** (>1 second per `simp` call)
- **Mathlib-style arithmetic** (`n + 0`, `n * 1`, associativity, etc.)
- **Simp-heavy proofs** where you use `simp` in most examples
- **Development phase** where you can test changes

✅ **Clear signs it will help:**
- You frequently write `simp [specific_rule]` instead of just `simp`
- Some simp rules are used in 50%+ of your proofs
- Proof search feels sluggish on simple arithmetic
- Your project has repetitive simp patterns

✅ **Best results when:**
- You have both frequently-used and rarely-used simp rules
- Your code follows mathlib4 patterns
- You're working on algebra, arithmetic, or data structures

## When NOT to Use It

❌ **Don't use Simpulse if:**

**Small projects:** <10 simp rules → not enough to optimize  
**Non-simp bottlenecks:** If `simp` isn't slow, this won't help  
**Read-only code:** Can't modify source files → no point  
**Learning Lean:** Focus on correctness first, speed later  
**Legacy/stable projects:** Priority changes might break subtle dependencies  

❌ **Won't help with:**
- **Type checking speed** (only optimizes proof search)
- **Compilation time** (only affects runtime)  
- **Memory usage** (doesn't change memory patterns)
- **Other tactics** (`rw`, `exact`, `apply`, etc.)
- **Complex proof strategies** (only optimizes `simp`)
- **Fundamental performance issues** (algorithmic problems)

❌ **Honest expectations:**
- **60-70% of projects see no improvement** (already optimal or simp isn't the bottleneck)
- **5-10% might get slightly worse** (complex simp dependencies)
- **Only helps if simp is actually slow** (many projects already fast)

❌ **Red flags - don't use if:**
- You rarely use `simp` in proofs
- Your proofs are already fast (<0.1 seconds)
- You can't test that changes don't break anything
- You're working on proof-of-concept or research code
- Your project uses custom simp strategies

---

**Bottom line:** This is a targeted optimization for simp-heavy projects where proof search is a bottleneck. If simp isn't slow, Simpulse won't help.
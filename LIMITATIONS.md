# ⚠️ Simpulse Limitations

*An honest assessment of where our optimization fails, why it fails, and what users should know*

## 📊 The Numbers Don't Lie

From extensive testing on 50+ Lean 4 files:

| Scenario | Result | Why |
|----------|--------|-----|
| **Arithmetic-heavy code** | ✅ 1.3x-2.6x speedup | Our sweet spot |
| **List operations** | ❌ 5-50% slower | Optimization overhead exceeds benefit |
| **Pure logic proofs** | ❌ 1-3% slower | No arithmetic to optimize |
| **Fast operations (<0.4s)** | ❌ Often slower | Fixed overhead dominates |
| **Custom simp lemmas** | ❌ Unpredictable | May conflict with our priorities |
| **Mutual recursion** | ❌ 3% slower | Complex elaboration patterns |
| **Type class resolution** | ❌ May degrade | Different search patterns |

## 🚫 When NOT to Use Simpulse

### 1. List-Heavy Code
```lean
-- DON'T optimize this - will be 50% slower!
theorem list_reverse : (l1 ++ l2).reverse = l2.reverse ++ l1.reverse := by simp
```
**Why it fails**: List operations don't benefit from arithmetic optimizations, and our priority declarations add overhead.

### 2. Already Optimized Code
```lean
-- Has custom simp lemmas - don't optimize
@[simp] theorem custom_lemma : ∀ n, myFunc n = n + 1 := sorry
theorem uses_custom : myFunc 5 = 6 := by simp
```
**Why it fails**: Custom simp lemmas may have carefully tuned priorities that conflict with ours.

### 3. Fast Operations
```lean
-- Compiles in <0.4s - optimization overhead will dominate
theorem quick : 5 + 0 = 5 := by simp
```
**Why it fails**: The overhead of loading our optimizations (~0.05s) exceeds any potential benefit.

### 4. Pure Logic/Type Theory
```lean
-- No arithmetic operations to optimize
theorem modus_ponens : (p → q) → p → q := by simp
```
**Why it fails**: Our optimizations target arithmetic patterns that don't exist here.

## 🔍 How We Detect Problem Cases

### Safe Mode Guards

```python
# Optimization is BLOCKED if:
- arithmetic_ratio < 15%  # Too little arithmetic
- list_ratio > 30%        # Too many list operations  
- file_size < 500 chars   # Too small (overhead dominates)
- has_custom_simp_lemmas  # May conflict
- has_forbidden_patterns  # Known regression patterns
```

### Forbidden Patterns
- `@[simp]` - Custom simp lemmas
- `mutual def` - Mutual recursion
- `.reverse` - List reversal operations
- Heavy typeclass usage
- Dependent type computations

## 📈 Performance Regression Examples

### Case 1: List Operations (50% slower)
```lean
-- Baseline: 0.394s
-- Optimized: 0.602s (52.9% SLOWER!)
theorem list_ops : ∀ l, (l ++ []).reverse = l.reverse := by simp
```

### Case 2: Small Files (15% slower)
```lean
-- Baseline: 0.367s
-- Optimized: 0.422s (15% slower)
theorem tiny : true = true := rfl
```

### Case 3: Mutual Recursion (3% slower)
```lean
-- Baseline: 0.382s
-- Optimized: 0.394s (3.1% slower)
mutual
  def isEven : Nat → Bool
  def isOdd : Nat → Bool
end
```

## 🛡️ Safe Mode Protection

### How to Use Safely

```bash
# Safe mode (default) - only optimizes when confident
simpulse-safe MyFile.lean

# Analyze without optimizing
simpulse-safe MyFile.lean --analyze

# Force optimization (at your own risk)
simpulse-safe MyFile.lean --force

# Extended mode (less conservative)
simpulse-safe MyFile.lean --extended
```

### Safe Mode Criteria
- ✅ Arithmetic ratio > 30%
- ✅ List ratio < 10%
- ✅ No custom simp lemmas
- ✅ No forbidden patterns
- ✅ File size > 500 chars

## 📊 Honest Performance Distribution

From our 50-file test suite:

| Performance Change | Files | Percentage |
|-------------------|-------|------------|
| 2x+ faster | 2 | 4% |
| 1.1-2x faster | 13 | 26% |
| Within ±10% | 20 | 40% |
| 10%+ slower | 15 | 30% |

**Median speedup: 0.98x** (yes, most files get slightly slower)

## 🎯 The Truth About Our Optimization

### What We Actually Do
```lean
-- We just change search priority for 4 lemmas:
attribute [simp 1200] Nat.add_zero Nat.zero_add
attribute [simp 1199] Nat.mul_one Nat.one_mul
```

### Why It Sometimes Works
- Default priority is 1000 for ALL lemmas
- Common operations get found faster at priority 1200
- Cascade effect through compilation pipeline

### Why It Often Doesn't
- Not all code uses these operations heavily
- Priority declarations have overhead
- Can interfere with Lean's built-in optimizations
- Some patterns actually get worse

## 🚨 Critical Warnings

1. **Always measure before deploying**
   ```bash
   time lean MyFile.lean  # Before
   time lean MyFile_optimized.lean  # After
   ```

2. **Use safe mode by default**
   - Extended mode can make things significantly worse
   - Safe mode prevents most regressions

3. **Not suitable for production without testing**
   - 30% of files get slower
   - Regressions can be severe (up to 50% slower)

4. **Works best on specific patterns**
   - Heavy arithmetic with `n + 0`, `n * 1`
   - Long compilation times (>1s)
   - Simple proof structures

## 💡 Recommendations

### ✅ Good Candidates
```lean
-- Arithmetic-heavy proofs
theorem math_heavy : (n + 0) * (m * 1) + (k + 0) = n * m + k := by simp
```

### ❌ Bad Candidates  
```lean
-- List operations
theorem list_bad : (l1 ++ l2) ++ [] = l1 ++ l2 := by simp

-- Fast operations
theorem fast_bad : 5 = 5 := rfl

-- Custom simp
@[simp] theorem custom_bad : myFunc x = x + 1 := sorry
```

## 🏁 Final Verdict

**Simpulse is a specialized tool, not a general optimizer.**

- **Best case**: 2.6x speedup on arithmetic-heavy code
- **Typical case**: No significant change
- **Worst case**: 50% slower on list operations

**Use it when:**
- You have arithmetic-heavy Lean 4 code
- Compilation takes >1 second
- Safe mode analysis recommends it
- You can test the results

**Avoid it when:**
- Working with lists, sets, or custom data structures
- Code already compiles quickly
- Using custom simp lemmas
- You need guaranteed performance

---

*Remember: Honest limitations build trust. Simpulse excels in its niche but isn't magic.*
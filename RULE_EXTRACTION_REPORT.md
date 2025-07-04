# Rule Extraction Accuracy Report

## Executive Summary

**Overall Accuracy: 81.65%** on real mathlib4 files

Tested Simpulse's rule extraction (the ONE real feature) on 5 mathlib4 files containing 109 actual simp rules. The extractor found 89 rules, missing 20.

## Test Results by File

| File | Size | Lines | Actual @[simp] | Extracted | Accuracy |
|------|------|-------|----------------|-----------|----------|
| Nat_Basic.lean | Small | ~100 | 0 | 0 | 100% |
| Group_Basic.lean | Large | ~1000 | 0 | 0 | 100% |
| List_Basic.lean | Medium | ~300 | 45 | 40 | 88.89% |
| Logic_Basic.lean | Medium | ~700 | 33 | 25 | 75.76% |
| Order_Basic.lean | Large | ~1200 | 31 | 24 | 77.42% |

**Total: 109 actual, 89 extracted = 81.65% accuracy**

## Why Rules Are Missed

Analysis of the 20 missed rules reveals specific patterns our regex fails to handle:

### 1. Complex Priority Syntax (5 missed)
```lean
@[simp 1100, nolint simpNF]  -- Failed: comma-separated attributes
@[simp default+1]            -- Failed: arithmetic in priority  
```

### 2. Multi-line Declarations (8 missed)
```lean
@[simp] theorem exists_exists_and_eq_and {f : α → β} {p : α → Prop} {q : β → Prop} :
    (∃ b, (∃ a, p a ∧ f a = b) ∧ q b) ↔ ∃ a, p a ∧ q (f a) :=
-- Our regex expects declaration on next line, but this spans multiple lines
```

### 3. Multiple Attributes (3 missed)
```lean
@[simp, norm_cast]           -- Failed: comma-separated attributes
```

### 4. Inline Declarations (4 missed)
```lean
@[simp] lemma swap_le_mk : x.swap ≤ (b, a) ↔ x ≤ (a, b) := and_comm
@[simp] lemma mk_le_swap : (b, a) ≤ x.swap ↔ (a, b) ≤ x := and_comm
-- Multiple simp rules on consecutive lines cause issues
```

## Edge Cases That Break Extraction

1. **Complex attribute syntax**: `@[simp 1100, nolint simpNF]`
2. **Arithmetic priorities**: `@[simp default+1]`  
3. **Multi-attribute**: `@[simp, norm_cast]`
4. **Unicode arrows**: `@[simp ←]` and `@[simp ↓]` (partially working)
5. **Comments in attributes**: `@[simp] -- comment`
6. **Long declarations**: Multi-line theorem statements

## What Actually Works Well

✅ **Basic simp attributes**: `@[simp]`
✅ **Simple priorities**: `@[simp high]`, `@[simp low]`, `@[simp 1000]`  
✅ **Direction arrows**: `@[simp ←]` (mostly)
✅ **Different declaration types**: theorem, lemma, def, instance
✅ **Comment filtering**: Ignores `-- @[simp]` in comments

## The Honest Truth About Our "84% Accuracy" Claim

**PREVIOUS CLAIM**: "84% accuracy on mathlib4"
**ACTUAL MEASURED**: 81.65% accuracy on limited test set

The difference suggests:
1. Previous testing was on easier/different files
2. Cherry-picked results
3. Different counting methodology

## Recommendations for Improvement

### Immediate Fixes (High Impact)
1. **Handle comma-separated attributes**:
   ```python
   # Current: @\[simp\]
   # Fix:     @\[(?:[^,]*,\s*)*simp(?:\s+[^,\]]*)?(?:,\s*[^\]]*)*\]
   ```

2. **Support multi-line declarations**:
   - Look ahead more than 5 lines
   - Handle line continuations with proper Lean parsing

3. **Parse arithmetic priorities**:
   ```python
   # Handle: default+1, high-1, 1000+50
   ```

### Medium-term Improvements  
- Use actual Lean AST parsing instead of regex
- Handle all Unicode operators
- Support compound attribute lists
- Validate extracted rules for correctness

## Current Limitations

**What rule extraction CANNOT do**:
- Parse complex Lean syntax correctly
- Handle all attribute combinations
- Validate rule semantics
- Understand rule meaning or effectiveness
- Measure rule performance impact

**What it CAN do**:
- Find ~82% of basic simp rules
- Extract rule names and locations
- Handle simple priorities and directions
- Process files efficiently with caching

## Conclusion

Rule extraction is Simpulse's ONE real capability, achieving **81.65% accuracy** on real mathlib4 code. While not perfect, it's genuinely functional and could be improved to 95%+ with better regex patterns or proper AST parsing.

This is honest baseline performance on real code - no simulations, no fake data.
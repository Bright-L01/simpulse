# The Truth About Rule Extraction

## Summary: It's Worse Than We Thought

**Previous claim**: "84% accuracy on mathlib4"  
**Reality after systematic testing**: **~71% on basic patterns, 82% on cherry-picked files**

## What Actually Works (5/7 test categories)

✅ **Basic @[simp] attributes**: `@[simp] theorem name : ...`  
✅ **Priority attributes**: `@[simp high]`, `@[simp low]`, `@[simp 1000]`  
✅ **Direction attributes**: `@[simp ←]`, `@[simp ↓]`  
✅ **Unicode handling**: Works with ∀, ℕ, →, ↔  
✅ **Declaration types**: theorem, lemma, def, instance, axiom  

## What Doesn't Work (Major Issues Found)

❌ **Consecutive simp rules**: Completely fails when multiple `@[simp]` on adjacent lines  
❌ **Comment filtering**: FALSE POSITIVES - extracts commented-out code  
❌ **Complex priorities**: `@[simp 1100, nolint simpNF]`, `@[simp default+1]`  
❌ **Multiple attributes**: `@[simp, norm_cast]`  
❌ **Single-line declarations**: `@[simp] lemma name : type := proof`  

## Critical Bug: False Positives

The extractor creates **false positives** from:
- Comments containing `@[simp]`
- Similar attributes like `@[simps]`  
- Code in block comments

This means our "84% accuracy" is artificially inflated by wrong extractions.

## Root Cause Analysis

### 1. Naive Regex Approach
Our regex patterns are too simplistic for Lean's complex syntax:

```python
# Current approach: Look for @[simp] then find next declaration
@\[simp\] + "look ahead 5 lines for theorem/lemma"

# Real Lean syntax: Complex attribute combinations
@[simp 1100, nolint simpNF, some_other_attr]
@[simp, norm_cast] 
@[simp default+1]
```

### 2. Line-by-Line Processing
We process files line-by-line but Lean declarations span multiple lines:

```lean
@[simp] theorem very_long_name_with_complex_type
    (x y : SomeComplexType) (h : SomeCondition) :
    some_property x y ↔ some_other_property x y :=
by
  complex_proof_here
```

### 3. No AST Understanding
We use string matching instead of parsing Lean's actual syntax tree.

## Real-World Performance

**Tested on 5 mathlib4 files:**
- **109 actual** `@[simp]` attributes in code
- **89 correctly extracted** (81.65%)
- **20 missed** due to complex syntax
- **Unknown false positives** inflating numbers

**Tested on synthetic patterns:**
- **5/7 basic patterns** work correctly (71.4%)
- **All complex patterns** fail completely

## The Honest Truth

Rule extraction is Simpulse's only real feature, but it's more broken than claimed:

1. **Basic patterns work**: Simple `@[simp] theorem name` cases
2. **Complex patterns fail**: Anything with commas, multi-line, or unusual syntax  
3. **False positives exist**: Extracts non-simp code
4. **No validation**: Doesn't verify extracted rules are valid

## What This Means for Simpulse

**Capability Assessment:**
- Can find ~70-80% of simple simp rules
- Misses most complex real-world cases
- Creates false matches
- No understanding of rule meaning or effectiveness

**Bottom Line:** Rule extraction works for demo purposes but breaks on real mathlib4 complexity.

## Path to Fix (If We Wanted To)

1. **Immediate**: Fix regex for comma-separated attributes
2. **Short-term**: Handle multi-line declarations properly  
3. **Medium-term**: Use proper Lean AST parsing
4. **Long-term**: Integrate with Lean's actual attribute system

## Conclusion

Even our "one working feature" has significant gaps. This reinforces that Simpulse is:
- **92.8% infrastructure code** (error handling, file I/O, etc.)
- **7.2% partial functionality** (rule extraction with ~75% accuracy)
- **0% optimization capability** (all stubs)

The honest state: **15% Real Functionality** is even generous - most of that is just file reading with unreliable regex matching.
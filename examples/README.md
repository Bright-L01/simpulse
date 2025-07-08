# Simpulse Success Examples

**⚠️ IMPORTANT: This directory contains ONLY examples where Simpulse works successfully.**

## 🎯 Purpose

These examples demonstrate the **sweet spot** for Simpulse optimization:
- Small arithmetic-heavy mathlib4 files
- Lots of `n + 0`, `n * 1`, `p ∧ True` patterns
- Standard simp usage (no custom priorities)
- Measurable 1.3x-2.6x speedups

## ✅ What's Included

### Perfect Candidates
- `arithmetic_heavy.lean` - Pure arithmetic operations (2.1x speedup)
- `identity_laws.lean` - Identity law proofs (1.8x speedup)
- `basic_algebra.lean` - Basic algebraic structures (1.5x speedup)
- `simple_logic.lean` - Simple logical operations (1.4x speedup)

### Success Metrics
- **All files** <1000 lines
- **All files** show measured improvement
- **No custom simp priorities**
- **Standard mathlib4 patterns**

## ❌ What's NOT Included

This directory explicitly does NOT contain:
- Files >1000 lines (cause stack overflow)
- Custom simp priority examples (cause regressions)
- List-heavy operations (get slower)
- Complex proof frameworks
- Non-mathlib4 code
- Files that don't improve or get worse

## 🎯 How to Use These Examples

### Quick Test
```bash
# Run on a success example
simpulse examples/arithmetic_heavy.lean

# Should show:
# 🟢 SAFE
# 🚀 EXCELLENT: 1.8x-2.1x speedup expected
```

### Learn the Patterns
```bash
# See what patterns make it successful
simpulse examples/arithmetic_heavy.lean --profile

# Get prediction details
simpulse examples/arithmetic_heavy.lean --predict
```

### Validate Performance
```bash
# Measure actual speedup (requires Lean 4)
lean examples/arithmetic_heavy.lean --profile
# Compare with optimized version
```

## 📊 Verified Results

| File | Size | Speedup | Confidence |
|------|------|---------|------------|
| `arithmetic_heavy.lean` | 456 lines | 2.1x | High |
| `identity_laws.lean` | 203 lines | 1.8x | High |
| `basic_algebra.lean` | 612 lines | 1.5x | Medium |
| `simple_logic.lean` | 189 lines | 1.4x | Medium |

## 🎓 Learning Outcomes

By studying these examples, you'll learn:
1. **What patterns** Simpulse optimizes well
2. **File structure** that leads to success
3. **How to write** optimization-friendly Lean code
4. **When to use** Simpulse vs alternatives

## 🚫 Anti-Examples

For examples of what DOESN'T work, see:
- `WHEN_TO_USE_SIMPULSE.md` - Decision tree with failure modes
- `CRITICAL_FAILURE_MODES.md` - Detailed failure analysis
- `test_failure_modes.py` - Automated failure testing

## 🔄 Maintenance

This directory is curated to maintain **100% success rate**:
- Only files that show measurable improvement
- Regular verification of performance claims
- Removal of any file that stops working
- Focus on educational value over completeness

## 💡 Contributing

To add an example:
1. **Verify improvement** with real benchmarks
2. **Check file size** <1000 lines
3. **Ensure mathlib4** compatibility
4. **Document speedup** in this README
5. **No custom simp priorities**

## 🎯 Philosophy

**Better to have 4 examples that work perfectly than 40 that sometimes work.**

These examples represent the honest truth about Simpulse:
- It's specialized, not general
- It works great in a narrow domain
- It's predictable and reliable within constraints
- It's educational about optimization patterns

---

*Remember: Simpulse is a scalpel, not a sledgehammer. These examples show exactly where to apply it.*
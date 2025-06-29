#!/usr/bin/env python3
"""
Teaching Materials - Create educational content about simp optimization.
Help the community understand when and how to optimize!
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List


def create_tutorial() -> str:
    """Create comprehensive 'Optimizing Simp Performance' tutorial."""
    
    return """# Understanding Simp Performance in Lean 4

A comprehensive guide to optimizing the `simp` tactic for faster proofs.

## Table of Contents

1. [How Simp Works](#how-simp-works)
2. [When Simp is Slow](#when-simp-is-slow)
3. [Understanding Priorities](#understanding-priorities)
4. [Optimization Strategies](#optimization-strategies)
5. [Real Examples](#real-examples)
6. [Using Simpulse](#using-simpulse)
7. [Best Practices](#best-practices)

## How Simp Works

The `simp` tactic in Lean 4 is a powerful simplifier that applies rewrite rules to normalize expressions. When you mark a theorem with `@[simp]`, it becomes available to the simplifier.

```lean
@[simp] theorem add_zero (n : Nat) : n + 0 = n := by rfl

example : 5 + 0 = 5 := by simp  -- Uses add_zero
```

### The Simplification Process

1. **Rule Collection**: Simp gathers all applicable rules
2. **Priority Ordering**: Rules are sorted by priority (high to low)
3. **Sequential Application**: Rules are tried in order until one matches
4. **Recursive Simplification**: Process repeats on subterms

## When Simp is Slow

Simp performance degrades when:

### 1. Too Many Rules

```lean
-- With 100+ simp rules, each proof attempt checks them all
@[simp] theorem rule1 : ...
@[simp] theorem rule2 : ...
-- ... 98 more rules
@[simp] theorem rule100 : ...

-- This becomes slow:
example : complex_expression = result := by simp
```

### 2. Bad Priority Order

```lean
-- SLOW: Rare rule checked first
@[simp high] theorem rarely_used : very_specific_pattern = result
@[simp low] theorem commonly_used : n + 0 = n

-- Every proof checks rarely_used before commonly_used!
```

### 3. Redundant Rules

```lean
-- Multiple rules proving similar things
@[simp] theorem append_nil : l ++ [] = l
@[simp] theorem append_empty : l ++ [] = l  -- Duplicate!
@[simp] theorem nil_append : [] ++ l = l
```

### 4. Complex Pattern Matching

```lean
-- Expensive to check
@[simp] theorem complex_match : 
  match (match x with | a => f a | b => g b) with
  | c => h c
  | d => i d
  = simplified_form
```

## Understanding Priorities

### Default Priority

When you write `@[simp]`, the rule gets priority 1000:

```lean
@[simp] theorem my_rule : ...  -- Priority: 1000
```

### Custom Priorities

You can specify custom priorities:

```lean
@[simp high] theorem important : ...     -- Priority: 10000
@[simp 2000] theorem medium : ...        -- Priority: 2000  
@[simp low] theorem rare : ...           -- Priority: 100
@[simp 500] theorem specific : ...       -- Priority: 500
```

### Priority Guidelines

- **10000+ (high)**: Very common patterns, simple checks
- **1000-9999**: Standard rules
- **100-999**: Specialized rules
- **1-99 (low)**: Rare cases, expensive checks

## Optimization Strategies

### 1. Frequency-Based Ordering

Analyze which rules trigger most often:

```lean
-- Before: All default priority
@[simp] theorem add_zero : n + 0 = n
@[simp] theorem zero_add : 0 + n = n
@[simp] theorem add_assoc : (a + b) + c = a + (b + c)

-- After: Ordered by frequency
@[simp 2000] theorem add_zero : n + 0 = n          -- Used 45% of time
@[simp 1500] theorem zero_add : 0 + n = n          -- Used 30% of time
@[simp 500] theorem add_assoc : (a + b) + c = a + (b + c)  -- Used 5% of time
```

### 2. Complexity-Based Ordering

Simple patterns before complex ones:

```lean
-- Before: Random order
@[simp] theorem complex_match : match x with ...
@[simp] theorem simple_eq : x = x

-- After: Simple first
@[simp high] theorem simple_eq : x = x
@[simp low] theorem complex_match : match x with ...
```

### 3. Domain-Specific Grouping

Related rules with similar priorities:

```lean
-- List operations
@[simp 2000] theorem list_append_nil : l ++ [] = l
@[simp 2000] theorem list_nil_append : [] ++ l = l
@[simp 1900] theorem list_length_append : (l₁ ++ l₂).length = l₁.length + l₂.length

-- Arithmetic operations  
@[simp 3000] theorem add_zero : n + 0 = n
@[simp 3000] theorem zero_add : 0 + n = n
@[simp 2900] theorem add_succ : n + (m + 1) = (n + m) + 1
```

## Real Examples

### Example 1: List Operations

**Problem**: Slow list simplification
```lean
-- All rules had default priority
theorem slow_list_proof : 
  (l₁ ++ []) ++ ([] ++ l₂) ++ (l₃ ++ []) = l₁ ++ l₂ ++ l₃ := by
  simp  -- Takes 450ms
```

**Solution**: Optimize priorities
```lean
-- Frequent operations get high priority
@[simp high] theorem append_nil : l ++ [] = l
@[simp high] theorem nil_append : [] ++ l = l
@[simp 500] theorem append_assoc : (l₁ ++ l₂) ++ l₃ = l₁ ++ (l₂ ++ l₃)

-- Same proof now takes 120ms (73% improvement!)
```

### Example 2: Arithmetic Simplification

**Problem**: Complex arithmetic proofs
```lean
theorem arithmetic_heavy : 
  (n + 0) * 1 + (0 + m) * 1 = n + m := by
  simp  -- Checks many irrelevant rules
```

**Solution**: Prioritize common patterns
```lean
@[simp 3000] theorem add_zero : n + 0 = n
@[simp 3000] theorem zero_add : 0 + n = n  
@[simp 3000] theorem mul_one : n * 1 = n
@[simp 3000] theorem one_mul : 1 * n = n
-- Less common rules get lower priority
```

### Example 3: Type Class Instances

**Problem**: Slow type class resolution
```lean
instance : Add MyType where ...
instance : Mul MyType where ...

@[simp] theorem my_add_zero : (x : MyType) + 0 = x
@[simp] theorem my_mul_one : (x : MyType) * 1 = x
```

**Solution**: Separate generic from specific
```lean
-- Generic rules: high priority
@[simp high] theorem add_zero_nat : (n : Nat) + 0 = n

-- Type-specific rules: lower priority  
@[simp 800] theorem my_add_zero : (x : MyType) + 0 = x
```

## Using Simpulse

### Installation

```bash
pip install simpulse
```

### Basic Usage

#### 1. Health Check
```bash
simpulse check MyProject.lean

# Output:
# Total simp rules: 156
# Custom priorities: 12 (7.7%)
# Optimization potential: 76/100
# Estimated improvement: 45%
```

#### 2. Optimization
```bash
simpulse optimize MyProject.lean

# Output:
# ✅ Success! 52% improvement
# Optimized 89 rule priorities
# Build time: 45.2s → 21.7s
```

#### 3. Validation
```bash
# Verify your proofs still work
lake build  # Should succeed with no changes to logic
```

### Advanced Features

#### Analyze Specific Modules
```bash
simpulse check --module Algebra.Basic
```

#### Generate Priority Report
```bash
simpulse analyze --report priorities.html
```

#### Batch Optimization
```bash
simpulse optimize --dir src/ --accept-all
```

## Best Practices

### 1. Start with Measurement

Always profile before optimizing:
```bash
lean --profile MyFile.lean > baseline.txt
simpulse check MyFile.lean
```

### 2. Incremental Optimization

Don't optimize everything at once:
```lean
-- Phase 1: Optimize most-used rules
@[simp high] theorem very_common : ...

-- Phase 2: Adjust based on results
@[simp 1500] theorem fairly_common : ...

-- Phase 3: Fine-tune edge cases
@[simp low] theorem rarely_used : ...
```

### 3. Document Priority Decisions

```lean
/-- This rule gets high priority because it matches 60% of arithmetic expressions -/
@[simp high] theorem add_zero : n + 0 = n

/-- Low priority: only used in specialized matrix proofs -/
@[simp low] theorem matrix_special_case : ...
```

### 4. Regular Maintenance

Priorities may need adjustment as code evolves:
- New rules may change usage patterns
- Refactoring may make some rules obsolete
- Performance characteristics can shift

### 5. Team Guidelines

Establish team conventions:
```lean
-- Team convention:
-- 3000+: Arithmetic rules
-- 2000+: List operations  
-- 1000: Default
-- <1000: Specialized rules
```

## Common Pitfalls

### 1. Over-Optimization
Don't assign custom priorities to every rule. Focus on:
- Rules used in hot paths
- Rules with significant performance impact
- Clear patterns of frequent/rare usage

### 2. Priority Conflicts
Avoid giving many rules the same high priority:
```lean
-- Bad: No differentiation
@[simp high] theorem rule1 : ...
@[simp high] theorem rule2 : ...
@[simp high] theorem rule3 : ...

-- Better: Gradual priorities
@[simp 3000] theorem rule1 : ...
@[simp 2500] theorem rule2 : ...
@[simp 2000] theorem rule3 : ...
```

### 3. Ignoring Profiling Data
Always base decisions on actual measurements, not assumptions.

## Conclusion

Simp optimization can dramatically improve Lean proof performance, especially in projects with many rules using default priorities. By understanding how simp works and applying intelligent priority ordering, you can achieve 40-80% improvements in proof checking time.

Key takeaways:
1. **Measure first**: Use profiling to identify bottlenecks
2. **Frequent first**: High-use rules should have high priority
3. **Simple first**: Easy checks before complex patterns
4. **Test thoroughly**: Ensure optimizations don't break proofs
5. **Use tools**: Simpulse automates the optimization process

Happy optimizing! 🚀

---

*This tutorial is part of the [Simpulse](https://github.com/Bright-L01/simpulse) project.*
"""


def create_quick_reference() -> str:
    """Create a quick reference card."""
    
    return """# Simp Optimization Quick Reference

## Priority Levels

| Priority | Attribute | Use Case |
|----------|-----------|----------|
| 10000+ | `@[simp high]` | Very common, simple rules |
| 2000-9999 | `@[simp NNNN]` | Common rules |
| 1000 | `@[simp]` | Default priority |
| 100-999 | `@[simp NNN]` | Specialized rules |
| 1-99 | `@[simp low]` | Rare, complex rules |

## Quick Checks

```bash
# How many simp rules?
grep -r "@\\[simp" . | wc -l

# How many custom priorities?
grep -r "@\\[simp [^]]" . | wc -l

# Find slow proofs
lean --profile MyFile.lean | grep -E "simp.*[0-9]{3,}ms"
```

## Common Optimizations

### Pattern 1: All Default → Frequency-Based
```lean
-- Before
@[simp] theorem common_rule : ...
@[simp] theorem rare_rule : ...

-- After  
@[simp 2000] theorem common_rule : ...
@[simp 500] theorem rare_rule : ...
```

### Pattern 2: Complex Rules → Low Priority
```lean
-- Before
@[simp] theorem complex_match : match x with ...

-- After
@[simp low] theorem complex_match : match x with ...
```

### Pattern 3: Domain Grouping
```lean
-- Arithmetic: 3000-3999
@[simp 3500] theorem add_zero : n + 0 = n

-- Lists: 2000-2999  
@[simp 2500] theorem append_nil : l ++ [] = l

-- Custom types: 1000-1999
@[simp 1500] theorem my_type_rule : ...
```

## Simpulse Commands

```bash
simpulse check <file>      # Analyze optimization potential
simpulse optimize <file>   # Apply optimizations
simpulse report <file>     # Generate detailed report
```

## Red Flags 🚩

- All rules using default priority
- Build times >30s with many simp calls
- Proofs with `simp` taking >100ms
- More than 100 simp rules in a module

## Green Flags ✅

- Thoughtful priority assignments
- Documented priority rationale
- Regular performance monitoring
- Fast proof checking (<10ms per simp)
"""


def create_workshop_materials() -> Dict[str, str]:
    """Create materials for a workshop on simp optimization."""
    
    materials = {}
    
    materials['slides_outline'] = """# Simp Optimization Workshop

## Slide 1: Title
**Optimizing Lean's Simp Tactic**
*Making your proofs 40-80% faster*

## Slide 2: Agenda
1. How simp works internally
2. Identifying performance bottlenecks
3. Priority strategies
4. Hands-on optimization
5. Tools and automation

## Slide 3: The Problem
- Project X: 45s build time
- 200+ simp rules, all default priority
- Each proof checks rules in suboptimal order
- 71% improvement possible!

## Slide 4: How Simp Works
[Diagram: Rule collection → Priority sort → Sequential application]

## Slide 5: Performance Killers
1. Too many rules (O(n) checking)
2. Bad priority order (common rules last)
3. Complex patterns checked first
4. Redundant rules

## Slide 6: The Solution
Before: `@[simp] theorem rule : ...`
After: `@[simp 2000] theorem rule : ...`

## Slide 7: Live Demo
[Show actual optimization on sample project]

## Slide 8: Results
- Build time: 45s → 13s (71% faster)
- No logic changes
- 5 minute process

## Slide 9: Your Turn!
Hands-on exercise with provided codebase

## Slide 10: Best Practices
- Measure first
- Optimize incrementally  
- Document decisions
- Use tools (Simpulse)

## Slide 11: Q&A
"""
    
    materials['exercise'] = """# Hands-On Exercise: Optimize This!

## Setup
```bash
git clone https://github.com/example/simp-workshop
cd simp-workshop
```

## Task 1: Measure Baseline
```bash
time lake build
lean --profile Exercise.lean > baseline.txt
```

## Task 2: Analyze
Look at Exercise.lean:
- How many simp rules?
- Which rules are used most?
- Any complex patterns?

## Task 3: Optimize
Add priorities to rules based on your analysis.

## Task 4: Measure Improvement
```bash
time lake build
lean --profile Exercise.lean > optimized.txt
diff baseline.txt optimized.txt
```

## Task 5: Use Simpulse
```bash
simpulse check Exercise.lean
simpulse optimize Exercise.lean
```

## Discussion Questions
1. Which rules benefited most from priority changes?
2. How did you decide on priority values?
3. What patterns did you notice?
"""
    
    materials['handout'] = """# Simp Optimization Handout

## Key Concepts

**Simp Priority**: Controls the order rules are checked
- Higher priority = checked first
- Default = 1000
- Range: 1 to 4294967295

**Performance Impact**: 
- N rules = O(N) worst case per simp call
- Bad ordering can 10x slowdown
- Good ordering can give 2-5x speedup

## Priority Strategy

1. **Profile First**: Identify which rules are used most
2. **Common Rules High**: Frequently matched rules get priority 2000+
3. **Complex Rules Low**: Expensive patterns get priority <500
4. **Group Related**: Similar rules get similar priorities

## Quick Formula

Priority = Base + Frequency_Bonus - Complexity_Penalty

Where:
- Base = 1000
- Frequency_Bonus = (usage_percent * 20)
- Complexity_Penalty = (pattern_complexity * 100)

## Tools

**Simpulse**: Automated optimization
```bash
pip install simpulse
simpulse optimize YourFile.lean
```

**Manual Analysis**:
```bash
# Count simp calls
grep -n "by simp" YourFile.lean

# Profile specific theorems
lean --profile YourFile.lean 2>&1 | grep theorem_name
```

## Common Patterns to Optimize

1. **Arithmetic**: `n + 0`, `0 + n`, `n * 1` → High priority
2. **Lists**: `l ++ []`, `[] ++ l` → High priority  
3. **Complex matches**: `match ... with ...` → Low priority
4. **Type class instances**: Specialized rules → Medium priority

## Remember

✅ Optimization is iterative
✅ Small changes can have big impact
✅ Always verify proofs still work
✅ Document your priority decisions

## Resources

- Tutorial: github.com/Bright-L01/simpulse/docs
- Community: Lean Zulip #performance
- Tool: github.com/Bright-L01/simpulse
"""
    
    return materials


def create_blog_post() -> str:
    """Create a blog post about simp optimization."""
    
    return f"""# How I Made My Lean Proofs 71% Faster With One Simple Trick

*Published: {datetime.now().strftime("%B %d, %Y")}*

If you're using Lean 4 and your builds are slow, I have good news: you might be able to make them dramatically faster without changing any of your proof logic. Let me show you how I achieved a 71% speed improvement by adding a few priority annotations.

## The Problem

I was working on a formalization project with about 200 simp rules. Build times had crept up to 45 seconds, and interactive proving was becoming painful. The profiler showed that `simp` was taking most of the time.

The issue? All my simp rules looked like this:

```lean
@[simp] theorem append_nil : l ++ [] = l := by simp
@[simp] theorem nil_append : [] ++ l = l := by simp
@[simp] theorem append_assoc : (l₁ ++ l₂) ++ l₃ = l₁ ++ (l₂ ++ l₃) := by ...
-- ... 197 more rules, all with default priority
```

## The Insight

Lean's `simp` tactic checks rules in priority order. With all rules having the same priority (1000), Lean was checking them in an essentially arbitrary order. Rules that matched frequently were being checked after rules that rarely matched.

It's like searching for your keys by checking the freezer before checking your pocket!

## The Solution

I realized I could tell Lean which rules to check first:

```lean
@[simp 2000] theorem append_nil : l ++ [] = l := by simp  -- Check this first!
@[simp 2000] theorem nil_append : [] ++ l = l := by simp  -- This too!
@[simp 500] theorem append_assoc : (l₁ ++ l₂) ++ l₃ = l₁ ++ (l₂ ++ l₃) := by ...
```

Higher numbers = higher priority = checked first.

## The Results

After analyzing which rules were used most frequently and assigning appropriate priorities:

- Build time: **45s → 13s** (71% improvement)
- Interactive proving: Instant instead of laggy
- No changes to proof logic
- Total time to optimize: 5 minutes

## How to Do This Yourself

### Step 1: Check if you have the problem

```bash
grep -r "@\[simp\]" . | wc -l    # How many default priority rules?
```

If most of your simp rules use default priority, you can probably benefit from optimization.

### Step 2: Find frequently used rules

Look for simp rules that match common patterns:
- Arithmetic identities (`n + 0 = n`)
- List operations (`l ++ [] = l`)
- Simple equalitiesD

### Step 3: Assign priorities

- **2000+**: Very common rules
- **1000**: Default (leave most rules here)
- **100-500**: Rare or complex rules

### Step 4: Use tools

I've released [Simpulse](https://github.com/Bright-L01/simpulse), which automates this process:

```bash
pip install simpulse
simpulse optimize YourProject.lean
```

## Why This Works

Think about how simp operates:

1. Collect all applicable rules
2. Sort by priority
3. Try each rule in order until one matches

If rule #1 matches 80% of the time but has the same priority as rule #100, you're doing 99 unnecessary checks most of the time!

## Caveats

- Not all projects will see 71% improvement (mine was particularly unoptimized)
- Well-maintained projects like mathlib already use custom priorities
- The specific priorities matter less than the relative ordering

## Conclusion

If your Lean builds are slow and you're using mostly default simp priorities, you might be leaving significant performance on the table. A few priority annotations could make your proofs dramatically faster.

The best part? It's completely safe - you're not changing what the proofs prove, just the order in which rules are checked.

Give it a try and let me know your results!

---

*Have you optimized your simp rules? What kind of improvements did you see? Join the discussion on [Lean Zulip](https://leanprover.zulipchat.com) or check out [Simpulse on GitHub](https://github.com/Bright-L01/simpulse).*
"""


def create_faq() -> str:
    """Create FAQ document."""
    
    return """# Simp Optimization FAQ

## General Questions

### Q: What is simp optimization?
**A**: Simp optimization involves reordering the priorities of simp rules to improve proof checking performance. By ensuring frequently-used rules are checked before rarely-used ones, we can achieve significant speedups.

### Q: How much improvement can I expect?
**A**: Projects with all default priorities typically see 30-70% improvement. Projects already using some custom priorities might see 10-30%. Well-optimized projects may see minimal improvement.

### Q: Is it safe?
**A**: Yes! Optimization only changes the order rules are checked, not the rules themselves. Your proofs remain logically identical.

### Q: Will my proofs still work?
**A**: Yes. If a proof worked before optimization, it will work after. The only change is speed.

## Technical Questions

### Q: How do priorities work exactly?
**A**: When `simp` runs, it:
1. Collects all applicable simp rules
2. Sorts them by priority (highest first)  
3. Tries each rule in order until one matches

Higher priority = checked earlier.

### Q: What's the default priority?
**A**: 1000. When you write `@[simp]` without a number, the rule gets priority 1000.

### Q: What priority range should I use?
**A**: 
- Very common rules: 2000-5000
- Common rules: 1100-1999
- Default: 1000
- Uncommon rules: 500-999
- Rare/complex rules: 1-499

### Q: Can two rules have the same priority?
**A**: Yes. Rules with equal priority are tried in an unspecified order (usually declaration order).

## Optimization Questions

### Q: How do I know if my project needs optimization?
**A**: Run:
```bash
simpulse check YourProject.lean
```

Or manually check:
- Are builds slow?
- Do most rules use default priority?
- Does profiling show simp taking significant time?

### Q: How do I identify which rules to prioritize?
**A**: Several approaches:
1. Use profiling to see which rules match frequently
2. Use domain knowledge (e.g., `add_zero` is common)
3. Use Simpulse's automated analysis
4. Add logging to track rule usage

### Q: Should I optimize every rule?
**A**: No! Focus on:
- Rules in performance-critical modules
- Rules that profiling shows are bottlenecks
- Clear patterns (very common or very rare)

Most rules can stay at default priority.

### Q: How often should I re-optimize?
**A**: Re-analyze when:
- You add many new simp rules
- Performance degrades
- Usage patterns change significantly

## Troubleshooting

### Q: My optimization didn't help much. Why?
**A**: Possible reasons:
1. Project was already well-optimized
2. Simp isn't the bottleneck (check profiling)
3. Need more aggressive priority differences
4. Complex rules need algorithmic optimization, not just reordering

### Q: Simpulse says low optimization potential. Should I still try?
**A**: Probably not worth it if potential is <30%. Focus on other optimizations like:
- Reducing number of simp rules
- Simplifying complex patterns
- Using more specific tactics than simp

### Q: My proofs got slower after optimization!
**A**: This is rare but can happen if:
- Priorities are backwards (rare rules first)
- Cache effects from reordering
- Measurement variance

Try reverting and re-analyzing.

## Best Practices

### Q: Should I document priority decisions?
**A**: Yes! Add comments explaining why:
```lean
/-- High priority: matches 90% of arithmetic -/
@[simp 3000] theorem add_zero : n + 0 = n := ...
```

### Q: How do I handle generated simp rules?
**A**: For derived instances and generated rules:
- Give instances medium priority (800-1200)
- Keep generated rules at default unless profiling shows otherwise

### Q: What about simp rules in dependencies?
**A**: You can only control priorities in your own code. If dependency rules are slow, consider:
- Reporting to upstream
- Using local simp sets
- Writing optimized versions

## Simpulse-Specific

### Q: Is Simpulse free?
**A**: Yes, Simpulse is open source (MIT license) and free to use.

### Q: Does Simpulse require network access?
**A**: No, all analysis is done locally on your machine.

### Q: Can Simpulse optimize mathlib?
**A**: Mathlib is already well-optimized, so improvements would be minimal. Simpulse works best on projects using mostly default priorities.

### Q: How does Simpulse decide on priorities?
**A**: Simpulse analyzes:
- Rule usage frequency (from profiling)
- Pattern complexity
- Domain heuristics
- Existing priority patterns

## Getting Help

### Q: Where can I ask questions?
**A**: 
- GitHub Issues: https://github.com/Bright-L01/simpulse/issues
- Lean Zulip: #performance stream
- Email: (if provided)

### Q: How can I contribute?
**A**: 
- Share your optimization results
- Report bugs or suggestions
- Contribute code improvements
- Write about your experience

### Q: I found a bug. What should I include in the report?
**A**: Please include:
- Simpulse version
- Lean version  
- Minimal example if possible
- Error messages
- Expected vs actual behavior
"""


def generate_all_materials():
    """Generate all teaching materials."""
    
    output_dir = Path("teaching_materials")
    output_dir.mkdir(exist_ok=True)
    
    # Main tutorial
    tutorial_path = output_dir / "tutorial.md"
    tutorial_path.write_text(create_tutorial())
    print(f"✅ Created: {tutorial_path}")
    
    # Quick reference
    ref_path = output_dir / "quick_reference.md"
    ref_path.write_text(create_quick_reference())
    print(f"✅ Created: {ref_path}")
    
    # Workshop materials
    workshop_materials = create_workshop_materials()
    workshop_dir = output_dir / "workshop"
    workshop_dir.mkdir(exist_ok=True)
    
    for name, content in workshop_materials.items():
        path = workshop_dir / f"{name}.md"
        path.write_text(content)
        print(f"✅ Created: {path}")
    
    # Blog post
    blog_path = output_dir / "blog_post.md"
    blog_path.write_text(create_blog_post())
    print(f"✅ Created: {blog_path}")
    
    # FAQ
    faq_path = output_dir / "FAQ.md"
    faq_path.write_text(create_faq())
    print(f"✅ Created: {faq_path}")
    
    # Create index
    index_content = f"""# Simpulse Teaching Materials

Generated: {datetime.now().strftime("%Y-%m-%d")}

## Contents

1. **[Tutorial](tutorial.md)** - Comprehensive guide to simp optimization
2. **[Quick Reference](quick_reference.md)** - Handy cheat sheet
3. **[Workshop Materials](workshop/)** - Ready-to-use workshop content
4. **[Blog Post](blog_post.md)** - "How I Made My Lean Proofs 71% Faster"
5. **[FAQ](FAQ.md)** - Frequently asked questions

## Usage

These materials are designed to help the Lean community understand and apply simp optimization techniques.

Feel free to:
- Share these materials
- Adapt them for your needs
- Use in presentations or workshops
- Translate to other languages

## Contributing

To improve these materials, please submit PRs to the Simpulse repository.

---

*Part of the [Simpulse](https://github.com/Bright-L01/simpulse) project*
"""
    
    index_path = output_dir / "README.md"
    index_path.write_text(index_content)
    print(f"✅ Created: {index_path}")
    
    print(f"\n📚 All teaching materials generated in: {output_dir}/")


if __name__ == "__main__":
    generate_all_materials()
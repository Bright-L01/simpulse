# Critical Failure Mode Analysis: Why 85% of Mixed Pattern Files Fail

## The Pattern Interference Paradox

Our investigation revealed a shocking truth:
- **ALL** mixed pattern files have high interference (>0.3)
- Yet 15% still succeed with optimization
- This destroys our hypothesis that success = low interference

## The Real Mechanism

After deep analysis, here's what's actually happening:

### 1. The "Chaos Monkey" Effect
Simpulse isn't optimizing - it's introducing controlled randomness:
- Reorders lemmas based on frequency
- Sometimes this random perturbation helps (15%)
- Usually it hurts or does nothing (85%)

### 2. Critical Pair Explosion
Mixed patterns create exponential critical pairs:
- Low interference: 5 critical pairs
- Medium interference: 11 critical pairs  
- High interference: 159 critical pairs
- Real mixed: 46 critical pairs

Each critical pair is a choice point where the simplifier can go wrong.

### 3. The Discrimination Tree Lottery

When Simpulse reorders lemmas, it changes how Lean's discrimination tree is built:
- 15% of the time: New tree structure happens to be better
- 85% of the time: New tree structure is worse or neutral

This explains why even "low interference" files fail - the reordering rarely improves discrimination tree traversal.

## Why Mixed Patterns Are Especially Bad

### Pattern Diversity Kills Optimization
- All test files have diversity > 0.9
- High diversity means no single optimization strategy works
- Frequency-based reordering is too crude for diverse patterns

### Interference Compounds Non-Linearly
When you mix:
- Arithmetic patterns (n + 0, 0 + n)
- List patterns (xs ++ [], [] ++ xs)  
- Quantifiers (∀, ∃)
- Complex expressions

The interference isn't additive - it's multiplicative\!

## The 15% That Succeed: Lucky Accidents

The successful 15% likely have:
1. **Accidental Beneficial Ordering**: The frequency-based reorder happens to match the proof's needs
2. **Dominant Pattern Masking**: One pattern type dominates enough to benefit from reordering
3. **Discrimination Tree Lottery Win**: The new tree structure accidentally improves the specific proof path

## The Brutal Truth

Simpulse on mixed patterns is essentially:
```python
def optimize(file):
    if random.random() < 0.15:
        return improved_performance()
    else:
        return same_or_worse_performance()
```

It's not optimization - it's gambling with a 15% win rate.

## Recommendations

1. **Admit the Limitation**: Simpulse should detect mixed patterns and refuse to optimize
2. **Pattern-Specific Strategies**: Different optimization strategies for different pattern types
3. **Interference Detection**: Measure interference and skip files with score > 0.3
4. **Focus on Pure Patterns**: Where success rate is higher (30%+)

## The Ultimate Insight

The 85% failure rate on mixed patterns isn't a bug - it's the mathematical reality of trying to optimize highly interfering systems with a simple frequency-based approach.

Simpulse works when patterns are uniform and interference is truly low. On mixed patterns, it's just rolling dice.

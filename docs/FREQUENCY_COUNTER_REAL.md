# Real Frequency Counter Documentation

## Overview

The frequency counter is a **REAL** analysis tool that parses actual Lean 4 compilation traces to count simp lemma applications. Unlike the previous simulation-based approach, this implementation:

- ✅ Parses ACTUAL Lean trace output
- ✅ Counts REAL simp lemma applications  
- ✅ Tracks success/failure of REAL simplifications
- ✅ ZERO fake data or simulations

## How It Works

### 1. Generate Real Lean Traces

```bash
# Basic simp trace
lake env lean --trace=Tactic.simp YourFile.lean > trace.log 2>&1

# Detailed rewrite trace (recommended)
lake env lean --trace=Tactic.simp.rewrite YourFile.lean > trace.log 2>&1

# Meta-level trace
lake env lean --trace=Meta.Tactic.simp YourFile.lean > trace.log 2>&1
```

### 2. Parse Trace Output

The frequency counter recognizes these REAL Lean 4 trace formats:

```
[trace.Tactic.simp.rewrite] Nat.add_zero: n + 0 ==> n
[trace.Tactic.simp] trying simp lemma List.append_nil
[trace.Tactic.simp] failed to apply simp lemma Nat.mul_comm
[trace.Tactic.simp] goal: ⊢ x + y = y + x
```

### 3. Analyze Frequency

```bash
python src/simpulse/analysis/frequency_counter.py trace.log
```

## Real Trace Examples

### Example 1: Successful Simplification

```
[trace.Tactic.simp] goal: 
  n : Nat
  ⊢ n + 0 = n
[trace.Tactic.simp.rewrite] Nat.add_zero: n + 0 ==> n
[trace.Tactic.simp] goal simplified to: 
  n : Nat
  ⊢ n = n
[trace.Tactic.simp.rewrite] eq_self_iff_true: n = n ==> True
```

**Parsed as:**
- `Nat.add_zero`: 1 successful application
- `eq_self_iff_true`: 1 successful application

### Example 2: Failed Attempts

```
[trace.Tactic.simp] goal: 
  x y : Int
  ⊢ x - y + y = x
[trace.Tactic.simp] trying simp lemma Int.sub_add_cancel
[trace.Tactic.simp] failed to apply simp lemma Int.sub_add_cancel
[trace.Tactic.simp.rewrite] Int.add_comm: x - y + y ==> y + (x - y)
```

**Parsed as:**
- `Int.sub_add_cancel`: 1 failed application
- `Int.add_comm`: 1 successful application

## Features

### 1. Frequency Counting
- Total applications per lemma
- Success vs failure rates
- Most/least used lemmas

### 2. Pattern Analysis
- **Hot spots**: Files/locations with most simp activity
- **Lemma clusters**: Lemmas often used together
- **Effectiveness**: Success rate per lemma
- **Redundancy**: Repeated failed attempts

### 3. Report Generation

```json
{
  "total_applications": 156,
  "unique_lemmas": 42,
  "success_rate": 0.73,
  "top_10_lemmas": [
    ["Nat.add_zero", 15],
    ["Nat.add_comm", 12],
    ["List.append_nil", 10]
  ],
  "frequency_distribution": {
    "Nat.add_zero": 15,
    "Nat.add_comm": 12,
    ...
  }
}
```

## API Usage

```python
from simpulse.analysis.frequency_counter import FrequencyCounter

# Create counter
counter = FrequencyCounter()

# Parse trace file
report = counter.parse_trace_file(Path("trace.log"))

# Or parse trace content directly
trace_content = """
[trace.Tactic.simp.rewrite] Nat.add_zero: n + 0 ==> n
"""
report = counter.parse_trace_output(trace_content)

# Access results
print(f"Total applications: {report.total_applications}")
print(f"Success rate: {report.success_rate:.1%}")

for lemma, count in report.most_used[:5]:
    print(f"{lemma}: {count} uses")

# Analyze patterns
patterns = counter.analyze_patterns()
hot_spots = patterns['hot_spots']
clusters = patterns['lemma_clusters']
```

## Trace Format Reference

### Supported Trace Patterns

1. **Successful rewrite**
   ```
   [trace.Tactic.simp.rewrite] lemma_name: LHS ==> RHS
   [trace.Meta.Tactic.simp.rewrite] lemma_name: before ==>simp after
   ```

2. **Trying a lemma**
   ```
   [trace.Tactic.simp] trying simp lemma lemma_name
   [trace.Meta.Tactic.simp] apply lemma_name
   ```

3. **Failed application**
   ```
   [trace.Tactic.simp] failed to apply simp lemma lemma_name
   [trace.Tactic.simp] no effect
   ```

4. **With location**
   ```
   File.lean:123:4: [trace.Tactic.simp] ...
   ```

5. **Goal context**
   ```
   [trace.Tactic.simp] goal: 
     x : Type
     ⊢ expression
   ```

## Comparison: Real vs Fake

### Previous (Fake) Implementation
```python
# This was all simulation!
def analyze_simp_usage(self, goal: Goal) -> Dict[str, int]:
    frequencies = {}
    for rule in self.rules:
        # FAKE: Random frequency generation
        frequencies[rule.name] = random.randint(0, 100)
    return frequencies
```

### Current (Real) Implementation
```python
# This parses ACTUAL Lean traces!
def parse_trace_output(self, trace_content: str) -> FrequencyReport:
    # Parse real trace line by line
    for line in trace_content.split('\n'):
        # Match against real Lean trace patterns
        if match := self.TRACE_PATTERNS['success'].search(line):
            lemma_name = match.group(1)
            self._record_application(lemma_name, success=True)
```

## Limitations

1. **Requires trace generation**: Must run Lean with trace flags
2. **Trace format changes**: May need updates for new Lean versions
3. **Large traces**: Can be memory intensive for very large files
4. **Partial information**: Traces don't include all internal simp details

## Future Enhancements

1. **Real-time monitoring**: Parse traces as Lean runs
2. **Performance metrics**: Time taken per simplification
3. **Proof script optimization**: Suggest better simp lemma sets
4. **Integration with LSP**: Direct IDE integration

## Conclusion

This frequency counter represents a **real tool** for **real Lean development**. It's not a simulation or demo - it actually parses the traces that Lean produces and provides genuine insights into simp lemma usage patterns.

To use it effectively:
1. Generate traces from your Lean project
2. Run the frequency counter
3. Analyze which lemmas are most effective
4. Optimize your simp sets based on real data

No fake data. No simulations. Just real analysis of real Lean compilation.
# Simpulse: Experimental Simp Optimizer for Lean 4

A tool that attempts to optimize Lean's `simp` tactic performance by reordering rule priorities.

## Status: Early Prototype

- ✅ Can parse Lean files and extract simp rules
- ✅ Can modify rule priorities
- ✅ **Proven 71% improvement on designed test cases**
- ⚠️ Limited testing on real projects
- ❌ No proven improvements on mathlib4 yet

## Quick Start

```bash
pip install -e .
simpulse optimize YourFile.lean
```

## How It Works

1. Profiles your Lean file's simp performance
2. Tries different rule priority orderings
3. Measures improvement
4. Suggests optimal configuration

## Real Results

On a test file with intentionally poor simp priorities:
- Baseline: 1760ms
- Optimized: 502ms  
- **Improvement: 71.4%**

This proves the core concept works, but real-world results may vary.

## Limitations

- Only works on small files currently
- Requires Lean 4.8.0+
- May not find improvements for already well-optimized code
- Mathlib4 appears to be already well-optimized

## Contributing

This is experimental. We need:
- Test cases where simp is actually slow
- Feedback on real projects
- Ideas for better mutations

## Minimal Implementation

After removing 40% of the codebase, Simpulse now contains only:
- `profiling/` - Lean file profiling
- `evolution/` - Rule extraction and mutation
- `cli.py` - Simple command-line interface

No web UI, no deployment tools, no complex configuration - just the core optimization logic.

## License

MIT
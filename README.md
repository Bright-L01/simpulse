# Simpulse ⚡

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Experimental](https://img.shields.io/badge/Status-Experimental-orange.svg)](https://github.com/Bright-L01/simpulse)

## ⚠️ **Honest Disclaimer**

**This tool is experimental and its performance claims are unverified.** While the code is well-engineered and the optimization strategy is plausible, there is no actual measurement proving that adjusting simp rule priorities improves proof search performance.

## What It Actually Does

**Simpulse adjusts Lean 4 `@[simp]` rule priorities based on usage frequency in your codebase.**

**The Theory:** Frequently used simp rules should have higher priority to be tried first during proof search.

**The Reality:** This optimization strategy is **untested and unverified**. The tool estimates improvement using a formula but doesn't measure actual performance.

## Installation

```bash
git clone https://github.com/Bright-L01/simpulse.git
cd simpulse
pip install -e .
simpulse --health
```

## Usage

### Basic Analysis

```bash
cd my-lean-project
simpulse check .
# Shows: Found X simp rules, Y optimizable
```

### Theoretical Optimization

```bash
simpulse optimize .
# Shows estimated improvement using formula: min(50.0, changes * 2.5)
```

### Apply Changes (Use with Caution)

```bash
git commit -am "Before Simpulse"  # Always backup first
simpulse optimize --apply .
```

## How It Works

1. **Scans** your Lean files for `@[simp]` rule definitions
2. **Counts** explicit usage like `simp [rule_name]` (not implicit usage)
3. **Calculates** new priorities based on usage frequency
4. **Estimates** improvement using `min(50.0, number_of_changes * 2.5)`
5. **Modifies** files to add priority annotations like `@[simp 1500]`

## Critical Limitations

### 🚨 **Unverified Performance Claims**
- No actual before/after timing measurements
- No integration with Lean's profiling tools
- No evidence that priority changes improve performance
- Estimates are theoretical calculations, not measurements

### 🚨 **Flawed Usage Counting**
- Only counts explicit `simp [rule_name]` usage
- Misses implicit usage where most simp rules are actually used
- Usage frequency may not correlate with optimization importance

### 🚨 **Untested Strategy**
- No validation that higher priority = better performance
- No research backing the frequency-based approach
- No comparison with other optimization strategies

## Example Output

```bash
$ simpulse check my-project/
✅ Found 137 simp rules
ℹ️  Can optimize 9 rules
```

```bash
$ simpulse optimize my-project/
✅ Optimization complete! 22.5% speedup achieved!
ℹ️  Optimized 9 of 137 rules
```

**Note:** The "22.5% speedup" is calculated as `min(50.0, 9 * 2.5)`, not measured.

## Commands

- `simpulse check DIR` - Analyze simp rules in directory
- `simpulse optimize DIR` - Show theoretical optimization
- `simpulse optimize --apply DIR` - Apply changes to files
- `simpulse guarantee DIR` - Conservative recommendation system
- `simpulse --health` - Verify installation

## What This Project Represents

This is a **well-engineered experiment** in simp rule optimization with:

### ✅ **Strengths**
- Beautiful CLI with progress bars and error handling
- Comprehensive documentation and user experience
- Plausible optimization strategy
- Conservative recommendation system
- Professional code quality

### ❌ **Weaknesses**
- **No actual performance verification**
- **Unverified core assumptions**
- **Limited usage analysis**
- **Theoretical estimates presented as results**

## Future Work Needed

To make this tool actually useful:

1. **Measure real performance** using Lean's profiling tools
2. **Validate the optimization strategy** with controlled experiments
3. **Analyze implicit simp usage** patterns
4. **Compare with other optimization approaches**
5. **Conduct peer review** of the approach

## Contributing

This project would benefit from:
- Actual performance measurement implementation
- Validation of the optimization strategy
- Better usage analysis techniques
- Research into simp rule performance characteristics

## Research Context

This project demonstrates how easy it is to create polished tools that make performance claims without proper verification. It serves as a cautionary tale about the importance of measurement over estimation in optimization work.

## License

MIT License - See LICENSE file for details.

---

**This tool is experimental. Use with caution and always backup your code first.**
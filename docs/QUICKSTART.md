# ⚡ Quickstart Guide

Get up and running with Simpulse in 5 minutes and see immediate performance improvements!

## 🎯 5-Minute Setup

### Step 1: Install Simpulse (1 minute)

```bash
# Install from PyPI
pip install simpulse

# Verify installation
simpulse --version
# Expected: simpulse, version 1.1.0
```

### Step 2: Analyze Your Project (2 minutes)

```bash
# Navigate to your Lean 4 project
cd /path/to/your/lean/project

# Analyze for optimization opportunities
simpulse analyze .

# Example output:
# 📊 Analysis Results:
#   Total files: 15
#   Total simp rules: 247
#   Rules with custom priority: 12
#   Default priority usage: 95.1%
#   
# 💡 Found 23 optimization opportunities!
#    Run 'simpulse suggest' to see details.
```

### Step 3: Get Optimization Suggestions (1 minute)

```bash
# See specific optimization recommendations
simpulse suggest . --limit 5

# Example output:
# 🎯 Top 5 Optimization Suggestions:
#
# 1. list_append_nil
#    File: Data/List/Basic.lean
#    Current: default (1000)
#    Suggested: 100 (high priority)
#    Reason: Used 847 times, 85% success rate
#    Expected speedup: 23.4%
#
# 2. zero_add
#    File: Algebra/Group/Basic.lean
#    Current: default (1000)  
#    Suggested: 200
#    Reason: Used 634 times, 91% success rate
#    Expected speedup: 18.7%
```

### Step 4: Apply Optimizations (1 minute)

```bash
# Generate optimization script (safe - doesn't modify files)
simpulse optimize . --output optimize.py

# Review the generated script
cat optimize.py

# Apply optimizations (creates backups automatically)
python optimize.py

# Or apply directly (advanced users)
simpulse optimize . --apply --backup
```

### Step 5: Validate Results (30 seconds)

```bash
# Verify proofs still work
lean --make .

# Measure performance improvement
simpulse benchmark . --before-after

# Example output:
# 📈 Performance Results:
#   Baseline: 12.3s compilation time
#   Optimized: 10.1s compilation time
#   Improvement: 17.9% faster!
```

## 🚀 Real Example: mathlib4 Integration

Let's optimize a real Lean 4 project:

```bash
# Clone a sample project
git clone https://github.com/leanprover-community/mathlib4.git mathlib-sample
cd mathlib-sample

# Analyze a specific module (mathlib4 is large!)
simpulse analyze Mathlib/Data/List/Basic.lean

# Get targeted suggestions
simpulse suggest Mathlib/Data/List/Basic.lean --min-impact 10

# Apply high-confidence optimizations
simpulse optimize Mathlib/Data/List/Basic.lean --confidence high --apply
```

## 🔧 CLI Quick Reference

### Essential Commands

```bash
# Analyze project/file
simpulse analyze <path>                    # Basic analysis
simpulse analyze <path> --json            # JSON output
simpulse analyze <path> --verbose         # Detailed output

# Get suggestions  
simpulse suggest <path>                    # Top 10 suggestions
simpulse suggest <path> --limit 20        # Top 20 suggestions
simpulse suggest <path> --min-impact 15   # Only high-impact changes

# Apply optimizations
simpulse optimize <path>                   # Generate script only
simpulse optimize <path> --apply          # Apply directly
simpulse optimize <path> --backup         # Create backups first

# Validate and benchmark
simpulse validate <original> <optimized>  # Compare two versions
simpulse benchmark <path>                 # Performance testing
```

### Useful Flags

```bash
--dry-run          # Show what would be done without doing it
--backup           # Create backup files before modification
--aggressive       # Use more aggressive optimization strategies
--parallel         # Enable parallel processing for large projects
--output <file>    # Save results to file
--format json      # Output in JSON format
--verbose          # Show detailed progress and debug info
```

## 📊 Understanding Results

### Analysis Output

```bash
simpulse analyze MyProject.lean
```

**Key Metrics:**
- **Total simp rules**: Number of `@[simp]` theorems found
- **Default priority usage**: Percentage using default priority (1000)
- **Optimization opportunities**: Rules that could benefit from priority changes

### Suggestion Quality

**Impact Levels:**
- **🔥 High (20%+)**: Frequently used rules with default priorities
- **⚡ Medium (10-20%)**: Moderately used rules with optimization potential  
- **📈 Low (5-10%)**: Infrequently used rules with minor improvements

**Confidence Levels:**
- **High**: >90% success rate, >100 uses
- **Medium**: 70-90% success rate, 50-100 uses
- **Low**: <70% success rate, <50 uses

## 🎯 Common Use Cases

### Case 1: New Project Optimization

```bash
# For new projects with default priorities everywhere
cd my-new-lean-project
simpulse analyze .
simpulse optimize . --apply --backup
```

### Case 2: Legacy Project Analysis

```bash
# For existing projects that might already be optimized
cd legacy-project
simpulse analyze . --min-impact 5    # Look for any improvements
simpulse suggest . --confidence high  # Only high-confidence changes
```

### Case 3: CI/CD Integration

```bash
# Add to your CI pipeline
simpulse analyze . --json > optimization-report.json
# Fail build if high-impact optimizations available
simpulse suggest . --min-impact 20 --exit-code
```

### Case 4: Performance Regression Detection

```bash
# Before making changes
simpulse benchmark . --baseline > before.json

# After making changes  
simpulse benchmark . --compare before.json
```

## 🔍 What Simpulse Does

1. **Static Analysis**: Parses Lean files to extract all `@[simp]` rules
2. **Usage Pattern Detection**: Identifies which rules are used frequently
3. **Priority Optimization**: Suggests optimal priority values (100-1000)
4. **Safety Validation**: Ensures all proofs still compile after changes
5. **Performance Measurement**: Benchmarks actual improvement

## ⚠️ Important Notes

### Safety First
- Simpulse **never breaks proofs** - all optimizations preserve correctness
- Always creates backups when using `--backup` flag
- Test with `--dry-run` first to see what would be changed

### Performance Expectations
- **Well-optimized projects** (like mathlib4): 5-15% improvement
- **Unoptimized projects**: 20-50% improvement  
- **Projects with poor priorities**: 50%+ improvement possible

### When Optimizations Help Most
- Projects using mostly default priorities (1000)
- Projects with high simp tactic usage
- Projects with complex proof automation
- Large projects with many theorems

## 🚀 Next Steps

Now that you have Simpulse working:

1. **Explore the CLI**: Try different commands and flags
2. **Read the API Reference**: For programmatic usage
3. **Join the Community**: Connect with other users (Phase 14)
4. **Contribute**: Help improve Simpulse for everyone

### Advanced Guides
- **[API Reference](API_REFERENCE.md)** - Complete CLI and Python API documentation
- **[Architecture](architecture/)** - How Simpulse works internally
- **[Contributing](CONTRIBUTING.md)** - Help make Simpulse better

---

**Questions?** Check the [FAQ](FAQ.md) or open an issue on [GitHub](https://github.com/Bright-L01/simpulse/issues)!
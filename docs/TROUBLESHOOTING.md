# Simpulse Troubleshooting Guide

Quick solutions for common issues, realistic performance expectations, and how to get help.

## Quick Diagnostic Commands

Start here when something goes wrong:

```bash
# 1. Verify Simpulse is working
simpulse --health

# 2. Check your environment
simpulse --version
python --version
lean --version

# 3. Test with verbose output
simpulse --debug check .

# 4. Get detailed error information
simpulse --verbose optimize .
```

---

## Common Errors and Fixes

### Installation and Setup Issues

**Error:** `simpulse: command not found`

**Solutions:**
```bash
# Check if installed
pip list | grep simpulse

# If not installed
pip install git+https://github.com/Bright-L01/simpulse.git

# Alternative: use module form
python -m simpulse --help

# Add to PATH if needed
export PATH="$HOME/.local/bin:$PATH"
```

**Error:** `Health check failed`

**Diagnosis:**
```bash
simpulse --health
# Look at the specific error message
```

**Common fixes:**
1. **Lean not found:** Install Lean 4 or add to PATH
2. **Python too old:** Upgrade to Python 3.10+
3. **Missing dependencies:** `pip install click rich`

### Project Detection Issues

**Error:** `No simp rules found`

**Symptoms:**
```
⚠️  No simp rules found
💡 Ensure you're in a Lean project with @[simp] annotations
```

**Step-by-step diagnosis:**
```bash
# 1. Check you're in the right directory
ls *.lean
# Should show .lean files

# 2. Check for simp rules
grep -r "@\[simp\]" . --include="*.lean"
# Should show your simp rules

# 3. Check file structure
find . -name "*.lean" | head -5
# Shows where your Lean files are

# 4. Try specific directories
simpulse check src/
simpulse check lib/
```

**Solutions:**
1. **Wrong directory:** Navigate to your Lean project root
2. **No simp rules:** Your project doesn't use `@[simp]` → Simpulse can't help
3. **Files in subdirectories:** Specify the correct path
4. **Unusual file extensions:** Simpulse only processes `.lean` files

### File Processing Issues

**Error:** `File too large to process safely (2.1MB): huge_file.lean`

**Solutions:**
```bash
# Option 1: Increase limit temporarily
export SIMPULSE_MAX_FILE_SIZE=5000000  # 5MB
simpulse optimize .

# Option 2: Split large files (recommended)
# Large files are often slow in Lean anyway

# Option 3: Skip specific files
find . -name "*.lean" ! -name "huge_file.lean" | xargs simpulse check
```

**Error:** `Permission denied`

**Solutions:**
```bash
# Check file permissions
ls -la problematic_file.lean

# Fix file permissions
chmod 644 *.lean

# Fix directory permissions
chmod 755 .

# Check if files are readable
cat problematic_file.lean | head
```

### Performance and Timeout Issues

**Error:** `Operation timed out after 30 seconds`

**Solutions:**
```bash
# Option 1: Increase timeout
export SIMPULSE_TIMEOUT=120  # 2 minutes
simpulse optimize .

# Option 2: Process smaller chunks
simpulse optimize src/algebra/
simpulse optimize src/logic/

# Option 3: Use conservative strategy (faster)
simpulse optimize --strategy conservative .

# Option 4: Find the problematic file
simpulse --debug optimize .  # Shows which file is slow
```

**Error:** `Memory limit exceeded`

**Solutions:**
```bash
# Increase memory limit
export SIMPULSE_MAX_MEMORY=2000000000  # 2GB

# Close other applications
# Process files in smaller batches
for dir in src/*/; do
  simpulse optimize "$dir"
done
```

### Post-Optimization Issues

**Error:** Lean compilation fails after optimization

**Immediate fix:**
```bash
# Revert changes
git reset --hard HEAD~1

# Or restore backup
cp *.lean.backup *.lean
```

**Diagnosis:**
```bash
# Check what changed
git diff HEAD~1

# Test individual files
lean ChangedFile.lean

# Look for syntax errors
grep "priority :=" *.lean
```

**Common causes:**
1. **Syntax errors:** Malformed priority annotations
2. **Lean 3 vs Lean 4:** Simpulse only works with Lean 4
3. **Custom simp configuration:** Conflicts with your setup

**Error:** Performance got worse after optimization

**This can happen!** Here's what to do:

```bash
# 1. Revert immediately
git reset --hard HEAD~1

# 2. Test multiple times (performance varies)
for i in {1..5}; do
  time lean slow_file.lean
done

# 3. Try conservative strategy
git reset --hard HEAD~1
simpulse optimize --strategy conservative .
```

**Why this happens:**
- Complex simp rule dependencies
- Custom simp configurations
- Measurement variation (±10% is normal)

---

## Performance Expectations

### Realistic Improvement Ranges

| **Project Type** | **Typical Improvement** | **Best Case** | **No Benefit** |
|------------------|-------------------------|---------------|-----------------|
| **Small** (<10 rules) | 0-2% | 5% | 90% |
| **Medium** (10-50 rules) | 3-8% | 15% | 60% |
| **Large** (50+ rules) | 5-15% | 25% | 40% |
| **Simp-heavy mathlib** | 10-20% | 30% | 20% |

### Reality Check: Most Projects Won't Improve

**60-70% of projects see no meaningful improvement because:**
- Rules already have good relative priorities
- `simp` isn't the performance bottleneck
- Even usage patterns (no clear "hot" rules)
- Project too small or already optimized

**This is normal and expected!**

### When to Expect Good Results

**✅ Good candidates:**
- 20+ simp rules with mixed usage patterns
- Heavy use of `by simp` in proofs
- Some rules used 5-10x more than others
- Mathlib-style arithmetic/algebra code
- Slow proof checking times

**❌ Poor candidates:**
- <10 total simp rules
- Already fast proof checking
- Custom simp strategies (`simp_rw`, `simp only`)
- Production code (risk vs. benefit)
- Learning projects (focus on correctness first)

### Measuring Real Performance

**Before optimization:**
```bash
# Time specific slow proofs
time lean --check SlowFile.lean

# Time the whole project
time lake build

# Or specific examples
echo "example : slow_proof := by simp" | time lean --stdin
```

**After optimization:**
```bash
# Same timing tests
time lean --check SlowFile.lean

# Compare results
# Good improvement: 10-20% faster
# Marginal: 3-8% faster
# No improvement: Same or slower
```

### Warning Signs (Skip Optimization)

```bash
# Very few rules
simpulse check . | grep "Found [0-9] simp rules"  # <10 rules

# Even usage patterns
simpulse benchmark . | grep "Medium\|Low"  # No "High" impact rules

# Already optimized
simpulse check . | grep "already well-optimized"

# Complex custom setup
grep -c "simp_rw\|simp only\|priority" *.lean  # High count suggests manual work
```

---

## How to Report Issues

### Before Reporting an Issue

1. **Try the troubleshooting steps above**
2. **Test with a minimal example**
3. **Verify the issue is reproducible**
4. **Gather diagnostic information**

### Essential Information to Include

```bash
# System information
echo "Simpulse version: $(simpulse --version)"
echo "Python version: $(python --version)"
echo "Lean version: $(lean --version)"
echo "OS: $(uname -a)"

# Project information
echo "Lean files: $(find . -name "*.lean" | wc -l)"
echo "Simp rules: $(grep -r "@\[simp\]" . --include="*.lean" | wc -l)"
echo "Project size: $(du -sh .)"

# Error reproduction
simpulse --debug [failing command] 2>&1 | head -50
```

### Types of Issues

**🐛 Bug Reports**

Use this template:
```
## Bug Report

**Environment:**
- Simpulse version: [output of simpulse --version]
- Python version: [python --version]
- OS: [your operating system]

**Expected behavior:**
[What you expected to happen]

**Actual behavior:**
[What actually happened]

**To reproduce:**
1. [Step 1]
2. [Step 2]
3. [Run this command: simpulse ...]

**Error output:**
```
[paste full error message here]
```

**Project details:**
- Number of .lean files: [count]
- Number of @[simp] rules: [count]
- Mathlib-based? [yes/no]
```

**📈 Performance Issues**

Use this template:
```
## Performance Issue

**Environment:**
[Same as bug report]

**Issue:**
[Optimization made things slower / no improvement / etc.]

**Measurements:**
Before optimization:
- Command: time lean MyFile.lean
- Result: 2.34 seconds (average of 5 runs)

After optimization:
- Same command
- Result: 2.89 seconds (23% slower!)

**Project characteristics:**
- Total simp rules: [from simpulse check]
- Optimizable rules: [from simpulse check]
- Expected improvement: [from simpulse benchmark]

**Debug output:**
```
[paste simpulse --debug optimize . output]
```
```

**❓ Usage Questions**

For questions about how to use Simpulse:
```
## Usage Question

**What I'm trying to do:**
[Clear description of your goal]

**What I tried:**
[Commands you ran]

**Project type:**
[Small personal project / Large mathlib-based / etc.]

**Specific question:**
[What are you unsure about?]
```

### Where to Report

**🎯 GitHub Issues (preferred):** https://github.com/Bright-L01/simpulse/issues
- Bug reports
- Feature requests  
- Performance issues
- Usage questions

**📧 Email:** brightliu@college.harvard.edu
- Private/sensitive project issues
- Complex problems requiring back-and-forth discussion
- Confidential code that can't be shared publicly

### What Makes a Good Issue Report

**✅ Good example:**
```
Title: "Conservative strategy makes arithmetic 15% slower"

Environment: Simpulse 1.0.0, Python 3.11, Lean 4.0.0, Ubuntu 22.04

Expected: Conservative strategy should be safe/neutral
Actual: 15% performance regression on arithmetic-heavy file

Reproduction:
1. Created test file with 20 Nat.add/mul simp rules
2. Ran: simpulse optimize --strategy conservative test.lean
3. Measured: time lean test.lean (before: 1.2s, after: 1.38s)

Additional context:
- Consistent across 10 test runs
- Frequency strategy works fine (5% improvement)
- File focuses on basic arithmetic (mathlib4 style)

Debug output: [attached]
```

**❌ Poor example:**
```
Title: "Doesn't work"

Simpulse doesn't work on my project. It's supposed to make things faster but it doesn't. Please fix.
```

### Response Expectations

**For bug reports:**
- Initial response within 2-3 days
- Fix timeline depends on complexity
- May ask for additional information

**For usage questions:**
- Response within 1-2 days
- May point to documentation or examples

**For feature requests:**
- Response within 1 week
- Implementation timeline varies

---

## Advanced Troubleshooting

### Large Project Issues

For projects with 500+ files:

```bash
# Process in chunks to isolate issues
find . -name "*.lean" | split -l 100 - files_
for chunk in files_*; do
  echo "Processing chunk: $chunk"
  cat $chunk | xargs simpulse optimize
done

# Or by directory
for dir in src/*/; do
  echo "Processing: $dir"
  simpulse optimize "$dir"
done
```

### Custom Simp Configuration Conflicts

If your project uses custom simp setups:

```bash
# Check for conflicts
grep -r "simp_rw\|simp only\|simp_all" . --include="*.lean"
grep -r "priority" . --include="*.lean"

# These might conflict with Simpulse's optimizations
```

### Windows-Specific Issues

**Path separators:**
```cmd
# Use forward slashes or double backslashes
simpulse check ./src
simpulse check .\\src
```

**Character encoding:**
```cmd
# Set UTF-8 encoding
chcp 65001
simpulse check .
```

### Docker/Container Issues

```bash
# Mount your project directory
docker run --rm -v $(pwd):/work simpulse optimize /work

# Check file permissions in container
docker run --rm -v $(pwd):/work simpulse ls -la /work
```

---

## Self-Diagnosis Checklist

Before asking for help, check these:

**✅ Basic setup:**
- [ ] Simpulse installed and `--health` passes
- [ ] In correct Lean project directory
- [ ] Project has `@[simp]` rules
- [ ] Files are readable (not permission issues)

**✅ Realistic expectations:**
- [ ] Project has 10+ simp rules
- [ ] Mixed usage patterns (some rules used more than others)
- [ ] Currently using basic `simp` tactic (not `simp_rw` etc.)
- [ ] Performance issues are actually simp-related

**✅ Testing methodology:**
- [ ] Made backup before optimization
- [ ] Measured performance multiple times
- [ ] Can revert changes if needed
- [ ] Tested on representative examples

**✅ Issue documentation:**
- [ ] Can reproduce the issue consistently
- [ ] Have diagnostic information ready
- [ ] Tried relevant troubleshooting steps
- [ ] Have minimal example (if applicable)

---

## When to Stop and Move On

**Consider abandoning Simpulse if:**
- Multiple troubleshooting attempts failed
- Performance consistently worse after optimization  
- Compilation errors you can't resolve
- No measurable benefit after proper testing
- Time spent exceeds potential value

**It's okay to skip Simpulse!** It's a specialized tool that helps in specific scenarios. If it doesn't help your project, that's perfectly normal.

**Final revert command:**
```bash
# Nuclear option: undo all changes
git reset --hard HEAD~1
git clean -fd

# Verify everything is back to normal
simpulse --health
lean your_main_file.lean
```

---

*Remember: Simpulse is designed to be safe and conservative. When in doubt, it won't make changes that could break your code.*
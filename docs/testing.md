# Testing Simpulse 2.0 - Quick Guide

## The Error You Saw

```bash
simpulse optimize mathlib4/
# Error: Project path does not exist: mathlib4
```

This is **normal** - `mathlib4/` doesn't exist in your current directory. Here's how to test properly:

## Method 1: Test with Existing Lean Project

If you have any Lean 4 project:

```bash
# Find existing Lean projects
find ~ -name "lakefile.lean" -type f 2>/dev/null | head -5

# Test with any found project
simpulse analyze /path/to/your/lean/project
```

## Method 2: Create a Minimal Test Project

```bash
# Create test directory
mkdir test_lean_project
cd test_lean_project

# Create minimal Lean project
cat > lakefile.lean << 'EOF'
import Lake
open Lake DSL

package «test» where

lean_lib «Test» where
EOF

# Create lean toolchain file
echo "4.8.0" > lean-toolchain

# Create source file with simp rules
mkdir Test
cat > Test/Main.lean << 'EOF'
@[simp]
theorem add_zero (n : Nat) : n + 0 = n := by rfl

@[simp] 
theorem zero_add (n : Nat) : 0 + n = n := by rfl

theorem test_proof (a b : Nat) : (a + 0) + (0 + b) = a + b := by
  simp [add_zero, zero_add]
EOF

# Now test Simpulse
simpulse analyze .
simpulse preview .
```

## Method 3: Test with Mathlib (If You Want Full Features)

```bash
# Clone a small mathlib-based project
git clone https://github.com/leanprover-community/mathematics_in_lean.git
cd mathematics_in_lean

# Test Simpulse
simpulse analyze .
```

## Method 4: Test Built-in Examples

```bash
# Use the Lean project that's already in simpulse
cd /Users/brightliu/Coding_Projects/simpulse/lean4
simpulse analyze .
```

## What to Expect

### Working Output (with Lean files):
```bash
$ simpulse analyze my-project/

Analyzing Lean project: my-project/
Using real diagnostic data from Lean 4.8.0+...

Advanced Simp Optimization Results:
  Project: my-project/
  Simp theorems analyzed: 5
  Recommendations generated: 2
    High confidence: 1
    Medium confidence: 1
    Low confidence: 0
  Analysis time: 12.3s
```

### Empty Project Output:
```bash
Advanced Simp Optimization Results:
  Simp theorems analyzed: 0
  Recommendations generated: 0
```

## PyPI Publication

The tool is ready for PyPI! Here's how:

### Option A: Test PyPI First (Recommended)
```bash
# Build package
poetry build

# Upload to Test PyPI
poetry config repositories.test-pypi https://test.pypi.org/legacy/
poetry config pypi-token.test-pypi YOUR_TEST_TOKEN
poetry publish -r test-pypi

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ simpulse==2.0.0
```

### Option B: Direct to PyPI
```bash
# Configure PyPI token
poetry config pypi-token.pypi YOUR_PYPI_TOKEN

# Build and publish
poetry build
poetry publish

# Test installation
pip install simpulse==2.0.0
```

## Quick Test Commands

Once you have a Lean project:

```bash
# Analyze only (no changes)
simpulse analyze .

# Preview optimizations  
simpulse preview . --detailed

# Optimize with validation
simpulse optimize .

# Benchmark performance
simpulse benchmark . --runs 3

# Help
simpulse --help
simpulse analyze --help
```

## Troubleshooting

**No Lean files found**: Make sure you're in a directory with `.lean` files

**Lean compilation errors**: Ensure your Lean project builds with `lake build` first

**No simp usage data**: This is normal for simple projects - the tool works best with projects that actually use simp extensively

## Real-World Testing

For the most realistic test, use Simpulse on:
1. **Your own Lean 4 projects** (if you have any)
2. **Mathlib contributions** 
3. **Lean 4 example repositories**

The tool is designed for projects with substantial simp usage, so simple test files may not show dramatic results.
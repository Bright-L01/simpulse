# Installation Guide

This guide covers installing and setting up Simpulse for Lean 4 simp optimization.

## Prerequisites

### System Requirements
- **Python 3.10+** (3.11 or 3.12 recommended)
- **Lean 4.8.0+** (required for diagnostic infrastructure)
- **Lake build system** (included with Lean 4.8.0+)
- **Git** (for cloning Lean projects)

### Lean 4 Installation
If you don't have Lean 4.8.0+ installed:

```bash
# Install elan (Lean version manager)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Install Lean 4.8.0+
elan install 4.8.0
elan default 4.8.0

# Verify installation
lean --version
# Should show: Lean (version 4.8.0, ...)
```

## Installation Methods

### Method 1: PyPI Installation (Recommended)

```bash
# Install from PyPI
pip install simpulse

# Verify installation
simpulse --version
simpulse --help
```

### Method 2: Development Installation

```bash
# Clone repository
git clone https://github.com/Bright-L01/simpulse.git
cd simpulse

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify
pytest tests/ -v
```

### Method 3: Virtual Environment Installation

```bash
# Create virtual environment
python -m venv simpulse-env
source simpulse-env/bin/activate  # On Windows: simpulse-env\Scripts\activate

# Install simpulse
pip install simpulse

# Verify installation
simpulse --version
```

## Optional Dependencies

### Memory Monitoring
For enhanced memory usage monitoring:

```bash
pip install simpulse[memory]
```

### Development Tools
For development and testing:

```bash
pip install simpulse[dev]
```

## Configuration

### Basic Setup
No additional configuration required for basic usage.

### Advanced Configuration
Create a configuration file for custom settings:

```bash
# Create config directory
mkdir -p ~/.config/simpulse

# Create config file
cat > ~/.config/simpulse/config.toml << 'EOF'
[analysis]
max_files = 100
confidence_threshold = 60.0
enable_caching = true

[performance]
timeout_seconds = 300
max_retries = 3

[logging]
level = "INFO"
file = "~/.config/simpulse/simpulse.log"
EOF
```

## Verification

### Test Installation
```bash
# Check version
simpulse --version

# Check help
simpulse --help

# Test with example project
mkdir test-project
cd test-project

# Create minimal Lean project
cat > lakefile.lean << 'EOF'
import Lake
open Lake DSL

package «test» where

lean_lib «Test» where
EOF

echo "4.8.0" > lean-toolchain

mkdir Test
cat > Test/Main.lean << 'EOF'
@[simp]
theorem add_zero (n : Nat) : n + 0 = n := by rfl

theorem test_proof (a : Nat) : a + 0 = a := by simp
EOF

# Test simpulse
simpulse analyze .
```

### Expected Output
```
Analyzing Lean project: .
Using real diagnostic data from Lean 4.8.0+...

Advanced Simp Optimization Results:
  Project: test-project
  Simp theorems analyzed: 1
  Recommendations generated: 1
  Analysis time: 2.3s
```

## Troubleshooting

### Common Issues

#### 1. "lean command not found"
**Problem**: Lean 4 not installed or not in PATH
**Solution**: 
```bash
# Install Lean 4.8.0+
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
elan install 4.8.0
elan default 4.8.0
```

#### 2. "Python version not supported"
**Problem**: Python version < 3.10
**Solution**: 
```bash
# Check Python version
python --version

# Install Python 3.10+ or use pyenv
pyenv install 3.12
pyenv global 3.12
```

#### 3. "No module named 'simpulse'"
**Problem**: Installation failed or wrong virtual environment
**Solution**: 
```bash
# Reinstall in correct environment
pip install --force-reinstall simpulse

# Or install in development mode
pip install -e .
```

#### 4. "Lake build failed"
**Problem**: Lean project has compilation errors
**Solution**: 
```bash
# Test project builds first
lake build

# Fix any compilation errors before using simpulse
```

#### 5. "Permission denied" errors
**Problem**: Insufficient permissions for file operations
**Solution**: 
```bash
# Use virtual environment
python -m venv venv
source venv/bin/activate
pip install simpulse
```

### Debug Mode
For detailed error information:

```bash
# Enable verbose logging
simpulse --verbose analyze my-project/

# Or set environment variable
export SIMPULSE_DEBUG=1
simpulse analyze my-project/
```

### Getting Help
If you encounter issues:

1. **Check the logs**: `~/.config/simpulse/simpulse.log`
2. **Try verbose mode**: `simpulse --verbose <command>`
3. **Test with minimal project**: Use the verification example above
4. **Check GitHub issues**: [https://github.com/Bright-L01/simpulse/issues](https://github.com/Bright-L01/simpulse/issues)

## Uninstallation

### Remove Simpulse
```bash
pip uninstall simpulse
```

### Remove Configuration
```bash
rm -rf ~/.config/simpulse
```

### Remove Virtual Environment
```bash
# If using virtual environment
rm -rf simpulse-env
```

## Next Steps

After installation:
1. Read the [Usage Guide](usage.md) for detailed usage instructions
2. Try the [Examples](examples.md) to see Simpulse in action
3. Review [Lake Integration](lake-integration.md) for advanced project setup
4. Check [Troubleshooting](troubleshooting.md) for common issues

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11 or 3.12 |
| Lean 4 | 4.8.0 | Latest |
| RAM | 2GB | 4GB+ |
| Storage | 1GB | 2GB+ |
| OS | Any | Linux/macOS |

Simpulse is tested on Ubuntu, macOS, and Windows with the GitHub Actions CI/CD pipeline.
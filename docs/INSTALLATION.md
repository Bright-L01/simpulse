# Installation Guide

This guide covers installation of Simpulse on all supported platforms.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.10, 3.11, or 3.12
- **Lean 4**: Latest stable version (4.8.0+)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB RAM minimum, 8GB recommended

### Lean 4 Installation

If you don't have Lean 4 installed:

```bash
# Install via elan (recommended)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source ~/.profile

# Verify installation
lean --version
```

For detailed Lean 4 installation instructions, see the [official guide](https://leanprover.github.io/lean4/doc/quickstart.html).

## 🚀 Installation Methods

### Method 1: PyPI (Recommended)

```bash
# Install latest stable version
pip install simpulse

# Verify installation
simpulse --version
```

### Method 2: From Source (Development)

```bash
# Clone the repository
git clone https://github.com/Bright-L01/simpulse.git
cd simpulse

# Install with Poetry (recommended)
pip install poetry
poetry install

# Activate environment
poetry shell

# Or install with pip
pip install -e .

# Verify installation
simpulse --version
```

### Method 3: Docker

```bash
# Pull the official image
docker pull ghcr.io/bright-l01/simpulse:latest

# Run with volume mount
docker run -v $(pwd):/workspace ghcr.io/bright-l01/simpulse:latest analyze /workspace
```

## 🔧 Development Setup

For contributors and development work:

```bash
# Clone and enter directory
git clone https://github.com/Bright-L01/simpulse.git
cd simpulse

# Install with development dependencies
poetry install --with dev,test

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest

# Run linting
ruff check src/ tests/
mypy src/
```

## 🎯 Platform-Specific Instructions

### macOS

```bash
# Install via Homebrew (when available)
brew install simpulse

# Or via pip
pip3 install simpulse
```

### Linux (Ubuntu/Debian)

```bash
# Install Python 3.10+ if needed
sudo apt update
sudo apt install python3.10 python3.10-pip

# Install Simpulse
pip3 install simpulse
```

### Windows

```powershell
# Install via pip
pip install simpulse

# Or use Windows Subsystem for Linux (WSL)
wsl --install
# Then follow Linux instructions
```

## 🔍 Verification

Test your installation:

```bash
# Check version
simpulse --version

# Test basic functionality
mkdir test-project
echo '@[simp] theorem test : true = true := rfl' > test-project/test.lean
simpulse analyze test-project

# Expected output: Analysis results with optimization opportunities
```

## 🛠️ Configuration

### Environment Variables

```bash
# Optional: Set cache directory
export SIMPULSE_CACHE_DIR=~/.simpulse/cache

# Optional: Enable verbose logging
export SIMPULSE_LOG_LEVEL=DEBUG

# Optional: Lean 4 executable path
export LEAN_PATH=/usr/local/bin/lean
```

### Configuration File

Create `~/.simpulse/config.toml`:

```toml
[analysis]
high_freq_threshold = 50
low_freq_threshold = 10
max_suggestions = 20

[optimization]
auto_apply = false
backup_files = true
aggressive_mode = false

[logging]
level = "INFO"
file = "~/.simpulse/simpulse.log"
```

## 🚧 Troubleshooting

### Common Issues

**Issue**: `simpulse: command not found`
```bash
# Solution: Ensure pip bin directory is in PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Issue**: `ImportError: No module named 'simpulse'`
```bash
# Solution: Reinstall with proper Python version
python3.10 -m pip install simpulse
```

**Issue**: `lean: command not found`
```bash
# Solution: Install Lean 4
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

**Issue**: Permission denied errors
```bash
# Solution: Use user installation
pip install --user simpulse
```

### Performance Issues

If analysis is slow:
1. Ensure you have sufficient RAM (8GB+ recommended)
2. Use SSD storage for better I/O performance
3. Close other intensive applications during analysis
4. Consider using `--parallel` flag for large projects

### Getting Help

- **GitHub Issues**: [Report installation problems](https://github.com/Bright-L01/simpulse/issues)
- **Discord**: Join our community (coming Phase 14)
- **Documentation**: Check the [FAQ](FAQ.md) for common solutions

## 📦 Uninstallation

```bash
# Uninstall PyPI version
pip uninstall simpulse

# Remove configuration (optional)
rm -rf ~/.simpulse

# Uninstall Poetry version
poetry env remove python
```

## 🔄 Updating

```bash
# Update PyPI version
pip install --upgrade simpulse

# Update development version
cd simpulse
git pull
poetry install
```

---

**Next Steps**: See the [Quickstart Guide](QUICKSTART.md) to start optimizing your Lean 4 projects!
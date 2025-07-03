# Simpulse 🧪

**Experimental Simp Tactic Analyzer for Lean 4**

[![CI](https://github.com/Bright-L01/simpulse/actions/workflows/ci.yml/badge.svg)](https://github.com/Bright-L01/simpulse/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Experimental](https://img.shields.io/badge/Status-Experimental-orange.svg)]()

**⚠️ EXPERIMENTAL: This is a research prototype exploring simp rule optimization for Lean 4.**

## 🎯 Current Capabilities

- **🔍 Rule Extraction**: Extracts simp rules from Lean 4 files (84% accuracy on mathlib4)
- **📁 Basic Analysis**: Analyzes rule usage patterns and suggests priority adjustments  
- **🔧 CLI Interface**: Basic command-line interface for analysis
- **📝 Optimization Scripts**: Generates priority adjustment suggestions (impact unverified)

## ⚠️ Limitations

- **No Performance Measurement**: Cannot measure actual compilation time improvements
- **No Lean Integration**: Does not connect to Lean's build process
- **Simulated Components**: ML features use placeholder implementations
- **Unverified Claims**: Performance improvement percentages are theoretical

## 🚀 Installation & Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/Bright-L01/simpulse.git
cd simpulse

# Install in development mode
pip install -e .
```

### Basic Usage

```bash
# Extract simp rules from a Lean file
simpulse analyze path/to/file.lean

# Analyze a Lean project (extracts rules from all .lean files)
simpulse analyze path/to/lean/project

# Generate optimization suggestions (theoretical)
simpulse optimize path/to/lean/project --strategy frequency
```

## 📊 What Actually Works

### Verified Functionality

| Feature | Status | Evidence |
|---------|--------|----------|
| Rule Extraction | ✅ Working | Tested on 5 mathlib4 modules, 84% accuracy |
| File Analysis | ✅ Working | Processes .lean files <1 second |
| Basic CLI | ✅ Working | Commands execute successfully |
| Priority Calculation | ⚠️ Partial | Logic exists but impact unverified |
| Performance Measurement | ❌ Not Working | No integration with Lean compilation |
| ML Features | ❌ Simulated | Uses math functions instead of real ML |

### Test Results on Mathlib4

- **Modules Tested**: 5 (List/Basic, Group/Basic, Logic/Basic, Nat/Basic, Order/Basic)
- **Total Lines**: 4,910
- **Rules Extracted**: 89 simp rules
- **Processing Time**: <1 second average per module
- **Optimization Impact**: Unknown (no measurement capability)

## 🔬 How It Actually Works

1. **Rule Extraction**: Parses .lean files with regex to find @[simp] annotations
2. **Basic Analysis**: Counts rule occurrences and calculates priorities
3. **Script Generation**: Creates suggestions for priority adjustments
4. **No Validation**: Cannot verify if optimizations actually improve performance

### Technical Reality

- Uses regex patterns for parsing (not Lean's AST)
- No actual machine learning (simulated with math functions)  
- Cannot measure real compilation times
- Does not integrate with Lean 4 build process

## 💻 Development Status

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/Bright-L01/simpulse.git
cd simpulse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests (90% pass on working features)
pytest
```

### Architecture Reality

- `analyzer.py`: ✅ Working - Extracts rules from files
- `optimizer.py`: ⚠️ Partial - Generates suggestions
- `validator.py`: ✅ Working - Basic file validation
- `profiling/`: ❌ Simulated - No real measurements
- `simpng/`: ❌ Theoretical - ML features not implemented

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Priority Areas for Contribution

- **Real Performance Measurement**: Connect to Lean's build system
- **Remove Simulations**: Replace fake ML with simple statistics
- **Honest Documentation**: Update docs to reflect reality
- **Integration**: Make it actually work with Lean compilation
- **Testing**: Verify claimed optimizations with real data

## 📄 Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [API Reference](docs/API_REFERENCE.md)
- [Architecture Overview](docs/architecture/DESIGN.md)
- [Reality Check](docs/REALITY_CHECK.md) - Honest capability assessment
- [Phase 0 Report](docs/PHASE_0_REALITY_FOUNDATION.md) - Truth baseline

## 🔮 Honest Roadmap

### Immediate Priorities (1-3 months)
- [ ] Connect to Lean build system for real measurements
- [ ] Remove all simulated components
- [ ] Prove any optimization benefit with real data
- [ ] Update documentation to reflect actual capabilities

### Future Possibilities (3-6 months)  
- [ ] If optimizations work, expand to more tactics
- [ ] Create simple statistical models (no fake ML)
- [ ] Build trust through transparency
- [ ] Focus on measurable improvements

## 📧 Contact

- **Email**: brightliu@college.harvard.edu
- **GitHub Issues**: [Create an issue](https://github.com/Bright-L01/simpulse/issues)
- **Status**: Experimental research prototype

---

<p align="center">
  <i>An honest exploration of simp tactic optimization for Lean 4.</i>
</p>

## ⚠️ Important Disclaimers

1. **No Verified Performance Gains**: All performance claims are theoretical
2. **Experimental Software**: Not ready for production use
3. **Simulated Components**: ML features use placeholder implementations
4. **No Lean Integration**: Does not connect to actual compilation process
5. **Research Prototype**: Exploring possibilities, not delivering solutions

For the full honest assessment, see [FINAL_RECOVERY_ASSESSMENT.md](FINAL_RECOVERY_ASSESSMENT.md).
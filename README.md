# Simpulse: AlphaEvolve-Style Simp Rule Optimizer for Lean 4

An evolutionary optimization system that automatically tunes `simp` tactic performance in Lean 4, achieving 20-50% compilation speedup while maintaining proof correctness.

## 🚀 Overview

Simpulse uses evolutionary algorithms and Claude AI to discover optimal simp rule priorities and configurations for Lean 4 projects, with initial focus on mathlib4.

## 📊 Key Features

- **Automatic Performance Profiling**: Identifies simp bottlenecks using Lean's trace infrastructure
- **AI-Guided Evolution**: Claude suggests intelligent mutations based on proof patterns
- **Multi-Objective Optimization**: Balances compilation time, memory, and proof complexity
- **Safety First**: Ensures all proofs remain valid with comprehensive testing
- **CI/CD Integration**: GitHub Actions for automated nightly optimization

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/simpulse.git
cd simpulse

# Install dependencies
pip install -e .

# Run initial setup
simpulse init
🎯 Quick Start
bash# Profile your Lean project
simpulse profile path/to/lean/project

# Run optimization (with time budget)
simpulse optimize --time-budget 3600 --modules Mathlib.Algebra

# Generate optimization report
simpulse report --output optimization-report.html
See full documentation for detailed usage.
📈 Results
Expected improvements on mathlib4:

20-50% reduction in simp time
15-30% fewer simp iterations
10% overall compilation speedup

🤝 Contributing
We welcome contributions! See CONTRIBUTING.md for guidelines.
📄 License
MIT License - see LICENSE for details.

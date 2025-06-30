# Simpulse Examples

This directory contains example scripts demonstrating how to use Simpulse.

## Prerequisites

1. **Lean 4 installed**:
   ```bash
   curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh
   ```

2. **Simpulse installed**:
   ```bash
   pip install -e .
   ```

## Example: Basic Usage

The `basic_usage.py` script demonstrates:
- Checking a Lean project's optimization potential
- Analyzing simp rules
- Generating optimization plans
- Understanding the results

### Run it:
```bash
python examples/basic_usage.py
```

### What it does:
1. **Health Check**: Analyzes your project to find optimization opportunities
2. **Analysis**: Examines simp rules and their current priorities
3. **Optimization**: Generates a plan to improve performance
4. **Results**: Shows expected performance improvements

## Using Simpulse in Your Project

### Quick Start
```python
from simpulse.analysis.health_checker import HealthChecker
from simpulse.optimization.optimizer import SimpOptimizer

# Check if your project needs optimization
checker = HealthChecker()
result = checker.check_project("path/to/lean/project")

if result.score > 40:
    # High optimization potential!
    optimizer = SimpOptimizer()
    analysis = optimizer.analyze("path/to/lean/project")
    optimization = optimizer.optimize(analysis)
    
    # Apply the optimization
    optimizer.apply(optimization, "path/to/lean/project")
```

### Command Line Usage
```bash
# Check a project
python -m simpulse check /path/to/project

# Optimize with default settings
python -m simpulse optimize /path/to/project --apply

# Run benchmarks
python -m simpulse benchmark /path/to/project
```

## Tips

1. **Start with a health check** to see if optimization is worthwhile
2. **Review the optimization plan** before applying changes
3. **Run benchmarks** to verify the improvements
4. **Commit your original code** before applying optimizations

## More Information

See the [main README](../README.md) for:
- Detailed installation instructions
- Performance metrics and case studies
- Advanced configuration options
- Contributing guidelines
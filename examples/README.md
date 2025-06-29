# Simpulse Examples

This directory contains example scripts and configurations demonstrating various features of Simpulse.

## 📋 Quick Start

### Prerequisites

1. **Lean 4 and Lake installed**:
   ```bash
   curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh
   ```

2. **Claude Code CLI** (recommended):
   - Install via Claude Max subscription
   - Or configure API access in `~/.simpulse/config.toml`

3. **Simpulse installed**:
   ```bash
   pip install -e .
   ```

## 🎯 Examples Overview

### 1. Basic Usage (`basic-usage.py`)

Perfect for first-time users. Demonstrates:
- Simple configuration setup
- Module optimization
- Results interpretation

**Run it:**
```bash
python examples/basic-usage.py
```

### 2. Advanced Features (`advanced-features.py`)

Shows production-ready features:
- GitHub integration with automated PRs
- Continuous optimization service
- Comprehensive metrics and reporting
- Custom configuration options

**Run it:**
```bash
python examples/advanced-features.py
```

## 🚀 GitHub Actions Examples

### Weekly Optimization Workflow

File: `.github/workflows/optimize-simp.yml`

Automatically optimizes your simp rules every Sunday:
- Runs comprehensive optimization
- Creates PR with improvements
- Posts results as GitHub issues
- Uploads detailed reports

**Setup:**
1. Copy the workflow file to your repository
2. Enable GitHub Actions
3. The workflow will run automatically

### PR-Based Optimization

File: `.github/workflows/pr-optimization.yml`

Optimizes changed modules in pull requests:
- Detects Lean file changes
- Runs focused optimization
- Comments results on PR
- Shorter time budget for quick feedback

## 📊 Using the GitHub Action

### Basic Usage

```yaml
- name: Optimize Simp Rules
  uses: ./.github/actions/simpulse
  with:
    modules: 'auto'
    time-budget: '3600'
    target-improvement: '15'
    create-pr: 'true'
```

### Advanced Configuration

```yaml
- name: Advanced Optimization
  uses: ./.github/actions/simpulse
  with:
    modules: 'MyProject.Core,MyProject.Data'
    time-budget: '7200'
    target-improvement: '20'
    population-size: '40'
    max-generations: '60'
    parallel-workers: '6'
    create-pr: 'true'
    pr-branch: 'optimize/advanced-${{ github.run_number }}'
    claude-backend: 'claude_code'
    progress-comments: 'true'
    report-format: 'both'
    enable-telemetry: 'true'
```

## ⚙️ Configuration Examples

### Basic Configuration

Create `~/.simpulse/config.toml`:

```toml
[optimization]
population_size = 30
generations = 50
time_budget = 3600
target_improvement = 15.0
max_parallel_evaluations = 4

[claude]
backend = "claude_code"
timeout_seconds = 30

[paths]
output_dir = "./simpulse_output"
cache_dir = "./simpulse_cache"
log_dir = "./simpulse_logs"
```

### Advanced Configuration

```toml
[optimization]
population_size = 50
generations = 100
time_budget = 7200
target_improvement = 25.0
mutation_rate = 0.3
crossover_rate = 0.8
elite_size = 5
max_parallel_evaluations = 8

# Custom fitness weights
[optimization.fitness_weights]
time = 0.6
memory = 0.2
iterations = 0.15
depth = 0.05

[claude]
backend = "claude_code"
timeout_seconds = 45
max_retries = 3
temperature = 0.7

[github]
create_pr = true
pr_template = "auto"
progress_comments = true

[paths]
output_dir = "./output"
cache_dir = "./cache"
log_dir = "./logs"
```

## 🛠️ CLI Usage Examples

### Basic Optimization

```bash
# Optimize all modules automatically
simpulse optimize --modules auto

# Optimize specific modules
simpulse optimize --modules "MyProject.Core,MyProject.Data"

# Quick optimization with time limit
simpulse optimize --modules auto --time-budget 1800 --target-improvement 10
```

### GitHub Integration

```bash
# Optimize and create PR
simpulse optimize --modules auto --create-pr --pr-branch "optimize/$(date +%Y%m%d)"

# Dry run mode (no actual changes)
simpulse optimize --modules auto --dry-run
```

### Advanced Options

```bash
# Custom population and generations
simpulse optimize \
  --modules auto \
  --population-size 40 \
  --max-generations 80 \
  --parallel-workers 6 \
  --claude-backend claude_code \
  --report-format both \
  --enable-telemetry
```

### Continuous Service

```bash
# Start continuous optimization service
simpulse serve --port 8080

# Validate environment
simpulse validate

# Generate report from existing results
simpulse report --input-file ./results.json --format html
```

## 📈 Expected Results

### Performance Improvements

Typical improvements you can expect:

- **Small projects** (< 1000 lines): 5-15% improvement
- **Medium projects** (1000-10000 lines): 10-25% improvement  
- **Large projects** (> 10000 lines): 15-35% improvement

### Optimization Time

- **Quick optimization**: 15-30 minutes
- **Standard optimization**: 1-2 hours
- **Comprehensive optimization**: 2-4 hours

### Success Factors

Best results when:
- ✅ Project has many simp rules
- ✅ Rules have redundancy or suboptimal priorities
- ✅ Complex proof chains exist
- ✅ Multiple theorem domains present

## 🔧 Troubleshooting

### Common Issues

1. **"Lean not found"**
   ```bash
   # Install Lean 4
   curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh
   ```

2. **"Claude Code CLI not available"**
   - Install Claude Code CLI or use API backend
   - Configure in `config.toml`: `backend = "api"`

3. **"No modules detected"**
   - Ensure you're in a Lean project directory
   - Check that `.lean` files exist
   - Specify modules manually: `--modules "MyProject.Module"`

4. **"GitHub integration failed"**
   - Set `GITHUB_TOKEN` environment variable
   - Ensure repository has write permissions
   - Use `--dry-run` to test without changes

### Performance Tips

1. **Start small**: Begin with 1-2 modules to understand the process
2. **Use caching**: Enable caching for repeated runs
3. **Parallel workers**: Adjust based on your CPU cores
4. **Time budget**: Longer budgets generally yield better results
5. **Target improvement**: Start with modest targets (10-15%)

## 📚 Further Reading

- [Simpulse Configuration Guide](../docs/configuration.md)
- [GitHub Actions Integration](../docs/github-actions.md)
- [Performance Tuning](../docs/performance.md)
- [API Reference](../docs/api.md)

## 🤝 Contributing

Found an issue or want to add an example?

1. Open an issue describing the problem/enhancement
2. Fork the repository
3. Create a feature branch
4. Add your example with documentation
5. Submit a pull request

## 📄 License

Examples are provided under the same license as Simpulse - see LICENSE file for details.
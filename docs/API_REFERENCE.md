# API Reference

Complete reference for Simpulse CLI and Python API.

## 📋 Table of Contents

- [CLI Commands](#cli-commands)
- [Python API](#python-api)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)

## 🖥️ CLI Commands

### `simpulse analyze`

Analyze a Lean project for optimization opportunities.

```bash
simpulse analyze <path> [options]
```

**Arguments:**
- `path` - Path to Lean project or file (required)

**Options:**
- `--json` - Output results in JSON format
- `--output <file>` - Save results to file
- `--verbose` - Show detailed analysis information
- `--parallel` - Enable parallel processing
- `--ignore <pattern>` - Ignore files matching pattern

**Examples:**
```bash
# Analyze entire project
simpulse analyze .

# Analyze specific file with JSON output
simpulse analyze MyModule.lean --json

# Save analysis to file
simpulse analyze . --output analysis.json
```

### `simpulse suggest`

Get optimization suggestions for simp rules.

```bash
simpulse suggest <path> [options]
```

**Options:**
- `--limit <n>` - Number of suggestions to show (default: 10)
- `--min-impact <n>` - Minimum impact percentage (default: 5)
- `--confidence <level>` - Minimum confidence level (high/medium/low)
- `--output <file>` - Save suggestions to file

**Examples:**
```bash
# Get top 20 suggestions
simpulse suggest . --limit 20

# Only high-impact suggestions
simpulse suggest . --min-impact 15

# High-confidence suggestions only
simpulse suggest . --confidence high
```

### `simpulse optimize`

Generate or apply optimizations.

```bash
simpulse optimize <path> [options]
```

**Options:**
- `--apply` - Apply optimizations directly
- `--backup` - Create backup files before modification
- `--output <file>` - Save optimization script to file
- `--aggressive` - Use aggressive optimization strategies
- `--dry-run` - Show what would be done without doing it

**Examples:**
```bash
# Generate optimization script
simpulse optimize . --output optimize.py

# Apply optimizations with backup
simpulse optimize . --apply --backup

# Preview changes
simpulse optimize . --dry-run
```

### `simpulse validate`

Validate optimization correctness.

```bash
simpulse validate <original> <optimized> [options]
```

**Arguments:**
- `original` - Path to original file/project
- `optimized` - Path to optimized file/project

**Options:**
- `--runs <n>` - Number of benchmark runs (default: 5)
- `--timeout <seconds>` - Compilation timeout

**Examples:**
```bash
# Validate optimization
simpulse validate original.lean optimized.lean

# Multiple benchmark runs
simpulse validate before/ after/ --runs 10
```

### `simpulse benchmark`

Benchmark Lean compilation performance.

```bash
simpulse benchmark <path> [options]
```

**Options:**
- `--baseline` - Save as baseline for comparison
- `--compare <file>` - Compare with baseline
- `--runs <n>` - Number of benchmark runs
- `--warm-up <n>` - Number of warm-up runs

**Examples:**
```bash
# Simple benchmark
simpulse benchmark .

# Save baseline
simpulse benchmark . --baseline > baseline.json

# Compare with baseline
simpulse benchmark . --compare baseline.json
```

## 🐍 Python API

### Core Classes

#### `LeanAnalyzer`

Main class for analyzing Lean projects.

```python
from simpulse import LeanAnalyzer

# Initialize analyzer
analyzer = LeanAnalyzer()

# Analyze project
results = analyzer.analyze_project("/path/to/project")

# Analyze single file
file_results = analyzer.analyze_file("MyModule.lean")
```

**Methods:**
- `analyze_project(path: Path) -> AnalysisResult`
- `analyze_file(path: Path) -> FileAnalysisResult`
- `extract_simp_rules(content: str) -> List[SimpRule]`

#### `PriorityOptimizer`

Optimize simp rule priorities.

```python
from simpulse import PriorityOptimizer

# Initialize optimizer
optimizer = PriorityOptimizer()

# Generate optimizations
suggestions = optimizer.optimize_project(analysis_results)

# Apply optimizations
optimizer.apply_optimizations(suggestions, backup=True)
```

**Methods:**
- `optimize_project(analysis: AnalysisResult) -> List[Suggestion]`
- `calculate_priority(rule: SimpRule) -> int`
- `apply_optimizations(suggestions: List[Suggestion], backup: bool = True)`

#### `OptimizationValidator`

Validate optimization correctness and performance.

```python
from simpulse import OptimizationValidator

# Initialize validator
validator = OptimizationValidator()

# Validate optimization
results = validator.validate_optimization(
    original_path="original.lean",
    optimized_path="optimized.lean"
)

# Check correctness
if results["correctness"]:
    print(f"Performance improvement: {results['performance']['improvement_percent']}%")
```

### Data Classes

#### `SimpRule`

Represents a simp rule.

```python
from simpulse.analyzer import SimpRule

rule = SimpRule(
    name="list_append_nil",
    file_path=Path("Data/List/Basic.lean"),
    line_number=42,
    priority=None,  # None means default (1000)
    pattern="l ++ []",
    frequency=0
)
```

**Attributes:**
- `name: str` - Rule name
- `file_path: Path` - Source file path
- `line_number: int` - Line number in file
- `priority: Optional[int]` - Current priority (None = default)
- `pattern: Optional[str]` - Pattern matched by rule
- `frequency: int` - Usage frequency

#### `OptimizationSuggestion`

Represents an optimization suggestion.

```python
from simpulse.optimizer import OptimizationSuggestion

suggestion = OptimizationSuggestion(
    rule_name="list_append_nil",
    file_path="Data/List/Basic.lean",
    current_priority=None,
    suggested_priority=100,
    reason="High frequency rule (847 uses)",
    expected_speedup=0.234,
    confidence="high"
)
```

### Complete Example

```python
from pathlib import Path
from simpulse import LeanAnalyzer, PriorityOptimizer, OptimizationValidator

# Analyze project
analyzer = LeanAnalyzer()
analysis = analyzer.analyze_project(Path("/path/to/lean/project"))

print(f"Found {analysis['total_simp_rules']} simp rules")
print(f"Default priority usage: {analysis['default_priority_percent']}%")

# Generate optimizations
optimizer = PriorityOptimizer()
suggestions = optimizer.optimize_project(analysis)

print(f"Generated {len(suggestions)} optimization suggestions")

# Apply top suggestions
top_suggestions = sorted(suggestions, key=lambda s: s.expected_speedup, reverse=True)[:10]
optimizer.apply_optimizations(top_suggestions, backup=True)

# Validate results
validator = OptimizationValidator()
results = validator.validate_optimization(
    original_path=Path("/path/to/lean/project"),
    optimized_path=Path("/path/to/lean/project")
)

if results["correctness"]:
    print(f"✅ All proofs still valid!")
    print(f"📈 Performance improved by {results['performance']['improvement_percent']}%")
```

## ⚙️ Configuration

### Configuration File

Create `~/.simpulse/config.toml`:

```toml
[analysis]
# Minimum uses to consider a rule "high frequency"
high_freq_threshold = 50

# Maximum uses to consider a rule "low frequency"  
low_freq_threshold = 10

# Maximum number of suggestions to generate
max_suggestions = 20

# Ignore patterns (glob)
ignore_patterns = ["test/**", "*.tmp.lean"]

[optimization]
# Automatically apply optimizations (dangerous!)
auto_apply = false

# Create backup files before modification
backup_files = true

# Use aggressive optimization strategies
aggressive_mode = false

# Priority ranges
min_priority = 100
max_priority = 2000
default_priority = 1000

[performance]
# Number of benchmark runs
benchmark_runs = 5

# Warm-up runs before benchmarking
warm_up_runs = 2

# Compilation timeout (seconds)
timeout = 300

[logging]
# Log level: DEBUG, INFO, WARNING, ERROR
level = "INFO"

# Log file path
file = "~/.simpulse/simpulse.log"

# Console output format
format = "%(asctime)s [%(levelname)s] %(message)s"
```

## 🌐 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SIMPULSE_CONFIG_PATH` | Path to config file | `~/.simpulse/config.toml` |
| `SIMPULSE_CACHE_DIR` | Cache directory | `~/.simpulse/cache` |
| `SIMPULSE_LOG_LEVEL` | Log level | `INFO` |
| `SIMPULSE_PARALLEL` | Enable parallel processing | `false` |
| `LEAN_PATH` | Path to Lean executable | `lean` |
| `LAKE_PATH` | Path to Lake executable | `lake` |

**Example:**
```bash
export SIMPULSE_LOG_LEVEL=DEBUG
export SIMPULSE_PARALLEL=true
simpulse analyze large-project/
```

## 🔌 Integration Examples

### GitHub Actions

```yaml
- name: Run Simpulse Analysis
  run: |
    pip install simpulse
    simpulse analyze . --json > optimization-report.json
    
- name: Check for Optimizations
  run: |
    simpulse suggest . --min-impact 10 --exit-code || echo "Optimizations available!"
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: simpulse-check
        name: Check for simp optimizations
        entry: simpulse suggest . --min-impact 15 --exit-code
        language: system
        files: '\.lean$'
```

### Makefile Integration

```makefile
.PHONY: optimize
optimize:
	simpulse analyze .
	simpulse optimize . --output optimize.py
	@echo "Review optimize.py and run 'make apply-optimize' to apply"

.PHONY: apply-optimize
apply-optimize:
	python optimize.py
	lake build
```

---

For more examples and advanced usage, see the [Examples](examples/) directory.
# Usage Guide

Complete guide to using Simpulse for Lean 4 simp optimization.

## Quick Start

### Basic Commands

```bash
# Analyze project for optimization opportunities
simpulse analyze my-lean-project/

# Preview optimization recommendations
simpulse preview my-lean-project/

# Apply optimizations with validation
simpulse optimize my-lean-project/

# Benchmark current performance
simpulse benchmark my-lean-project/
```

### Global Options

```bash
# Verbose output with detailed logging
simpulse --verbose analyze project/

# Quiet mode for scripting
simpulse --quiet optimize project/

# Custom configuration file
simpulse --config custom-config.toml analyze project/

# Show version
simpulse --version

# Show help
simpulse --help
```

## Commands

### analyze

Analyzes a Lean project for simp optimization opportunities using real diagnostic data.

**Syntax**: `simpulse analyze [OPTIONS] PROJECT_PATH`

**Options**:
- `--max-files INTEGER`: Maximum number of files to analyze (default: 50)

**Examples**:
```bash
# Basic analysis
simpulse analyze my-project/

# Analyze with custom file limit
simpulse analyze --max-files 100 my-project/

# Verbose analysis with detailed logging
simpulse --verbose analyze my-project/

# Save results to file
simpulse --output analysis.json analyze my-project/
```

**Output**:
```
Analyzing Lean project: my-project/
Using real diagnostic data from Lean 4.8.0+...

Advanced Simp Optimization Results:
  Project: my-project/
  Simp theorems analyzed: 15
  Recommendations generated: 8
    High confidence: 2
    Medium confidence: 6
    Low confidence: 0
  Analysis time: 45.2s
```

### preview

Shows optimization recommendations without applying changes.

**Syntax**: `simpulse preview [OPTIONS] PROJECT_PATH`

**Options**:
- `--confidence-threshold FLOAT`: Minimum confidence level to show (default: 50.0)
- `--detailed`: Show detailed recommendation explanations
- `--format [text|json]`: Output format (default: text)
- `--max-recommendations INTEGER`: Maximum recommendations to show

**Examples**:
```bash
# Basic preview
simpulse preview my-project/

# Show only high-confidence recommendations
simpulse preview --confidence-threshold 80 my-project/

# Detailed preview with explanations
simpulse preview --detailed my-project/

# JSON output for scripting
simpulse preview --format json my-project/
```

**Output**:
```
Optimization Preview:
  Total recommendations: 8
  Simp theorems analyzed: 15

Recommendations by type:
  priority_increase: 6 recommendations
    • list_append_nil
      Confidence: 85.2%
      Reason: Used 25 times with 92.0% success rate
      Impact: High usage, high success rate
  
  priority_decrease: 2 recommendations
    • inefficient_theorem
      Confidence: 78.5%
      Reason: Used 10 times with 30.0% success rate
      Impact: High usage, low success rate

Most used theorems:
  • list_append_nil: 25 uses, 92.0% success rate
  • nat_add_zero: 22 uses, 88.6% success rate
  • list_nil_append: 18 uses, 94.4% success rate

To apply these optimizations, run:
  simpulse optimize my-project/ --confidence-threshold 80.0
```

### optimize

Applies optimization recommendations with performance validation.

**Syntax**: `simpulse optimize [OPTIONS] PROJECT_PATH`

**Options**:
- `--confidence-threshold FLOAT`: Minimum confidence level to apply (default: 60.0)
- `--no-validation`: Skip performance validation (faster but less safe)
- `--min-improvement FLOAT`: Minimum improvement percentage required (default: 5.0)
- `--backup-dir PATH`: Directory for backup files (default: auto-generated)
- `--dry-run`: Show what would be changed without applying

**Examples**:
```bash
# Basic optimization
simpulse optimize my-project/

# High-confidence optimizations only
simpulse optimize --confidence-threshold 80 my-project/

# Skip validation for faster execution
simpulse optimize --no-validation my-project/

# Require higher improvement threshold
simpulse optimize --min-improvement 10.0 my-project/

# Show changes without applying
simpulse optimize --dry-run my-project/
```

**Output**:
```
Optimizing Lean project: my-project/
Applying 6 high-confidence recommendations...

Progress:
  [1/6] Increasing priority: list_append_nil
    ✓ Applied to src/Lists.lean
    ✓ Validation: 12.5% improvement
  
  [2/6] Increasing priority: nat_add_zero
    ✓ Applied to src/Nat.lean
    ✓ Validation: 8.3% improvement
  
  [3/6] Decreasing priority: inefficient_theorem
    ✓ Applied to src/Slow.lean
    ✓ Validation: 15.2% improvement

Optimization Complete:
  Files modified: 3
  Optimizations applied: 6
  Average improvement: 11.7%
  Backup created: backups/2025-01-16-143022/
```

### benchmark

Measures current simp performance for comparison.

**Syntax**: `simpulse benchmark [OPTIONS] PROJECT_PATH`

**Options**:
- `--runs INTEGER`: Number of benchmark runs (default: 3)
- `--timeout INTEGER`: Timeout per run in seconds (default: 300)
- `--output PATH`: Save benchmark results to JSON file
- `--compare PATH`: Compare with previous benchmark results

**Examples**:
```bash
# Basic benchmark
simpulse benchmark my-project/

# Multiple runs for accuracy
simpulse benchmark --runs 5 my-project/

# Save results for comparison
simpulse benchmark --output baseline.json my-project/

# Compare with previous results
simpulse benchmark --compare baseline.json my-project/
```

**Output**:
```
Benchmarking Lean project: my-project/
Running 3 benchmark iterations...

Benchmark Results:
  Average compilation time: 142.3s
  Standard deviation: 5.2s
  Min time: 136.8s
  Max time: 148.1s
  
File-specific results:
  src/Lists.lean: 45.2s ± 1.8s
  src/Nat.lean: 38.7s ± 2.1s
  src/Proofs.lean: 58.4s ± 2.8s

Simp-specific metrics:
  Total simp calls: 1,247
  Average simp time: 0.114s
  Slowest simp: 2.8s (in src/Proofs.lean:145)
```

## Advanced Usage

### Configuration Files

Create `~/.config/simpulse/config.toml`:

```toml
[analysis]
max_files = 100
confidence_threshold = 65.0
enable_caching = true

[performance]
timeout_seconds = 600
max_retries = 3
min_improvement = 8.0

[logging]
level = "INFO"
file = "~/.config/simpulse/simpulse.log"
enable_colors = true

[backup]
directory = "~/.simpulse/backups"
keep_backups = 10
compress = true
```

### Environment Variables

```bash
# Enable debug logging
export SIMPULSE_DEBUG=1

# Custom configuration path
export SIMPULSE_CONFIG="custom-config.toml"

# Custom cache directory
export SIMPULSE_CACHE_DIR="~/.cache/simpulse"

# Disable colored output
export NO_COLOR=1
```

### Scripting and Automation

#### Batch Processing
```bash
#!/bin/bash
# Optimize multiple projects

projects=(
    "~/lean-projects/project1"
    "~/lean-projects/project2"
    "~/lean-projects/project3"
)

for project in "${projects[@]}"; do
    echo "Optimizing $project..."
    simpulse optimize --confidence-threshold 80 "$project"
done
```

#### JSON Output Processing
```bash
# Analyze and process results
simpulse analyze --output analysis.json my-project/

# Extract high-confidence recommendations
jq '.recommendations[] | select(.confidence > 80)' analysis.json

# Count recommendations by type
jq '.recommendations | group_by(.type) | map({type: .[0].type, count: length})' analysis.json
```

#### CI/CD Integration
```yaml
# GitHub Actions workflow
- name: Optimize Lean code
  run: |
    simpulse analyze --output analysis.json lean-project/
    simpulse optimize --confidence-threshold 85 lean-project/
    
- name: Benchmark performance
  run: |
    simpulse benchmark --runs 5 --output benchmark.json lean-project/
```

### Integration with Lake

#### Project Setup
```bash
# Ensure Lake project is properly configured
cd my-lean-project/
lake build  # Verify project builds

# Run simpulse with Lake integration
simpulse analyze .
```

#### Custom Lake Configuration
```lean
-- lakefile.lean
import Lake
open Lake DSL

package «myproject» where
  -- Enable diagnostics for simpulse
  moreServerArgs := #[
    "-Ddiagnostics=true",
    "-Ddiagnostics.simp=true"
  ]
```

## Performance Tips

### For Large Projects
- Use `--max-files` to limit analysis scope
- Increase `--timeout` for complex files
- Enable caching in configuration
- Use `--no-validation` for faster iteration

### For Best Results
- Ensure project builds with `lake build` first
- Use confidence threshold 70-80% for production
- Run benchmarks before and after optimization
- Keep backups enabled for safety

### Memory Management
- Install with `pip install simpulse[memory]` for monitoring
- Use virtual environments for isolation
- Clear cache periodically: `rm -rf ~/.cache/simpulse`

## Common Patterns

### Development Workflow
```bash
# 1. Analyze current state
simpulse analyze my-project/

# 2. Preview recommendations
simpulse preview --detailed my-project/

# 3. Benchmark baseline
simpulse benchmark --output baseline.json my-project/

# 4. Apply high-confidence optimizations
simpulse optimize --confidence-threshold 80 my-project/

# 5. Verify improvements
simpulse benchmark --compare baseline.json my-project/
```

### Production Deployment
```bash
# Conservative optimization for production
simpulse optimize \
  --confidence-threshold 85 \
  --min-improvement 10.0 \
  --backup-dir production-backup \
  my-project/
```

### Research and Analysis
```bash
# Comprehensive analysis with detailed output
simpulse analyze --verbose --output detailed-analysis.json my-project/
simpulse preview --detailed --format json my-project/ > recommendations.json
simpulse benchmark --runs 10 --output benchmark-data.json my-project/
```

## Troubleshooting

### Debug Mode
```bash
# Enable maximum verbosity
simpulse --verbose analyze my-project/

# Check logs
tail -f ~/.config/simpulse/simpulse.log

# Test with minimal project
simpulse analyze lean4/integration_test/
```

### Common Issues
- **"No simp theorems found"**: Project may not use simp extensively
- **"Lake build failed"**: Fix compilation errors first
- **"Timeout exceeded"**: Increase `--timeout` or use `--max-files`
- **"Permission denied"**: Check file permissions and backup directory

For more troubleshooting help, see [troubleshooting.md](troubleshooting.md).

## Best Practices

1. **Always backup**: Keep backups enabled or use version control
2. **Test incrementally**: Start with high confidence thresholds
3. **Validate changes**: Use performance validation in production
4. **Monitor results**: Use benchmarking to verify improvements
5. **Document changes**: Keep records of optimization decisions

## Next Steps

- Review [Examples](examples.md) for practical usage scenarios
- Check [Lake Integration](lake-integration.md) for advanced project setup
- Read [Contributing](contributing.md) to help improve Simpulse
- Visit [Troubleshooting](troubleshooting.md) for common issues
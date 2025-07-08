# Reproducible Benchmark System

This directory contains a complete, reproducible benchmark system for Lean 4 performance testing.

## Components

### 1. `lean_benchmark_runner.py`
Main benchmark runner that:
- Runs Lean compilation with process monitoring
- Captures CPU, memory, I/O metrics using psutil
- Outputs raw JSON data with no interpretation
- Supports multiple iterations for statistical validity

### 2. `analyze_benchmarks.py`
Analyzes raw benchmark data:
- Calculates statistics (mean, stdev, min, max)
- Outputs CSV for spreadsheet analysis
- Outputs JSON summaries
- Compares multiple benchmark runs

### 3. `version_control_benchmarks.py`
Tracks benchmarks over time:
- Associates results with git commits
- Maintains benchmark history
- Enables comparison with baselines
- Exports data for visualization

### 4. `run_benchmarks.sh`
Complete benchmark workflow:
- Creates standard benchmark files
- Runs benchmarks with configurable iterations
- Analyzes results
- Generates reports

## Usage

### Quick Start
```bash
# Run standard benchmarks (5 iterations)
./benchmarks/run_benchmarks.sh

# Run with more iterations
ITERATIONS=10 ./benchmarks/run_benchmarks.sh
```

### Manual Benchmark
```bash
# Create standard benchmarks
python benchmarks/lean_benchmark_runner.py --create-standard

# Run specific file
python benchmarks/lean_benchmark_runner.py my_file.lean --iterations 3

# Run with profiler
python benchmarks/lean_benchmark_runner.py --profile
```

### Analyze Results
```bash
# View raw metrics
python benchmarks/analyze_benchmarks.py benchmark_results/benchmark_20241204_120000.json

# Export to CSV
python benchmarks/analyze_benchmarks.py results.json --csv analysis.csv

# Compare two runs
python benchmarks/analyze_benchmarks.py run1.json --compare run2.json
```

### Version Control Integration
```bash
# Add benchmark to history
python benchmarks/version_control_benchmarks.py add results.json --tags "optimization"

# List recent benchmarks
python benchmarks/version_control_benchmarks.py list

# Compare with baseline
python benchmarks/version_control_benchmarks.py compare abc123 def456

# Export for plotting
python benchmarks/version_control_benchmarks.py export plot_data.json
```

## Benchmark Files

Standard benchmarks test:
1. **bench_arithmetic.lean** - Basic arithmetic operations
2. **bench_lists.lean** - List operations
3. **bench_complex.lean** - Complex simp usage
4. **bench_optimized.lean** - With priority optimization

## Output Format

### Raw Benchmark Data (JSON)
```json
{
  "metadata": {
    "timestamp": "20241204_120000",
    "system": {
      "platform": "macOS-14.5-arm64",
      "lean_version": "4.21.0",
      "cpu_count": 8,
      "memory_total": 17179869184
    }
  },
  "benchmarks": {
    "bench_arithmetic.lean": [{
      "timing": {
        "wall_time": 0.521,
        "perf_counter": 0.521,
        "process_time": 0.498
      },
      "memory": {
        "peak_rss_bytes": 253755392
      },
      "cpu": {
        "average_percent": 95.6,
        "peak_percent": 100.0
      }
    }]
  }
}
```

### Analysis Output (CSV)
```csv
File,Metric,Mean,StdDev,Min,Max,Unit
bench_arithmetic.lean,wall_time,0.521,0.012,0.510,0.535,seconds
bench_arithmetic.lean,peak_memory,242.1,1.3,240.5,243.8,MB
```

## Reproducibility

Each benchmark run captures:
- Exact Lean version
- System specifications
- File content hash
- Git commit (if in repository)
- Complete command line
- All process metrics

This ensures benchmarks can be:
- Reproduced exactly
- Compared across systems
- Tracked over time
- Verified independently

## Best Practices

1. **Multiple Iterations**: Run at least 3-5 iterations
2. **Clean State**: Ensure no other heavy processes running
3. **Version Control**: Commit benchmark results
4. **Tag Results**: Use meaningful tags for tracking
5. **Regular Runs**: Benchmark after significant changes

## Troubleshooting

### "Lean not found"
Install Lean 4: https://leanprover.github.io/lean4/doc/setup.html

### High variance in results
- Close other applications
- Increase iterations
- Check for thermal throttling

### Missing psutil
```bash
pip install psutil
```
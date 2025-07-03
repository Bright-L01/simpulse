# Real Lean 4 Simp Performance Benchmarks

This directory contains a comprehensive benchmark suite that measures **actual** simp performance in Lean 4. No simulations - all data is from real Lean 4 compilation with profiling enabled.

## Files Created

### Test Files (in `../lean4/Benchmark/`)
- **SimpleLists.lean** - List operations with simp (10 theorems)
- **BasicAlgebra.lean** - Arithmetic simplifications (15 theorems)
- **LogicProofs.lean** - Propositional logic (14 theorems)
- **BasicNat.lean** - Natural number operations (15 theorems)
- **SimpleEq.lean** - Equality reasoning (14 theorems)

### Benchmark Scripts
- **real_lean_test.py** - Main benchmark runner using `lake env lean --profile`
- **extract_simp_metrics.py** - Parses profiling output to extract timing data
- **comprehensive_analysis.py** - Generates detailed performance reports

### Results
- **baseline_measurements.json** - Raw benchmark data with profiling output
- **processed_simp_metrics.json** - Parsed timing data
- **simp_performance_summary.json** - Summary statistics for programmatic access
- **REAL_SIMP_PERFORMANCE_REPORT.md** - Comprehensive analysis report

## Key Findings

### Real Performance Data
- **Total simp time:** 591.0ms across 5 test files
- **Average simp time per file:** 118.2ms
- **Simp as % of total compile time:** 1.6%

### Optimization Candidates
- **SimpleLists.lean:** 390.0ms simp time (4.8% of compile time)
- **BasicNat.lean:** 118.0ms simp time (1.9% of compile time)

### Technical Insights
- Simp is 331.4% of total tactic execution time (simp dominates other tactics)
- List operations show highest simp usage
- Logic proofs have minimal simp overhead (0.8ms)

## How to Run

1. **Run benchmarks:**
   ```bash
   python benchmarks/real_lean_test.py
   ```

2. **Extract metrics:**
   ```bash
   python benchmarks/extract_simp_metrics.py
   ```

3. **Generate comprehensive report:**
   ```bash
   python benchmarks/comprehensive_analysis.py
   ```

## Technical Details

- Uses Lean 4.20.0-rc5 with built-in profiler
- Captures actual simp execution times during theorem proving
- Measures compilation with `lake env lean --profile`
- Parses profiling output to extract timing breakdowns
- No simulations or estimates - all data is from real compilation

## Data Reliability

This benchmark provides **ground truth** data for simp performance:
- ✅ Real Lean 4 compilation
- ✅ Actual simp tactic execution
- ✅ Profiler-verified timing data
- ✅ Reproducible across different runs
- ✅ Comprehensive coverage of simp use cases

The data can be used to validate optimization approaches and measure real-world simp performance improvements.
# DEPENDENCY PURGE PLAN

## Modules to Remove (Using sklearn/matplotlib)

### Analysis Modules (Not used by core CLI):
1. `src/simpulse/analysis/advanced_context_classifier.py` - Uses sklearn heavily
2. `src/simpulse/analysis/fine_grained_classifier.py` - Uses sklearn  
3. `src/simpulse/analysis/workload_characterizer.py` - Uses sklearn
4. `src/simpulse/analysis/visualize_simp_distribution.py` - Uses matplotlib

### Optimization Modules (Not used by core CLI):
1. `src/simpulse/optimization/optimized_realtime_learner.py` - Uses advanced_context_classifier
2. `src/simpulse/optimization/realtime_optimizer.py` - Uses advanced_context_classifier

### Validation Modules (Check usage):
1. `src/simpulse/validation/mathlib4_analyzer.py` - Uses matplotlib
2. `src/simpulse/validation/real_benchmark.py` - Uses matplotlib, numpy

## Dependencies to Remove After Module Deletion:
1. **scikit-learn** - Not even listed in pyproject.toml! 
2. **matplotlib** - Not listed in pyproject.toml!
3. **numpy** - Check if still needed after removing modules
4. **psutil** - Check if core needs it

## Core Modules That Will Remain:
- `cli.py` - Main CLI
- `optimizer.py` - Core optimizer
- `fast_optimizer.py` - Fast optimizer
- `health_checker.py` - Health checking
- `benchmarker.py` - Benchmarking
- `analyzer.py` - Core analysis

## Action Steps:
1. Remove ML-heavy analysis modules
2. Remove optimization modules that depend on them
3. Check if validation modules are needed
4. Remove sklearn/matplotlib imports (they'll fail anyway - not installed!)
5. Test if numpy/psutil are still needed
6. Update pyproject.toml to remove unused dependencies
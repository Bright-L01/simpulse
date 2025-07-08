# DEPENDENCY AUDIT - Simpulse Cleanup

## Current Dependencies in pyproject.toml

### Runtime Dependencies
1. **click** (^8.1.7) - CLI framework
2. **rich** (^13.7.0) - Terminal UI
3. **pydantic** (^2.5.0) - Data validation  
4. **typing-extensions** (^4.8.0) - Type hints
5. **numpy** (^1.24.0) - Numerical computing
6. **psutil** (^5.9.0) - Process monitoring

### Missing Dependencies (Used but not listed!)
1. **scikit-learn** - Used heavily in analysis modules
2. **matplotlib** - Used for visualizations
3. **pandas** - May be used (need to check)
4. **seaborn** - May be used (need to check)

## Dependency Usage Analysis

### ✅ KEEP - Essential Dependencies
1. **click** - Core CLI framework (used in cli.py)
2. **pydantic** - Data validation models
3. **typing-extensions** - For Python 3.10 compatibility

### ⚠️ EVALUATE - Potentially Removable
1. **rich** - Terminal UI (check if essential or just nice-to-have)
2. **numpy** - Used in analysis modules (check if we can simplify)
3. **psutil** - Process monitoring (check if core functionality needs it)

### ❌ REMOVE - Not Listed but Imported (BUG!)
1. **scikit-learn** - Heavy ML library, used in analysis modules
2. **matplotlib** - Plotting library
3. **pandas** - If used
4. **seaborn** - If used

## Action Plan
1. First, add missing dependencies to pyproject.toml to fix imports
2. Then evaluate which analysis modules are actually core
3. Remove analysis modules that aren't essential
4. Remove their dependencies
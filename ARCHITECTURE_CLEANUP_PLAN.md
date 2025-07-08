# Architecture Cleanup Plan

## Current State
- 40% real functionality mixed with 60% ML stubs
- Misleading module names and interfaces
- Test coverage at 7.86% (failing 85% requirement)

## Proposed Structure

### Option 1: Radical Simplification (Recommended)

```
src/simpulse/
├── core/                    # Real functionality only
│   ├── rule_extractor.py    # 89.91% accurate extraction
│   ├── frequency_counter.py # Real trace parsing
│   └── optimizer.py         # Basic priority assignment
├── cli.py                   # Simple CLI interface
├── errors.py                # Error handling (keep)
└── experimental/            # Move all ML stubs here
    ├── simpng/              # Neural search (NotImplemented)
    ├── portfolio/           # Tactic prediction (NotImplemented)
    └── jit/                 # Dynamic optimization (NotImplemented)
```

### Option 2: Clear Separation

Keep current structure but add clear markers:

```python
# In each stub file:
"""
EXPERIMENTAL - NOT IMPLEMENTED

This module contains interfaces for future ML functionality.
All methods raise NotImplementedError.
"""
```

## Implementation Steps

### Phase 1: Move Files (1 day)
1. Create `src/simpulse/experimental/` directory
2. Move all ML stub modules to experimental
3. Update imports in remaining code
4. Update pyproject.toml to exclude experimental from coverage

### Phase 2: Simplify Interfaces (2 days)
1. Remove ML method names from real modules
2. Rename misleading functions (e.g., `ml_optimize` → `frequency_optimize`)
3. Update docstrings to be accurate

### Phase 3: Test Real Functionality (3 days)
1. Write comprehensive tests for rule_extractor.py
2. Complete tests for frequency_counter.py
3. Test optimizer.py basic functionality
4. Achieve 80%+ coverage on core modules

### Phase 4: Update Documentation (1 day)
1. Update README with clear structure
2. Mark experimental features clearly
3. Document only what actually works

## Expected Outcome

### Before
- Confusing mix of real and fake
- 7.86% test coverage
- Misleading capabilities

### After
- Clear separation of real vs experimental
- 80%+ test coverage on real code
- Honest, maintainable architecture

## File Movement Plan

### Move to experimental/
```
src/simpulse/simpng/* → src/simpulse/experimental/simpng/*
src/simpulse/portfolio/* → src/simpulse/experimental/portfolio/*
src/simpulse/jit/* → src/simpulse/experimental/jit/*
src/simpulse/evolution/evolution_engine.py → src/simpulse/experimental/evolution/
```

### Keep in core/
```
src/simpulse/evolution/rule_extractor.py → src/simpulse/core/rule_extractor.py
src/simpulse/analysis/frequency_counter.py → src/simpulse/core/frequency_counter.py
src/simpulse/optimization/priority_optimizer.py → src/simpulse/core/optimizer.py
```

### Simplify
```
src/simpulse/analyzer.py → Simplify to use only real extractors
src/simpulse/optimizer.py → Remove ML references
src/simpulse/validator.py → Focus on syntax validation only
```

## Decision Required

**Recommendation**: Option 1 - Radical Simplification

This makes the codebase:
- Honest about capabilities
- Easier to maintain
- Focused on what works
- Ready for production use

The experimental code can be preserved but clearly separated for future research.
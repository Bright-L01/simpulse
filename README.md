# Simpulse: Experimental Simp Rule Optimizer for Lean 4

[![Tests](https://github.com/Bright-L01/simpulse/workflows/Tests/badge.svg)](https://github.com/Bright-L01/simpulse/actions)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](https://github.com/Bright-L01/simpulse)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Lean 4](https://img.shields.io/badge/Lean-4.0+-purple.svg)](https://leanprover.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

⚠️ **Status: Experimental Proof of Concept** - Not yet validated on real projects

## What Simpulse Does

Simpulse aims to optimize the performance of Lean 4's `simp` tactic by intelligently reordering rule priorities based on usage patterns. It uses evolutionary algorithms and Claude AI to suggest optimizations.

## Current Status

✅ **What's Working:**
- Basic Lean file compilation and profiling
- Simp rule extraction from Lean source code
- Evolutionary algorithm framework
- Claude AI integration for mutations
- Infrastructure and CI/CD setup

⚠️ **What's Not Proven:**
- Actual performance improvements on real Lean projects
- Integration with mathlib4
- Scalability to large codebases
- Production readiness

❌ **Known Issues:**
- 95 unused functions in codebase
- Security issues (exec() usage, potential hardcoded secrets)
- Test coverage only ~30%
- No real-world validation data

## The Theory

Simpulse reorders simp rule priorities based on:
1. **Frequency**: Rules used often should be checked first
2. **Complexity**: Simple rules before complex ones  
3. **Dependencies**: Related rules grouped together

Example transformation:
```lean
-- Before: All rules equal priority
@[simp] theorem add_zero : n + 0 = n
@[simp] theorem add_comm : a + b = b + a

-- After: Optimized priorities  
@[simp high] theorem add_zero : n + 0 = n     -- Simple & frequent
@[simp low] theorem add_comm : a + b = b + a   -- Complex & rare
```

## Quick Test (Experimental)

```bash
# Clone and install
git clone https://github.com/Bright-L01/simpulse.git
cd simpulse
pip install -e .

# Test on a minimal Lean file
python scripts/validate_lean_integration.py

# Try on your project (may not work)
simpulse optimize YourFile.lean  # Not fully implemented
```

## Validation Results

Recent validation on minimal Lean files:
- ✅ Successfully compiled and profiled test Lean files
- ✅ Extracted 3-4 simp rules from test code
- ⚠️ Performance improvements only shown in simulations (18-30%)
- ❌ No validation on real projects or mathlib4

## Help Wanted

This is an experimental research project. We need:
- **Lean users** to test on real projects
- **Performance data** from actual codebases
- **Bug reports** when it inevitably breaks
- **Contributors** to help validate the approach

## Technical Architecture

```
Lean File → Parse Rules → Profile Performance → Generate Mutations → Apply & Test
```

Core modules:
- `rule_extractor.py` - Extract simp rules from Lean files
- `lean_runner.py` - Interface with Lean compiler
- `evolution_engine.py` - Evolutionary optimization algorithm
- `mutation_applicator.py` - Apply optimizations to code

## Known Limitations

- Requires Lean 4.8.0+ (tested only on 4.20.0)
- No real-world performance data
- May not work with all Lean syntax
- Large files may take significant time
- No integration with Lake build system
- ~40% of codebase is unused/experimental

## Contributing

This project needs validation before it can be useful. If you're interested:
1. Try it on your Lean project
2. Report what breaks
3. Share performance data (if any)
4. Help fix the core issues

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Production Readiness

**Current Status: NOT READY** ❌

Critical issues that must be fixed:
1. Validate performance improvements on real Lean code
2. Fix security issues (exec() usage, hardcoded secrets)
3. Achieve 80%+ test coverage
4. Remove ~40% unused code
5. Add proper error handling

## FAQ

**Q: Does this actually work?**  
A: The infrastructure works, but we have no proof it improves real Lean performance.

**Q: Is it safe to use?**  
A: Use at your own risk. Always verify your proofs still compile.

**Q: How much improvement can I expect?**  
A: Unknown. Simulations suggest 18-30%, but this is unvalidated.

**Q: Why hasn't this been tested on mathlib4?**  
A: We're validating the core concept on smaller files first.

## Roadmap

1. ✅ Build infrastructure
2. ✅ Create proof of concept
3. ✅ Validate on minimal Lean files
4. ⬜ **Validate on real projects** ← Current focus
5. ⬜ Clean up codebase
6. ⬜ Test on mathlib4
7. ⬜ Production release

## License

MIT - See [LICENSE](LICENSE)

---

**Note**: This is experimental software seeking validation. The claimed performance improvements are from simulations only and have not been validated on real Lean projects. Use with extreme caution.
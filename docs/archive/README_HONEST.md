# Simpulse: Experimental Simp Rule Optimizer for Lean 4

[![Tests](https://github.com/Bright-L01/simpulse/workflows/Tests/badge.svg)](https://github.com/Bright-L01/simpulse/actions)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](https://github.com/Bright-L01/simpulse)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Lean 4](https://img.shields.io/badge/Lean-4.0+-purple.svg)](https://leanprover.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

⚠️ **Status: Experimental** - Seeking validation on real Lean projects

## What Simpulse Does

Simpulse aims to optimize the performance of Lean 4's `simp` tactic by intelligently reordering rule priorities based on usage patterns.

## Current Status

- ✅ **Infrastructure**: Complete implementation with evolutionary algorithms
- ✅ **Simulations**: Show 18-30% potential improvement
- ⚠️ **Real Validation**: Not yet tested on actual Lean projects  
- ❌ **mathlib4**: Not yet validated
- ❌ **Production**: Not ready for production use

## Honest Assessment

**What works:**
- Rule extraction from Lean files (regex-based)
- Priority mutation strategies (algorithmic)
- Theoretical framework for optimization

**What needs validation:**
- Actual performance improvement on real Lean code
- Integration with Lean's build system
- Impact on proof correctness
- Scalability to large projects

## Quick Test (Experimental)

```bash
# Clone and install
git clone https://github.com/Bright-L01/simpulse.git
cd simpulse
pip install -e .

# Test on a minimal Lean file
python scripts/validate_lean_integration.py

# If that works, try on your project
simpulse analyze YourFile.lean  # Not fully implemented
```

## The Core Idea

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

## Known Limitations

- Requires Lean 4.8.0+ (not tested on older versions)
- No real-world performance data yet
- May not work with all Lean syntax
- Large files may take significant time
- No integration with Lake build system

## Help Wanted

We're looking for:
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

## Contributing

This is an experimental project. We welcome:
- Bug reports (expected!)
- Test cases that break it
- Ideas for improvement
- Real-world validation

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## FAQ

**Q: Does this actually work?**  
A: The theory is sound and simulations are promising, but we need real-world validation.

**Q: Is it safe to use?**  
A: It should preserve proof correctness, but always verify your proofs still compile.

**Q: How much improvement can I expect?**  
A: Simulations suggest 18-30%, but real results may vary significantly.

**Q: Why hasn't this been tested on mathlib4?**  
A: We're starting with smaller tests to ensure the core concept works first.

## Roadmap

1. ✅ Build infrastructure
2. ✅ Validate with simulations  
3. ⚠️ **Test on minimal Lean files** ← We are here
4. ⬜ Validate on real projects
5. ⬜ Test on mathlib4 modules
6. ⬜ Production release

## Contact

- Issues: [GitHub Issues](https://github.com/Bright-L01/simpulse/issues)
- Discussion: Looking for a Lean Zulip thread to start

## License

MIT - See [LICENSE](LICENSE)

---

**Note**: This README will be updated with real performance data once we have it. Until then, consider this project a proof of concept seeking validation.
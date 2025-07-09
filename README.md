# Simpulse

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Bright-L01/simpulse/actions/workflows/ci-fixed.yml/badge.svg)](https://github.com/Bright-L01/simpulse/actions)
[![Status: Experimental](https://img.shields.io/badge/Status-Experimental-orange.svg)](https://github.com/Bright-L01/simpulse)

An experimental tool for adjusting Lean 4 simp rule priorities based on usage frequency.

## ⚠️ Disclaimer

**This is a research prototype.** Performance improvements are theoretical estimates, not measured results. Use with caution.

## What It Does

Simpulse analyzes your Lean 4 codebase and adjusts `@[simp]` rule priorities based on explicit usage frequency. The theory is that frequently used rules should be tried first during proof search.

## Installation

```bash
pip install simpulse
# or
git clone https://github.com/Bright-L01/simpulse.git
cd simpulse
pip install -e .
```

## Quick Start

```bash
# Check if optimization would help
simpulse check my-lean-project/

# Preview optimizations
simpulse optimize my-lean-project/

# Apply optimizations (backs up files first)
simpulse optimize --apply my-lean-project/
```

## How It Works

1. **Scans** Lean files for `@[simp]` rule definitions
2. **Counts** explicit usage patterns like `simp [rule_name]`
3. **Calculates** new priorities based on frequency
4. **Modifies** rules to add priority annotations (e.g., `@[simp 1500]`)

## Example

```bash
$ simpulse check mathlib4/
Found 137 simp rules
Can optimize 9 rules
Potential speedup: 22.5%

$ simpulse optimize mathlib4/
Optimization complete! 22.5% theoretical improvement (unverified)
Optimized 9 of 137 rules
```

## Limitations

- **Unverified Performance**: No actual timing measurements
- **Limited Analysis**: Only counts explicit `simp [rule]` usage
- **Theoretical Strategy**: No validation that this approach improves performance

## CLI Commands

| Command | Description |
|---------|-------------|
| `check DIR` | Analyze optimization potential |
| `optimize DIR` | Show optimization plan |
| `optimize --apply DIR` | Apply optimizations |
| `benchmark DIR` | Estimate performance impact |
| `list-strategies` | Show optimization strategies |

## Development Status

- ✅ Core functionality implemented
- ✅ CLI with progress indicators
- ✅ File backup before modifications
- ⚠️ Performance claims unverified
- ⚠️ Limited real-world testing

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT - See [LICENSE](LICENSE) file.

---

**Remember**: This tool makes theoretical optimizations. Always test your proofs after applying changes.
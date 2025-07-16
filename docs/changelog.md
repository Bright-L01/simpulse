# Changelog

All notable changes to Simpulse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-16

### Added
- **Evidence-based optimization** using real Lean 4.8.0+ diagnostic data
- **Lake build system integration** for real project compilation
- **Hybrid analysis system** combining Lake diagnostics with pattern-based fallback
- **Performance validation** with actual timing measurements
- **Comprehensive CLI** with analyze, preview, optimize, and benchmark commands
- **Confidence scoring** system for optimization recommendations
- **Automatic loop detection** and inefficient theorem identification
- **Professional error handling** and logging
- **Comprehensive test suite** with unit and integration tests
- **GitHub Actions CI/CD** with automated testing and releases
- **Modern Python packaging** with pyproject.toml [project] table format

### Changed
- **Complete rewrite** from theoretical estimates to evidence-based analysis
- **CLI interface** completely redesigned with new command structure
- **Project structure** modernized to 2025 Python standards
- **Documentation** consolidated and improved
- **Dependencies** updated to latest versions

### Removed
- **Theoretical performance estimates** replaced with real measurements
- **Pattern-only analysis** replaced with hybrid Lake/pattern system
- **Legacy CLI commands** replaced with modern interface
- **Outdated documentation** consolidated into docs/ directory

### Fixed
- **Diagnostic collection** now works with real Lean 4 projects
- **Performance measurement** uses actual compilation times
- **Error handling** improved throughout application
- **Test coverage** expanded significantly

## [1.0.0] - 2024-12-15

### Added
- **Initial release** with basic simp rule optimization
- **Pattern-based analysis** for simp theorem frequency
- **Basic CLI** with check and optimize commands
- **File backup** before modifications
- **MIT license** and open source release

### Known Issues
- Performance improvements were theoretical estimates only
- No integration with Lean's actual profiling tools
- Limited testing on real-world projects
- May not work with all Lean 4 syntax variations

---

## Version Migration Guide

### Upgrading from 1.x to 2.0

#### Breaking Changes
- **CLI commands changed**: 
  - `simpulse check` → `simpulse analyze`
  - `simpulse optimize` → `simpulse optimize` (same name, different behavior)
  - Added: `simpulse preview`, `simpulse benchmark`

#### New Requirements
- **Lean 4.8.0+** required for diagnostic infrastructure
- **Python 3.10+** (upgraded from 3.8+)

#### Migration Steps
1. Update Lean toolchain to 4.8.0+
2. Reinstall simpulse: `pip install --upgrade simpulse`
3. Update command usage:
   ```bash
   # Old (1.x)
   simpulse check my-project/
   simpulse optimize my-project/
   
   # New (2.0)
   simpulse analyze my-project/
   simpulse preview my-project/
   simpulse optimize my-project/
   ```

#### Benefits of Upgrading
- **Real diagnostic data** instead of theoretical estimates
- **Performance validation** with actual measurements
- **Lake integration** for real project compilation
- **Confidence scoring** for optimization recommendations
- **Better error handling** and user experience

---

## Development Timeline

- **2024-12-15**: Initial concept and prototype (v1.0.0)
- **2025-01-10**: Major rewrite decision after performance theater realization  
- **2025-01-12**: Lake integration implementation
- **2025-01-14**: Hybrid diagnostic system completion
- **2025-01-16**: Production-ready release with CI/CD (v2.0.0)

---

## Acknowledgments

This project evolved from recognizing the importance of evidence-based optimization over theoretical estimates. The 2.0 rewrite represents a complete paradigm shift toward using real diagnostic data from Lean 4.8.0+ to make verifiable performance improvements.

Special thanks to the Lean 4 community for the diagnostic infrastructure and the Lake build system that made this evidence-based approach possible.
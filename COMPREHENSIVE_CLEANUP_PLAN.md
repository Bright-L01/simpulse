# Comprehensive Cleanup Plan - Simpulse 2.0
## Industry-Standard Repository Structure and Best Practices

### 🎯 Objective
Transform the Simpulse repository into a production-ready, industry-standard Python project with proper CI/CD, documentation, and code organization following 2025 best practices.

## 📋 Current Issues Identified

### 1. Documentation Problems
- **Too many root-level MD files** (8 files): CHANGELOG.md, CONTRIBUTING.md, LAKE_INTEGRATION_GUIDE.md, PROJECT_STATUS.md, PYPI_RELEASE.md, README.md, TESTING_GUIDE.md
- **Outdated content**: PROJECT_STATUS.md describes v0.1.0 when we're at v2.0.0
- **Scattered information**: Documentation spread across multiple files without clear organization

### 2. Python Project Structure Issues
- **Mixed dependency management**: Both poetry.lock AND requirements.txt present
- **Outdated pyproject.toml**: Using old [tool.poetry] format instead of new [project] format (Poetry 2.0+ standard)
- **Missing CI/CD**: No GitHub Actions workflows for automated testing
- **Test organization**: Could be improved with better structure and coverage

### 3. Lean4 Project Structure Issues
- **Multiple nested projects**: SimpulseJIT/, TacticPortfolio/, TestProject/, integration_test/
- **Redundant lakefile configurations**: Multiple lakefile.lean files
- **Scattered examples**: Various example files in different locations
- **Missing CI/CD integration**: No Lean 4 specific GitHub Actions

### 4. Git History Issues
- **Claude references**: Need to remove all Claude co-author references from commit history
- **Messy commit history**: Should be cleaned up and squashed where appropriate

## 🚀 Implementation Plan

### Phase 1: Python Project Standardization (HIGH PRIORITY)

#### 1.1 Update pyproject.toml to 2025 Standards
- **Migrate from [tool.poetry] to [project] table** (Poetry 2.0+ standard)
- **Remove requirements.txt** (redundant with poetry.lock)
- **Update build system configuration**
- **Add proper tool configurations** for ruff, mypy, pytest

#### 1.2 Implement CI/CD with GitHub Actions
- **Create .github/workflows/ci.yml** with matrix testing (Python 3.10, 3.11, 3.12)
- **Add Lean 4 testing workflow** using leanprover/lean-action
- **Implement caching** for both Python and Lean dependencies
- **Add automated testing, linting, and type checking**

#### 1.3 Improve Test Organization
- **Consolidate test files** into proper unit/integration structure
- **Remove redundant test status files** (TEST_REALITY_STATUS.md, TEST_STATUS.md)
- **Add comprehensive test coverage** for all modules
- **Implement test parameterization** for better coverage

### Phase 2: Documentation Consolidation (MEDIUM PRIORITY)

#### 2.1 Create Proper Documentation Structure
```
docs/
├── README.md                    # Main documentation
├── installation.md             # Installation guide  
├── usage.md                    # Usage examples
├── development.md              # Development setup
├── lake-integration.md         # Lake integration guide
├── troubleshooting.md          # Common issues
├── changelog.md                # Version history
└── contributing.md             # Contribution guidelines
```

#### 2.2 Consolidate Root-Level Documentation
- **Move specialized guides** to docs/ folder
- **Remove redundant files**: PROJECT_STATUS.md, PYPI_RELEASE.md, TESTING_GUIDE.md
- **Update README.md** to be the single source of truth
- **Create proper CHANGELOG.md** with version history

### Phase 3: Lean4 Project Structure Optimization (MEDIUM PRIORITY)

#### 3.1 Consolidate Lean4 Structure
- **Merge similar projects**: Combine SimpulseJIT/, TacticPortfolio/ into main structure
- **Centralize examples**: Move all examples to examples/ directory
- **Standardize lakefile configurations**: Use consistent patterns
- **Remove redundant test projects**: Keep only essential integration tests

#### 3.2 Implement Lean4 Best Practices
- **Add proper lean-toolchain file** with consistent version
- **Implement Lake caching strategy** for CI/CD
- **Add Lean 4 specific GitHub Actions** workflow
- **Standardize module organization**

### Phase 4: Code Quality and Modularization (MEDIUM PRIORITY)

#### 4.1 Improve Code Organization
- **Review src/simpulse/ modules** for redundancy
- **Consolidate similar functionality**: cli.py vs advanced_cli.py
- **Implement proper module interfaces**
- **Add comprehensive docstrings** and type hints

#### 4.2 Remove Redundant Code
- **Identify overlapping functionality** between modules
- **Consolidate optimizer implementations**
- **Remove dead code** and unused imports
- **Standardize error handling**

### Phase 5: Git History Cleanup (LOW PRIORITY)

#### 5.1 Clean Commit History
- **Remove Claude references** from all commit messages
- **Squash related commits** for cleaner history
- **Rewrite commit messages** to follow conventional commits
- **Ensure all commits are authored by user only**

#### 5.2 Repository Hygiene
- **Remove unnecessary files** from git tracking
- **Add proper .gitignore** for Python and Lean
- **Clean up branch structure**
- **Add proper git hooks** for pre-commit checks

## 📁 Target Repository Structure

```
simpulse/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Python CI/CD
│       └── lean.yml            # Lean 4 CI/CD
├── docs/
│   ├── installation.md
│   ├── usage.md
│   ├── development.md
│   ├── lake-integration.md
│   ├── troubleshooting.md
│   ├── changelog.md
│   └── contributing.md
├── examples/
│   ├── basic/
│   ├── advanced/
│   └── benchmarks/
├── lean4/
│   ├── Simpulse/              # Main Lean library
│   ├── examples/              # Lean examples
│   ├── tests/                 # Lean integration tests
│   ├── lakefile.lean          # Main lake configuration
│   └── lean-toolchain         # Lean version specification
├── src/
│   └── simpulse/
│       ├── __init__.py
│       ├── cli.py             # Unified CLI
│       ├── diagnostic_parser.py
│       ├── lake_integration.py
│       ├── optimization_engine.py
│       ├── performance_measurement.py
│       └── config.py
├── tests/
│   ├── unit/
│   │   ├── test_cli.py
│   │   ├── test_diagnostic_parser.py
│   │   └── test_optimization_engine.py
│   ├── integration/
│   │   ├── test_lake_integration.py
│   │   └── test_end_to_end.py
│   └── conftest.py
├── pyproject.toml             # Modern Python configuration
├── README.md                  # Main documentation
├── LICENSE                    # MIT license
└── .gitignore                 # Proper ignore patterns
```

## 🛠️ Implementation Steps

### Step 1: Backup and Preparation
1. **Create backup branch**: `git checkout -b cleanup-backup`
2. **Document current state**: Save current structure for reference
3. **Run full test suite**: Ensure everything works before changes

### Step 2: Python Standardization
1. **Update pyproject.toml** to [project] table format
2. **Remove requirements.txt** 
3. **Create GitHub Actions workflows**
4. **Reorganize test structure**

### Step 3: Documentation Consolidation
1. **Create docs/ directory structure**
2. **Move and consolidate documentation**
3. **Remove redundant root-level files**
4. **Update README.md**

### Step 4: Lean4 Optimization
1. **Consolidate Lean4 projects**
2. **Standardize lakefile configurations**
3. **Add Lean4 GitHub Actions**
4. **Organize examples**

### Step 5: Code Quality
1. **Review and consolidate modules**
2. **Remove redundant code**
3. **Improve type hints and documentation**
4. **Add comprehensive tests**

### Step 6: Git History Cleanup
1. **Interactive rebase** to clean history
2. **Remove Claude references**
3. **Squash related commits**
4. **Update .gitignore**

## ✅ Success Criteria

### Technical Standards
- [ ] Python project follows 2025 best practices with [project] table
- [ ] CI/CD workflows pass on all supported Python versions
- [ ] Lean 4 integration works with official leanprover/lean-action
- [ ] Test coverage >90% for core functionality
- [ ] All linting and type checking passes

### Documentation Standards
- [ ] Single source of truth in README.md
- [ ] Comprehensive docs/ directory with proper organization
- [ ] No redundant or outdated documentation files
- [ ] Clear installation and usage instructions

### Code Quality Standards
- [ ] No redundant or dead code
- [ ] Proper modularization and separation of concerns
- [ ] Comprehensive type hints and docstrings
- [ ] Consistent error handling patterns

### Git Standards
- [ ] Clean commit history with no Claude references
- [ ] Conventional commit messages
- [ ] Proper .gitignore coverage
- [ ] All commits authored by user only

## 🎯 Next Development Phases

### Phase A: PyPI Publication (Immediate)
- Automated PyPI publication via GitHub Actions
- Proper versioning and release management
- Documentation for installation and usage

### Phase B: Community Features (Short-term)
- Improved Lake integration with better error handling
- Support for more Lean 4 project structures
- Enhanced diagnostic data collection

### Phase C: Advanced Features (Medium-term)
- Real-time performance monitoring
- Integration with Lean 4 language server
- Advanced optimization algorithms

### Phase D: Ecosystem Integration (Long-term)
- VS Code extension integration
- Mathlib-specific optimizations
- Community contribution framework

## 📊 Timeline

- **Phase 1 (Python Standardization)**: 2 hours
- **Phase 2 (Documentation)**: 1 hour
- **Phase 3 (Lean4 Optimization)**: 1.5 hours
- **Phase 4 (Code Quality)**: 1 hour
- **Phase 5 (Git Cleanup)**: 1 hour

**Total estimated time**: 6.5 hours

## 🚨 Risk Mitigation

1. **Backup strategy**: Multiple backup branches before major changes
2. **Incremental testing**: Test after each phase completion
3. **Rollback plan**: Clear steps to revert if issues arise
4. **Documentation**: Document all changes for future reference

This plan transforms Simpulse from an experimental tool into a production-ready, industry-standard Python project that follows all 2025 best practices for both Python and Lean 4 development.
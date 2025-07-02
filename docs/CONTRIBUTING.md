# Contributing to Simpulse

Thank you for your interest in contributing to Simpulse! This guide will help you get started.

## 🤝 How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **🐛 Bug Reports**: Found a bug? Report it!
- **✨ Feature Requests**: Have an idea for improvement? Share it!
- **📝 Documentation**: Help improve our docs
- **🔧 Code Contributions**: Fix bugs or implement features
- **🧪 Testing**: Add tests or improve test coverage
- **🎯 Performance**: Optimize performance or add benchmarks

### Before You Start

1. **Check existing issues** to avoid duplicate work
2. **Discuss large changes** by opening an issue first
3. **Read our [Code of Conduct](CODE_OF_CONDUCT.md)**
4. **Review our [Architecture docs](architecture/)** to understand the codebase

## 🚀 Getting Started

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/simpulse.git
cd simpulse

# Install with development dependencies
pip install poetry
poetry install --with dev,test

# Install pre-commit hooks
poetry run pre-commit install

# Run tests to verify setup
poetry run pytest

# Check code quality
poetry run ruff check src/ tests/
poetry run mypy src/
```

### Project Structure

```
simpulse/
├── src/simpulse/           # Main source code
│   ├── analyzer.py         # Lean file analysis
│   ├── optimizer.py        # Priority optimization
│   ├── validator.py        # Correctness validation
│   └── cli.py             # Command-line interface
├── tests/                  # Test suite
├── docs/                   # Documentation
├── examples/               # Usage examples
├── lean4/                  # Lean 4 package
└── pyproject.toml         # Project configuration
```

## 🔧 Development Workflow

### 1. Create a Branch

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or bug fix branch
git checkout -b fix/issue-123
```

### 2. Make Changes

- **Write tests first** (TDD approach encouraged)
- **Follow code style** (enforced by pre-commit hooks)
- **Update documentation** when needed
- **Add type hints** for all new code

### 3. Test Your Changes

```bash
# Run full test suite
poetry run pytest

# Run with coverage
poetry run pytest --cov=simpulse --cov-report=html

# Run specific tests
poetry run pytest tests/test_analyzer.py

# Run integration tests
poetry run pytest tests/integration/
```

### 4. Check Code Quality

```bash
# Format code
poetry run black src/ tests/
poetry run isort src/ tests/

# Lint code
poetry run ruff check src/ tests/

# Type check
poetry run mypy src/

# Security scan
poetry run bandit -r src/

# Run all checks
poetry run pre-commit run --all-files
```

### 5. Commit Changes

```bash
# Add changes
git add .

# Commit with conventional commit message
git commit -m "feat: add new optimization strategy"

# Or for bug fixes
git commit -m "fix: handle empty Lean files correctly"
```

**Conventional Commit Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding tests
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

### 6. Submit Pull Request

```bash
# Push branch
git push origin feature/your-feature-name

# Create PR on GitHub
# - Describe what you changed and why
# - Link to related issues
# - Include test results if relevant
```

## 📋 Pull Request Guidelines

### PR Checklist

- [ ] **Tests pass**: All existing tests still pass
- [ ] **New tests added**: For new functionality or bug fixes
- [ ] **Documentation updated**: For user-facing changes
- [ ] **Type hints added**: For all new code
- [ ] **Performance considered**: No significant regressions
- [ ] **Backwards compatibility**: Maintained unless breaking change is justified

### PR Description Template

```markdown
## Description
Brief description of changes.

## Related Issue
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Added tests for new functionality
- [ ] All tests pass
- [ ] Manual testing performed

## Performance Impact
- [ ] No performance impact
- [ ] Performance improved
- [ ] Performance regression (justified)

## Breaking Changes
- [ ] No breaking changes
- [ ] Breaking changes (documented)
```

## 🧪 Testing Guidelines

### Test Types

**Unit Tests** (`tests/unit/`):
- Test individual functions/classes
- Fast execution (<1s each)
- No external dependencies

**Integration Tests** (`tests/integration/`):
- Test component interactions
- May require Lean 4 installation
- Test actual file processing

**Performance Tests** (`tests/performance/`):
- Benchmark critical paths
- Regression detection
- Large file handling

### Writing Tests

```python
import pytest
from pathlib import Path
from simpulse.analyzer import LeanAnalyzer

def test_analyzer_basic_functionality():
    """Test basic analyzer functionality."""
    analyzer = LeanAnalyzer()
    
    # Test with simple Lean content
    content = "@[simp] theorem test : true = true := rfl"
    rules = analyzer.extract_simp_rules(content)
    
    assert len(rules) == 1
    assert rules[0].name == "test"
    assert rules[0].priority is None  # Default priority

def test_analyzer_with_custom_priority():
    """Test analyzer handles custom priorities."""
    analyzer = LeanAnalyzer()
    
    content = "@[simp, priority := 100] theorem test : true = true := rfl"
    rules = analyzer.extract_simp_rules(content)
    
    assert len(rules) == 1
    assert rules[0].priority == 100

@pytest.mark.integration
def test_analyzer_real_file():
    """Test analyzer with real Lean file."""
    analyzer = LeanAnalyzer()
    
    # Use example file
    file_path = Path("examples/test.lean")
    if file_path.exists():
        result = analyzer.analyze_file(file_path)
        assert result is not None
```

### Test Coverage

- **Target**: 85%+ coverage
- **Critical paths**: 95%+ coverage
- **New code**: 90%+ coverage

```bash
# Generate coverage report
poetry run pytest --cov=simpulse --cov-report=html
open htmlcov/index.html
```

## 📝 Documentation Guidelines

### Documentation Types

**User Documentation**:
- Installation and setup guides
- API reference
- Tutorials and examples
- FAQ and troubleshooting

**Developer Documentation**:
- Architecture overview
- Code organization
- Testing procedures
- Release processes

### Writing Style

- **Clear and concise**: Use simple language
- **Examples included**: Show don't just tell
- **Up to date**: Keep docs synchronized with code
- **Accessible**: Consider different skill levels

### Documentation Updates

Update docs for:
- New CLI commands or options
- API changes
- New configuration options
- Performance improvements
- Breaking changes

## 🏗️ Architecture Guidelines

### Code Organization

**Separation of Concerns**:
- `analyzer.py`: Lean file parsing and analysis
- `optimizer.py`: Priority optimization algorithms
- `validator.py`: Correctness and performance validation
- `cli.py`: Command-line interface

**Dependencies**:
- **Minimize external dependencies**
- **Prefer standard library** when possible
- **Justify each new dependency**

### Performance Considerations

- **Profile before optimizing**
- **Use appropriate data structures**
- **Cache expensive computations**
- **Support parallel processing** for large projects

### Error Handling

```python
# Good: Specific, actionable errors
if not file_path.exists():
    raise FileNotFoundError(f"Lean file not found: {file_path}")

# Bad: Generic errors
if not file_path.exists():
    raise Exception("File error")
```

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes

### Release Checklist

1. **Update CHANGELOG.md**
2. **Update version in pyproject.toml**
3. **Run full test suite**
4. **Update documentation**
5. **Create release PR**
6. **Tag release after merge**
7. **GitHub Actions handles PyPI publishing**

## 🎯 Areas Needing Help

### High Priority
- **VS Code Extension**: Integrate with Lean 4 VS Code extension
- **mathlib4 Testing**: More comprehensive mathlib4 compatibility tests
- **Performance Optimization**: Faster analysis for large projects
- **Error Handling**: Better error messages and recovery

### Medium Priority
- **Documentation**: More examples and tutorials
- **CI/CD**: Windows testing and ARM64 support
- **Benchmarking**: More comprehensive performance tests
- **LSP Integration**: Language Server Protocol support

### Research Projects
- **ML Optimization**: Machine learning-based priority suggestions
- **Dynamic Optimization**: Runtime adaptation to usage patterns
- **Multi-Tactic Support**: Extend beyond simp to other tactics

## 🙋 Getting Help

### Community Resources
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Discord**: Real-time community chat (coming Phase 14)

### Maintainer Contact
- **Bright Liu**: bright.liu@example.com
- **GitHub**: @Bright-L01

### Response Time
- **Urgent bugs**: 24-48 hours
- **Feature requests**: 1-2 weeks
- **General questions**: 2-3 days

## 📜 Code of Conduct

### Our Pledge

We are committed to making participation in our project a harassment-free experience for everyone, regardless of:
- Age, body size, disability, ethnicity
- Gender identity and expression
- Level of experience, education, socio-economic status
- Nationality, personal appearance, race, religion
- Sexual identity and orientation

### Expected Behavior

- **Be respectful** and inclusive
- **Welcome newcomers** and help them learn
- **Focus on constructive feedback**
- **Respect different viewpoints** and experiences
- **Show empathy** towards other community members

### Unacceptable Behavior

- Harassment of any kind
- Discriminatory language or actions
- Personal attacks or political arguments
- Public or private harassment
- Publishing others' information without permission

### Enforcement

Violations may result in:
1. **Warning**: First offense or minor violations
2. **Temporary ban**: Repeated or serious violations
3. **Permanent ban**: Severe or repeated violations

Report violations to: conduct@simpulse.dev

## 🏆 Recognition

### Contributors

All contributors are recognized in:
- **README.md**: Major contributors listed
- **CHANGELOG.md**: Contributions noted in releases
- **GitHub**: Contributor graphs and statistics

### Maintainer Status

Active contributors may be invited to become maintainers with:
- **Commit access**: Ability to merge PRs
- **Release authority**: Help with releases
- **Community leadership**: Guide project direction

---

Thank you for contributing to Simpulse! Together we're making Lean 4 theorem proving faster and more efficient. 🚀
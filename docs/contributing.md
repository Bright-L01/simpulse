# Contributing to Simpulse

Thank you for your interest in contributing to Simpulse! We welcome contributions from the community.

## How to Contribute

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/yourusername/simpulse.git
cd simpulse
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Make Your Changes

- Create a feature branch: `git checkout -b feature/your-feature`
- Make your changes following our code style
- Add tests for new functionality
- Ensure all tests pass: `pytest tests/`

### 4. Code Style

- Use Black for formatting: `black src/ tests/`
- Type hints required for new code
- Follow PEP 8 guidelines
- Run linters: `flake8 src/ tests/`

### 5. Testing

- Write tests for all new functionality
- Ensure 85%+ code coverage
- Run tests: `pytest tests/ --cov=simpulse`

### 6. Submit Pull Request

- Push your branch: `git push origin feature/your-feature`
- Open a Pull Request on GitHub
- Describe your changes clearly
- Link any related issues

## Development Guidelines

### Code Organization

- Keep modules focused and cohesive
- Follow existing patterns in the codebase
- Document complex algorithms

### Commit Messages

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring

### Testing Lean Integration

If your changes affect Lean integration:
1. Install Lean 4 (see README)
2. Run integration tests
3. Test on real Lean projects

## Getting Help

- Open an issue for bugs or features
- Join our Discord for discussions
- Check existing issues and PRs

## Code of Conduct

Please be respectful and professional in all interactions.
# PyPI Release Instructions for Simpulse 2.0

## Prerequisites

1. **PyPI Account**: Ensure you have a PyPI account at https://pypi.org
2. **API Token**: Generate an API token in PyPI account settings
3. **Poetry**: Ensure Poetry is installed and configured

## Release Process

### 1. Pre-Release Checks

```bash
# Update version in pyproject.toml (already done: 2.0.0)
# Ensure README.md is updated (already done)
# Run tests
python -m pytest tests/ -v

# Check package building
poetry build
```

### 2. Configure PyPI Authentication

```bash
# Configure PyPI token (one-time setup)
poetry config pypi-token.pypi YOUR_API_TOKEN_HERE
```

### 3. Build and Publish

```bash
# Clean previous builds
rm -rf dist/

# Build package
poetry build

# Check built package
ls dist/
# Should show:
# simpulse-2.0.0-py3-none-any.whl
# simpulse-2.0.0.tar.gz

# Publish to PyPI
poetry publish
```

### 4. Verify Publication

```bash
# Install from PyPI in a clean environment
pip install simpulse==2.0.0

# Test installation
simpulse --help
```

## Version Management

Current version: **2.0.0**

To update for future releases:
1. Edit `pyproject.toml` version field
2. Edit `src/simpulse/__init__.py` version field
3. Follow release process above

## Package Information

- **Name**: `simpulse`
- **Version**: `2.0.0`
- **Description**: Advanced Lean 4 simp optimization using real diagnostic data
- **Keywords**: lean4, theorem-proving, simp, optimization, diagnostics, performance
- **Status**: Beta (Development Status :: 4 - Beta)

## Important Notes

- This is a **major version release** (1.0 → 2.0) due to breaking changes
- All theoretical estimates replaced with evidence-based analysis
- Requires Lean 4.8.0+ for diagnostic infrastructure
- CLI completely rewritten with new commands

## Post-Release

After successful publication:

1. **Update GitHub Release**:
   - Create release tag `v2.0.0`
   - Include changelog and breaking changes
   - Attach built wheel/tarball

2. **Update Documentation**:
   - Ensure README reflects PyPI installation
   - Update any version references

3. **Test Installation**:
   ```bash
   pip install simpulse
   simpulse analyze --help
   ```

## Troubleshooting

**Build Failures**:
- Check `pyproject.toml` syntax
- Ensure all dependencies are specified
- Verify package structure with `poetry check`

**Upload Failures**:
- Verify PyPI token is correct
- Check if version already exists (versions are immutable)
- Ensure sufficient account permissions

**Installation Issues**:
- Verify Python version compatibility (3.10+)
- Check dependency conflicts
- Test in clean virtual environment

## Development Status

- **Alpha** (3): Early development, unstable API
- **Beta** (4): **Current** - Feature complete, testing phase  
- **Production** (5): Stable for production use

Simpulse 2.0 is marked as **Beta** because while the core functionality is complete and tested, it's a major rewrite that needs broader community validation.
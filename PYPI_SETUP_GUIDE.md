# PyPI Trusted Publishing Setup Guide

## Overview
This guide provides step-by-step instructions for setting up PyPI Trusted Publishing for the Simpulse project using modern 2025 best practices.

## Prerequisites
- GitHub repository: `Bright-L01/simpulse`
- PyPI account with permissions to create new projects
- GitHub Actions workflow configured (already done)

## Step 1: Initial PyPI Project Setup

### 1.1 Create PyPI Account
If you don't have a PyPI account:
1. Go to https://pypi.org/account/register/
2. Create account with email `brightliu@college.harvard.edu`
3. Verify email address

### 1.2 Create Project on PyPI
1. Go to https://pypi.org/manage/projects/
2. Click "Create new project"
3. Project name: `simpulse`
4. Or upload the first release manually to create the project

## Step 2: Configure Trusted Publishing

### 2.1 Access Trusted Publishers
1. Go to https://pypi.org/manage/project/simpulse/settings/
2. Scroll to "Trusted Publishers" section
3. Click "Add a new trusted publisher"

### 2.2 GitHub Actions Configuration
Configure the trusted publisher with these exact settings:

**Repository owner**: `Bright-L01`
**Repository name**: `simpulse`
**Workflow filename**: `release.yml`
**Environment (optional)**: `pypi`

### 2.3 TestPyPI Configuration (Optional)
For testing releases:
1. Go to https://test.pypi.org/manage/project/simpulse/settings/
2. Add trusted publisher with same settings but environment: `testpypi`

## Step 3: GitHub Repository Settings

### 3.1 Create Environment for PyPI
1. Go to GitHub repository settings
2. Navigate to "Environments"
3. Create new environment: `pypi`
4. Add protection rules if desired (e.g., require reviews)

### 3.2 Create Environment for TestPyPI
1. Create new environment: `testpypi`
2. No special protection needed for testing

## Step 4: Test the Setup

### 4.1 Test with TestPyPI
1. Create a test tag:
   ```bash
   git tag v2.0.0-test
   git push origin v2.0.0-test
   ```

2. Monitor GitHub Actions for successful TestPyPI upload

### 4.2 Production Release
1. Create production tag:
   ```bash
   git tag v2.0.0
   git push origin v2.0.0
   ```

2. Monitor GitHub Actions for successful PyPI upload

## Step 5: Verification

### 5.1 Check PyPI Project
1. Visit https://pypi.org/project/simpulse/
2. Verify package information and version

### 5.2 Test Installation
```bash
pip install simpulse==2.0.0
simpulse --help
```

## GitHub Actions Workflow Details

The configured workflow (`release.yml`) includes:

- **Trigger**: On tag pushes starting with `v` (e.g., `v2.0.0`)
- **Testing**: Runs full test suite before publishing
- **Building**: Creates both wheel and source distributions
- **TestPyPI**: Publishes to TestPyPI on all releases
- **PyPI**: Publishes to PyPI only on tag pushes
- **GitHub Release**: Creates GitHub release with artifacts

## Security Features

### Trusted Publishing Benefits
- **No API tokens**: No need to manage API tokens
- **Automatic expiration**: Tokens expire after use
- **Scoped permissions**: Limited to specific repository and workflow
- **Audit trail**: Full audit trail of all publishes

### PEP 740 Attestations
The workflow automatically generates PEP 740-compatible attestations for enhanced security verification.

## Troubleshooting

### Common Issues

#### 1. "Trusted publisher not configured"
- Verify PyPI project exists
- Check trusted publisher settings match exactly
- Ensure environment name matches workflow

#### 2. "Permission denied" errors
- Verify `id-token: write` permission in workflow
- Check environment protection rules
- Ensure repository has correct permissions

#### 3. "Package already exists" errors
- Version numbers must be unique
- Cannot republish same version
- Increment version number for new releases

#### 4. "Workflow not found" errors
- Verify workflow filename matches PyPI configuration
- Check workflow is in `.github/workflows/` directory
- Ensure workflow has been pushed to repository

### Debug Steps

1. **Check GitHub Actions logs** for detailed error messages
2. **Verify PyPI trusted publisher configuration**
3. **Test with TestPyPI first** before production
4. **Check repository settings** for environment configuration

## Maintenance

### Regular Tasks
- Monitor PyPI project for security updates
- Review trusted publisher configurations quarterly
- Update workflow dependencies regularly
- Test release process with TestPyPI

### Version Updates
- Follow semantic versioning (2.0.0, 2.0.1, 2.1.0, etc.)
- Update version in `pyproject.toml`
- Create corresponding git tag
- GitHub Actions handles the rest automatically

## Next Steps

1. **Set up PyPI trusted publisher** (manual step)
2. **Create initial release** with `git tag v2.0.0`
3. **Monitor GitHub Actions** for successful publishing
4. **Test installation** from PyPI
5. **Announce release** to Lean 4 community

## References

- [PyPI Trusted Publishing Documentation](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions PyPI Publish Action](https://github.com/pypa/gh-action-pypi-publish)
- [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [PEP 740 Attestations](https://peps.python.org/pep-0740/)

---

**Important**: This setup eliminates the need for API tokens and provides the most secure method for publishing Python packages to PyPI in 2025.
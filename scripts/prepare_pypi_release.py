#!/usr/bin/env python3
"""
Prepare Simpulse for PyPI release.

This script handles:
- Package metadata validation
- Version management
- Build process
- Distribution creation
- Pre-release testing
- Upload to PyPI
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import toml
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ReleaseConfig:
    """PyPI release configuration."""
    version: str
    test_pypi: bool = True
    dry_run: bool = False
    skip_tests: bool = False
    token: Optional[str] = None


class PyPIReleaser:
    """Prepare and release Simpulse to PyPI."""
    
    def __init__(self, project_root: Path, config: ReleaseConfig):
        """Initialize PyPI releaser.
        
        Args:
            project_root: Root directory of the project
            config: Release configuration
        """
        self.project_root = project_root
        self.config = config
        self.dist_dir = project_root / "dist"
        
    async def prepare_release(self) -> bool:
        """Complete PyPI release preparation process."""
        logger.info(f"Preparing PyPI release for version {self.config.version}...")
        
        try:
            # Step 1: Validate package metadata
            logger.info("\n📋 Validating package metadata...")
            if not self.validate_metadata():
                return False
            
            # Step 2: Update version
            logger.info("\n🔢 Updating version...")
            self.update_version()
            
            # Step 3: Run tests
            if not self.config.skip_tests:
                logger.info("\n🧪 Running tests...")
                if not self.run_tests():
                    return False
            
            # Step 4: Build distributions
            logger.info("\n🏗️ Building distributions...")
            if not self.build_distributions():
                return False
            
            # Step 5: Validate distributions
            logger.info("\n✅ Validating distributions...")
            if not self.validate_distributions():
                return False
            
            # Step 6: Test installation
            logger.info("\n📦 Testing installation...")
            if not self.test_installation():
                return False
            
            # Step 7: Create release notes
            logger.info("\n📝 Creating release notes...")
            self.create_release_notes()
            
            # Step 8: Upload to PyPI
            if not self.config.dry_run:
                logger.info("\n🚀 Uploading to PyPI...")
                if not await self.upload_to_pypi():
                    return False
            else:
                logger.info("\n🔍 Dry run - skipping upload")
            
            logger.info("\n✅ PyPI release preparation complete!")
            return True
            
        except Exception as e:
            logger.error(f"Release preparation failed: {e}")
            return False
    
    def validate_metadata(self) -> bool:
        """Validate package metadata in pyproject.toml."""
        pyproject_path = self.project_root / "pyproject.toml"
        
        if not pyproject_path.exists():
            logger.error("pyproject.toml not found")
            return False
        
        try:
            with open(pyproject_path, 'r') as f:
                data = toml.load(f)
            
            # Check required fields
            project = data.get("project", {})
            required_fields = [
                "name", "version", "description", "authors",
                "license", "readme", "classifiers", "dependencies"
            ]
            
            missing_fields = []
            for field in required_fields:
                if field not in project:
                    missing_fields.append(field)
            
            if missing_fields:
                logger.error(f"Missing required fields: {', '.join(missing_fields)}")
                return False
            
            # Validate authors format
            authors = project.get("authors", [])
            if not authors:
                logger.error("At least one author required")
                return False
            
            # Check classifiers
            classifiers = project.get("classifiers", [])
            required_classifiers = [
                "Development Status",
                "Intended Audience",
                "License",
                "Programming Language :: Python"
            ]
            
            classifier_prefixes = [c.split(" ::")[0] for c in classifiers]
            for req in required_classifiers:
                if not any(c.startswith(req) for c in classifier_prefixes):
                    logger.warning(f"Missing classifier category: {req}")
            
            # Validate URLs
            urls = project.get("urls", {})
            if "Homepage" not in urls:
                logger.warning("No homepage URL specified")
            
            logger.info("✓ Package metadata validated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate metadata: {e}")
            return False
    
    def update_version(self) -> None:
        """Update version in all relevant files."""
        files_to_update = [
            (self.project_root / "pyproject.toml", 'version = "{}"'),
            (self.project_root / "src" / "simpulse" / "__init__.py", '__version__ = "{}"'),
            (self.project_root / "docs" / "conf.py", 'release = "{}"'),
        ]
        
        for file_path, pattern in files_to_update:
            if file_path.exists():
                content = file_path.read_text()
                
                # Update version
                if pattern.format("") in content:
                    # Extract current version
                    import re
                    current_version = re.search(
                        pattern.format(r'([^"]+)'),
                        content
                    )
                    if current_version:
                        old_version = current_version.group(1)
                        content = content.replace(
                            pattern.format(old_version),
                            pattern.format(self.config.version)
                        )
                        file_path.write_text(content)
                        logger.info(f"Updated version in {file_path.name}: {old_version} → {self.config.version}")
    
    def run_tests(self) -> bool:
        """Run test suite before release."""
        try:
            # Run pytest
            result = subprocess.run(
                ["pytest", "tests/", "-v", "--tb=short"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error("Tests failed - cannot proceed with release")
                logger.error(result.stdout[-1000:])  # Last 1000 chars
                return False
            
            # Check coverage
            result = subprocess.run(
                ["pytest", "--cov=simpulse", "--cov-report=term"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Extract coverage percentage
            import re
            coverage_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', result.stdout)
            if coverage_match:
                coverage = int(coverage_match.group(1))
                if coverage < 85:
                    logger.warning(f"Coverage is {coverage}% (recommended: 85%+)")
                else:
                    logger.info(f"✓ Test coverage: {coverage}%")
            
            logger.info("✓ All tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return False
    
    def build_distributions(self) -> bool:
        """Build source and wheel distributions."""
        try:
            # Clean previous builds
            if self.dist_dir.exists():
                shutil.rmtree(self.dist_dir)
            
            build_dir = self.project_root / "build"
            if build_dir.exists():
                shutil.rmtree(build_dir)
            
            # Build distributions
            result = subprocess.run(
                ["python", "-m", "build"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error("Build failed")
                logger.error(result.stderr)
                return False
            
            # Check created files
            dist_files = list(self.dist_dir.glob("*"))
            if len(dist_files) < 2:
                logger.error("Expected both wheel and sdist, found: " + str(dist_files))
                return False
            
            for dist_file in dist_files:
                logger.info(f"✓ Built: {dist_file.name} ({dist_file.stat().st_size / 1024:.1f} KB)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build distributions: {e}")
            return False
    
    def validate_distributions(self) -> bool:
        """Validate built distributions."""
        try:
            # Check with twine
            result = subprocess.run(
                ["twine", "check", "dist/*"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error("Distribution validation failed")
                logger.error(result.stdout)
                return False
            
            logger.info("✓ Distributions validated")
            
            # Additional checks
            for dist_file in self.dist_dir.glob("*.whl"):
                # Check wheel contents
                import zipfile
                with zipfile.ZipFile(dist_file, 'r') as zf:
                    files = zf.namelist()
                    
                    # Check for required files
                    if not any("simpulse/__init__.py" in f for f in files):
                        logger.error(f"Package files missing in {dist_file.name}")
                        return False
                    
                    # Check for metadata
                    if not any(".dist-info/METADATA" in f for f in files):
                        logger.error(f"Metadata missing in {dist_file.name}")
                        return False
                    
                    logger.info(f"✓ Wheel contains {len(files)} files")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate distributions: {e}")
            return False
    
    def test_installation(self) -> bool:
        """Test package installation in clean environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Create virtual environment
                venv_path = Path(tmpdir) / "test_venv"
                subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_path)],
                    check=True
                )
                
                # Get pip path
                if sys.platform == "win32":
                    pip_path = venv_path / "Scripts" / "pip"
                    python_path = venv_path / "Scripts" / "python"
                else:
                    pip_path = venv_path / "bin" / "pip"
                    python_path = venv_path / "bin" / "python"
                
                # Install from wheel
                wheel_file = next(self.dist_dir.glob("*.whl"))
                result = subprocess.run(
                    [str(pip_path), "install", str(wheel_file)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error("Installation failed")
                    logger.error(result.stderr)
                    return False
                
                # Test import
                test_script = """
import simpulse
print(f"Version: {simpulse.__version__}")

# Basic functionality test
from simpulse.config import Config
config = Config()
print(f"Config created: {config.project_name}")
"""
                
                result = subprocess.run(
                    [str(python_path), "-c", test_script],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error("Import test failed")
                    logger.error(result.stderr)
                    return False
                
                logger.info("✓ Package installed and imported successfully")
                logger.info(f"  {result.stdout.strip()}")
                
                return True
                
            except Exception as e:
                logger.error(f"Installation test failed: {e}")
                return False
    
    def create_release_notes(self) -> None:
        """Create release notes for this version."""
        notes_path = self.project_root / f"RELEASE_NOTES_{self.config.version}.md"
        
        # Extract from CHANGELOG if exists
        changelog_path = self.project_root / "CHANGELOG.md"
        changelog_section = ""
        
        if changelog_path.exists():
            content = changelog_path.read_text()
            # Try to extract section for this version
            import re
            pattern = f"## \\[?{re.escape(self.config.version)}\\]?.*?\n(.*?)(?=## \\[|$)"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                changelog_section = match.group(1).strip()
        
        # Generate release notes
        lines = [
            f"# Simpulse {self.config.version} Release Notes",
            "",
            f"**Release Date**: {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "## What's New",
            "",
            changelog_section if changelog_section else "See CHANGELOG.md for details.",
            "",
            "## Installation",
            "",
            "```bash",
            "pip install --upgrade simpulse",
            "```",
            "",
            "## Verification",
            "",
            "After installation, verify with:",
            "",
            "```python",
            "import simpulse",
            f"assert simpulse.__version__ == '{self.config.version}'",
            "```",
            "",
            "## PyPI Links",
            "",
            f"- [simpulse on PyPI](https://pypi.org/project/simpulse/{self.config.version}/)",
            "- [Download files](https://pypi.org/project/simpulse/#files)",
            "",
            "## Support",
            "",
            "- [Documentation](https://simpulse.dev)",
            "- [GitHub Issues](https://github.com/yourusername/simpulse/issues)",
            "- [Discussions](https://github.com/yourusername/simpulse/discussions)",
        ]
        
        notes_path.write_text('\n'.join(lines))
        logger.info(f"Created release notes: {notes_path}")
    
    async def upload_to_pypi(self) -> bool:
        """Upload distributions to PyPI."""
        try:
            # Determine repository URL
            if self.config.test_pypi:
                repository_url = "https://test.pypi.org/legacy/"
                logger.info("Uploading to Test PyPI...")
            else:
                repository_url = "https://upload.pypi.org/legacy/"
                logger.info("Uploading to PyPI...")
            
            # Build command
            cmd = [
                "twine", "upload",
                "--repository-url", repository_url,
                "dist/*"
            ]
            
            # Add authentication
            if self.config.token:
                cmd.extend(["--username", "__token__", "--password", self.config.token])
            elif os.environ.get("PYPI_API_TOKEN"):
                cmd.extend(["--username", "__token__", "--password", os.environ["PYPI_API_TOKEN"]])
            
            # Upload
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error("Upload failed")
                logger.error(result.stderr)
                return False
            
            logger.info("✓ Successfully uploaded to PyPI")
            
            # Display URLs
            if self.config.test_pypi:
                logger.info(f"View at: https://test.pypi.org/project/simpulse/{self.config.version}/")
            else:
                logger.info(f"View at: https://pypi.org/project/simpulse/{self.config.version}/")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload: {e}")
            return False
    
    def generate_release_checklist(self) -> None:
        """Generate post-release checklist."""
        checklist_path = self.project_root / "POST_RELEASE_CHECKLIST.md"
        
        lines = [
            f"# Post-Release Checklist for v{self.config.version}",
            "",
            "## Immediate Tasks",
            "",
            "- [ ] Verify package on PyPI",
            "- [ ] Test installation: `pip install simpulse=={}`".format(self.config.version),
            "- [ ] Create GitHub release",
            "- [ ] Tag the release: `git tag -a v{} -m 'Release v{}'`".format(self.config.version, self.config.version),
            "- [ ] Push tag: `git push origin v{}`".format(self.config.version),
            "",
            "## Communication",
            "",
            "- [ ] Announce on GitHub Discussions",
            "- [ ] Post on Lean Zulip",
            "- [ ] Update documentation site",
            "- [ ] Tweet announcement (if applicable)",
            "- [ ] Update mathlib4 PR (if applicable)",
            "",
            "## Verification",
            "",
            "- [ ] Check PyPI statistics",
            "- [ ] Monitor GitHub issues for problems",
            "- [ ] Verify documentation reflects new version",
            "- [ ] Test example notebooks/scripts",
            "",
            "## Next Steps",
            "",
            "- [ ] Update version to next development version",
            "- [ ] Create milestone for next release",
            "- [ ] Plan features for next version",
            "",
            "## Rollback Plan",
            "",
            "If issues are discovered:",
            "",
            "1. `pip install simpulse=={previous_version}`",
            "2. Yank release on PyPI if critical",
            "3. Fix issues and release patch version",
            "4. Communicate to users",
        ]
        
        checklist_path.write_text('\n'.join(lines))
        logger.info(f"Created post-release checklist: {checklist_path}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare Simpulse for PyPI release"
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version to release (e.g., 1.0.0)"
    )
    parser.add_argument(
        "--test-pypi",
        action="store_true",
        help="Upload to Test PyPI instead of production"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform all steps except upload"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests (not recommended)"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="PyPI API token (or use PYPI_API_TOKEN env var)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    # Validate version format
    import re
    if not re.match(r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$', args.version):
        logger.error(f"Invalid version format: {args.version}")
        logger.error("Use semantic versioning (e.g., 1.0.0, 1.0.0-beta.1)")
        sys.exit(1)
    
    # Create configuration
    config = ReleaseConfig(
        version=args.version,
        test_pypi=args.test_pypi,
        dry_run=args.dry_run,
        skip_tests=args.skip_tests,
        token=args.token
    )
    
    # Prepare release
    releaser = PyPIReleaser(args.project_root, config)
    success = await releaser.prepare_release()
    
    if success:
        releaser.generate_release_checklist()
        logger.info("\n" + "="*60)
        logger.info("PYPI RELEASE PREPARATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Version: {config.version}")
        if config.test_pypi:
            logger.info("Repository: Test PyPI")
        else:
            logger.info("Repository: PyPI")
        logger.info("="*60)
    else:
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
#!/usr/bin/env python3
"""
Automated release preparation checklist for Simpulse.

This script ensures all necessary steps are completed before
creating a new release.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Status of a checklist item."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class CheckResult:
    """Result of a single check."""
    name: str
    status: CheckStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    fix_command: Optional[str] = None


class ReleaseChecklist:
    """Automated release preparation checklist."""
    
    def __init__(self, project_root: Path, version: Optional[str] = None):
        """Initialize release checklist.
        
        Args:
            project_root: Root directory of the project
            version: Target release version
        """
        self.project_root = project_root
        self.version = version
        self.checks: List[CheckResult] = []
        
    def run_all_checks(self) -> Tuple[bool, List[CheckResult]]:
        """Run all release preparation checks.
        
        Returns:
            Tuple of (all_passed, check_results)
        """
        logger.info("Running release preparation checklist...")
        
        # Define all checks
        check_functions = [
            self.check_version_number,
            self.check_git_status,
            self.check_branch,
            self.check_tests,
            self.check_code_quality,
            self.check_documentation,
            self.check_changelog,
            self.check_dependencies,
            self.check_security,
            self.check_benchmarks,
            self.check_examples,
            self.check_license,
            self.check_package_metadata,
            self.check_github_workflows,
            self.check_docker_build
        ]
        
        # Run each check
        for check_func in check_functions:
            try:
                result = check_func()
                self.checks.append(result)
                
                # Log result
                if result.status == CheckStatus.PASSED:
                    logger.info(f"✅ {result.name}: {result.message}")
                elif result.status == CheckStatus.WARNING:
                    logger.warning(f"⚠️  {result.name}: {result.message}")
                elif result.status == CheckStatus.FAILED:
                    logger.error(f"❌ {result.name}: {result.message}")
                    if result.fix_command:
                        logger.info(f"   Fix: {result.fix_command}")
                elif result.status == CheckStatus.SKIPPED:
                    logger.info(f"⏭️  {result.name}: {result.message}")
                    
            except Exception as e:
                logger.error(f"Error running check {check_func.__name__}: {e}")
                self.checks.append(CheckResult(
                    name=check_func.__name__.replace('check_', '').replace('_', ' ').title(),
                    status=CheckStatus.FAILED,
                    message=f"Check failed with error: {e}"
                ))
        
        # Calculate overall status
        failed_checks = [c for c in self.checks if c.status == CheckStatus.FAILED]
        warning_checks = [c for c in self.checks if c.status == CheckStatus.WARNING]
        
        all_passed = len(failed_checks) == 0
        
        # Print summary
        print("\n" + "="*60)
        print("RELEASE CHECKLIST SUMMARY")
        print("="*60)
        print(f"Total Checks: {len(self.checks)}")
        print(f"Passed: {len([c for c in self.checks if c.status == CheckStatus.PASSED])}")
        print(f"Failed: {len(failed_checks)}")
        print(f"Warnings: {len(warning_checks)}")
        print(f"Skipped: {len([c for c in self.checks if c.status == CheckStatus.SKIPPED])}")
        print("="*60)
        
        if all_passed:
            print("\n✅ All checks passed! Ready for release.")
        else:
            print("\n❌ Some checks failed. Please fix issues before releasing.")
            print("\nFailed checks:")
            for check in failed_checks:
                print(f"  - {check.name}: {check.message}")
                if check.fix_command:
                    print(f"    Fix: {check.fix_command}")
        
        return all_passed, self.checks
    
    def check_version_number(self) -> CheckResult:
        """Check if version number is properly set."""
        if not self.version:
            return CheckResult(
                name="Version Number",
                status=CheckStatus.FAILED,
                message="No version number specified",
                fix_command="Specify version with --version flag"
            )
        
        # Validate version format (semantic versioning)
        version_pattern = re.compile(r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$')
        if not version_pattern.match(self.version):
            return CheckResult(
                name="Version Number",
                status=CheckStatus.FAILED,
                message=f"Invalid version format: {self.version}",
                fix_command="Use semantic versioning (e.g., 1.0.0, 1.0.0-beta.1)"
            )
        
        # Check version in pyproject.toml
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            if f'version = "{self.version}"' not in content:
                return CheckResult(
                    name="Version Number",
                    status=CheckStatus.WARNING,
                    message=f"Version {self.version} not found in pyproject.toml",
                    fix_command=f"Update version in pyproject.toml to {self.version}"
                )
        
        return CheckResult(
            name="Version Number",
            status=CheckStatus.PASSED,
            message=f"Version {self.version} is valid"
        )
    
    def check_git_status(self) -> CheckResult:
        """Check Git repository status."""
        try:
            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                return CheckResult(
                    name="Git Status",
                    status=CheckStatus.FAILED,
                    message="Uncommitted changes detected",
                    details={"changes": result.stdout.strip().split('\n')},
                    fix_command="Commit or stash all changes"
                )
            
            # Check if tag already exists
            if self.version:
                tag_result = subprocess.run(
                    ["git", "tag", "-l", f"v{self.version}"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if tag_result.stdout.strip():
                    return CheckResult(
                        name="Git Status",
                        status=CheckStatus.FAILED,
                        message=f"Tag v{self.version} already exists",
                        fix_command=f"Remove existing tag or use a different version"
                    )
            
            return CheckResult(
                name="Git Status",
                status=CheckStatus.PASSED,
                message="Working directory clean"
            )
            
        except Exception as e:
            return CheckResult(
                name="Git Status",
                status=CheckStatus.FAILED,
                message=f"Failed to check Git status: {e}"
            )
    
    def check_branch(self) -> CheckResult:
        """Check if on correct branch for release."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            current_branch = result.stdout.strip()
            
            # Define allowed release branches
            allowed_branches = ["main", "master", "release", "develop"]
            
            if current_branch not in allowed_branches:
                return CheckResult(
                    name="Git Branch",
                    status=CheckStatus.WARNING,
                    message=f"Not on standard release branch: {current_branch}",
                    fix_command=f"Switch to main/master branch for release"
                )
            
            # Check if branch is up to date with remote
            fetch_result = subprocess.run(
                ["git", "fetch"],
                cwd=self.project_root,
                capture_output=True
            )
            
            status_result = subprocess.run(
                ["git", "status", "-uno"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if "Your branch is behind" in status_result.stdout:
                return CheckResult(
                    name="Git Branch",
                    status=CheckStatus.FAILED,
                    message="Branch is behind remote",
                    fix_command="Pull latest changes: git pull"
                )
            
            return CheckResult(
                name="Git Branch",
                status=CheckStatus.PASSED,
                message=f"On branch {current_branch} and up to date"
            )
            
        except Exception as e:
            return CheckResult(
                name="Git Branch",
                status=CheckStatus.FAILED,
                message=f"Failed to check branch: {e}"
            )
    
    def check_tests(self) -> CheckResult:
        """Check if all tests pass."""
        try:
            # Run pytest
            result = subprocess.run(
                ["pytest", "-v", "--tb=short"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Extract failure summary
                output_lines = result.stdout.split('\n')
                failed_tests = [line for line in output_lines if "FAILED" in line]
                
                return CheckResult(
                    name="Tests",
                    status=CheckStatus.FAILED,
                    message=f"Tests failed ({len(failed_tests)} failures)",
                    details={"failed_tests": failed_tests[:5]},  # First 5 failures
                    fix_command="Fix failing tests before release"
                )
            
            # Check test coverage
            coverage_result = subprocess.run(
                ["pytest", "--cov=simpulse", "--cov-report=term"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Extract coverage percentage
            coverage_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', coverage_result.stdout)
            if coverage_match:
                coverage = int(coverage_match.group(1))
                if coverage < 85:
                    return CheckResult(
                        name="Tests",
                        status=CheckStatus.WARNING,
                        message=f"Test coverage is {coverage}% (target: 85%)",
                        fix_command="Add more tests to improve coverage"
                    )
            
            return CheckResult(
                name="Tests",
                status=CheckStatus.PASSED,
                message="All tests passed with good coverage"
            )
            
        except FileNotFoundError:
            return CheckResult(
                name="Tests",
                status=CheckStatus.FAILED,
                message="pytest not found",
                fix_command="Install pytest: pip install pytest pytest-cov"
            )
        except Exception as e:
            return CheckResult(
                name="Tests",
                status=CheckStatus.FAILED,
                message=f"Failed to run tests: {e}"
            )
    
    def check_code_quality(self) -> CheckResult:
        """Check code quality with linters."""
        issues = []
        
        # Run flake8
        try:
            result = subprocess.run(
                ["flake8", "src/simpulse", "--count"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                issues.append(f"flake8: {result.stdout.strip().split()[-1]} issues")
        except FileNotFoundError:
            issues.append("flake8 not installed")
        
        # Run mypy
        try:
            result = subprocess.run(
                ["mypy", "src/simpulse", "--ignore-missing-imports"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                error_count = len([l for l in result.stdout.split('\n') if ': error:' in l])
                if error_count > 0:
                    issues.append(f"mypy: {error_count} type errors")
        except FileNotFoundError:
            issues.append("mypy not installed")
        
        # Run black check
        try:
            result = subprocess.run(
                ["black", "--check", "src/simpulse"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                issues.append("black: formatting issues")
        except FileNotFoundError:
            issues.append("black not installed")
        
        if issues:
            return CheckResult(
                name="Code Quality",
                status=CheckStatus.WARNING,
                message=f"Code quality issues: {', '.join(issues)}",
                fix_command="Run: black . && flake8 && mypy src/simpulse"
            )
        
        return CheckResult(
            name="Code Quality",
            status=CheckStatus.PASSED,
            message="Code quality checks passed"
        )
    
    def check_documentation(self) -> CheckResult:
        """Check documentation completeness."""
        issues = []
        
        # Check README
        readme_path = self.project_root / "README.md"
        if not readme_path.exists():
            issues.append("README.md missing")
        else:
            readme_content = readme_path.read_text()
            # Check for essential sections
            required_sections = ["Installation", "Usage", "Documentation", "License"]
            for section in required_sections:
                if section.lower() not in readme_content.lower():
                    issues.append(f"README missing {section} section")
        
        # Check API documentation
        docs_dir = self.project_root / "docs"
        if not docs_dir.exists():
            issues.append("docs/ directory missing")
        else:
            # Check for key documentation files
            expected_docs = ["index.md", "api_reference.md", "examples.md"]
            for doc in expected_docs:
                if not (docs_dir / doc).exists():
                    issues.append(f"Missing documentation: {doc}")
        
        # Check docstrings
        src_dir = self.project_root / "src" / "simpulse"
        if src_dir.exists():
            py_files = list(src_dir.rglob("*.py"))
            files_without_docstring = []
            
            for py_file in py_files[:10]:  # Check first 10 files
                content = py_file.read_text()
                if not content.strip().startswith('"""'):
                    files_without_docstring.append(py_file.name)
            
            if files_without_docstring:
                issues.append(f"{len(files_without_docstring)} files missing module docstrings")
        
        if issues:
            return CheckResult(
                name="Documentation",
                status=CheckStatus.WARNING,
                message=f"Documentation issues: {', '.join(issues[:3])}",
                fix_command="Update documentation before release"
            )
        
        return CheckResult(
            name="Documentation",
            status=CheckStatus.PASSED,
            message="Documentation is complete"
        )
    
    def check_changelog(self) -> CheckResult:
        """Check if CHANGELOG is updated."""
        changelog_path = self.project_root / "CHANGELOG.md"
        
        if not changelog_path.exists():
            return CheckResult(
                name="Changelog",
                status=CheckStatus.FAILED,
                message="CHANGELOG.md not found",
                fix_command="Create CHANGELOG.md with release notes"
            )
        
        content = changelog_path.read_text()
        
        # Check if version is mentioned
        if self.version and self.version not in content:
            return CheckResult(
                name="Changelog",
                status=CheckStatus.FAILED,
                message=f"Version {self.version} not found in CHANGELOG",
                fix_command=f"Add release notes for version {self.version}"
            )
        
        # Check for unreleased section
        if "unreleased" in content.lower() and len(content.split("unreleased")[-1].strip()) < 50:
            return CheckResult(
                name="Changelog",
                status=CheckStatus.WARNING,
                message="Unreleased section is empty",
                fix_command="Document changes in CHANGELOG"
            )
        
        return CheckResult(
            name="Changelog",
            status=CheckStatus.PASSED,
            message="CHANGELOG is up to date"
        )
    
    def check_dependencies(self) -> CheckResult:
        """Check dependency specifications."""
        issues = []
        
        # Check pyproject.toml
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            if "[tool.poetry.dependencies]" not in content and "[project.dependencies]" not in content:
                issues.append("No dependencies section in pyproject.toml")
        else:
            issues.append("pyproject.toml not found")
        
        # Check requirements files
        req_files = ["requirements.txt", "requirements-dev.txt"]
        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                # Check for unpinned dependencies
                content = req_path.read_text()
                unpinned = [line for line in content.split('\n') 
                          if line.strip() and not line.startswith('#') 
                          and '==' not in line and '>=' not in line]
                if unpinned:
                    issues.append(f"{req_file} has {len(unpinned)} unpinned dependencies")
        
        if issues:
            return CheckResult(
                name="Dependencies",
                status=CheckStatus.WARNING,
                message=f"Dependency issues: {', '.join(issues)}",
                fix_command="Pin all dependencies for reproducible builds"
            )
        
        return CheckResult(
            name="Dependencies",
            status=CheckStatus.PASSED,
            message="Dependencies are properly specified"
        )
    
    def check_security(self) -> CheckResult:
        """Check for security issues."""
        issues = []
        
        # Check for hardcoded secrets
        patterns = [
            (r'api_key\s*=\s*["\'][^"\']+["\']', "hardcoded API key"),
            (r'password\s*=\s*["\'][^"\']+["\']', "hardcoded password"),
            (r'token\s*=\s*["\'][^"\']+["\']', "hardcoded token"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "hardcoded secret")
        ]
        
        src_dir = self.project_root / "src"
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                content = py_file.read_text()
                for pattern, desc in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(f"{desc} in {py_file.name}")
        
        # Run safety check if available
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                vulnerabilities = json.loads(result.stdout)
                if vulnerabilities:
                    issues.append(f"{len(vulnerabilities)} known vulnerabilities")
        except FileNotFoundError:
            pass  # safety not installed
        
        if issues:
            return CheckResult(
                name="Security",
                status=CheckStatus.FAILED,
                message=f"Security issues found: {', '.join(issues[:3])}",
                fix_command="Fix security issues before release"
            )
        
        return CheckResult(
            name="Security",
            status=CheckStatus.PASSED,
            message="No security issues detected"
        )
    
    def check_benchmarks(self) -> CheckResult:
        """Check if benchmarks are up to date."""
        benchmark_dir = self.project_root / "benchmarks"
        
        if not benchmark_dir.exists():
            return CheckResult(
                name="Benchmarks",
                status=CheckStatus.WARNING,
                message="No benchmarks directory found",
                fix_command="Add performance benchmarks"
            )
        
        # Check for recent benchmark results
        result_files = list(benchmark_dir.glob("*results*.json"))
        if not result_files:
            return CheckResult(
                name="Benchmarks",
                status=CheckStatus.WARNING,
                message="No benchmark results found",
                fix_command="Run benchmarks: python benchmarks/performance_optimization.py"
            )
        
        # Check if results are recent (within 7 days)
        import time
        latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
        age_days = (time.time() - latest_result.stat().st_mtime) / (24 * 3600)
        
        if age_days > 7:
            return CheckResult(
                name="Benchmarks",
                status=CheckStatus.WARNING,
                message=f"Benchmark results are {int(age_days)} days old",
                fix_command="Run fresh benchmarks before release"
            )
        
        return CheckResult(
            name="Benchmarks",
            status=CheckStatus.PASSED,
            message="Benchmarks are up to date"
        )
    
    def check_examples(self) -> CheckResult:
        """Check if examples are working."""
        examples_dir = self.project_root / "examples"
        
        if not examples_dir.exists():
            return CheckResult(
                name="Examples",
                status=CheckStatus.WARNING,
                message="No examples directory found",
                fix_command="Add usage examples"
            )
        
        # Check example files
        example_files = list(examples_dir.glob("*.py"))
        if not example_files:
            return CheckResult(
                name="Examples",
                status=CheckStatus.WARNING,
                message="No example files found",
                fix_command="Add Python example files"
            )
        
        # Try to run examples (syntax check only)
        failed_examples = []
        for example in example_files[:5]:  # Check first 5
            try:
                result = subprocess.run(
                    ["python", "-m", "py_compile", str(example)],
                    capture_output=True
                )
                if result.returncode != 0:
                    failed_examples.append(example.name)
            except:
                failed_examples.append(example.name)
        
        if failed_examples:
            return CheckResult(
                name="Examples",
                status=CheckStatus.FAILED,
                message=f"{len(failed_examples)} examples have syntax errors",
                fix_command="Fix example code"
            )
        
        return CheckResult(
            name="Examples",
            status=CheckStatus.PASSED,
            message=f"{len(example_files)} examples verified"
        )
    
    def check_license(self) -> CheckResult:
        """Check if LICENSE file exists."""
        license_path = self.project_root / "LICENSE"
        
        if not license_path.exists():
            # Also check for LICENSE.txt or LICENSE.md
            alt_licenses = ["LICENSE.txt", "LICENSE.md"]
            for alt in alt_licenses:
                if (self.project_root / alt).exists():
                    license_path = self.project_root / alt
                    break
            else:
                return CheckResult(
                    name="License",
                    status=CheckStatus.FAILED,
                    message="LICENSE file not found",
                    fix_command="Add LICENSE file"
                )
        
        # Check if license is mentioned in README
        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            readme_content = readme_path.read_text().lower()
            if "license" not in readme_content:
                return CheckResult(
                    name="License",
                    status=CheckStatus.WARNING,
                    message="License not mentioned in README",
                    fix_command="Add license section to README"
                )
        
        return CheckResult(
            name="License",
            status=CheckStatus.PASSED,
            message="LICENSE file exists"
        )
    
    def check_package_metadata(self) -> CheckResult:
        """Check package metadata completeness."""
        issues = []
        
        # Check pyproject.toml
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            
            # Required fields
            required_fields = [
                "name", "version", "description", "authors",
                "license", "readme", "repository", "keywords"
            ]
            
            for field in required_fields:
                if f'{field} =' not in content:
                    issues.append(f"Missing {field} in pyproject.toml")
        else:
            issues.append("pyproject.toml not found")
        
        # Check setup.py if exists
        setup_path = self.project_root / "setup.py"
        if setup_path.exists():
            content = setup_path.read_text()
            if "long_description" not in content:
                issues.append("Missing long_description in setup.py")
        
        if issues:
            return CheckResult(
                name="Package Metadata",
                status=CheckStatus.WARNING,
                message=f"Metadata issues: {', '.join(issues[:3])}",
                fix_command="Complete package metadata"
            )
        
        return CheckResult(
            name="Package Metadata",
            status=CheckStatus.PASSED,
            message="Package metadata is complete"
        )
    
    def check_github_workflows(self) -> CheckResult:
        """Check GitHub Actions workflows."""
        workflows_dir = self.project_root / ".github" / "workflows"
        
        if not workflows_dir.exists():
            return CheckResult(
                name="GitHub Workflows",
                status=CheckStatus.WARNING,
                message="No GitHub workflows found",
                fix_command="Add CI/CD workflows"
            )
        
        # Check for essential workflows
        expected_workflows = ["ci.yml", "release.yml"]
        missing_workflows = []
        
        for workflow in expected_workflows:
            workflow_path = workflows_dir / workflow
            if not workflow_path.exists():
                # Check for alternative names
                alternatives = [
                    workflow.replace('.yml', '.yaml'),
                    f"test.yml" if workflow == "ci.yml" else workflow,
                    f"publish.yml" if workflow == "release.yml" else workflow
                ]
                
                found = False
                for alt in alternatives:
                    if (workflows_dir / alt).exists():
                        found = True
                        break
                
                if not found:
                    missing_workflows.append(workflow)
        
        if missing_workflows:
            return CheckResult(
                name="GitHub Workflows",
                status=CheckStatus.WARNING,
                message=f"Missing workflows: {', '.join(missing_workflows)}",
                fix_command="Add GitHub Actions workflows"
            )
        
        return CheckResult(
            name="GitHub Workflows",
            status=CheckStatus.PASSED,
            message="GitHub workflows configured"
        )
    
    def check_docker_build(self) -> CheckResult:
        """Check if Docker image builds successfully."""
        dockerfile_path = self.project_root / "Dockerfile"
        
        if not dockerfile_path.exists():
            return CheckResult(
                name="Docker Build",
                status=CheckStatus.SKIPPED,
                message="No Dockerfile found"
            )
        
        # Try to build Docker image
        try:
            result = subprocess.run(
                ["docker", "build", "-t", "simpulse-test:latest", "."],
                cwd=self.project_root,
                capture_output=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                return CheckResult(
                    name="Docker Build",
                    status=CheckStatus.FAILED,
                    message="Docker build failed",
                    fix_command="Fix Dockerfile issues"
                )
            
            # Clean up test image
            subprocess.run(
                ["docker", "rmi", "simpulse-test:latest"],
                capture_output=True
            )
            
            return CheckResult(
                name="Docker Build",
                status=CheckStatus.PASSED,
                message="Docker image builds successfully"
            )
            
        except FileNotFoundError:
            return CheckResult(
                name="Docker Build",
                status=CheckStatus.SKIPPED,
                message="Docker not available"
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                name="Docker Build",
                status=CheckStatus.FAILED,
                message="Docker build timeout",
                fix_command="Optimize Docker build"
            )
        except Exception as e:
            return CheckResult(
                name="Docker Build",
                status=CheckStatus.FAILED,
                message=f"Docker build error: {e}"
            )
    
    def generate_report(self, output_path: Path) -> bool:
        """Generate release checklist report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            True if successful
        """
        try:
            lines = [
                "# Simpulse Release Checklist Report",
                "",
                f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Version**: {self.version or 'Not specified'}",
                f"**Project**: {self.project_root}",
                "",
                "## Summary",
                "",
                f"- **Total Checks**: {len(self.checks)}",
                f"- **Passed**: {len([c for c in self.checks if c.status == CheckStatus.PASSED])}",
                f"- **Failed**: {len([c for c in self.checks if c.status == CheckStatus.FAILED])}",
                f"- **Warnings**: {len([c for c in self.checks if c.status == CheckStatus.WARNING])}",
                f"- **Skipped**: {len([c for c in self.checks if c.status == CheckStatus.SKIPPED])}",
                "",
                "## Detailed Results",
                ""
            ]
            
            # Group by status
            for status in CheckStatus:
                status_checks = [c for c in self.checks if c.status == status]
                if status_checks:
                    lines.append(f"### {status.value.title()}")
                    lines.append("")
                    
                    for check in status_checks:
                        icon = {
                            CheckStatus.PASSED: "✅",
                            CheckStatus.FAILED: "❌",
                            CheckStatus.WARNING: "⚠️",
                            CheckStatus.SKIPPED: "⏭️",
                            CheckStatus.PENDING: "⏳"
                        }.get(check.status, "❓")
                        
                        lines.append(f"- {icon} **{check.name}**: {check.message}")
                        if check.fix_command:
                            lines.append(f"  - Fix: `{check.fix_command}`")
                        if check.details:
                            lines.append(f"  - Details: {json.dumps(check.details, indent=2)}")
                    
                    lines.append("")
            
            # Add action items
            failed_checks = [c for c in self.checks if c.status == CheckStatus.FAILED]
            if failed_checks:
                lines.extend([
                    "## Action Items",
                    "",
                    "The following issues must be fixed before release:",
                    ""
                ])
                
                for i, check in enumerate(failed_checks, 1):
                    lines.append(f"{i}. **{check.name}**: {check.message}")
                    if check.fix_command:
                        lines.append(f"   - Run: `{check.fix_command}`")
                    lines.append("")
            
            # Write report
            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"Release checklist report saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run automated release preparation checklist"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Target release version (e.g., 1.0.0)"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("release_checklist_report.md"),
        help="Output file for report"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to auto-fix issues"
    )
    
    args = parser.parse_args()
    
    # Run checklist
    checklist = ReleaseChecklist(args.project_root, args.version)
    all_passed, results = checklist.run_all_checks()
    
    # Generate report
    checklist.generate_report(args.report)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Git setup script for Simpulse development.

This script configures Git hooks, branch protection, and development
workflow for the Simpulse project.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class GitSetup:
    """Handles Git configuration for Simpulse project."""
    
    def __init__(self, project_root: Path):
        """Initialize Git setup.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.git_dir = project_root / ".git"
        self.hooks_dir = self.git_dir / "hooks"
        
    def is_git_repo(self) -> bool:
        """Check if directory is a Git repository."""
        return self.git_dir.exists() and self.git_dir.is_dir()
    
    def setup_pre_commit_hook(self) -> bool:
        """Setup pre-commit hook for code quality checks."""
        hook_content = '''#!/bin/bash
# Simpulse pre-commit hook

echo "Running pre-commit checks..."

# Check for Python files
if git diff --cached --name-only | grep -q '\.py$'; then
    echo "Checking Python code..."
    
    # Run Black formatter check
    if command -v black &> /dev/null; then
        echo "  - Running Black formatter check..."
        black --check --diff $(git diff --cached --name-only | grep '\.py$')
        if [ $? -ne 0 ]; then
            echo "❌ Black formatting check failed. Run 'black .' to fix."
            exit 1
        fi
    fi
    
    # Run flake8 linter
    if command -v flake8 &> /dev/null; then
        echo "  - Running flake8 linter..."
        flake8 $(git diff --cached --name-only | grep '\.py$')
        if [ $? -ne 0 ]; then
            echo "❌ Flake8 linting failed."
            exit 1
        fi
    fi
    
    # Run mypy type checker
    if command -v mypy &> /dev/null; then
        echo "  - Running mypy type checker..."
        mypy $(git diff --cached --name-only | grep '\.py$') --ignore-missing-imports
        if [ $? -ne 0 ]; then
            echo "❌ Mypy type checking failed."
            exit 1
        fi
    fi
fi

# Check for Lean files
if git diff --cached --name-only | grep -q '\.lean$'; then
    echo "Checking Lean code..."
    
    # Run lean check if available
    if command -v lake &> /dev/null; then
        echo "  - Running Lean syntax check..."
        lake build --no-build
        if [ $? -ne 0 ]; then
            echo "❌ Lean syntax check failed."
            exit 1
        fi
    fi
fi

# Check for sensitive data
echo "Checking for sensitive data..."
if git diff --cached --name-only | xargs grep -E -i "(api_key|apikey|password|secret|token|credential)" 2>/dev/null; then
    echo "⚠️  Warning: Possible sensitive data detected. Please review before committing."
    echo "Continue anyway? (y/N)"
    read -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ All pre-commit checks passed!"
'''
        
        try:
            hook_path = self.hooks_dir / "pre-commit"
            with open(hook_path, 'w') as f:
                f.write(hook_content)
            
            # Make executable
            os.chmod(hook_path, 0o755)
            
            logger.info("✅ Pre-commit hook installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to install pre-commit hook: {e}")
            return False
    
    def setup_pre_push_hook(self) -> bool:
        """Setup pre-push hook for test verification."""
        hook_content = '''#!/bin/bash
# Simpulse pre-push hook

echo "Running pre-push checks..."

# Run tests before push
echo "Running tests..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v --tb=short
    if [ $? -ne 0 ]; then
        echo "❌ Tests failed. Fix failing tests before pushing."
        exit 1
    fi
else
    echo "⚠️  pytest not found. Skipping tests."
fi

# Check branch protection
protected_branches=("main" "master" "production")
current_branch=$(git symbolic-ref HEAD | sed -e 's,.*/\\(.*\\),\\1,')

for branch in "${protected_branches[@]}"; do
    if [[ "$current_branch" == "$branch" ]]; then
        echo "⚠️  You're pushing to protected branch '$branch'."
        echo "Are you sure? (y/N)"
        read -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
done

echo "✅ All pre-push checks passed!"
'''
        
        try:
            hook_path = self.hooks_dir / "pre-push"
            with open(hook_path, 'w') as f:
                f.write(hook_content)
            
            # Make executable
            os.chmod(hook_path, 0o755)
            
            logger.info("✅ Pre-push hook installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to install pre-push hook: {e}")
            return False
    
    def setup_commit_msg_hook(self) -> bool:
        """Setup commit message hook for conventional commits."""
        hook_content = '''#!/bin/bash
# Simpulse commit message hook

commit_regex='^(feat|fix|docs|style|refactor|test|chore|perf)(\(.+\))?: .{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "❌ Invalid commit message format!"
    echo ""
    echo "Commit message must follow conventional commits format:"
    echo "  <type>(<scope>): <subject>"
    echo ""
    echo "Types:"
    echo "  feat:     A new feature"
    echo "  fix:      A bug fix"
    echo "  docs:     Documentation changes"
    echo "  style:    Code style changes (formatting, etc)"
    echo "  refactor: Code refactoring"
    echo "  test:     Adding or updating tests"
    echo "  chore:    Maintenance tasks"
    echo "  perf:     Performance improvements"
    echo ""
    echo "Example: feat(evolution): add adaptive mutation strategy"
    exit 1
fi
'''
        
        try:
            hook_path = self.hooks_dir / "commit-msg"
            with open(hook_path, 'w') as f:
                f.write(hook_content)
            
            # Make executable
            os.chmod(hook_path, 0o755)
            
            logger.info("✅ Commit message hook installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to install commit-msg hook: {e}")
            return False
    
    def configure_git_attributes(self) -> bool:
        """Configure .gitattributes for proper file handling."""
        attributes_content = '''# Auto detect text files and perform LF normalization
* text=auto

# Python files
*.py text diff=python
*.pyw text diff=python
*.pyx text diff=python
*.pyi text diff=python

# Lean files
*.lean text
*.olean binary

# Documentation
*.md text
*.rst text
*.txt text

# Configuration
*.json text
*.yaml text
*.yml text
*.toml text
*.ini text
*.cfg text

# Scripts
*.sh text eol=lf
*.bash text eol=lf

# Data files
*.csv text
*.tsv text
*.parquet binary
*.pickle binary
*.pkl binary

# Archives
*.7z binary
*.gz binary
*.tar binary
*.zip binary

# Images
*.jpg binary
*.jpeg binary
*.png binary
*.gif binary
*.svg text

# Notebooks
*.ipynb text
'''
        
        try:
            attributes_path = self.project_root / ".gitattributes"
            with open(attributes_path, 'w') as f:
                f.write(attributes_content)
            
            logger.info("✅ .gitattributes configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure .gitattributes: {e}")
            return False
    
    def configure_git_ignore(self) -> bool:
        """Configure .gitignore for Simpulse project."""
        ignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
ENV/
env/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.project
.pydevproject

# Testing
.coverage
.pytest_cache/
htmlcov/
.tox/
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# Lean
*.olean
build/
lake-packages/

# Simpulse specific
simpulse_output/
logs/
.simpulse/
*.profile.json
benchmarks/results/
cache/

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Secrets
.env
.env.local
.env.*.local
*.key
*.pem
secrets/

# Documentation
docs/_build/
site/

# Temporary files
*.tmp
*.temp
*.log
*.bak
.backup/
'''
        
        try:
            gitignore_path = self.project_root / ".gitignore"
            
            # Append to existing .gitignore if it exists
            existing_content = ""
            if gitignore_path.exists():
                existing_content = gitignore_path.read_text()
            
            # Only add if not already present
            if "# Simpulse specific" not in existing_content:
                with open(gitignore_path, 'a') as f:
                    if existing_content and not existing_content.endswith('\n'):
                        f.write('\n')
                    f.write('\n' + ignore_content)
            
            logger.info("✅ .gitignore configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure .gitignore: {e}")
            return False
    
    def setup_git_config(self) -> bool:
        """Setup recommended Git configuration."""
        configs = [
            # Better diff algorithm
            ("diff.algorithm", "histogram"),
            
            # Helpful aliases
            ("alias.co", "checkout"),
            ("alias.br", "branch"),
            ("alias.ci", "commit"),
            ("alias.st", "status"),
            ("alias.unstage", "reset HEAD --"),
            ("alias.last", "log -1 HEAD"),
            ("alias.visual", "log --graph --oneline --all"),
            
            # Auto-stash on rebase
            ("rebase.autoStash", "true"),
            
            # Better merge conflict markers
            ("merge.conflictStyle", "diff3"),
            
            # Prune on fetch
            ("fetch.prune", "true"),
            
            # Use SSH for GitHub
            ("url.git@github.com:.insteadOf", "https://github.com/"),
        ]
        
        try:
            for key, value in configs:
                result = subprocess.run(
                    ["git", "config", key, value],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.warning(f"Failed to set {key}: {result.stderr}")
            
            logger.info("✅ Git configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update Git configuration: {e}")
            return False
    
    def create_branch_structure(self) -> bool:
        """Create recommended branch structure."""
        branches = [
            ("develop", "Development branch"),
            ("feature/template", "Template for feature branches"),
            ("hotfix/template", "Template for hotfix branches"),
        ]
        
        try:
            current_branch = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            ).stdout.strip()
            
            for branch_name, description in branches:
                if branch_name.endswith("/template"):
                    # Create template branches as orphans
                    subprocess.run(
                        ["git", "checkout", "--orphan", branch_name],
                        cwd=self.project_root,
                        capture_output=True
                    )
                    
                    # Create README for template
                    readme_content = f"# {branch_name}\n\n{description}\n\nDelete this file and start your work."
                    readme_path = self.project_root / "README_TEMPLATE.md"
                    readme_path.write_text(readme_content)
                    
                    subprocess.run(
                        ["git", "add", "README_TEMPLATE.md"],
                        cwd=self.project_root,
                        capture_output=True
                    )
                    
                    subprocess.run(
                        ["git", "commit", "-m", f"chore: create {branch_name}"],
                        cwd=self.project_root,
                        capture_output=True
                    )
                else:
                    # Create regular branches
                    subprocess.run(
                        ["git", "checkout", "-b", branch_name],
                        cwd=self.project_root,
                        capture_output=True
                    )
            
            # Return to original branch
            subprocess.run(
                ["git", "checkout", current_branch],
                cwd=self.project_root,
                capture_output=True
            )
            
            logger.info("✅ Branch structure created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create branch structure: {e}")
            return False
    
    def run_full_setup(self) -> bool:
        """Run complete Git setup."""
        if not self.is_git_repo():
            logger.error("❌ Not a Git repository. Run 'git init' first.")
            return False
        
        logger.info("Starting Git setup for Simpulse...")
        
        # Run all setup steps
        steps = [
            ("Pre-commit hook", self.setup_pre_commit_hook),
            ("Pre-push hook", self.setup_pre_push_hook),
            ("Commit message hook", self.setup_commit_msg_hook),
            (".gitattributes", self.configure_git_attributes),
            (".gitignore", self.configure_git_ignore),
            ("Git configuration", self.setup_git_config),
        ]
        
        success = True
        for step_name, step_func in steps:
            logger.info(f"Setting up {step_name}...")
            if not step_func():
                success = False
        
        if success:
            logger.info("\n✅ Git setup completed successfully!")
            logger.info("\nRecommended next steps:")
            logger.info("1. Install development dependencies:")
            logger.info("   pip install black flake8 mypy pytest pre-commit")
            logger.info("2. Set up pre-commit framework:")
            logger.info("   pre-commit install")
            logger.info("3. Create feature branch:")
            logger.info("   git checkout -b feature/your-feature")
        else:
            logger.error("\n❌ Git setup completed with errors.")
        
        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup Git configuration for Simpulse development"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--skip-branches",
        action="store_true",
        help="Skip creating branch structure"
    )
    
    args = parser.parse_args()
    
    setup = GitSetup(args.project_root)
    
    if not args.skip_branches:
        # Ask before creating branches
        response = input("Create recommended branch structure? (y/N): ")
        if response.lower() == 'y':
            setup.create_branch_structure()
    
    success = setup.run_full_setup()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Comprehensive production readiness audit for Simpulse.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class ProductionAudit:
    """Audit Simpulse for production readiness."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.src_path = repo_path / "src" / "simpulse"
        self.issues = {"critical": [], "warning": [], "info": []}
        
    def run_full_audit(self) -> Dict[str, any]:
        """Run comprehensive production audit."""
        print("SIMPULSE PRODUCTION READINESS AUDIT")
        print("="*70)
        
        results = {
            "test_coverage": self.check_test_coverage(),
            "documentation": self.verify_documentation(),
            "security": self.security_audit(),
            "user_experience": self.user_experience_test(),
            "code_quality": self.check_code_quality(),
            "dependencies": self.check_dependencies(),
            "core_functionality": self.verify_core_functionality()
        }
        
        # Generate summary
        results["summary"] = self.generate_summary()
        results["ready_for_production"] = self._is_ready_for_production()
        
        return results
    
    def check_test_coverage(self) -> Dict[str, any]:
        """Verify actual test coverage."""
        print("\n1. CHECKING TEST COVERAGE")
        print("-"*50)
        
        # Check if tests exist
        test_files = list((self.repo_path / "tests").rglob("test_*.py"))
        print(f"Found {len(test_files)} test files")
        
        # Try to run coverage
        try:
            result = subprocess.run(
                ["python", "-m", "coverage", "run", "-m", "pytest", "tests/"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Get coverage report
                cov_result = subprocess.run(
                    ["python", "-m", "coverage", "report"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                
                # Parse coverage
                coverage_match = re.search(r'TOTAL.*?(\d+)%', cov_result.stdout)
                if coverage_match:
                    coverage = int(coverage_match.group(1))
                    print(f"Test coverage: {coverage}%")
                    
                    if coverage < 50:
                        self.issues["critical"].append(f"Test coverage only {coverage}% (need 80%+)")
                    elif coverage < 80:
                        self.issues["warning"].append(f"Test coverage {coverage}% (recommend 80%+)")
                    
                    return {"coverage": coverage, "status": "measured"}
                    
        except subprocess.CalledProcessError:
            pass
        
        # Fallback: estimate based on file count
        src_files = list(self.src_path.rglob("*.py"))
        coverage_estimate = (len(test_files) / max(len(src_files), 1)) * 100
        
        self.issues["critical"].append("No real test coverage data - tests may not be running")
        
        return {
            "coverage": coverage_estimate,
            "status": "estimated",
            "test_files": len(test_files),
            "src_files": len(src_files)
        }
    
    def verify_documentation(self) -> Dict[str, any]:
        """Ensure all docs are accurate."""
        print("\n2. VERIFYING DOCUMENTATION")
        print("-"*50)
        
        docs_status = {}
        
        # Check essential files
        essential_docs = ["README.md", "LICENSE", "CONTRIBUTING.md", "CHANGELOG.md"]
        for doc in essential_docs:
            doc_path = self.repo_path / doc
            if doc_path.exists():
                content = doc_path.read_text()
                docs_status[doc] = {
                    "exists": True,
                    "size": len(content),
                    "has_content": len(content) > 100
                }
                
                # Specific checks
                if doc == "README.md":
                    if "pip install simpulse" in content:
                        if "experimental" not in content.lower() and "alpha" not in content.lower():
                            self.issues["warning"].append("README should clearly state experimental status")
                    
                    if "30%" in content and "simulation" not in content.lower():
                        self.issues["critical"].append("README claims 30% improvement without mentioning it's from simulation")
                        
            else:
                docs_status[doc] = {"exists": False}
                self.issues["critical"].append(f"Missing essential documentation: {doc}")
        
        # Check if examples work
        example_files = list((self.repo_path / "examples").glob("*.py"))
        print(f"Found {len(example_files)} examples")
        
        return docs_status
    
    def security_audit(self) -> Dict[str, any]:
        """Final security check."""
        print("\n3. SECURITY AUDIT")
        print("-"*50)
        
        security_issues = []
        
        # Check for dangerous patterns
        dangerous_patterns = {
            r"eval\s*\(": "eval() usage",
            r"exec\s*\(": "exec() usage",
            r"__import__": "dynamic imports",
            r"subprocess\.call\s*\(": "unsafe subprocess usage",
            r"shell\s*=\s*True": "shell injection risk",
            r"api_key|secret|password|token": "potential hardcoded secrets"
        }
        
        for py_file in self.src_path.rglob("*.py"):
            content = py_file.read_text()
            for pattern, description in dangerous_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    # Check if it's in a validation/security context
                    if "validate" not in py_file.name and "security" not in py_file.name:
                        security_issues.append(f"{py_file.name}: {description}")
                        self.issues["critical"].append(f"Security: {description} in {py_file.name}")
        
        # Check file permissions
        for script in self.repo_path.rglob("*.py"):
            if script.stat().st_mode & 0o111:  # Executable
                if script.name not in ["setup.py", "cli_v2.py"]:
                    self.issues["info"].append(f"Unexpected executable: {script.name}")
        
        print(f"Found {len(security_issues)} potential security issues")
        
        return {
            "issues": security_issues,
            "secure": len(security_issues) == 0
        }
    
    def user_experience_test(self) -> Dict[str, any]:
        """Test from new user perspective."""
        print("\n4. USER EXPERIENCE TEST")
        print("-"*50)
        
        ux_issues = []
        
        # Check installation instructions
        readme = self.repo_path / "README.md"
        if readme.exists():
            content = readme.read_text()
            
            # Check for clear installation steps
            if "pip install" not in content:
                ux_issues.append("No clear installation instructions")
                self.issues["critical"].append("README missing installation instructions")
            
            # Check for quick start
            if "quick start" not in content.lower() and "getting started" not in content.lower():
                ux_issues.append("No quick start guide")
                self.issues["warning"].append("README should have a Quick Start section")
        
        # Check if CLI has help
        cli_path = self.src_path / "cli_v2.py"
        if cli_path.exists():
            content = cli_path.read_text()
            if "--help" not in content and "argparse" not in content and "click" not in content:
                ux_issues.append("CLI may lack help documentation")
                self.issues["warning"].append("CLI should have --help option")
        
        # Check error messages
        error_patterns = [
            r'raise\s+\w+\s*\(\s*["\']',  # Basic exception with message
            r'logging\.error',              # Logged errors
            r'print\s*\(\s*["\']Error',    # Print errors
        ]
        
        error_count = 0
        for py_file in self.src_path.rglob("*.py"):
            content = py_file.read_text()
            for pattern in error_patterns:
                error_count += len(re.findall(pattern, content))
        
        print(f"Found {error_count} error messages in code")
        if error_count < 10:
            self.issues["warning"].append("Very few error messages - may have poor error handling")
        
        return {
            "issues": ux_issues,
            "error_handling": error_count > 10,
            "documentation": len(ux_issues) == 0
        }
    
    def check_code_quality(self) -> Dict[str, any]:
        """Check overall code quality."""
        print("\n5. CODE QUALITY CHECK")
        print("-"*50)
        
        quality_metrics = {}
        
        # Count TODOs and FIXMEs
        todo_count = 0
        fixme_count = 0
        
        for py_file in self.src_path.rglob("*.py"):
            content = py_file.read_text()
            todo_count += len(re.findall(r'#\s*TODO', content, re.IGNORECASE))
            fixme_count += len(re.findall(r'#\s*FIXME', content, re.IGNORECASE))
        
        quality_metrics["todos"] = todo_count
        quality_metrics["fixmes"] = fixme_count
        
        if fixme_count > 5:
            self.issues["warning"].append(f"Found {fixme_count} FIXMEs in code")
        
        # Check for proper typing
        typed_functions = 0
        total_functions = 0
        
        for py_file in self.src_path.rglob("*.py"):
            content = py_file.read_text()
            # Simple heuristic
            total_functions += len(re.findall(r'def\s+\w+\s*\(', content))
            typed_functions += len(re.findall(r'def\s+\w+\s*\([^)]*\)\s*->', content))
        
        typing_ratio = typed_functions / max(total_functions, 1)
        quality_metrics["typing_ratio"] = typing_ratio
        
        if typing_ratio < 0.5:
            self.issues["warning"].append(f"Only {typing_ratio:.0%} of functions have type hints")
        
        print(f"TODOs: {todo_count}, FIXMEs: {fixme_count}")
        print(f"Type hints: {typed_functions}/{total_functions} ({typing_ratio:.0%})")
        
        return quality_metrics
    
    def check_dependencies(self) -> Dict[str, any]:
        """Check dependency health."""
        print("\n6. DEPENDENCY CHECK")
        print("-"*50)
        
        # Check requirements.txt
        req_file = self.repo_path / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text()
            if content.strip():
                self.issues["info"].append("requirements.txt has dependencies - good for simple projects")
        
        # Check pyproject.toml
        pyproject = self.repo_path / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            if "dependencies = [" in content and '"]' in content:
                # Extract dependencies section
                if 'dependencies = [\n    # Core functionality uses only Python standard library\n]' in content:
                    print("✓ No external dependencies - excellent!")
                else:
                    self.issues["info"].append("Project has external dependencies")
        
        return {"minimal_dependencies": True}
    
    def verify_core_functionality(self) -> Dict[str, any]:
        """Verify core functionality exists and could work."""
        print("\n7. CORE FUNCTIONALITY CHECK")
        print("-"*50)
        
        core_modules = {
            "rule_extractor": self.src_path / "evolution" / "rule_extractor.py",
            "lean_runner": self.src_path / "profiling" / "lean_runner.py",
            "evolution_engine": self.src_path / "evolution" / "evolution_engine.py",
            "mutation_applicator": self.src_path / "evolution" / "mutation_applicator.py"
        }
        
        core_status = {}
        for name, path in core_modules.items():
            if path.exists():
                content = path.read_text()
                # Check if it has actual implementation
                has_impl = "pass" not in content or len(content) > 1000
                core_status[name] = {
                    "exists": True,
                    "has_implementation": has_impl,
                    "size": len(content)
                }
                
                if not has_impl:
                    self.issues["critical"].append(f"Core module {name} appears to be a stub")
            else:
                core_status[name] = {"exists": False}
                self.issues["critical"].append(f"Missing core module: {name}")
        
        # Check if we can actually work with Lean files
        lean_integration = any(
            "lean" in str(f).lower() 
            for f in self.src_path.rglob("*.py")
        )
        
        if not lean_integration:
            self.issues["critical"].append("No clear Lean integration found")
        
        return core_status
    
    def _is_ready_for_production(self) -> bool:
        """Determine if ready for production."""
        # Critical issues must be zero
        return len(self.issues["critical"]) == 0
    
    def generate_summary(self) -> Dict[str, any]:
        """Generate audit summary."""
        return {
            "critical_issues": len(self.issues["critical"]),
            "warnings": len(self.issues["warning"]),
            "info": len(self.issues["info"]),
            "issues": self.issues
        }
    
    def generate_report(self) -> str:
        """Generate detailed audit report."""
        results = self.run_full_audit()
        
        lines = [
            "SIMPULSE PRODUCTION READINESS AUDIT REPORT",
            "="*70,
            "",
            f"Date: {Path.cwd().name}",
            f"Ready for Production: {'NO ❌' if not results['ready_for_production'] else 'YES ✅'}",
            "",
            "SUMMARY",
            "-"*70,
            f"Critical Issues: {results['summary']['critical_issues']}",
            f"Warnings: {results['summary']['warnings']}",
            f"Info: {results['summary']['info']}",
            "",
        ]
        
        # Critical issues
        if results['summary']['critical_issues'] > 0:
            lines.extend([
                "CRITICAL ISSUES (Must Fix)",
                "-"*70,
            ])
            for issue in self.issues["critical"]:
                lines.append(f"❌ {issue}")
            lines.append("")
        
        # Warnings
        if results['summary']['warnings'] > 0:
            lines.extend([
                "WARNINGS (Should Fix)",
                "-"*70,
            ])
            for issue in self.issues["warning"]:
                lines.append(f"⚠️  {issue}")
            lines.append("")
        
        # Detailed results
        lines.extend([
            "DETAILED RESULTS",
            "-"*70,
            "",
            "1. Test Coverage:",
            f"   Status: {results['test_coverage']['status']}",
            f"   Coverage: {results['test_coverage'].get('coverage', 0):.0f}%",
            "",
            "2. Security:",
            f"   Secure: {'Yes' if results['security']['secure'] else 'No'}",
            f"   Issues: {len(results['security']['issues'])}",
            "",
            "3. Code Quality:",
            f"   TODOs: {results['code_quality']['todos']}",
            f"   FIXMEs: {results['code_quality']['fixmes']}",
            f"   Type Hints: {results['code_quality']['typing_ratio']:.0%}",
            "",
            "4. Core Functionality:",
        ])
        
        for module, status in results['core_functionality'].items():
            if isinstance(status, dict) and status.get('exists'):
                lines.append(f"   ✓ {module}: exists")
            else:
                lines.append(f"   ✗ {module}: missing or incomplete")
        
        # Action items
        lines.extend([
            "",
            "REQUIRED ACTIONS BEFORE RELEASE",
            "-"*70,
            "1. Fix all critical issues listed above",
            "2. Validate on real Lean 4 code (not just simulations)",
            "3. Update README to clearly state experimental status",
            "4. Add proper error handling and user feedback",
            "5. Achieve at least 80% test coverage",
            "6. Document actual performance on real projects",
            "",
            "RECOMMENDATION: Do not release until core Lean integration is proven to work."
        ])
        
        return "\n".join(lines)


def main():
    """Run production readiness audit."""
    repo_path = Path(__file__).parent.parent
    auditor = ProductionAudit(repo_path)
    
    print("Starting production readiness audit...")
    report = auditor.generate_report()
    
    # Save report
    report_path = repo_path / "PRODUCTION_AUDIT.md"
    report_path.write_text(report)
    
    print(f"\nReport saved to: {report_path}")
    print("\n" + report)
    
    # Generate action plan if not ready
    if auditor.issues["critical"]:
        print("\n" + "="*70)
        print("ACTION PLAN")
        print("="*70)
        print("Fix these critical issues first:")
        for i, issue in enumerate(auditor.issues["critical"][:5], 1):
            print(f"{i}. {issue}")


if __name__ == "__main__":
    main()
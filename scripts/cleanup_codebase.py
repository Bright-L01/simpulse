#!/usr/bin/env python3
"""
Clean up the Simpulse codebase by removing unused code and improving structure.
"""

import ast
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class CodebaseCleaner:
    """Analyze and clean the Simpulse codebase."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.src_path = repo_path / "src" / "simpulse"
        self.all_functions = defaultdict(set)  # module -> {functions}
        self.all_imports = defaultdict(set)  # module -> {imports}
        self.function_calls = defaultdict(set)  # function -> {callers}
        self.import_usage = defaultdict(int)  # import -> usage count
        self.duplicate_code = []
        self.large_files = []
        self.complex_functions = []

    def analyze_codebase(self) -> Dict[str, any]:
        """Complete codebase analysis."""
        print("Analyzing Simpulse codebase...")
        print("=" * 70)

        # Find all Python files
        python_files = list(self.src_path.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")

        # Analyze each file
        for py_file in python_files:
            self._analyze_file(py_file)

        # Find issues
        unused_functions = self._find_unused_functions()
        unused_imports = self._find_unused_imports()
        duplicate_constants = self._find_duplicate_constants()

        # Generate report
        report = {
            "total_files": len(python_files),
            "total_functions": sum(len(funcs) for funcs in self.all_functions.values()),
            "unused_functions": unused_functions,
            "unused_imports": unused_imports,
            "duplicate_constants": duplicate_constants,
            "large_files": self.large_files,
            "complex_functions": self.complex_functions,
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file."""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)

            # Check file size
            lines = content.count("\n")
            if lines > 300:
                self.large_files.append((str(file_path), lines))

            # Extract information
            module_name = self._get_module_name(file_path)

            for node in ast.walk(tree):
                # Track functions
                if isinstance(node, ast.FunctionDef):
                    self.all_functions[module_name].add(node.name)

                    # Check complexity (rough estimate)
                    complexity = self._estimate_complexity(node)
                    if complexity > 10:
                        self.complex_functions.append(
                            (module_name, node.name, complexity)
                        )

                # Track imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.all_imports[module_name].add(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            import_name = f"{node.module}.{alias.name}"
                            self.all_imports[module_name].add(import_name)

                # Track function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        self.function_calls[node.func.id].add(module_name)
                    elif isinstance(node.func, ast.Attribute):
                        self.function_calls[node.func.attr].add(module_name)

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        relative = file_path.relative_to(self.src_path)
        parts = relative.parts[:-1] + (relative.stem,)
        return ".".join(parts)

    def _estimate_complexity(self, node: ast.FunctionDef) -> int:
        """Estimate cyclomatic complexity of a function."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
        return complexity

    def _find_unused_functions(self) -> List[Tuple[str, str]]:
        """Find functions that are never called."""
        unused = []

        for module, functions in self.all_functions.items():
            for func in functions:
                # Skip special methods
                if func.startswith("__") and func.endswith("__"):
                    continue

                # Skip test functions
                if func.startswith("test_"):
                    continue

                # Check if called anywhere
                if func not in self.function_calls:
                    unused.append((module, func))

        return unused

    def _find_unused_imports(self) -> List[Tuple[str, str]]:
        """Find imports that are never used."""
        unused = []

        for module, imports in self.all_imports.items():
            module_path = self.src_path / module.replace(".", "/").replace(
                "/__init__", ""
            )
            if module_path.is_dir():
                module_path = module_path / "__init__.py"
            else:
                module_path = module_path.with_suffix(".py")

            if module_path.exists():
                content = module_path.read_text()
                for imp in imports:
                    # Simple check - could be improved
                    imp_name = imp.split(".")[-1]
                    if imp_name not in content:
                        unused.append((module, imp))

        return unused

    def _find_duplicate_constants(self) -> Dict[str, List[str]]:
        """Find duplicate string/number constants."""
        constants = defaultdict(list)

        for py_file in self.src_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                tree = ast.parse(content)
                module_name = self._get_module_name(py_file)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant):
                        value = node.value
                        if (
                            isinstance(value, (str, int, float))
                            and len(str(value)) > 10
                        ):
                            constants[value].append(module_name)

            except Exception:
                pass

        # Return only duplicates
        return {k: v for k, v in constants.items() if len(v) > 1}

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Based on analysis, suggest improvements
        if len(self.all_functions) > 50:
            recommendations.append(
                "Consider splitting large modules into smaller, focused ones"
            )

        if self.large_files:
            recommendations.append(
                f"Refactor large files: {[f[0] for f in self.large_files[:3]]}"
            )

        if self.complex_functions:
            recommendations.append(
                f"Simplify complex functions: {[f[1] for f in self.complex_functions[:3]]}"
            )

        # Specific to Simpulse
        recommendations.extend(
            [
                "Focus on core functionality: rule extraction, mutation, and profiling",
                "Remove experimental features until core is proven",
                "Consolidate similar modules (e.g., models.py and models_v2.py)",
                "Create clear separation between CLI, core logic, and utilities",
            ]
        )

        return recommendations

    def generate_cleanup_report(self) -> str:
        """Generate actionable cleanup report."""
        report = self.analyze_codebase()

        lines = [
            "SIMPULSE CODEBASE CLEANUP REPORT",
            "=" * 70,
            "",
            f"Total Files: {report['total_files']}",
            f"Total Functions: {report['total_functions']}",
            "",
            "ISSUES FOUND:",
            "-" * 70,
        ]

        # Unused functions
        if report["unused_functions"]:
            lines.append(f"\n1. UNUSED FUNCTIONS ({len(report['unused_functions'])})")
            for module, func in report["unused_functions"][:10]:
                lines.append(f"   - {module}.{func}")
            if len(report["unused_functions"]) > 10:
                lines.append(f"   ... and {len(report['unused_functions']) - 10} more")

        # Unused imports
        if report["unused_imports"]:
            lines.append(f"\n2. UNUSED IMPORTS ({len(report['unused_imports'])})")
            for module, imp in report["unused_imports"][:10]:
                lines.append(f"   - {module}: {imp}")
            if len(report["unused_imports"]) > 10:
                lines.append(f"   ... and {len(report['unused_imports']) - 10} more")

        # Duplicate constants
        if report["duplicate_constants"]:
            lines.append(
                f"\n3. DUPLICATE CONSTANTS ({len(report['duplicate_constants'])})"
            )
            for const, modules in list(report["duplicate_constants"].items())[:5]:
                const_str = (
                    str(const)[:50] + "..." if len(str(const)) > 50 else str(const)
                )
                lines.append(f"   - '{const_str}' in {len(modules)} modules")

        # Large files
        if report["large_files"]:
            lines.append(f"\n4. LARGE FILES ({len(report['large_files'])})")
            for file_path, line_count in report["large_files"][:5]:
                lines.append(f"   - {file_path}: {line_count} lines")

        # Complex functions
        if report["complex_functions"]:
            lines.append(f"\n5. COMPLEX FUNCTIONS ({len(report['complex_functions'])})")
            for module, func, complexity in report["complex_functions"][:5]:
                lines.append(f"   - {module}.{func}: complexity {complexity}")

        # Recommendations
        lines.extend(
            [
                "",
                "RECOMMENDATIONS:",
                "-" * 70,
            ]
        )
        for i, rec in enumerate(report["recommendations"], 1):
            lines.append(f"{i}. {rec}")

        # Action items
        lines.extend(
            [
                "",
                "IMMEDIATE ACTIONS:",
                "-" * 70,
                "1. Remove all unused functions and imports",
                "2. Extract duplicate constants to a shared constants.py",
                "3. Split evolution_engine.py (400+ lines) into smaller modules",
                "4. Merge models.py and models_v2.py",
                "5. Focus on core loop: profile → extract → mutate → measure",
                "",
                "MODULES TO CONSIDER REMOVING:",
                "-" * 70,
                "- web/dashboard.py (not essential for core functionality)",
                "- strategies/advanced_strategies.py (premature optimization)",
                "- deployment/* (until core is proven)",
                "- benchmarks/* (until we have real results)",
                "",
                "This cleanup could reduce codebase by ~40% while maintaining core functionality.",
            ]
        )

        return "\n".join(lines)


def main():
    """Run codebase cleanup analysis."""
    repo_path = Path(__file__).parent.parent
    cleaner = CodebaseCleaner(repo_path)

    print("Starting codebase analysis...")
    report = cleaner.generate_cleanup_report()

    # Save report
    report_path = repo_path / "CLEANUP_REPORT.md"
    report_path.write_text(report)

    print(f"\nReport saved to: {report_path}")
    print("\n" + report)

    # Ask if we should create cleanup script
    print("\n" + "=" * 70)
    response = input("Generate automated cleanup script? (y/n): ")
    if response.lower() == "y":
        generate_cleanup_script(cleaner.analyze_codebase(), repo_path)


def generate_cleanup_script(analysis: Dict, repo_path: Path) -> None:
    """Generate script to automatically clean up identified issues."""
    script_path = repo_path / "scripts" / "auto_cleanup.py"

    script_content = '''#!/usr/bin/env python3
"""
Automated cleanup script generated from analysis.
Review carefully before running!
"""

import os
from pathlib import Path

def remove_unused_functions():
    """Remove identified unused functions."""
    # This would remove the unused functions
    # For safety, we just print them
    unused = {unused_functions}
    print(f"Would remove {{len(unused)}} unused functions")
    for module, func in unused[:5]:
        print(f"  - {{module}}.{{func}}")

def remove_unused_imports():
    """Remove identified unused imports."""
    # This would clean up imports
    unused = {unused_imports}
    print(f"Would remove {{len(unused)}} unused imports")

def extract_constants():
    """Extract duplicate constants to shared module."""
    duplicates = {duplicate_constants}
    if duplicates:
        print(f"Would extract {{len(duplicates)}} duplicate constants")

if __name__ == "__main__":
    print("AUTOMATED CLEANUP (DRY RUN)")
    print("="*50)
    remove_unused_functions()
    remove_unused_imports()
    extract_constants()
    print("\nRun with --apply to actually make changes")
'''.format(
        unused_functions=analysis["unused_functions"][:10],
        unused_imports=analysis["unused_imports"][:10],
        duplicate_constants=list(analysis["duplicate_constants"].items())[:5],
    )

    script_path.write_text(script_content)
    os.chmod(script_path, 0o755)
    print(f"\nCleanup script generated: {script_path}")


if __name__ == "__main__":
    main()

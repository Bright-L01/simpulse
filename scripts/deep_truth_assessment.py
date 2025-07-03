#!/usr/bin/env python3
"""
Deep Truth Assessment Tool
Comprehensive analysis and execution testing of the entire codebase.
"""

import ast
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class FunctionAnalysis:
    """Detailed analysis of a single function."""

    name: str
    module: str
    file_path: str
    line_number: int
    docstring: Optional[str]
    parameters: List[str]
    returns_annotation: Optional[str]

    # Analysis results
    is_placeholder: bool = False
    uses_random: bool = False
    uses_simulation: bool = False
    has_todo: bool = False
    has_not_implemented: bool = False
    returns_hardcoded: bool = False
    returns_none: bool = False
    is_empty: bool = False

    # Execution results
    execution_tested: bool = False
    execution_success: bool = False
    execution_error: Optional[str] = None
    actual_behavior: Optional[str] = None

    # Pattern detections
    suspicious_patterns: List[str] = field(default_factory=list)
    deception_indicators: List[str] = field(default_factory=list)


class DeepTruthAssessor:
    """Comprehensive truth assessment of the codebase."""

    def __init__(self):
        self.analyses: Dict[str, FunctionAnalysis] = {}
        self.module_cache: Dict[str, Any] = {}
        self.test_inputs = self._generate_test_inputs()

    def _generate_test_inputs(self) -> Dict[str, List[Any]]:
        """Generate diverse test inputs for different parameter types."""
        return {
            "int": [0, 1, -1, 42, 1000000],
            "float": [0.0, 1.0, -1.0, 3.14, 1e6],
            "str": ["", "test", "hello world", "x" * 100],
            "bool": [True, False],
            "list": [[], [1, 2, 3], ["a", "b", "c"], list(range(100))],
            "dict": [{}, {"key": "value"}, {"a": 1, "b": 2}],
            "tuple": [(), (1,), (1, 2, 3)],
            "None": [None],
            "array": [np.array([1, 2, 3]), np.zeros(10), np.random.rand(5, 5)],
            "dataframe": [pd.DataFrame(), pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})],
            "path": [Path("/tmp/test"), Path("."), Path("test.txt")],
        }

    def analyze_codebase(self, src_dir: str = "src") -> Dict[str, Any]:
        """Analyze entire codebase with deep inspection."""
        src_path = project_root / src_dir

        # Find all Python files
        py_files = list(src_path.rglob("*.py"))

        print(f"Found {len(py_files)} Python files to analyze")

        for py_file in py_files:
            if "__pycache__" in str(py_file):
                continue

            print(f"\nAnalyzing: {py_file.relative_to(project_root)}")
            self._analyze_file(py_file)

        # Generate comprehensive report
        return self._generate_report()

    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content, str(file_path))

            # Extract all functions and methods
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._analyze_function(node, file_path, content)

        except Exception as e:
            print(f"  Error parsing {file_path}: {e}")

    def _analyze_function(self, node: ast.AST, file_path: Path, content: str):
        """Deeply analyze a single function."""
        func_name = node.name
        module_path = self._get_module_path(file_path)

        analysis = FunctionAnalysis(
            name=func_name,
            module=module_path,
            file_path=str(file_path),
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            parameters=[arg.arg for arg in node.args.args],
            returns_annotation=ast.unparse(node.returns) if node.returns else None,
        )

        # Static analysis
        self._analyze_static_patterns(node, analysis, content)

        # Dynamic analysis - try to execute the function
        self._analyze_dynamic_behavior(analysis, file_path)

        # Store analysis
        key = f"{module_path}.{func_name}"
        self.analyses[key] = analysis

    def _analyze_static_patterns(self, node: ast.AST, analysis: FunctionAnalysis, content: str):
        """Analyze static patterns in function code."""
        # Get function source
        try:
            func_source = ast.unparse(node)
        except:
            func_source = content.splitlines()[node.lineno - 1 : node.end_lineno]
            func_source = "\n".join(func_source)

        # Check for placeholder patterns
        if len(node.body) == 1:
            stmt = node.body[0]

            # Empty function with just pass
            if isinstance(stmt, ast.Pass):
                analysis.is_empty = True
                analysis.is_placeholder = True
                analysis.suspicious_patterns.append("Empty function with only 'pass'")

            # Just returns None
            elif isinstance(stmt, ast.Return) and stmt.value is None:
                analysis.returns_none = True
                analysis.is_placeholder = True
                analysis.suspicious_patterns.append("Returns None immediately")

            # Returns constant
            elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant):
                analysis.returns_hardcoded = True
                analysis.suspicious_patterns.append(f"Returns hardcoded value: {stmt.value.value}")

            # Raises NotImplementedError
            elif isinstance(stmt, ast.Raise):
                if isinstance(stmt.exc, ast.Call) and isinstance(stmt.exc.func, ast.Name):
                    if stmt.exc.func.id == "NotImplementedError":
                        analysis.has_not_implemented = True
                        analysis.is_placeholder = True
                        analysis.suspicious_patterns.append("Raises NotImplementedError")

        # Check for specific patterns in function body
        for node in ast.walk(node):
            # Random usage
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "random":
                    analysis.uses_random = True
                    analysis.deception_indicators.append("Uses random module")

            # np.random usage
            if isinstance(node, ast.Attribute) and node.attr == "random":
                if isinstance(node.value, ast.Name) and node.value.id == "np":
                    analysis.uses_random = True
                    analysis.deception_indicators.append("Uses numpy.random")

            # Simulation patterns
            if isinstance(node, ast.Name):
                if "simul" in node.id.lower() or "mock" in node.id.lower():
                    analysis.uses_simulation = True
                    analysis.deception_indicators.append(f"Uses simulation/mock: {node.id}")

        # Check for TODO/FIXME comments
        if "TODO" in func_source or "FIXME" in func_source or "XXX" in func_source:
            analysis.has_todo = True
            analysis.suspicious_patterns.append("Contains TODO/FIXME comments")

        # Check for deceptive patterns
        if analysis.docstring:
            # Function claims to do X but body suggests Y
            doc_lower = analysis.docstring.lower()

            # Claims optimization but uses random
            if ("optimiz" in doc_lower or "minimi" in doc_lower) and analysis.uses_random:
                analysis.deception_indicators.append("Claims optimization but uses random values")

            # Claims analysis but returns hardcoded
            if ("analyz" in doc_lower or "comput" in doc_lower) and analysis.returns_hardcoded:
                analysis.deception_indicators.append("Claims analysis but returns hardcoded value")

            # Claims implementation but is placeholder
            if "implement" in doc_lower and analysis.is_placeholder:
                analysis.deception_indicators.append("Claims implementation but is placeholder")

    def _analyze_dynamic_behavior(self, analysis: FunctionAnalysis, file_path: Path):
        """Try to execute the function and observe actual behavior."""
        try:
            # Load module
            module = self._load_module(file_path)
            if not module:
                return

            # Get function
            func = getattr(module, analysis.name, None)
            if not func or not callable(func):
                return

            # Try to execute with various inputs
            results = []
            errors = []

            # Determine input combinations based on parameters
            if not analysis.parameters or (
                len(analysis.parameters) == 1 and analysis.parameters[0] == "self"
            ):
                # No parameters or only self - try calling with no args
                try:
                    result = func()
                    results.append(("no_args", result))
                    analysis.execution_success = True
                except Exception as e:
                    errors.append(("no_args", str(e)))
            else:
                # Try various input combinations
                for i in range(min(5, len(analysis.parameters))):  # Test up to 5 times
                    test_args = self._generate_test_args(analysis.parameters)
                    try:
                        result = func(*test_args)
                        results.append((test_args, result))
                        analysis.execution_success = True
                    except Exception as e:
                        errors.append((test_args, str(e)))

            analysis.execution_tested = True

            # Analyze results
            if results:
                self._analyze_execution_results(analysis, results)

            if errors and not results:
                analysis.execution_error = f"All executions failed: {errors[0][1]}"

        except Exception as e:
            analysis.execution_tested = True
            analysis.execution_error = f"Failed to load/execute: {str(e)}"

    def _analyze_execution_results(
        self, analysis: FunctionAnalysis, results: List[Tuple[Any, Any]]
    ):
        """Analyze execution results for patterns."""
        # Check if all results are the same
        if len({str(r[1]) for r in results}) == 1:
            analysis.suspicious_patterns.append("Returns same value for different inputs")
            analysis.actual_behavior = f"Always returns: {results[0][1]}"

        # Check if all results are None
        if all(r[1] is None for r in results):
            analysis.returns_none = True
            analysis.actual_behavior = "Always returns None"

        # Check for random behavior
        if len(results) > 2:
            values = [r[1] for r in results if isinstance(r[1], (int, float))]
            if values and len(set(values)) == len(values):
                analysis.suspicious_patterns.append("Returns different numeric values each time")
                analysis.uses_random = True

        # Store sample behavior
        if len(results) > 0:
            analysis.actual_behavior = f"Sample output: {results[0][1]}"

    def _generate_test_args(self, parameters: List[str]) -> List[Any]:
        """Generate test arguments for function parameters."""
        args = []
        for param in parameters:
            if param == "self":
                continue

            # Try to guess type from parameter name
            if "path" in param or "file" in param:
                args.append(self.test_inputs["path"][0])
            elif "df" in param or "data" in param:
                args.append(self.test_inputs["dataframe"][0])
            elif "arr" in param or "matrix" in param:
                args.append(self.test_inputs["array"][0])
            elif "text" in param or "str" in param:
                args.append(self.test_inputs["str"][1])
            elif "num" in param or "val" in param:
                args.append(self.test_inputs["float"][1])
            elif "flag" in param or "is_" in param:
                args.append(self.test_inputs["bool"][0])
            else:
                # Default to simple value
                args.append(1)

        return args

    def _load_module(self, file_path: Path) -> Optional[Any]:
        """Dynamically load a Python module."""
        try:
            # Check cache
            if str(file_path) in self.module_cache:
                return self.module_cache[str(file_path)]

            # Create module spec
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                return None

            # Load module
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Cache it
            self.module_cache[str(file_path)] = module
            return module

        except Exception as e:
            print(f"    Failed to load module {file_path}: {e}")
            return None

    def _get_module_path(self, file_path: Path) -> str:
        """Get module path from file path."""
        try:
            rel_path = file_path.relative_to(project_root)
            parts = list(rel_path.parts[:-1]) + [rel_path.stem]
            return ".".join(parts)
        except:
            return file_path.stem

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive truth assessment report."""
        total_functions = len(self.analyses)

        # Categorize functions
        placeholders = []
        random_based = []
        simulations = []
        unimplemented = []
        hardcoded = []
        empty = []
        deceptive = []
        working = []
        untested = []

        for key, analysis in self.analyses.items():
            if analysis.is_placeholder or analysis.is_empty:
                placeholders.append(analysis)
            if analysis.uses_random:
                random_based.append(analysis)
            if analysis.uses_simulation:
                simulations.append(analysis)
            if analysis.has_not_implemented or analysis.has_todo:
                unimplemented.append(analysis)
            if analysis.returns_hardcoded:
                hardcoded.append(analysis)
            if analysis.is_empty:
                empty.append(analysis)
            if analysis.deception_indicators:
                deceptive.append(analysis)
            if analysis.execution_success and not any(
                [
                    analysis.is_placeholder,
                    analysis.uses_random,
                    analysis.returns_hardcoded,
                    analysis.is_empty,
                ]
            ):
                working.append(analysis)
            if not analysis.execution_tested:
                untested.append(analysis)

        # Calculate percentages
        placeholder_pct = len(placeholders) / total_functions * 100 if total_functions > 0 else 0
        random_pct = len(random_based) / total_functions * 100 if total_functions > 0 else 0
        working_pct = len(working) / total_functions * 100 if total_functions > 0 else 0

        report = {
            "summary": {
                "total_functions": total_functions,
                "placeholders": len(placeholders),
                "random_based": len(random_based),
                "simulations": len(simulations),
                "unimplemented": len(unimplemented),
                "hardcoded": len(hardcoded),
                "empty": len(empty),
                "deceptive": len(deceptive),
                "working": len(working),
                "untested": len(untested),
                "placeholder_percentage": placeholder_pct,
                "random_percentage": random_pct,
                "working_percentage": working_pct,
            },
            "details": {
                "placeholders": [self._format_function_detail(a) for a in placeholders[:10]],
                "random_based": [self._format_function_detail(a) for a in random_based[:10]],
                "deceptive": [self._format_function_detail(a) for a in deceptive[:10]],
                "working": [self._format_function_detail(a) for a in working[:10]],
            },
            "worst_offenders": self._find_worst_offenders(),
            "execution_stats": self._get_execution_stats(),
            "timestamp": datetime.now().isoformat(),
        }

        return report

    def _format_function_detail(self, analysis: FunctionAnalysis) -> Dict[str, Any]:
        """Format function analysis for report."""
        return {
            "function": f"{analysis.module}.{analysis.name}",
            "file": analysis.file_path.replace(str(project_root), ""),
            "line": analysis.line_number,
            "issues": analysis.suspicious_patterns + analysis.deception_indicators,
            "actual_behavior": analysis.actual_behavior,
            "execution_error": analysis.execution_error,
        }

    def _find_worst_offenders(self) -> List[Dict[str, Any]]:
        """Find the most problematic modules."""
        module_issues = {}

        for key, analysis in self.analyses.items():
            module = analysis.module.split(".")[0:3]  # Top 3 levels
            module_key = ".".join(module)

            if module_key not in module_issues:
                module_issues[module_key] = {
                    "total": 0,
                    "placeholders": 0,
                    "random": 0,
                    "deceptive": 0,
                }

            module_issues[module_key]["total"] += 1
            if analysis.is_placeholder:
                module_issues[module_key]["placeholders"] += 1
            if analysis.uses_random:
                module_issues[module_key]["random"] += 1
            if analysis.deception_indicators:
                module_issues[module_key]["deceptive"] += 1

        # Sort by total issues
        sorted_modules = sorted(
            module_issues.items(),
            key=lambda x: x[1]["placeholders"] + x[1]["random"] + x[1]["deceptive"],
            reverse=True,
        )

        return [
            {
                "module": module,
                "stats": stats,
                "issue_rate": (stats["placeholders"] + stats["random"] + stats["deceptive"])
                / stats["total"]
                * 100,
            }
            for module, stats in sorted_modules[:10]
        ]

    def _get_execution_stats(self) -> Dict[str, Any]:
        """Get execution testing statistics."""
        tested = sum(1 for a in self.analyses.values() if a.execution_tested)
        successful = sum(1 for a in self.analyses.values() if a.execution_success)
        failed = sum(
            1 for a in self.analyses.values() if a.execution_tested and not a.execution_success
        )

        return {
            "total_tested": tested,
            "successful": successful,
            "failed": failed,
            "test_coverage": tested / len(self.analyses) * 100 if self.analyses else 0,
            "success_rate": successful / tested * 100 if tested > 0 else 0,
        }

    def save_report(self, report: Dict[str, Any], output_file: str = "deep_truth_report.json"):
        """Save report to file."""
        output_path = project_root / output_file
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")

        # Also save a human-readable summary
        summary_path = project_root / "DEEP_TRUTH_SUMMARY.md"
        with open(summary_path, "w") as f:
            f.write(self._generate_markdown_summary(report))
        print(f"Summary saved to: {summary_path}")

    def _generate_markdown_summary(self, report: Dict[str, Any]) -> str:
        """Generate human-readable markdown summary."""
        s = report["summary"]

        markdown = f"""# Deep Truth Assessment Report

Generated: {report['timestamp']}

## 📊 Overall Statistics

- **Total Functions Analyzed**: {s['total_functions']}
- **Execution Test Coverage**: {report['execution_stats']['test_coverage']:.1f}%
- **Execution Success Rate**: {report['execution_stats']['success_rate']:.1f}%

## 🚨 Critical Findings

### Function Categories:
- **Placeholders**: {s['placeholders']} ({s['placeholder_percentage']:.1f}%)
- **Random-Based**: {s['random_based']} ({s['random_percentage']:.1f}%)
- **Hardcoded Returns**: {s['hardcoded']}
- **Empty Functions**: {s['empty']}
- **Unimplemented**: {s['unimplemented']}
- **Deceptive**: {s['deceptive']}
- **Actually Working**: {s['working']} ({s['working_percentage']:.1f}%)

## 🎭 Deceptive Functions

Functions that claim to do one thing but actually do another:

"""

        for func in report["details"]["deceptive"][:5]:
            markdown += f"\n### {func['function']}\n"
            markdown += f"- File: `{func['file']}` (line {func['line']})\n"
            markdown += f"- Issues: {', '.join(func['issues'])}\n"
            if func["actual_behavior"]:
                markdown += f"- Actual behavior: {func['actual_behavior']}\n"

        markdown += "\n## 🎲 Random-Based Functions\n\n"
        markdown += "Functions that rely on random values instead of real computation:\n\n"

        for func in report["details"]["random_based"][:5]:
            markdown += f"- `{func['function']}`: {', '.join(func['issues'])}\n"

        markdown += "\n## 📦 Worst Offending Modules\n\n"

        for module_info in report["worst_offenders"][:5]:
            m = module_info["module"]
            s = module_info["stats"]
            markdown += f"\n### {m}\n"
            markdown += f"- Total functions: {s['total']}\n"
            markdown += f"- Placeholders: {s['placeholders']}\n"
            markdown += f"- Random-based: {s['random']}\n"
            markdown += f"- Deceptive: {s['deceptive']}\n"
            markdown += f"- **Issue rate: {module_info['issue_rate']:.1f}%**\n"

        markdown += "\n## 💡 Conclusion\n\n"

        if s["working_percentage"] < 10:
            markdown += (
                "⚠️ **CRITICAL**: Less than 10% of functions appear to have real implementations.\n"
            )
        elif s["working_percentage"] < 30:
            markdown += (
                "⚠️ **WARNING**: Only {:.1f}% of functions have real implementations.\n".format(
                    s["working_percentage"]
                )
            )

        if s["random_percentage"] > 20:
            markdown += "🎲 **CONCERN**: {:.1f}% of functions rely on random values.\n".format(
                s["random_percentage"]
            )

        if s["deceptive"] > 10:
            markdown += "🎭 **DECEPTION**: {} functions have misleading documentation.\n".format(
                s["deceptive"]
            )

        return markdown


def main():
    """Run deep truth assessment."""
    print("🔍 Starting Deep Truth Assessment...")
    print("=" * 60)

    assessor = DeepTruthAssessor()
    report = assessor.analyze_codebase()

    print("\n" + "=" * 60)
    print("📊 ASSESSMENT COMPLETE")
    print("=" * 60)

    s = report["summary"]
    print(f"\nTotal Functions: {s['total_functions']}")
    print(f"Placeholders: {s['placeholders']} ({s['placeholder_percentage']:.1f}%)")
    print(f"Random-based: {s['random_based']} ({s['random_percentage']:.1f}%)")
    print(f"Actually Working: {s['working']} ({s['working_percentage']:.1f}%)")

    print(f"\nExecution Coverage: {report['execution_stats']['test_coverage']:.1f}%")
    print(f"Success Rate: {report['execution_stats']['success_rate']:.1f}%")

    # Save detailed report
    assessor.save_report(report)

    print("\n✅ Assessment complete! Check DEEP_TRUTH_SUMMARY.md for details.")


if __name__ == "__main__":
    main()

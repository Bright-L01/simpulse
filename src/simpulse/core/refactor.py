"""
Code refactoring utilities for Simpulse.

This module provides tools for automated code cleanup, optimization,
and refactoring to maintain code quality.
"""

import ast
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RefactorType(Enum):
    """Types of refactoring operations."""

    REMOVE_DEAD_CODE = "remove_dead_code"
    EXTRACT_CONSTANTS = "extract_constants"
    SIMPLIFY_CONDITIONALS = "simplify_conditionals"
    REMOVE_DUPLICATE_CODE = "remove_duplicate_code"
    OPTIMIZE_IMPORTS = "optimize_imports"
    ADD_TYPE_HINTS = "add_type_hints"
    EXTRACT_FUNCTIONS = "extract_functions"
    RENAME_VARIABLES = "rename_variables"


@dataclass
class RefactoringSuggestion:
    """A suggested refactoring operation."""

    type: RefactorType
    file_path: Path
    line_start: int
    line_end: int
    description: str
    original_code: str
    suggested_code: str
    confidence: float
    estimated_impact: Dict[str, Any]


class CodeRefactor:
    """Automated code refactoring tool."""

    def __init__(self):
        """Initialize code refactor."""
        self.suggestions: List[RefactoringSuggestion] = []

    def analyze_file(self, file_path: Path) -> List[RefactoringSuggestion]:
        """Analyze a Python file for refactoring opportunities.

        Args:
            file_path: Path to Python file

        Returns:
            List of refactoring suggestions
        """
        if not file_path.exists() or not file_path.suffix == ".py":
            return []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content, filename=str(file_path))

            # Run various analyzers
            suggestions = []
            suggestions.extend(self._find_dead_code(tree, file_path, content))
            suggestions.extend(self._find_duplicate_constants(tree, file_path, content))
            suggestions.extend(
                self._find_complex_conditionals(tree, file_path, content)
            )
            suggestions.extend(self._find_unused_imports(tree, file_path, content))
            suggestions.extend(self._find_long_functions(tree, file_path, content))

            return suggestions

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return []

    def _find_dead_code(
        self, tree: ast.AST, file_path: Path, content: str
    ) -> List[RefactoringSuggestion]:
        """Find dead code (unreachable code after return/raise).

        Args:
            tree: AST tree
            file_path: Source file path
            content: File content

        Returns:
            List of suggestions
        """
        suggestions = []

        class DeadCodeFinder(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Check for code after return/raise
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, (ast.Return, ast.Raise)):
                        if i < len(node.body) - 1:
                            # Found dead code
                            dead_start = node.body[i + 1].lineno
                            dead_end = node.body[-1].end_lineno or dead_start

                            lines = content.split("\n")
                            dead_code = "\n".join(lines[dead_start - 1 : dead_end])

                            suggestions.append(
                                RefactoringSuggestion(
                                    type=RefactorType.REMOVE_DEAD_CODE,
                                    file_path=file_path,
                                    line_start=dead_start,
                                    line_end=dead_end,
                                    description=f"Unreachable code after {type(stmt).__name__.lower()}",
                                    original_code=dead_code,
                                    suggested_code="",
                                    confidence=0.95,
                                    estimated_impact={
                                        "lines_removed": dead_end - dead_start + 1
                                    },
                                )
                            )
                        break

                self.generic_visit(node)

        finder = DeadCodeFinder()
        finder.visit(tree)

        return suggestions

    def _find_duplicate_constants(
        self, tree: ast.AST, file_path: Path, content: str
    ) -> List[RefactoringSuggestion]:
        """Find duplicate constant values that could be extracted.

        Args:
            tree: AST tree
            file_path: Source file path
            content: File content

        Returns:
            List of suggestions
        """
        suggestions = []
        constants: Dict[Any, List[Tuple[int, str]]] = {}

        class ConstantFinder(ast.NodeVisitor):
            def visit_Constant(self, node):
                if isinstance(node.value, (str, int, float)) and node.value != "":
                    # Skip small integers and single chars
                    if isinstance(node.value, int) and -10 <= node.value <= 10:
                        return
                    if isinstance(node.value, str) and len(node.value) <= 1:
                        return

                    key = node.value
                    if key not in constants:
                        constants[key] = []

                    # Get context (parent node)
                    context = ""
                    if hasattr(node, "parent"):
                        context = type(node.parent).__name__

                    constants[key].append((node.lineno, context))

                self.generic_visit(node)

        # Add parent references
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        finder = ConstantFinder()
        finder.visit(tree)

        # Find duplicates
        for value, locations in constants.items():
            if len(locations) >= 3:  # At least 3 occurrences
                lines = [loc[0] for loc in locations]

                # Suggest extracting to constant
                const_name = self._suggest_constant_name(value)

                suggestions.append(
                    RefactoringSuggestion(
                        type=RefactorType.EXTRACT_CONSTANTS,
                        file_path=file_path,
                        line_start=min(lines),
                        line_end=max(lines),
                        description=f"Extract repeated value {repr(value)} to constant",
                        original_code=f"# Value {repr(value)} appears {len(locations)} times",
                        suggested_code=f"{const_name} = {repr(value)}",
                        confidence=0.8,
                        estimated_impact={
                            "occurrences": len(locations),
                            "readability_improvement": "high",
                        },
                    )
                )

        return suggestions

    def _find_complex_conditionals(
        self, tree: ast.AST, file_path: Path, content: str
    ) -> List[RefactoringSuggestion]:
        """Find complex conditionals that could be simplified.

        Args:
            tree: AST tree
            file_path: Source file path
            content: File content

        Returns:
            List of suggestions
        """
        suggestions = []

        class ConditionalComplexityAnalyzer(ast.NodeVisitor):
            def visit_If(self, node):
                # Count boolean operators
                op_count = self._count_bool_ops(node.test)

                if op_count >= 3:
                    # Complex conditional
                    lines = content.split("\n")
                    original = lines[node.lineno - 1 : node.end_lineno]

                    suggestions.append(
                        RefactoringSuggestion(
                            type=RefactorType.SIMPLIFY_CONDITIONALS,
                            file_path=file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            description="Complex conditional with multiple boolean operators",
                            original_code="\n".join(original),
                            suggested_code="# Consider extracting to separate boolean variables",
                            confidence=0.7,
                            estimated_impact={
                                "complexity_reduction": op_count,
                                "readability_improvement": "medium",
                            },
                        )
                    )

                self.generic_visit(node)

            def _count_bool_ops(self, node):
                if isinstance(node, ast.BoolOp):
                    count = len(node.values) - 1
                    for value in node.values:
                        count += self._count_bool_ops(value)
                    return count
                elif isinstance(node, ast.Compare):
                    return len(node.ops)
                else:
                    return 0

        analyzer = ConditionalComplexityAnalyzer()
        analyzer.visit(tree)

        return suggestions

    def _find_unused_imports(
        self, tree: ast.AST, file_path: Path, content: str
    ) -> List[RefactoringSuggestion]:
        """Find unused imports.

        Args:
            tree: AST tree
            file_path: Source file path
            content: File content

        Returns:
            List of suggestions
        """
        suggestions = []

        # Collect all imports
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imports[name] = (node.lineno, alias.name, None)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imports[name] = (node.lineno, alias.name, node.module)

        # Find all name usages
        used_names = set()

        class NameCollector(ast.NodeVisitor):
            def visit_Name(self, node):
                used_names.add(node.id)
                self.generic_visit(node)

            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
                self.generic_visit(node)

        collector = NameCollector()
        collector.visit(tree)

        # Find unused imports
        for import_name, (line, original_name, module) in imports.items():
            if import_name not in used_names:
                lines = content.split("\n")
                import_line = lines[line - 1]

                suggestions.append(
                    RefactoringSuggestion(
                        type=RefactorType.OPTIMIZE_IMPORTS,
                        file_path=file_path,
                        line_start=line,
                        line_end=line,
                        description=f"Unused import: {import_name}",
                        original_code=import_line,
                        suggested_code="",
                        confidence=0.9,
                        estimated_impact={"lines_removed": 1},
                    )
                )

        return suggestions

    def _find_long_functions(
        self, tree: ast.AST, file_path: Path, content: str
    ) -> List[RefactoringSuggestion]:
        """Find functions that are too long and should be split.

        Args:
            tree: AST tree
            file_path: Source file path
            content: File content

        Returns:
            List of suggestions
        """
        suggestions = []

        class FunctionLengthAnalyzer(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Calculate function length
                if node.body:
                    start_line = node.lineno
                    end_line = node.body[-1].end_lineno or start_line
                    func_length = end_line - start_line + 1

                    if func_length > 50:  # Functions longer than 50 lines
                        lines = content.split("\n")
                        func_preview = "\n".join(
                            lines[start_line - 1 : min(start_line + 4, end_line)]
                        )

                        suggestions.append(
                            RefactoringSuggestion(
                                type=RefactorType.EXTRACT_FUNCTIONS,
                                file_path=file_path,
                                line_start=start_line,
                                line_end=end_line,
                                description=f"Function '{node.name}' is {func_length} lines long",
                                original_code=func_preview + "\n...",
                                suggested_code="# Consider breaking into smaller functions",
                                confidence=0.6,
                                estimated_impact={
                                    "lines": func_length,
                                    "complexity_reduction": "high",
                                    "testability_improvement": "high",
                                },
                            )
                        )

                self.generic_visit(node)

        analyzer = FunctionLengthAnalyzer()
        analyzer.visit(tree)

        return suggestions

    def _suggest_constant_name(self, value: Any) -> str:
        """Suggest a name for an extracted constant.

        Args:
            value: Constant value

        Returns:
            Suggested constant name
        """
        if isinstance(value, str):
            # Use string content as hint
            name = re.sub(r"[^a-zA-Z0-9_]", "_", value[:20])
            name = name.upper()
            if name and name[0].isdigit():
                name = "CONST_" + name
            return name or "STRING_CONSTANT"
        elif isinstance(value, (int, float)):
            return f"NUMERIC_CONSTANT_{abs(int(value))}"
        else:
            return "EXTRACTED_CONSTANT"

    def apply_suggestion(self, suggestion: RefactoringSuggestion) -> bool:
        """Apply a refactoring suggestion to the code.

        Args:
            suggestion: Refactoring suggestion to apply

        Returns:
            True if successful
        """
        try:
            with open(suggestion.file_path, encoding="utf-8") as f:
                lines = f.readlines()

            # Apply based on suggestion type
            if suggestion.type == RefactorType.REMOVE_DEAD_CODE:
                # Remove lines
                del lines[suggestion.line_start - 1 : suggestion.line_end]

            elif suggestion.type == RefactorType.OPTIMIZE_IMPORTS:
                # Remove import line
                if suggestion.line_start <= len(lines):
                    del lines[suggestion.line_start - 1]

            # Write back
            with open(suggestion.file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            logger.info(f"Applied refactoring: {suggestion.description}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply refactoring: {e}")
            return False

    def analyze_project(
        self,
        project_path: Path,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze entire project for refactoring opportunities.

        Args:
            project_path: Project root path
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude

        Returns:
            Analysis summary
        """
        include_patterns = include_patterns or ["**/*.py"]
        exclude_patterns = exclude_patterns or ["**/test_*", "**/__pycache__/**"]

        all_suggestions = []
        files_analyzed = 0

        # Find Python files
        for pattern in include_patterns:
            for file_path in project_path.glob(pattern):
                # Check exclusions
                if any(file_path.match(excl) for excl in exclude_patterns):
                    continue

                if file_path.is_file():
                    suggestions = self.analyze_file(file_path)
                    all_suggestions.extend(suggestions)
                    files_analyzed += 1

        # Group by type
        by_type = {}
        for suggestion in all_suggestions:
            if suggestion.type not in by_type:
                by_type[suggestion.type] = []
            by_type[suggestion.type].append(suggestion)

        # Calculate impact
        total_lines_to_remove = sum(
            s.estimated_impact.get("lines_removed", 0) for s in all_suggestions
        )

        return {
            "files_analyzed": files_analyzed,
            "total_suggestions": len(all_suggestions),
            "suggestions_by_type": {
                t.value: len(suggestions) for t, suggestions in by_type.items()
            },
            "estimated_line_reduction": total_lines_to_remove,
            "high_confidence_suggestions": len(
                [s for s in all_suggestions if s.confidence >= 0.8]
            ),
            "suggestions": all_suggestions,
        }

    def generate_refactoring_report(
        self, analysis: Dict[str, Any], output_path: Path
    ) -> bool:
        """Generate a refactoring report.

        Args:
            analysis: Analysis results from analyze_project
            output_path: Path to save report

        Returns:
            True if successful
        """
        try:
            report_lines = [
                "# Simpulse Code Refactoring Report",
                "",
                f"**Files Analyzed**: {analysis['files_analyzed']}",
                f"**Total Suggestions**: {analysis['total_suggestions']}",
                f"**High Confidence**: {analysis['high_confidence_suggestions']}",
                f"**Estimated Line Reduction**: {analysis['estimated_line_reduction']}",
                "",
                "## Suggestions by Type",
                "",
            ]

            for ref_type, count in analysis["suggestions_by_type"].items():
                report_lines.append(f"- **{ref_type}**: {count} suggestions")

            report_lines.extend(["", "## Detailed Suggestions", ""])

            # Group by file
            by_file = {}
            for suggestion in analysis["suggestions"]:
                if suggestion.file_path not in by_file:
                    by_file[suggestion.file_path] = []
                by_file[suggestion.file_path].append(suggestion)

            for file_path, suggestions in by_file.items():
                report_lines.append(f"### {file_path}")
                report_lines.append("")

                for suggestion in suggestions:
                    report_lines.extend(
                        [
                            f"**Line {suggestion.line_start}-{suggestion.line_end}**: {suggestion.description}",
                            f"- Type: {suggestion.type.value}",
                            f"- Confidence: {suggestion.confidence:.1%}",
                            "",
                        ]
                    )

            # Write report
            with open(output_path, "w") as f:
                f.write("\n".join(report_lines))

            logger.info(f"Refactoring report saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return False

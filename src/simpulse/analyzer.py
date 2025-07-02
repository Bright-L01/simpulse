"""Lean 4 project analysis and simp rule extraction."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimpRule:
    """Represents a simp rule found in Lean code."""

    name: str
    file_path: Path
    line_number: int
    priority: int | None = None
    pattern: str | None = None
    frequency: int = 0

    def __hash__(self) -> int:
        """Make SimpRule hashable for use in sets/dicts."""
        return hash((self.name, str(self.file_path), self.line_number))


@dataclass
class LeanFileAnalysis:
    """Analysis results for a single Lean file."""

    file_path: Path
    simp_rules: list[SimpRule]
    total_lines: int
    syntax_valid: bool


class LeanAnalyzer:
    """Analyzes Lean 4 projects to extract simp rules and usage patterns."""

    def __init__(self, lean_executable: str = "lean"):
        """Initialize the analyzer.

        Args:
            lean_executable: Path to Lean 4 executable.
        """
        self.lean_executable = lean_executable
        self._simp_rule_pattern = re.compile(
            r"@\[simp(?:,\s*priority\s*:=\s*(\d+))?\]\s*theorem\s+(\w+)", re.MULTILINE
        )

    def extract_simp_rules(self, content: str) -> list[SimpRule]:
        """Extract simp rules from Lean code content.

        Args:
            content: Lean code content to analyze.

        Returns:
            List of SimpRule objects found in the content.
        """
        rules = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            match = self._simp_rule_pattern.search(line)
            if match:
                priority_str, rule_name = match.groups()
                priority = int(priority_str) if priority_str else None

                rule = SimpRule(
                    name=rule_name,
                    file_path=Path(""),  # Will be set by caller
                    line_number=line_num,
                    priority=priority,
                    pattern=line.strip(),
                )
                rules.append(rule)

        return rules

    def analyze_file(self, file_path: Path) -> LeanFileAnalysis:
        """Analyze a single Lean file.

        Args:
            file_path: Path to the Lean file to analyze.

        Returns:
            LeanFileAnalysis with extracted simp rules.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        rules = self.extract_simp_rules(content)

        # Set the correct file path for each rule
        for rule in rules:
            rule.file_path = file_path

        # Check syntax validity
        syntax_valid = self._validate_lean_syntax(file_path)

        return LeanFileAnalysis(
            file_path=file_path,
            simp_rules=rules,
            total_lines=len(content.split("\n")),
            syntax_valid=syntax_valid,
        )

    def analyze_project(self, project_path: Path) -> dict:
        """Analyze an entire Lean project.

        Args:
            project_path: Path to the Lean project directory.

        Returns:
            Dictionary with analysis results.
        """
        lean_files = self._get_lean_files(project_path)
        all_rules = []
        total_files = 0

        for file_path in lean_files:
            try:
                analysis = self.analyze_file(file_path)
                all_rules.extend(analysis.simp_rules)
                total_files += 1
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")

        # Calculate statistics
        stats = self._calculate_statistics(all_rules)

        # Get optimization opportunities
        opportunities = self._get_optimization_opportunities(all_rules)

        return {
            "total_files": total_files,
            "total_simp_rules": len(all_rules),
            "rules_with_custom_priority": stats["rules_with_custom_priority"],
            "default_priority_percent": stats["default_priority_percent"],
            "rules_by_frequency": sorted(all_rules, key=lambda r: r.frequency, reverse=True),
            "optimization_opportunities": len(opportunities),
            "estimated_improvement": min(0.7, len(opportunities) * 0.05),  # Cap at 70%
        }

    def _get_lean_files(self, directory: Path) -> list[Path]:
        """Get all .lean files in a directory recursively."""
        return list(directory.rglob("*.lean"))

    def _validate_lean_syntax(self, file_path: Path) -> bool:
        """Validate that a Lean file has correct syntax.

        Args:
            file_path: Path to the Lean file.

        Returns:
            True if syntax is valid, False otherwise.
        """
        try:
            result = subprocess.run(
                [self.lean_executable, "--check", str(file_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _calculate_statistics(self, rules: list[SimpRule]) -> dict:
        """Calculate statistics about simp rules.

        Args:
            rules: List of SimpRule objects.

        Returns:
            Dictionary with calculated statistics.
        """
        total_rules = len(rules)
        if total_rules == 0:
            return {
                "total_simp_rules": 0,
                "rules_with_custom_priority": 0,
                "default_priority_percent": 0.0,
                "total_usage_frequency": 0,
            }

        rules_with_priority = sum(1 for rule in rules if rule.priority is not None)
        default_priority_percent = ((total_rules - rules_with_priority) / total_rules) * 100
        total_frequency = sum(rule.frequency for rule in rules)

        return {
            "total_simp_rules": total_rules,
            "rules_with_custom_priority": rules_with_priority,
            "default_priority_percent": default_priority_percent,
            "total_usage_frequency": total_frequency,
        }

    def _get_optimization_opportunities(self, rules: list[SimpRule]) -> list[SimpRule]:
        """Identify rules that would benefit from priority optimization.

        Args:
            rules: List of SimpRule objects.

        Returns:
            List of rules that are optimization opportunities.
        """
        opportunities = []

        for rule in rules:
            # High frequency rules without custom priority are opportunities
            if rule.frequency > 30 and rule.priority is None:
                opportunities.append(rule)
            # Medium frequency rules without priority are also opportunities
            elif rule.frequency > 10 and rule.priority is None:
                opportunities.append(rule)

        return opportunities

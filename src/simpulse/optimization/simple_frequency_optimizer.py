"""Simple frequency-based optimizer that counts simp rule usage in proofs."""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


@dataclass
class RuleUsage:
    """Track usage of a single rule."""

    rule_name: str
    explicit_uses: int = 0  # Direct uses in simp [rule]
    implicit_uses: int = 0  # Potential uses in plain simp calls
    total_uses: int = 0
    files_used_in: Set[str] = None

    def __post_init__(self):
        if self.files_used_in is None:
            self.files_used_in = set()
        self.total_uses = self.explicit_uses + self.implicit_uses


@dataclass
class PrioritySuggestion:
    """Suggestion for changing a rule's priority."""

    rule_name: str
    current_priority: int
    suggested_priority: int
    usage_count: int
    reason: str
    file_path: str


class SimpleFrequencyOptimizer:
    """Count simp rule usage and suggest priority changes based on frequency."""

    def __init__(self):
        # Pattern to find simp applications with explicit rules
        self.simp_with_rules_pattern = re.compile(
            r"simp\s*\[([^\]]+)\]", re.MULTILINE | re.DOTALL  # simp [rule1, rule2, ...]
        )

        # Pattern to find plain simp calls
        self.plain_simp_pattern = re.compile(
            r"(?:^|\s)simp(?:\s|$|[^\w\[])", re.MULTILINE  # Plain simp (not simp_all, simp_rw, etc)
        )

        # Pattern to extract rule names from simp [...]
        self.rule_name_pattern = re.compile(r"[\w\.]+")

    def analyze_project(self, project_path: Path) -> Dict[str, RuleUsage]:
        """Analyze a Lean project to count simp rule usage.

        Returns dict mapping rule names to their usage information.
        """
        usage_stats = defaultdict(lambda: RuleUsage(""))

        # Find all Lean files
        lean_files = list(project_path.rglob("*.lean"))

        for lean_file in lean_files:
            # Skip dependencies
            if "lake-packages" in str(lean_file) or ".lake" in str(lean_file):
                continue

            try:
                content = lean_file.read_text(encoding="utf-8")
                self._analyze_file(lean_file, content, usage_stats)
            except Exception as e:
                print(f"Warning: Could not analyze {lean_file}: {e}")

        # Convert defaultdict to regular dict with proper rule names
        return {name: usage for name, usage in usage_stats.items() if usage.rule_name}

    def _analyze_file(self, file_path: Path, content: str, usage_stats: Dict[str, RuleUsage]):
        """Analyze a single file for simp usage."""

        # Count explicit simp rule uses: simp [rule1, rule2, ...]
        for match in self.simp_with_rules_pattern.finditer(content):
            rules_text = match.group(1)
            # Extract individual rule names
            for rule_match in self.rule_name_pattern.finditer(rules_text):
                rule_name = rule_match.group(0)
                # Skip common keywords that aren't rule names
                if rule_name not in ["only", "at", "with"]:
                    if rule_name not in usage_stats:
                        usage_stats[rule_name] = RuleUsage(rule_name=rule_name)
                    usage_stats[rule_name].explicit_uses += 1
                    usage_stats[rule_name].files_used_in.add(str(file_path))

        # Count plain simp calls (these might use any simp rule)
        plain_simp_count = len(self.plain_simp_pattern.findall(content))

        # For implicit uses, we need to know what simp rules exist in scope
        # For now, we'll mark this for rules we've seen used explicitly
        if plain_simp_count > 0:
            for rule_name, usage in usage_stats.items():
                if usage.files_used_in and str(file_path) in usage.files_used_in:
                    # This rule might be used implicitly in this file
                    usage.implicit_uses += plain_simp_count // 10  # Conservative estimate

    def suggest_priorities(
        self, usage_stats: Dict[str, RuleUsage], simp_rules: List
    ) -> List[PrioritySuggestion]:
        """Generate priority suggestions based on usage frequency.

        Args:
            usage_stats: Rule usage statistics
            simp_rules: List of SimpRule objects from analyzer

        Returns:
            List of priority change suggestions
        """
        suggestions = []

        # Create lookup for existing rules
        rule_info = {}
        for rule in simp_rules:
            # Handle different rule object types
            if hasattr(rule, "location") and rule.location:
                file_path = str(rule.location.file)
            elif hasattr(rule, "file_path"):
                file_path = str(rule.file_path)
            else:
                file_path = "unknown"

            # Get priority - handle both numeric and enum types
            if hasattr(rule, "priority"):
                if isinstance(rule.priority, int):
                    priority = rule.priority
                elif rule.priority is None:
                    priority = 1000
                else:
                    # Assume it's an enum with default value
                    priority = 1000
            else:
                priority = 1000

            rule_info[rule.name] = {"priority": priority, "file_path": file_path}

        # Sort rules by total usage
        sorted_rules = sorted(
            usage_stats.items(), key=lambda x: x[1].explicit_uses + x[1].implicit_uses, reverse=True
        )

        # Suggest priorities based on usage frequency
        for i, (rule_name, usage) in enumerate(sorted_rules):
            if rule_name not in rule_info:
                continue  # Skip rules we don't have info for

            current_priority = rule_info[rule_name]["priority"]
            total_uses = usage.explicit_uses + usage.implicit_uses

            # Skip rules with no usage
            if total_uses == 0:
                continue

            # Assign priorities based on usage tiers
            if total_uses >= 50:
                # Very frequently used - high priority
                suggested_priority = 2000
                reason = f"Very high usage ({usage.explicit_uses} explicit, {usage.implicit_uses} implicit)"
            elif total_uses >= 20:
                # Frequently used
                suggested_priority = 1500
                reason = (
                    f"High usage ({usage.explicit_uses} explicit, {usage.implicit_uses} implicit)"
                )
            elif total_uses >= 10:
                # Moderately used
                suggested_priority = 1200
                reason = f"Moderate usage ({usage.explicit_uses} explicit, {usage.implicit_uses} implicit)"
            elif total_uses >= 5:
                # Sometimes used - increase priority
                suggested_priority = 1100
                reason = f"Moderate usage ({usage.explicit_uses} explicit, {usage.implicit_uses} implicit)"
            elif total_uses >= 3:
                # Keep default for low-moderate usage
                suggested_priority = 1000
                reason = (
                    f"Some usage ({usage.explicit_uses} explicit, {usage.implicit_uses} implicit)"
                )
            else:
                # Rarely used - lower priority
                suggested_priority = 100
                reason = (
                    f"Low usage ({usage.explicit_uses} explicit, {usage.implicit_uses} implicit)"
                )

            # Only suggest change if different from current
            if suggested_priority != current_priority:
                suggestions.append(
                    PrioritySuggestion(
                        rule_name=rule_name,
                        current_priority=current_priority,
                        suggested_priority=suggested_priority,
                        usage_count=total_uses,
                        reason=reason,
                        file_path=rule_info[rule_name]["file_path"],
                    )
                )

        # Sort suggestions by usage count
        suggestions.sort(key=lambda x: x.usage_count, reverse=True)

        return suggestions

    def format_suggestions(self, suggestions: List[PrioritySuggestion]) -> str:
        """Format suggestions as readable text."""
        if not suggestions:
            return "No priority changes suggested - all rules seem appropriately prioritized."

        output = []
        output.append(
            f"Found {len(suggestions)} simp rules that could benefit from priority adjustment:\n"
        )

        for i, suggestion in enumerate(suggestions[:20], 1):  # Show top 20
            output.append(f"{i}. Rule: {suggestion.rule_name}")
            output.append(f"   File: {suggestion.file_path}")
            output.append(f"   Current priority: {suggestion.current_priority}")
            output.append(f"   Suggested priority: {suggestion.suggested_priority}")
            output.append(f"   Reason: {suggestion.reason}")
            output.append("")

        if len(suggestions) > 20:
            output.append(f"... and {len(suggestions) - 20} more suggestions")

        return "\n".join(output)

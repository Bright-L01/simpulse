"""Priority optimization for Lean 4 simp rules."""

import shutil
from dataclasses import dataclass
from pathlib import Path

from .analyzer import SimpRule


@dataclass
class OptimizationSuggestion:
    """Represents a suggested optimization for a simp rule."""

    rule_name: str
    file_path: str
    current_priority: int | None
    suggested_priority: int
    reason: str
    expected_speedup: float
    confidence: str  # "high", "medium", "low"

    def __str__(self) -> str:
        """String representation of the suggestion."""
        current = self.current_priority or "default (1000)"
        return (
            f"{self.rule_name}: {current} → {self.suggested_priority} "
            f"(+{self.expected_speedup:.1%} speedup, {self.confidence} confidence)"
        )


class PriorityOptimizer:
    """Optimizes simp rule priorities based on usage patterns."""

    def __init__(self, min_frequency_threshold: int = 5):
        """Initialize the optimizer.

        Args:
            min_frequency_threshold: Minimum frequency to consider for optimization.
        """
        self.min_frequency_threshold = min_frequency_threshold

    def calculate_priority(self, rule: SimpRule) -> int:
        """Calculate optimal priority for a simp rule based on frequency.

        Args:
            rule: SimpRule to calculate priority for.

        Returns:
            Suggested priority value (lower = higher priority).
        """
        frequency = rule.frequency

        # High frequency rules get low priority numbers (high priority)
        if frequency >= 100:
            return 100 + min(400, frequency // 5)  # 100-500 range
        elif frequency >= 50:
            return 500 + min(500, frequency * 5)  # 500-1000 range
        elif frequency >= 10:
            return 1000 + min(500, frequency * 20)  # 1000-1500 range
        else:
            return 1500 + min(500, frequency * 50)  # 1500-2000 range

    def optimize_project(self, analysis_result: dict) -> list[OptimizationSuggestion]:
        """Generate optimization suggestions for a project.

        Args:
            analysis_result: Project analysis results from LeanAnalyzer.

        Returns:
            List of OptimizationSuggestion objects.
        """
        rules = analysis_result.get("rules_by_frequency", [])
        return self._generate_suggestions(rules)

    def _generate_suggestions(self, rules: list[SimpRule]) -> list[OptimizationSuggestion]:
        """Generate optimization suggestions for a list of rules.

        Args:
            rules: List of SimpRule objects to optimize.

        Returns:
            List of OptimizationSuggestion objects.
        """
        suggestions = []

        for rule in rules:
            if rule.frequency < self.min_frequency_threshold:
                continue

            suggested_priority = self.calculate_priority(rule)
            expected_speedup = self._estimate_speedup(rule, suggested_priority)
            confidence = self._determine_confidence(rule, suggested_priority, expected_speedup)

            reason = self._generate_reason(rule, suggested_priority)

            suggestion = OptimizationSuggestion(
                rule_name=rule.name,
                file_path=str(rule.file_path),
                current_priority=rule.priority,
                suggested_priority=suggested_priority,
                reason=reason,
                expected_speedup=expected_speedup,
                confidence=confidence,
            )
            suggestions.append(suggestion)

        # Sort by expected speedup (highest first)
        return sorted(suggestions, key=lambda s: s.expected_speedup, reverse=True)

    def _estimate_speedup(self, rule: SimpRule, suggested_priority: int) -> float:
        """Estimate performance speedup from optimization.

        Args:
            rule: SimpRule being optimized.
            suggested_priority: Suggested new priority.

        Returns:
            Estimated speedup as a fraction (0.0 to 1.0).
        """
        frequency = rule.frequency
        current_priority = rule.priority or 1000

        # Higher frequency rules benefit more from better priorities
        frequency_factor = min(0.5, frequency / 200)  # Cap at 50%

        # Priority improvement factor
        if suggested_priority < current_priority:
            priority_factor = min(0.3, (current_priority - suggested_priority) / 1000)
        else:
            priority_factor = 0.0

        return frequency_factor * priority_factor

    def _determine_confidence(self, rule: SimpRule, suggested_priority: int, speedup: float) -> str:
        """Determine confidence level for the optimization suggestion.

        Args:
            rule: SimpRule being optimized.
            suggested_priority: Suggested new priority.
            speedup: Estimated speedup.

        Returns:
            Confidence level: "high", "medium", or "low".
        """
        if rule.frequency >= 100 and speedup >= 0.2:
            return "high"
        elif rule.frequency >= 50 and speedup >= 0.1:
            return "medium"
        else:
            return "low"

    def _generate_reason(self, rule: SimpRule, suggested_priority: int) -> str:
        """Generate human-readable reason for the optimization.

        Args:
            rule: SimpRule being optimized.
            suggested_priority: Suggested new priority.

        Returns:
            Human-readable explanation.
        """
        frequency = rule.frequency

        if frequency >= 100:
            return f"High frequency rule ({frequency} uses)"
        elif frequency >= 50:
            return f"Medium frequency rule ({frequency} uses)"
        else:
            return f"Low frequency rule ({frequency} uses)"

    def _apply_optimization_to_file(
        self, file_path: Path, suggestions: list[OptimizationSuggestion]
    ) -> bool:
        """Apply optimization suggestions to a file.

        Args:
            file_path: Path to the file to modify.
            suggestions: List of suggestions to apply.

        Returns:
            True if successful, False otherwise.
        """
        if not file_path.exists():
            return False

        try:
            content = file_path.read_text(encoding="utf-8")

            # Apply each suggestion
            for suggestion in suggestions:
                # Simple regex replacement (production would be more sophisticated)
                old_pattern = f"@[simp] theorem {suggestion.rule_name}"
                new_pattern = f"@[simp, priority := {suggestion.suggested_priority}] theorem {suggestion.rule_name}"
                content = content.replace(old_pattern, new_pattern)

            file_path.write_text(content, encoding="utf-8")
            return True

        except Exception:
            return False

    def _create_backup(self, file_path: Path) -> Path:
        """Create a backup of a file before modification.

        Args:
            file_path: Path to the file to backup.

        Returns:
            Path to the backup file.
        """
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        shutil.copy2(file_path, backup_path)
        return backup_path

    def generate_optimization_script(
        self, optimizations: dict[str, list[OptimizationSuggestion]], output_path: Path
    ) -> None:
        """Generate a Python script to apply optimizations.

        Args:
            optimizations: Dictionary mapping file paths to suggestions.
            output_path: Path where to save the generated script.
        """
        script_content = '''#!/usr/bin/env python3
"""Generated optimization script for Simpulse."""

import shutil
from pathlib import Path

def apply_optimizations():
    """Apply all optimizations with backup."""
    optimizations = {
'''

        for file_path, suggestions in optimizations.items():
            script_content += f'        "{file_path}": [\n'
            for suggestion in suggestions:
                script_content += f"            # {suggestion.reason}\n"
                script_content += (
                    f'            ("{suggestion.rule_name}", {suggestion.suggested_priority}),\n'
                )
            script_content += "        ],\n"

        script_content += """    }
    
    for file_path, changes in optimizations.items():
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: {file_path} not found")
            continue
            
        # Create backup
        backup = path.with_suffix(path.suffix + '.bak')
        shutil.copy2(path, backup)
        print(f"Created backup: {backup}")
        
        # Apply changes
        content = path.read_text()
        for rule_name, priority in changes:
            old = f"@[simp] theorem {rule_name}"
            new = f"@[simp, priority := {priority}] theorem {rule_name}"
            content = content.replace(old, new)
        
        path.write_text(content)
        print(f"Applied optimizations to: {file_path}")

if __name__ == "__main__":
    apply_optimizations()
"""

        output_path.write_text(script_content)
        output_path.chmod(0o755)  # Make executable

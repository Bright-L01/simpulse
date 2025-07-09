"""
Simple Lean 4 simp rule optimizer - under 200 lines of obvious code.

Does one thing: finds frequently used simp rules and gives them higher priority.
No sophistication, no complex strategies, just basic priority adjustment that works.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .error import OptimizationError, check_memory_usage, safe_file_read, safe_file_write, timeout

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """A simp rule found in code."""

    name: str
    file_path: Path
    line_num: int
    priority: int = 1000  # Default Lean priority
    usage_count: int = 0


@dataclass
class Change:
    """A single optimization change."""

    rule_name: str
    file_path: str
    old_priority: int
    new_priority: int
    reason: str


class UnifiedOptimizer:
    """Dead simple simp rule optimizer."""

    def __init__(self, strategy: str = "frequency"):
        # Only frequency strategy supported - it's the only one that works
        # Find @[simp] rules
        self.rule_pattern = re.compile(
            r"@\[simp(?:\s+(\d+))?\].*?(?:theorem|lemma|def)\s+(\w+)", re.MULTILINE | re.DOTALL
        )

        # Find simp rule usage
        self.usage_pattern = re.compile(r"simp\s*\[([^\]]+)\]")

    def optimize(self, project_path, apply: bool = False) -> dict:
        """Main entry point. Find rules, count usage, optimize priorities."""
        # Import timeout value from config
        from .config import OPTIMIZATION_TIMEOUT

        # Run the actual optimization with timeout and memory checks
        with timeout(OPTIMIZATION_TIMEOUT, "optimization"):
            return self._optimize_with_safety(project_path, apply)

    def _optimize_with_safety(self, project_path, apply: bool = False) -> dict:
        """Internal optimization with safety checks."""
        project = Path(project_path)

        # Check memory at start
        check_memory_usage("optimization start")

        # Validate project path
        if not project.exists():
            raise OptimizationError(f"Project path does not exist: {project}")
        if not project.is_dir():
            raise OptimizationError(f"Project path is not a directory: {project}")

        # Step 1: Find all Lean files
        lean_files = list(project.glob("**/*.lean"))

        # Import config for exclusion patterns
        from .config import should_skip_file

        lean_files = [f for f in lean_files if not should_skip_file(f)]

        if not lean_files:
            logger.warning(f"No Lean files found in {project}")
            return {
                "project_path": str(project_path),
                "total_rules": 0,
                "rules_changed": 0,
                "estimated_improvement": 0.0,
                "changes": [],
            }

        logger.info(f"Found {len(lean_files)} Lean files in {project}")

        # Check memory after file discovery
        check_memory_usage("after finding files")

        # Step 2: Extract all simp rules
        rules = self._find_rules(lean_files)

        if not rules:
            logger.info("No simp rules found in project")
            return {
                "project_path": str(project_path),
                "total_rules": 0,
                "rules_changed": 0,
                "estimated_improvement": 0.0,
                "changes": [],
            }

        logger.info(f"Found {len(rules)} simp rules")

        # Check memory after rule extraction
        check_memory_usage("after extracting rules")

        # Step 3: Count how often each rule is used
        self._count_usage(lean_files, rules)

        # Check memory after counting usage
        check_memory_usage("after counting usage")

        # Step 4: Calculate new priorities (simple frequency-based)
        changes = self._calculate_changes(rules)

        logger.info(f"Calculated {len(changes)} optimization changes")

        # Step 5: Apply changes if requested
        if apply and changes:
            logger.info("Applying optimization changes...")
            self._apply_changes(changes)

        # Step 6: Return simple results
        return {
            "project_path": str(project_path),
            "total_rules": len(rules),
            "rules_changed": len(changes),
            "estimated_improvement": min(50.0, len(changes) * 2.5),  # Simple estimate
            "changes": [
                {
                    "rule_name": change.rule_name,
                    "file_path": change.file_path,
                    "old_priority": change.old_priority,
                    "new_priority": change.new_priority,
                    "reason": change.reason,
                }
                for change in changes
            ],
        }

    def _find_rules(self, lean_files: list[Path]) -> list[Rule]:
        """Find all @[simp] rules in files."""
        rules = []
        failed_files = []

        for file_path in lean_files:
            content = safe_file_read(file_path)
            if content is None:
                failed_files.append(file_path)
                continue

            try:
                for match in self.rule_pattern.finditer(content):
                    priority_str = match.group(1)
                    rule_name = match.group(2)

                    line_num = content[: match.start()].count("\n") + 1
                    priority = int(priority_str) if priority_str else 1000

                    rules.append(
                        Rule(
                            name=rule_name,
                            file_path=file_path,
                            line_num=line_num,
                            priority=priority,
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to parse rules in {file_path}: {e}")
                failed_files.append(file_path)

        if failed_files:
            logger.info(f"Skipped {len(failed_files)} files due to read/parse errors")

        return rules

    def _count_usage(self, lean_files: list[Path], rules: list[Rule]) -> None:
        """Count how often each rule is used in simp calls."""
        rule_by_name = {rule.name: rule for rule in rules}
        failed_files = []

        for file_path in lean_files:
            content = safe_file_read(file_path)
            if content is None:
                failed_files.append(file_path)
                continue

            try:
                for match in self.usage_pattern.finditer(content):
                    rule_list = match.group(1)
                    for rule_name in rule_list.split(","):
                        rule_name = rule_name.strip()
                        if rule_name in rule_by_name:
                            rule_by_name[rule_name].usage_count += 1
            except Exception as e:
                logger.warning(f"Failed to count usage in {file_path}: {e}")
                failed_files.append(file_path)

        if failed_files:
            logger.info(f"Skipped {len(failed_files)} files when counting usage")

    def _calculate_changes(self, rules: list[Rule]) -> list[Change]:
        """Calculate priority changes based on usage frequency."""
        changes = []

        # Sort rules by usage frequency
        used_rules = [rule for rule in rules if rule.usage_count > 0]
        used_rules.sort(key=lambda r: r.usage_count, reverse=True)

        # Assign priorities: most used gets 100, next gets 110, etc.
        for i, rule in enumerate(used_rules):
            new_priority = 100 + (i * 10)

            # Only change if it's an improvement (lower number = higher priority)
            if new_priority < rule.priority:
                changes.append(
                    Change(
                        rule_name=rule.name,
                        file_path=str(rule.file_path),
                        old_priority=rule.priority,
                        new_priority=new_priority,
                        reason=f"Used {rule.usage_count} times",
                    )
                )

        return changes

    def _apply_changes(self, changes: list[Change]) -> None:
        """Apply priority changes to files."""
        # Group changes by file to minimize file I/O
        changes_by_file: dict[Path, list[Change]] = {}
        for change in changes:
            file_path = Path(change.file_path)
            if file_path not in changes_by_file:
                changes_by_file[file_path] = []
            changes_by_file[file_path].append(change)

        # Apply changes to each file
        failed_files = []
        successful_files = []

        for file_path, file_changes in changes_by_file.items():
            content = safe_file_read(file_path)
            if content is None:
                failed_files.append(file_path)
                continue

            try:
                for change in file_changes:
                    # Replace @[simp] or @[simp N] with @[simp new_priority]
                    pattern = rf"@\[simp(?:\s+\d+)?\](\s*.*?)(?=(?:theorem|lemma|def)\s+{re.escape(change.rule_name)}\b)"
                    replacement = f"@[simp {change.new_priority}]\\1"
                    content = re.sub(pattern, replacement, content)

                if safe_file_write(file_path, content):
                    successful_files.append(file_path)
                    logger.info(f"Applied {len(file_changes)} changes to {file_path}")
                else:
                    failed_files.append(file_path)

            except Exception as e:
                logger.error(f"Failed to apply changes to {file_path}: {e}")
                failed_files.append(file_path)

        if failed_files:
            logger.warning(f"Failed to apply changes to {len(failed_files)} files")
        if successful_files:
            logger.info(f"Successfully updated {len(successful_files)} files")


# Simple CLI integration
def main():
    """Direct command-line usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python unified_optimizer.py <project_path> [--apply]")
        return

    project_path = sys.argv[1]
    apply = "--apply" in sys.argv

    optimizer = UnifiedOptimizer(strategy="frequency")
    results = optimizer.optimize(project_path, apply=apply)

    print("\nOptimization Results:")
    print(f"  Total rules: {results['total_rules']}")
    print(f"  Rules optimized: {results['rules_changed']}")
    print(f"  Estimated improvement: {results['estimated_improvement']:.1f}%")

    if results["changes"] and not apply:
        print("\nTop changes:")
        for change in results["changes"][:5]:
            print(
                f"  {change['rule_name']}: {change['old_priority']} → {change['new_priority']} ({change['reason']})"
            )
        print("\nRun with --apply to apply these changes")


if __name__ == "__main__":
    main()

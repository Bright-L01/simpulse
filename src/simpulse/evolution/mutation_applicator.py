"""Mutation applicator for modifying Lean files with simp rules.

This module provides functionality to apply mutations to simp rules
by modifying the source files directly while preserving formatting.
"""

import asyncio
import difflib
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .models import MutationSuggestion, MutationType, SimpRule

logger = logging.getLogger(__name__)


@dataclass
class AppliedMutation:
    """Represents a successfully applied mutation."""

    original_rule: SimpRule
    suggestion: MutationSuggestion
    modified_content: str
    backup_content: str
    file_path: Path
    line_numbers: Tuple[int, int]  # (start, end)
    success: bool = True
    error_message: Optional[str] = None

    def rollback(self) -> bool:
        """Rollback the mutation by restoring original content."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(self.backup_content)
            return True
        except Exception as e:
            logger.error(f"Failed to rollback mutation: {e}")
            return False


@dataclass
class MutationSet:
    """Collection of mutations applied together."""

    mutations: List[AppliedMutation]
    workspace_path: Path
    success: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def rollback_all(self) -> bool:
        """Rollback all mutations in the set."""
        success = True
        for mutation in reversed(self.mutations):  # Rollback in reverse order
            if not mutation.rollback():
                success = False
        return success

    @property
    def modified_files(self) -> Set[Path]:
        """Get set of files modified by this mutation set."""
        return {m.file_path for m in self.mutations}


class MutationApplicator:
    """Applies mutations to simp rules in Lean source files."""

    def __init__(self, preserve_formatting: bool = True):
        """Initialize mutation applicator.

        Args:
            preserve_formatting: Whether to preserve original formatting
        """
        self.preserve_formatting = preserve_formatting
        self._simp_attr_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for simp attribute matching."""
        return {
            # Match @[simp] with optional priority and direction
            "full_attr": re.compile(
                r"@\[simp(?:\s+(high|low|\d+))?(?:\s*([↓←]))?\]", re.MULTILINE
            ),
            # Match just the simp part for replacement
            "simp_content": re.compile(r"simp(?:\s+(high|low|\d+))?(?:\s*([↓←]))?"),
            # Find declaration after attribute
            "declaration": re.compile(
                r"(theorem|lemma|def|instance|axiom)\s+([a-zA-Z_][a-zA-Z0-9_\']*)",
                re.MULTILINE,
            ),
        }

    async def apply_mutation(
        self, rule: SimpRule, suggestion: MutationSuggestion
    ) -> AppliedMutation:
        """Apply a single mutation to a rule.

        Args:
            rule: Original simp rule
            suggestion: Mutation to apply

        Returns:
            AppliedMutation with result details
        """
        if not rule.location or not rule.location.file:
            return AppliedMutation(
                original_rule=rule,
                suggestion=suggestion,
                modified_content="",
                backup_content="",
                file_path=Path(),
                line_numbers=(0, 0),
                success=False,
                error_message="Rule has no location information",
            )

        file_path = rule.location.file

        try:
            # Read original file content
            with open(file_path, encoding="utf-8") as f:
                original_content = f.read()

            # Apply the specific mutation
            modified_content, line_numbers = self._apply_mutation_to_content(
                original_content, rule, suggestion
            )

            # Write modified content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified_content)

            return AppliedMutation(
                original_rule=rule,
                suggestion=suggestion,
                modified_content=modified_content,
                backup_content=original_content,
                file_path=file_path,
                line_numbers=line_numbers,
                success=True,
            )

        except Exception as e:
            logger.error(f"Failed to apply mutation to {rule.name}: {e}")
            return AppliedMutation(
                original_rule=rule,
                suggestion=suggestion,
                modified_content="",
                backup_content="",
                file_path=file_path,
                line_numbers=(0, 0),
                success=False,
                error_message=str(e),
            )

    def _apply_mutation_to_content(
        self, content: str, rule: SimpRule, suggestion: MutationSuggestion
    ) -> Tuple[str, Tuple[int, int]]:
        """Apply mutation to file content.

        Args:
            content: Original file content
            rule: Rule to modify
            suggestion: Mutation to apply

        Returns:
            Tuple of (modified_content, line_numbers)
        """
        lines = content.split("\n")

        # Find the rule's location
        rule_line = rule.location.line - 1  # Convert to 0-based indexing

        # Find the @[simp] attribute line (may be before the rule)
        attr_line = self._find_simp_attribute_line(lines, rule_line, rule.name)

        if attr_line is None:
            raise ValueError(f"Could not find @[simp] attribute for rule {rule.name}")

        # Apply mutation based on type
        if suggestion.mutation_type == MutationType.PRIORITY_CHANGE:
            lines[attr_line] = self._apply_priority_change(lines[attr_line], suggestion)
        elif suggestion.mutation_type == MutationType.DIRECTION_CHANGE:
            lines[attr_line] = self._apply_direction_change(
                lines[attr_line], suggestion
            )
        elif suggestion.mutation_type == MutationType.CONDITION_ADD:
            # Condition changes require modifying the declaration, not just the attribute
            decl_line = self._find_declaration_line(lines, attr_line)
            if decl_line is not None:
                lines[decl_line] = self._apply_condition_add(
                    lines[decl_line], suggestion
                )
        elif suggestion.mutation_type == MutationType.CONDITION_REMOVE:
            decl_line = self._find_declaration_line(lines, attr_line)
            if decl_line is not None:
                lines[decl_line] = self._apply_condition_remove(
                    lines[decl_line], suggestion
                )
        elif suggestion.mutation_type == MutationType.RULE_DISABLE:
            # Comment out the @[simp] attribute
            lines[attr_line] = f"-- {lines[attr_line]}"
        else:
            raise ValueError(f"Unsupported mutation type: {suggestion.mutation_type}")

        modified_content = "\n".join(lines)
        line_numbers = (attr_line + 1, attr_line + 1)  # Convert back to 1-based

        return modified_content, line_numbers

    def _find_simp_attribute_line(
        self, lines: List[str], rule_line: int, rule_name: str
    ) -> Optional[int]:
        """Find the line containing the @[simp] attribute for a rule.

        Args:
            lines: File lines
            rule_line: Line where rule is declared
            rule_name: Name of the rule

        Returns:
            Line number (0-based) or None if not found
        """
        # Search backwards from rule line to find @[simp]
        for i in range(rule_line, max(-1, rule_line - 10), -1):
            if i < len(lines) and "@[simp" in lines[i]:
                return i

        # If not found, search forward (attribute might be on same line)
        for i in range(rule_line, min(len(lines), rule_line + 3)):
            if "@[simp" in lines[i]:
                return i

        return None

    def _find_declaration_line(self, lines: List[str], attr_line: int) -> Optional[int]:
        """Find the declaration line after an attribute.

        Args:
            lines: File lines
            attr_line: Line with attribute

        Returns:
            Line number (0-based) or None if not found
        """
        # Search forward from attribute line
        for i in range(attr_line, min(len(lines), attr_line + 5)):
            if self._simp_attr_patterns["declaration"].search(lines[i]):
                return i

        return None

    def _apply_priority_change(self, line: str, suggestion: MutationSuggestion) -> str:
        """Apply priority change to simp attribute.

        Args:
            line: Line with @[simp] attribute
            suggestion: Mutation suggestion

        Returns:
            Modified line
        """
        # Extract target priority from suggestion
        target_priority = self._extract_target_priority(suggestion)

        # Replace the simp attribute content
        def replace_simp(match):
            if target_priority == "default":
                return "simp"
            elif target_priority in ["high", "low"]:
                return f"simp {target_priority}"
            else:
                return f"simp {target_priority}"

        # Use the simp_content pattern to replace just the simp part
        modified = self._simp_attr_patterns["simp_content"].sub(replace_simp, line)

        return modified

    def _apply_direction_change(self, line: str, suggestion: MutationSuggestion) -> str:
        """Apply direction change to simp attribute.

        Args:
            line: Line with @[simp] attribute
            suggestion: Mutation suggestion

        Returns:
            Modified line
        """
        # Extract target direction from suggestion
        target_direction = self._extract_target_direction(suggestion)

        def replace_simp(match):
            current_priority = match.group(1) if match.group(1) else ""

            if target_direction == "backward":
                if current_priority:
                    return f"simp {current_priority} ↓"
                else:
                    return "simp ↓"
            else:  # forward direction
                if current_priority:
                    return f"simp {current_priority}"
                else:
                    return "simp"

        modified = self._simp_attr_patterns["simp_content"].sub(replace_simp, line)

        return modified

    def _apply_condition_add(self, line: str, suggestion: MutationSuggestion) -> str:
        """Add condition to rule declaration.

        Args:
            line: Declaration line
            suggestion: Mutation suggestion

        Returns:
            Modified line
        """
        # Extract condition from mutated_declaration in suggestion
        condition = self._extract_condition_from_suggestion(suggestion)

        if not condition:
            return line

        # Find where to insert the condition (usually after parameters)
        # This is a simplified approach - a full parser would be more robust
        if "(" in line and ")" in line:
            # Insert before the last closing paren
            last_paren = line.rfind(")")
            modified = line[:last_paren] + f" [{condition}]" + line[last_paren:]
        else:
            # Insert before the colon
            colon_pos = line.find(":")
            if colon_pos != -1:
                modified = line[:colon_pos] + f" [{condition}]" + line[colon_pos:]
            else:
                modified = line

        return modified

    def _apply_condition_remove(self, line: str, suggestion: MutationSuggestion) -> str:
        """Remove condition from rule declaration.

        Args:
            line: Declaration line
            suggestion: Mutation suggestion

        Returns:
            Modified line
        """
        # Remove bracketed conditions [...]
        condition_pattern = re.compile(r"\s*\[[^\]]+\]")
        modified = condition_pattern.sub("", line)

        return modified

    def _extract_target_priority(self, suggestion: MutationSuggestion) -> str:
        """Extract target priority from mutation suggestion.

        Args:
            suggestion: Mutation suggestion

        Returns:
            Target priority as string
        """
        # Parse the mutated_declaration to extract priority
        mutated = suggestion.mutated_declaration

        if "@[simp high]" in mutated:
            return "high"
        elif "@[simp low]" in mutated:
            return "low"
        elif "@[simp " in mutated:
            # Extract numeric priority
            match = re.search(r"@\[simp\s+(\d+)", mutated)
            if match:
                return match.group(1)

        return "default"

    def _extract_target_direction(self, suggestion: MutationSuggestion) -> str:
        """Extract target direction from mutation suggestion.

        Args:
            suggestion: Mutation suggestion

        Returns:
            Target direction as string
        """
        mutated = suggestion.mutated_declaration

        if "↓" in mutated or "←" in mutated:
            return "backward"
        else:
            return "forward"

    def _extract_condition_from_suggestion(self, suggestion: MutationSuggestion) -> str:
        """Extract condition to add from mutation suggestion.

        Args:
            suggestion: Mutation suggestion

        Returns:
            Condition string
        """
        mutated = suggestion.mutated_declaration

        # Find bracketed conditions in the mutated declaration
        condition_match = re.search(r"\[([^\]]+)\]", mutated)
        if condition_match:
            return condition_match.group(1)

        return ""

    async def apply_mutation_set(
        self,
        mutations: List[MutationSuggestion],
        rules: List[SimpRule],
        workspace: Path,
    ) -> MutationSet:
        """Apply multiple mutations atomically.

        Args:
            mutations: List of mutations to apply
            rules: List of corresponding rules
            workspace: Workspace directory

        Returns:
            MutationSet with results
        """
        if len(mutations) != len(rules):
            raise ValueError("Number of mutations must match number of rules")

        applied_mutations = []
        success = True
        validation_errors = []

        try:
            # Apply all mutations
            for mutation, rule in zip(mutations, rules):
                applied = await self.apply_mutation(rule, mutation)
                applied_mutations.append(applied)

                if not applied.success:
                    success = False
                    validation_errors.append(
                        f"Failed to apply mutation to {rule.name}: {applied.error_message}"
                    )

            # Validate syntax if all mutations applied successfully
            if success:
                syntax_errors = await self._validate_syntax(workspace)
                if syntax_errors:
                    success = False
                    validation_errors.extend(syntax_errors)

            # If any failures, rollback all changes
            if not success:
                logger.warning("Mutation set failed, rolling back changes")
                for applied in applied_mutations:
                    if applied.success:
                        applied.rollback()

        except Exception as e:
            logger.error(f"Error applying mutation set: {e}")
            success = False
            validation_errors.append(str(e))

            # Rollback any successful mutations
            for applied in applied_mutations:
                if applied.success:
                    applied.rollback()

        return MutationSet(
            mutations=applied_mutations,
            workspace_path=workspace,
            success=success,
            validation_errors=validation_errors,
        )

    async def _validate_syntax(self, workspace: Path) -> List[str]:
        """Validate syntax of modified files using lean check.

        Args:
            workspace: Workspace directory

        Returns:
            List of validation errors
        """
        errors = []

        try:
            # Run basic lean syntax check
            process = await asyncio.create_subprocess_exec(
                "lean",
                "--check",
                str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_output = stderr.decode("utf-8", errors="replace")
                errors.append(f"Syntax validation failed: {error_output}")

        except Exception as e:
            errors.append(f"Failed to run syntax validation: {e}")

        return errors

    def generate_patch(
        self, original: str, mutated: str, filename: str = "file"
    ) -> str:
        """Generate git-compatible patch.

        Args:
            original: Original file content
            mutated: Mutated file content
            filename: File name for patch header

        Returns:
            Git patch string
        """
        original_lines = original.splitlines(keepends=True)
        mutated_lines = mutated.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            mutated_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )

        return "".join(diff)

    def backup_files(
        self, file_paths: List[Path], backup_dir: Path
    ) -> Dict[Path, Path]:
        """Create backups of files before modification.

        Args:
            file_paths: Files to backup
            backup_dir: Directory for backups

        Returns:
            Mapping of original to backup paths
        """
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_mapping = {}

        for file_path in file_paths:
            if not file_path.exists():
                continue

            backup_path = backup_dir / f"{file_path.name}.backup"
            shutil.copy2(file_path, backup_path)
            backup_mapping[file_path] = backup_path

        return backup_mapping

    def restore_from_backups(self, backup_mapping: Dict[Path, Path]) -> bool:
        """Restore files from backups.

        Args:
            backup_mapping: Mapping of original to backup paths

        Returns:
            True if all files restored successfully
        """
        success = True

        for original_path, backup_path in backup_mapping.items():
            try:
                if backup_path.exists():
                    shutil.copy2(backup_path, original_path)
            except Exception as e:
                logger.error(f"Failed to restore {original_path} from backup: {e}")
                success = False

        return success

    def _is_valid_filename(self, filename: str) -> bool:
        """Check if filename is valid and safe.

        Args:
            filename: Filename to validate

        Returns:
            True if filename is valid
        """
        from ..security.validators import is_valid_filename

        return is_valid_filename(filename)

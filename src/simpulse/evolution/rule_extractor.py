"""Extractor for simp rules from Lean 4 source code.

This module provides functionality to parse Lean files and extract
simp rule declarations with their metadata.
"""

import logging
import re
from pathlib import Path

from .models import ModuleRules, SimpDirection, SimpPriority, SimpRule, SourceLocation

logger = logging.getLogger(__name__)


class RuleExtractor:
    """Extracts simp rules from Lean 4 source code."""

    # Regex patterns for parsing simp attributes
    SIMP_ATTR_PATTERNS = {
        "basic": re.compile(r"@\[simp\]", re.MULTILINE),
        "priority_named": re.compile(r"@\[simp\s+(high|low)\]", re.MULTILINE),
        "priority_numeric": re.compile(r"@\[simp\s+(\d+)\]", re.MULTILINE),
        "direction": re.compile(r"@\[simp\s*([↓←])\]", re.MULTILINE),
        "combined": re.compile(r"@\[simp\s*(high|low|\d+)?\s*([↓←])?\]", re.MULTILINE),
    }

    # Pattern for declarations that can have simp attributes
    DECLARATION_PATTERNS = {
        "theorem": re.compile(r"^(theorem|lemma)\s+([a-zA-Z_][a-zA-Z0-9_\']*)", re.MULTILINE),
        "def": re.compile(r"^def\s+([a-zA-Z_][a-zA-Z0-9_\']*)", re.MULTILINE),
        "instance": re.compile(r"^instance\s+([a-zA-Z_][a-zA-Z0-9_\']*)?", re.MULTILINE),
        "axiom": re.compile(r"^axiom\s+([a-zA-Z_][a-zA-Z0-9_\']*)", re.MULTILINE),
    }

    # Pattern for extracting full declarations
    FULL_DECL_PATTERN = re.compile(
        r"(@\[.*?\])?\s*(theorem|lemma|def|instance|axiom)\s+([^:]+):\s*([^:=]+)(?::=\s*(.+?)(?=\n(?:theorem|lemma|def|instance|axiom|@\[|\Z)))?",
        re.MULTILINE | re.DOTALL,
    )

    def __init__(self):
        """Initialize rule extractor."""
        self._cache: dict[Path, ModuleRules] = {}

    def extract_rules_from_file(self, file_path: Path) -> ModuleRules:
        """Extract simp rules from a single Lean file.

        Args:
            file_path: Path to Lean file

        Returns:
            ModuleRules containing extracted rules
        """
        # Check cache first
        if file_path in self._cache:
            return self._cache[file_path]

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return ModuleRules(module_name=file_path.stem, file_path=file_path, rules=[])

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return ModuleRules(module_name=file_path.stem, file_path=file_path, rules=[])

        # Extract module info
        module_name = self._extract_module_name(content, file_path)
        imports = self._extract_imports(content)

        # Extract simp rules
        rules = self._extract_simp_rules(content, file_path)

        module_rules = ModuleRules(
            module_name=module_name,
            file_path=file_path,
            rules=rules,
            imports=imports,
            metadata={
                "total_rules": len(rules),
                "file_size": len(content),
                "line_count": content.count("\n") + 1,
            },
        )

        # Cache result
        self._cache[file_path] = module_rules

        return module_rules

    def extract_rules_from_module(self, module_path: Path) -> ModuleRules:
        """Extract simp rules from a module directory.

        Args:
            module_path: Path to module directory or file

        Returns:
            ModuleRules containing all rules from the module
        """
        if module_path.is_file():
            return self.extract_rules_from_file(module_path)

        # Process directory
        all_rules = []
        all_imports = set()

        for lean_file in module_path.rglob("*.lean"):
            file_rules = self.extract_rules_from_file(lean_file)
            all_rules.extend(file_rules.rules)
            all_imports.update(file_rules.imports)

        return ModuleRules(
            module_name=module_path.name,
            file_path=module_path,
            rules=all_rules,
            imports=list(all_imports),
            metadata={
                "total_files": len(list(module_path.rglob("*.lean"))),
                "total_rules": len(all_rules),
            },
        )

    def _extract_module_name(self, content: str, file_path: Path) -> str:
        """Extract module name from file content or path."""
        # Look for namespace declaration
        namespace_match = re.search(r"namespace\s+([a-zA-Z_][a-zA-Z0-9_\.]*)", content)
        if namespace_match:
            return namespace_match.group(1)

        # Fall back to file name
        return file_path.stem

    def _extract_imports(self, content: str) -> list[str]:
        """Extract import statements from file content."""
        imports = []
        import_pattern = re.compile(r"^import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)", re.MULTILINE)

        for match in import_pattern.finditer(content):
            imports.append(match.group(1))

        return imports

    def _extract_simp_rules(self, content: str, file_path: Path) -> list[SimpRule]:
        """Extract simp rules from file content.

        Args:
            content: File content
            file_path: Path to file for location tracking

        Returns:
            List of extracted SimpRule objects
        """
        rules = []
        lines = content.split("\n")

        # Find all simp attributes and their locations
        simp_locations = self._find_simp_attributes(content)

        # For each simp attribute, find the associated declaration
        for attr_line, attr_info in simp_locations:
            declaration = self._find_declaration_after_attribute(lines, attr_line, attr_info)

            if declaration:
                rule = self._create_simp_rule(declaration, attr_info, file_path, attr_line + 1)
                if rule:
                    rules.append(rule)

        return rules

    def _find_simp_attributes(self, content: str) -> list[tuple[int, dict]]:
        """Find all simp attributes and their metadata.

        Returns:
            List of (line_number, attribute_info) tuples
        """
        locations = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            # Check for simp attributes
            simp_match = re.search(r"@\[simp(?:\s+(high|low|\d+))?(?:\s*([↓←]))?\]", line)
            if simp_match:
                priority_str = simp_match.group(1)
                direction_str = simp_match.group(2)

                # Parse priority
                priority = SimpPriority.DEFAULT
                if priority_str:
                    if priority_str == "high":
                        priority = SimpPriority.HIGH
                    elif priority_str == "low":
                        priority = SimpPriority.LOW
                    elif priority_str.isdigit():
                        priority = int(priority_str)

                # Parse direction
                direction = SimpDirection.FORWARD
                if direction_str in ["↓", "←"]:
                    direction = SimpDirection.BACKWARD

                attr_info = {
                    "priority": priority,
                    "direction": direction,
                    "raw_attr": simp_match.group(0),
                }

                locations.append((i, attr_info))

        return locations

    def _find_declaration_after_attribute(
        self, lines: list[str], attr_line: int, attr_info: dict
    ) -> dict | None:
        """Find the declaration following a simp attribute.

        Args:
            lines: File lines
            attr_line: Line number of the attribute
            attr_info: Parsed attribute information

        Returns:
            Declaration information or None
        """
        # Look for declaration in the next few lines
        for i in range(attr_line + 1, min(len(lines), attr_line + 5)):
            line = lines[i].strip()

            if not line or line.startswith("--"):
                continue

            # Check for various declaration types
            for decl_type, pattern in self.DECLARATION_PATTERNS.items():
                match = pattern.match(line)
                if match:
                    # Extract full declaration
                    full_decl = self._extract_full_declaration(lines, i)

                    return {
                        "type": decl_type,
                        "name": match.group(1) if match.lastindex else "unnamed",
                        "line": i,
                        "declaration": full_decl,
                        "first_line": line,
                    }

        return None

    def _extract_full_declaration(self, lines: list[str], start_line: int) -> str:
        """Extract the complete declaration starting from a line.

        Args:
            lines: File lines
            start_line: Starting line number

        Returns:
            Complete declaration text
        """
        decl_lines = []
        brace_count = 0
        paren_count = 0
        in_declaration = True

        for i in range(start_line, len(lines)):
            line = lines[i]
            decl_lines.append(line)

            # Track nested structures
            brace_count += line.count("{") - line.count("}")
            paren_count += line.count("(") - line.count(")")

            # Check for end of declaration
            if in_declaration:
                # Simple heuristic: declaration ends when we hit another top-level construct
                # or when brackets are balanced and we see certain keywords
                stripped = line.strip()
                if (
                    brace_count == 0
                    and paren_count == 0
                    and (
                        stripped.endswith(":=")
                        or stripped.startswith(("theorem", "lemma", "def", "instance", "@["))
                        or (stripped == "" and i > start_line)
                    )
                ):
                    break

        return "\n".join(decl_lines)

    def _create_simp_rule(
        self, declaration: dict, attr_info: dict, file_path: Path, line_num: int
    ) -> SimpRule | None:
        """Create a SimpRule object from extracted information.

        Args:
            declaration: Declaration information
            attr_info: Attribute information
            file_path: Source file path
            line_num: Line number

        Returns:
            SimpRule object or None if creation fails
        """
        try:
            # Parse the declaration to extract pattern and RHS
            pattern, rhs = self._parse_declaration_components(declaration["declaration"])

            # Extract conditions (type class constraints, etc.)
            conditions = self._extract_conditions(declaration["declaration"])

            location = SourceLocation(
                file=file_path, line=line_num, column=0, module=file_path.stem
            )

            rule = SimpRule(
                name=declaration["name"],
                declaration=declaration["declaration"],
                priority=attr_info["priority"],
                direction=attr_info["direction"],
                location=location,
                conditions=conditions,
                pattern=pattern,
                rhs=rhs,
                metadata={
                    "declaration_type": declaration["type"],
                    "raw_attribute": attr_info["raw_attr"],
                    "file_path": str(file_path),
                },
            )

            return rule

        except Exception as e:
            logger.warning(
                f"Failed to create simp rule from {declaration.get('name', 'unknown')}: {e}"
            )
            return None

    def _parse_declaration_components(self, declaration: str) -> tuple[str | None, str | None]:
        """Parse declaration to extract pattern and RHS.

        Args:
            declaration: Full declaration text

        Returns:
            Tuple of (pattern, rhs)
        """
        # Simple pattern matching for common forms
        # This is a basic implementation - a full parser would be more robust

        # Look for equality patterns: pattern = rhs
        eq_match = re.search(r":\s*([^=]+)\s*=\s*(.+)", declaration, re.DOTALL)
        if eq_match:
            return eq_match.group(1).strip(), eq_match.group(2).strip()

        # Look for type signatures: name : type := body
        sig_match = re.search(r":\s*([^:=]+)(?::=\s*(.+))?", declaration, re.DOTALL)
        if sig_match:
            pattern = sig_match.group(1).strip()
            rhs = sig_match.group(2).strip() if sig_match.group(2) else None
            return pattern, rhs

        return None, None

    def _extract_conditions(self, declaration: str) -> list[str]:
        """Extract type class constraints and other conditions.

        Args:
            declaration: Full declaration text

        Returns:
            List of condition strings
        """
        conditions = []

        # Look for type class constraints in square brackets
        constraint_pattern = re.compile(r"\[([^\]]+)\]")
        for match in constraint_pattern.finditer(declaration):
            conditions.append(match.group(1).strip())

        # Look for explicit conditions after ':'
        # This is a simplified approach
        if ":" in declaration:
            declaration.split(":")[1].split(":=")[0]
            # Extract conditions from type signature
            # This would need more sophisticated parsing for complex cases

        return conditions

    def get_rules_by_priority(self, rules: list[SimpRule]) -> dict[str, list[SimpRule]]:
        """Group rules by priority level.

        Args:
            rules: List of simp rules

        Returns:
            Dictionary mapping priority levels to rule lists
        """
        groups = {"high": [], "default": [], "low": [], "numeric": []}

        for rule in rules:
            if isinstance(rule.priority, int):
                groups["numeric"].append(rule)
            else:
                groups[rule.priority.value].append(rule)

        return groups

    def find_rules_by_pattern(self, rules: list[SimpRule], pattern: str) -> list[SimpRule]:
        """Find rules matching a specific pattern.

        Args:
            rules: List of simp rules to search
            pattern: Pattern to match (regex)

        Returns:
            List of matching rules
        """
        matching = []
        pattern_re = re.compile(pattern, re.IGNORECASE)

        for rule in rules:
            if (
                (rule.pattern and pattern_re.search(rule.pattern))
                or (rule.name and pattern_re.search(rule.name))
                or pattern_re.search(rule.declaration)
            ):
                matching.append(rule)

        return matching

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()

    def _is_valid_module_name(self, name: str) -> bool:
        """Validate a module name.

        Args:
            name: Module name to validate

        Returns:
            True if valid module name
        """
        from ..security.validators import validate_module_name

        return validate_module_name(name)

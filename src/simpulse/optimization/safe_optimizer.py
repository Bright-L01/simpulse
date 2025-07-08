"""
Safe optimization with guards to prevent performance regressions
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class OptimizationGuard:
    """Guards to prevent harmful optimizations."""

    min_arithmetic_ratio: float = 0.15  # At least 15% arithmetic operations
    min_file_size: int = 500  # At least 500 chars (avoid tiny files)
    max_list_ratio: float = 0.30  # No more than 30% list operations
    min_simp_usage: int = 3  # At least 3 simp calls
    forbidden_patterns: List[str] = None

    def __post_init__(self):
        if self.forbidden_patterns is None:
            self.forbidden_patterns = [
                r"@\[simp\]",  # Custom simp lemmas
                r"mutual\s+def",  # Mutual recursion
                r"TypeClass",  # Heavy typeclass usage
                r"\.reverse",  # List reversal operations
                r"dependent\s+type",  # Dependent types
            ]


@dataclass
class FileAnalysis:
    """Analysis results for a Lean file."""

    total_lines: int = 0
    arithmetic_ops: int = 0
    list_ops: int = 0
    simp_calls: int = 0
    custom_simp_lemmas: int = 0
    has_forbidden_patterns: bool = False
    arithmetic_ratio: float = 0.0
    list_ratio: float = 0.0


class SafeOptimizer:
    """Optimizer with safety guards to prevent regressions."""

    # Proven safe optimizations
    SAFE_OPTIMIZATIONS = """
-- Simpulse Safe Mode: Only proven optimizations
attribute [simp 1200] Nat.add_zero Nat.zero_add
attribute [simp 1199] Nat.mul_one Nat.one_mul
"""

    # Extended optimizations (use with caution)
    EXTENDED_OPTIMIZATIONS = """
-- Additional optimizations (may cause regressions)
attribute [simp 1198] eq_self_iff_true true_and and_true
attribute [simp 1197] List.append_nil
attribute [simp 1196] or_false false_or
"""

    def __init__(self, safe_mode: bool = True):
        self.safe_mode = safe_mode
        self.guard = OptimizationGuard()

        # Critical safety patterns (prevent catastrophic failures)
        self.critical_patterns = [
            r"@\[simp\s+\d+\]",  # Custom simp priorities (cause conflicts)
            r"declare_simp_like_tactic",  # Custom simp tactics
            r"mutual\s+def",  # Mutual recursion (causes regressions)
            r"@\[simp\]\s+def.*:.*→.*→",  # Recursive simp definitions
        ]

    def analyze_file(self, content: str) -> FileAnalysis:
        """Analyze a Lean file to determine optimization safety."""
        analysis = FileAnalysis()
        lines = content.split("\n")
        analysis.total_lines = len(lines)

        # Pattern counters
        arithmetic_patterns = [
            r"\+\s*0\b",
            r"\b0\s*\+",
            r"\*\s*1\b",
            r"\b1\s*\*",
            r"Nat\.add",
            r"Nat\.mul",
            r"Nat\.zero",
            r"Nat\.succ",
        ]

        list_patterns = [
            r"\+\+\s*\[\]",
            r"\[\]\s*\+\+",
            r"\.reverse",
            r"\.append",
            r"List\.",
            r":::",
            r"\.head",
            r"\.tail",
            r"\.length",
        ]

        for line in lines:
            # Skip comments
            if line.strip().startswith("--"):
                continue

            # Count arithmetic operations
            for pattern in arithmetic_patterns:
                analysis.arithmetic_ops += len(re.findall(pattern, line))

            # Count list operations
            for pattern in list_patterns:
                analysis.list_ops += len(re.findall(pattern, line))

            # Count simp calls
            analysis.simp_calls += len(re.findall(r"\bsimp\b", line))

            # Count custom simp lemmas
            if re.search(r"@\[simp(?:\s+\d+)?\]", line):
                analysis.custom_simp_lemmas += 1

            # Check critical failure patterns first
            for pattern in self.critical_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    analysis.has_forbidden_patterns = True
                    break

            # Check other forbidden patterns
            if not analysis.has_forbidden_patterns:
                for pattern in self.guard.forbidden_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        analysis.has_forbidden_patterns = True
                        break

        # Calculate ratios
        total_ops = analysis.arithmetic_ops + analysis.list_ops
        if total_ops > 0:
            analysis.arithmetic_ratio = analysis.arithmetic_ops / total_ops
            analysis.list_ratio = analysis.list_ops / total_ops

        return analysis

    def should_optimize(self, analysis: FileAnalysis) -> Tuple[bool, List[str]]:
        """Determine if optimization is safe based on analysis."""
        reasons = []

        # CRITICAL SAFETY CHECKS (prevent catastrophic failures)

        # Check for stack overflow risk (large files)
        if analysis.total_lines > 1000:
            reasons.append("File too large (>1000 lines, stack overflow risk)")
            return False, reasons

        # Check for critical failure patterns
        if analysis.has_forbidden_patterns:
            reasons.append(
                "Contains critical failure patterns (custom simp priorities, recursive definitions)"
            )
            return False, reasons

        # Check for custom simp infrastructure
        if analysis.custom_simp_lemmas > 0:
            reasons.append(
                f"Has custom simp lemmas ({analysis.custom_simp_lemmas} found) - may conflict"
            )
            return False, reasons

        # PERFORMANCE CHECKS

        # Check file size
        if analysis.total_lines < 5:
            reasons.append("File too small (overhead will dominate)")

        # Check arithmetic ratio
        if analysis.arithmetic_ratio < self.guard.min_arithmetic_ratio:
            reasons.append(
                f"Low arithmetic ratio ({analysis.arithmetic_ratio:.1%} < {self.guard.min_arithmetic_ratio:.1%})"
            )

        # Check list ratio
        if analysis.list_ratio > self.guard.max_list_ratio:
            reasons.append(
                f"High list operation ratio ({analysis.list_ratio:.1%} > {self.guard.max_list_ratio:.1%})"
            )

        # Check simp usage
        if analysis.simp_calls < self.guard.min_simp_usage:
            reasons.append(f"Low simp usage ({analysis.simp_calls} < {self.guard.min_simp_usage})")

        # In safe mode, be more conservative
        if self.safe_mode:
            # Require high arithmetic ratio and no red flags
            safe = (
                analysis.arithmetic_ratio > 0.3
                and analysis.list_ratio < 0.1
                and analysis.total_lines >= 5
                and analysis.simp_calls >= self.guard.min_simp_usage
            )
            if not safe and not reasons:
                reasons.append("Does not meet safe mode criteria")
            return safe, reasons

        # In normal mode, just avoid known bad patterns
        return len(reasons) == 0, reasons

    def get_optimizations(self, analysis: FileAnalysis) -> str:
        """Get appropriate optimizations based on analysis."""
        if self.safe_mode:
            return self.SAFE_OPTIMIZATIONS

        # In extended mode, still be cautious
        if analysis.arithmetic_ratio > 0.5 and analysis.list_ratio < 0.1:
            return self.SAFE_OPTIMIZATIONS + "\n" + self.EXTENDED_OPTIMIZATIONS
        else:
            return self.SAFE_OPTIMIZATIONS

    def optimize_file(self, file_path: Path) -> Tuple[bool, str, List[str]]:
        """
        Optimize a file with safety checks.
        Returns: (should_optimize, optimization_code, reasons)
        """
        try:
            content = file_path.read_text()
            analysis = self.analyze_file(content)
            should_opt, reasons = self.should_optimize(analysis)

            if should_opt:
                optimizations = self.get_optimizations(analysis)
                return (
                    True,
                    optimizations,
                    [
                        f"Optimization recommended: {analysis.arithmetic_ratio:.1%} arithmetic operations"
                    ],
                )
            else:
                return False, "", reasons

        except Exception as e:
            return False, "", [f"Error analyzing file: {str(e)}"]

    def generate_report(self, file_path: Path) -> str:
        """Generate a detailed analysis report for a file."""
        try:
            content = file_path.read_text()
            analysis = self.analyze_file(content)
            should_opt, reasons = self.should_optimize(analysis)

            report = f"""
Simpulse Optimization Analysis Report
=====================================
File: {file_path.name}
Safe Mode: {self.safe_mode}

File Metrics:
- Total lines: {analysis.total_lines}
- Arithmetic operations: {analysis.arithmetic_ops}
- List operations: {analysis.list_ops}
- Simp calls: {analysis.simp_calls}
- Custom simp lemmas: {analysis.custom_simp_lemmas}

Ratios:
- Arithmetic ratio: {analysis.arithmetic_ratio:.1%}
- List ratio: {analysis.list_ratio:.1%}

Safety Analysis:
- Has forbidden patterns: {analysis.has_forbidden_patterns}
- Recommendation: {'OPTIMIZE' if should_opt else 'SKIP'}

"""
            if reasons:
                report += "Reasons:\n"
                for reason in reasons:
                    report += f"  - {reason}\n"

            if should_opt:
                report += f"\nRecommended optimizations:\n{self.get_optimizations(analysis)}"

            return report

        except Exception as e:
            return f"Error generating report: {str(e)}"


def main():
    """Example usage of SafeOptimizer."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python safe_optimizer.py <lean_file> [--extended]")
        return

    file_path = Path(sys.argv[1])
    safe_mode = "--extended" not in sys.argv

    optimizer = SafeOptimizer(safe_mode=safe_mode)
    print(optimizer.generate_report(file_path))


if __name__ == "__main__":
    main()

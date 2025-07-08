"""
Compatibility checker for Simpulse optimization
Analyzes files BEFORE optimization to prevent failures
"""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class CompatibilityLevel(Enum):
    """Compatibility levels for optimization."""

    EXCELLENT = "excellent"  # Very likely to improve performance
    GOOD = "good"  # Likely to improve performance
    FAIR = "fair"  # May improve performance
    POOR = "poor"  # Unlikely to improve performance
    DANGEROUS = "dangerous"  # Likely to cause regressions or failures
    INCOMPATIBLE = "incompatible"  # Will cause failures, do not optimize


@dataclass
class CompatibilityIssue:
    """Represents a compatibility issue found in the file."""

    severity: str  # "critical", "warning", "info"
    category: str  # "size", "custom_simp", "domain_specific", etc.
    message: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class CompatibilityReport:
    """Complete compatibility analysis report."""

    file_path: Path
    compatibility_level: CompatibilityLevel
    score: int  # 0-100, higher is better
    issues: List[CompatibilityIssue]
    file_stats: Dict[str, any]
    recommendation: str
    estimated_speedup: Optional[str] = None
    risk_assessment: str = ""


class CompatibilityChecker:
    """Analyzes files for Simpulse optimization compatibility."""

    def __init__(self):
        self.critical_patterns = {
            "custom_simp_priorities": r"@\[simp\s+\d+\]",
            "custom_simp_tactics": r"declare_simp_like_tactic",
            "mutual_recursion": r"mutual\s+def",
            "recursive_simp_defs": r"@\[simp\]\s+def.*:.*→.*→",
            "dependent_types": r"Type\*|\{.*:.*Type.*\}",
            "universe_polymorphism": r"universe\s+\w+|\.{sort\s+\w+}",
            "float_operations": r"Float\.|Double\.",
            "sorry_axioms": r"\bsorry\b|\baxiom\b",
        }

        self.warning_patterns = {
            "list_heavy": r"List\.|\.reverse|\.append|\+\+\s*\[\]",
            "complex_tactics": r"induction\s+.*with|cases\s+.*with|match.*with",
            "typeclass_heavy": r"instance\s+.*:|class\s+.*:",
            "namespace_scoping": r"namespace\s+\w+|section\s+\w+",
            "meta_programming": r"#eval|#check|#print",
        }

        self.positive_patterns = {
            "arithmetic_ops": r"\+\s*0\b|\b0\s*\+|\*\s*1\b|\b1\s*\*|Nat\.add|Nat\.mul",
            "simple_simp": r"\bsimp\b(?!\s*\[)(?!\s*only)(?!\s*at)",
            "basic_theorems": r"theorem\s+\w+.*:.*:=\s*by\s+simp",
            "mathlib_imports": r"import\s+Mathlib\.",
        }

    def analyze_file(self, file_path: Path) -> CompatibilityReport:
        """Perform complete compatibility analysis on a file."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return CompatibilityReport(
                file_path=file_path,
                compatibility_level=CompatibilityLevel.INCOMPATIBLE,
                score=0,
                issues=[CompatibilityIssue("critical", "file_access", f"Cannot read file: {e}")],
                file_stats={},
                recommendation="Cannot analyze file - check file permissions and encoding",
            )

        # Basic file stats
        lines = content.split("\n")
        file_stats = {
            "total_lines": len(lines),
            "non_empty_lines": len([l for l in lines if l.strip()]),
            "comment_lines": len([l for l in lines if l.strip().startswith("--")]),
            "file_size_bytes": len(content.encode("utf-8")),
        }

        # Analyze patterns
        issues = []
        score = 100  # Start with perfect score, deduct for issues

        # Critical checks (immediate disqualification)
        for pattern_name, pattern in self.critical_patterns.items():
            matches = list(re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE))
            if matches:
                for match in matches[:3]:  # Report first 3 matches
                    line_num = content[: match.start()].count("\n") + 1
                    issue = self._create_critical_issue(pattern_name, line_num)
                    issues.append(issue)
                score = 0  # Critical issues = incompatible

        # Size checks (CRITICAL - can cause stack overflow)
        if file_stats["total_lines"] > 1000:
            issues.append(
                CompatibilityIssue(
                    "critical",
                    "size",
                    f"File has {file_stats['total_lines']} lines (>1000 limit) - will cause stack overflow",
                    suggestion="Split into smaller files or use different optimization approach",
                )
            )
            score = 0  # Critical size issue = incompatible
        elif file_stats["total_lines"] > 500:
            issues.append(
                CompatibilityIssue(
                    "warning",
                    "size",
                    f"Large file ({file_stats['total_lines']} lines) may cause performance issues",
                    suggestion="Consider splitting file or thorough testing",
                )
            )
            score -= 40  # Heavy penalty for large files
        elif file_stats["total_lines"] < 10:
            issues.append(
                CompatibilityIssue(
                    "warning",
                    "size",
                    f"Very small file ({file_stats['total_lines']} lines) - optimization overhead may dominate",
                    suggestion="Optimization likely not beneficial for tiny files",
                )
            )
            score -= 20

        # Warning pattern checks
        warning_count = 0
        for pattern_name, pattern in self.warning_patterns.items():
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if matches:
                warning_count += len(matches)
                if len(matches) > 5:  # Many matches = problematic
                    issues.append(
                        CompatibilityIssue(
                            "warning",
                            pattern_name,
                            f"Heavy use of {pattern_name.replace('_', ' ')} ({len(matches)} occurrences)",
                            suggestion=self._get_suggestion(pattern_name),
                        )
                    )
                    score -= 15

        # Positive pattern analysis
        positive_score = 0
        arithmetic_count = len(re.findall(self.positive_patterns["arithmetic_ops"], content))
        simp_count = len(re.findall(self.positive_patterns["simple_simp"], content))
        theorem_count = len(re.findall(self.positive_patterns["basic_theorems"], content))
        mathlib_imports = len(re.findall(self.positive_patterns["mathlib_imports"], content))

        file_stats.update(
            {
                "arithmetic_operations": arithmetic_count,
                "simple_simp_calls": simp_count,
                "basic_theorems": theorem_count,
                "mathlib_imports": mathlib_imports,
                "warning_patterns": warning_count,
            }
        )

        # Calculate positive contributions
        if arithmetic_count > 10:
            positive_score += 20
            issues.append(
                CompatibilityIssue(
                    "info",
                    "arithmetic",
                    f"Good arithmetic density ({arithmetic_count} operations)",
                    suggestion="This file should benefit from Simpulse optimization",
                )
            )

        if simp_count > 5:
            positive_score += 15

        if mathlib_imports > 0:
            positive_score += 10

        # Final score calculation
        score = max(0, min(100, score + positive_score))

        # Determine compatibility level
        compatibility_level = self._determine_compatibility_level(score, issues)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            compatibility_level, score, file_stats, issues
        )

        # Estimate speedup
        estimated_speedup = self._estimate_speedup(file_stats, compatibility_level)

        # Risk assessment
        risk_assessment = self._assess_risk(issues, file_stats)

        return CompatibilityReport(
            file_path=file_path,
            compatibility_level=compatibility_level,
            score=score,
            issues=issues,
            file_stats=file_stats,
            recommendation=recommendation,
            estimated_speedup=estimated_speedup,
            risk_assessment=risk_assessment,
        )

    def _create_critical_issue(self, pattern_name: str, line_num: int) -> CompatibilityIssue:
        """Create a critical compatibility issue."""
        messages = {
            "custom_simp_priorities": "Custom simp priorities detected - will conflict with optimization",
            "custom_simp_tactics": "Custom simp tactics detected - incompatible with Simpulse",
            "mutual_recursion": "Mutual recursion detected - causes performance regression",
            "recursive_simp_defs": "Recursive simp definitions detected - different optimization needed",
            "dependent_types": "Dependent types detected - may cause elaboration issues",
            "universe_polymorphism": "Universe polymorphism detected - may cause compilation issues",
            "float_operations": "Float operations detected - unsupported by current optimization",
            "sorry_axioms": "Sorry/axiom detected - file may be incomplete",
        }

        suggestions = {
            "custom_simp_priorities": "Remove custom priorities or use manual optimization",
            "custom_simp_tactics": "Use standard simp tactics or different optimization approach",
            "mutual_recursion": "Consider non-mutual definitions or manual optimization",
            "recursive_simp_defs": "Use different simp strategy for recursive functions",
            "dependent_types": "Test thoroughly or avoid optimization",
            "universe_polymorphism": "Ensure compatibility with Lean version",
            "float_operations": "Use integer arithmetic or different optimization",
            "sorry_axioms": "Complete proofs before optimizing",
        }

        return CompatibilityIssue(
            "critical",
            pattern_name,
            messages.get(pattern_name, f"Critical pattern {pattern_name} detected"),
            line_num,
            suggestions.get(pattern_name),
        )

    def _get_suggestion(self, pattern_name: str) -> str:
        """Get suggestion for warning patterns."""
        suggestions = {
            "list_heavy": "Consider different optimization strategy for list operations",
            "complex_tactics": "Complex tactics may not benefit from simp optimization",
            "typeclass_heavy": "Typeclass resolution may interfere with optimization",
            "namespace_scoping": "Ensure optimization is applied in correct scope",
            "meta_programming": "Meta-programming may cause elaboration issues",
        }
        return suggestions.get(pattern_name, "Consider manual review of this pattern")

    def _determine_compatibility_level(
        self, score: int, issues: List[CompatibilityIssue]
    ) -> CompatibilityLevel:
        """Determine overall compatibility level."""
        # Check for critical issues first
        if any(issue.severity == "critical" for issue in issues):
            return CompatibilityLevel.INCOMPATIBLE

        # Score-based determination
        if score >= 80:
            return CompatibilityLevel.EXCELLENT
        elif score >= 60:
            return CompatibilityLevel.GOOD
        elif score >= 40:
            return CompatibilityLevel.FAIR
        elif score >= 20:
            return CompatibilityLevel.POOR
        else:
            return CompatibilityLevel.DANGEROUS

    def _generate_recommendation(
        self, level: CompatibilityLevel, score: int, stats: Dict, issues: List[CompatibilityIssue]
    ) -> str:
        """Generate a recommendation based on analysis."""
        recommendations = {
            CompatibilityLevel.EXCELLENT: "✅ RECOMMENDED: This file is ideal for Simpulse optimization. "
            "High arithmetic density and mathlib4 patterns detected.",
            CompatibilityLevel.GOOD: "✅ SAFE: This file should benefit from optimization. "
            "Monitor performance and test thoroughly.",
            CompatibilityLevel.FAIR: "⚠️ CAUTION: Optimization may help but benefits are uncertain. "
            "Test carefully and measure performance impact.",
            CompatibilityLevel.POOR: "❌ NOT RECOMMENDED: Low probability of improvement. "
            "Consider manual optimization or different approach.",
            CompatibilityLevel.DANGEROUS: "🚨 AVOID: High risk of performance regression. "
            "Do not optimize without extensive testing.",
            CompatibilityLevel.INCOMPATIBLE: "🛑 INCOMPATIBLE: Will cause compilation errors or severe regressions. "
            "Do not optimize this file.",
        }

        base_rec = recommendations[level]

        # Add specific guidance
        critical_issues = [i for i in issues if i.severity == "critical"]
        if critical_issues:
            base_rec += f" Critical issues: {len(critical_issues)} found."

        if stats.get("arithmetic_operations", 0) > 15:
            base_rec += " High arithmetic density detected - excellent candidate."

        return base_rec

    def _estimate_speedup(self, stats: Dict, level: CompatibilityLevel) -> Optional[str]:
        """Estimate potential speedup based on file characteristics."""
        if level == CompatibilityLevel.INCOMPATIBLE:
            return "N/A (incompatible)"

        arithmetic_ops = stats.get("arithmetic_operations", 0)
        stats.get("simple_simp_calls", 0)

        if level == CompatibilityLevel.EXCELLENT and arithmetic_ops > 20:
            return "1.5x - 2.5x speedup expected"
        elif level == CompatibilityLevel.GOOD and arithmetic_ops > 10:
            return "1.2x - 1.8x speedup likely"
        elif level == CompatibilityLevel.FAIR:
            return "1.0x - 1.3x speedup possible"
        elif level == CompatibilityLevel.POOR:
            return "0.9x - 1.1x (may be slower)"
        else:
            return "0.5x - 0.9x (likely slower)"

    def _assess_risk(self, issues: List[CompatibilityIssue], stats: Dict) -> str:
        """Assess risk level of optimization."""
        critical_count = len([i for i in issues if i.severity == "critical"])
        warning_count = len([i for i in issues if i.severity == "warning"])

        if critical_count > 0:
            return "🔴 HIGH RISK: Critical compatibility issues detected"
        elif warning_count > 3:
            return "🟡 MEDIUM RISK: Multiple warning patterns detected"
        elif stats.get("total_lines", 0) > 500:
            return "🟡 MEDIUM RISK: Large file size"
        else:
            return "🟢 LOW RISK: No major compatibility issues"


def generate_compatibility_report_md(report: CompatibilityReport, output_path: Path) -> None:
    """Generate a markdown compatibility report."""
    content = f"""# Simpulse Compatibility Report

**File:** `{report.file_path.name}`  
**Analysis Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Compatibility Level:** {report.compatibility_level.value.upper()}  
**Score:** {report.score}/100  

## 🎯 Recommendation

{report.recommendation}

## 📊 Analysis Summary

| Metric | Value |
|--------|-------|
| **Compatibility Level** | {report.compatibility_level.value.upper()} |
| **Safety Score** | {report.score}/100 |
| **Risk Assessment** | {report.risk_assessment} |
| **Estimated Speedup** | {report.estimated_speedup or 'Unknown'} |

## 📈 File Statistics

| Attribute | Count |
|-----------|-------|
| Total Lines | {report.file_stats.get('total_lines', 0)} |
| Arithmetic Operations | {report.file_stats.get('arithmetic_operations', 0)} |
| Simple Simp Calls | {report.file_stats.get('simple_simp_calls', 0)} |
| Basic Theorems | {report.file_stats.get('basic_theorems', 0)} |
| Mathlib Imports | {report.file_stats.get('mathlib_imports', 0)} |
| Warning Patterns | {report.file_stats.get('warning_patterns', 0)} |

## 🔍 Issues Found

"""

    if not report.issues:
        content += "✅ No compatibility issues detected.\n\n"
    else:
        # Group issues by severity
        critical_issues = [i for i in report.issues if i.severity == "critical"]
        warning_issues = [i for i in report.issues if i.severity == "warning"]
        info_issues = [i for i in report.issues if i.severity == "info"]

        if critical_issues:
            content += "### 🚨 Critical Issues\n\n"
            for issue in critical_issues:
                content += f"- **{issue.category.replace('_', ' ').title()}**: {issue.message}\n"
                if issue.suggestion:
                    content += f"  - *Suggestion: {issue.suggestion}*\n"
                if issue.line_number:
                    content += f"  - *Line: {issue.line_number}*\n"
                content += "\n"

        if warning_issues:
            content += "### ⚠️ Warnings\n\n"
            for issue in warning_issues:
                content += f"- **{issue.category.replace('_', ' ').title()}**: {issue.message}\n"
                if issue.suggestion:
                    content += f"  - *Suggestion: {issue.suggestion}*\n"
                content += "\n"

        if info_issues:
            content += "### ℹ️ Information\n\n"
            for issue in info_issues:
                content += f"- **{issue.category.replace('_', ' ').title()}**: {issue.message}\n"
                content += "\n"

    content += f"""## 🎯 Next Steps

Based on this analysis:

"""

    if report.compatibility_level in [CompatibilityLevel.EXCELLENT, CompatibilityLevel.GOOD]:
        content += """1. ✅ **Proceed with optimization** - this file is a good candidate
2. 📊 **Monitor performance** before and after optimization  
3. 🧪 **Test thoroughly** in your specific environment
4. 📈 **Measure actual speedup** and compare with estimates
"""
    elif report.compatibility_level == CompatibilityLevel.FAIR:
        content += """1. ⚠️ **Consider optimization** but test extensively
2. 📊 **Benchmark carefully** to ensure no regressions
3. 🔄 **Have rollback plan** ready
4. 📋 **Address warnings** if possible before optimizing
"""
    else:
        content += """1. ❌ **Do not optimize** this file
2. 🔧 **Address critical issues** first
3. 🎯 **Consider alternative optimization strategies**
4. 📚 **Consult documentation** for specific patterns
"""

    content += f"""
## 🛠️ Technical Details

**Analysis Engine:** Simpulse Compatibility Checker v1.0  
**Pattern Database:** {len(report.issues)} patterns checked  
**Confidence Level:** {'High' if report.score > 70 or report.score < 30 else 'Medium'}

---

*This report was generated automatically. For questions or issues, consult the Simpulse documentation.*
"""

    output_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python compatibility_checker.py <lean_file>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    checker = CompatibilityChecker()
    report = checker.analyze_file(file_path)

    print(f"Compatibility: {report.compatibility_level.value}")
    print(f"Score: {report.score}/100")
    print(f"Recommendation: {report.recommendation}")

    # Generate markdown report
    report_path = file_path.parent / f"{file_path.stem}_compatibility_report.md"
    generate_compatibility_report_md(report, report_path)
    print(f"Report saved: {report_path}")

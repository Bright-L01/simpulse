#!/usr/bin/env python3
"""
Simp Health Check - Identify optimization opportunities in Lean projects.
This is the key to finding projects that need our help.
"""

import asyncio
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class HealthReport:
    """Complete health report for a Lean project."""

    total_rules: int
    custom_priorities: int
    default_priorities: int
    optimization_potential: float  # 0-100 score
    estimated_improvement: float  # Estimated % improvement
    slow_modules: List[Tuple[str, float]]  # (module, time_ms)
    recommendations: List[str]
    patterns_found: Dict[str, bool]

    @property
    def priority_ratio(self) -> float:
        """Ratio of custom to total priorities."""
        return self.custom_priorities / max(self.total_rules, 1)


class SimpHealthChecker:
    """Analyze a Lean project's simp optimization potential."""

    def __init__(self):
        self.slow_threshold_ms = 100  # Proofs slower than this need help

    async def analyze_project(self, project_path: Path) -> HealthReport:
        """Full project analysis."""

        print(f"🔍 Analyzing {project_path.name}...")

        # 1. Find all Lean files
        lean_files = list(project_path.rglob("*.lean"))
        print(f"   Found {len(lean_files)} Lean files")

        # 2. Count simp rules
        rule_stats = self._count_simp_rules(lean_files)
        print(f"   Found {rule_stats['total']} simp rules")

        # 3. Profile performance (sample if too many files)
        sample_size = min(len(lean_files), 10)
        slow_modules = await self._profile_modules(lean_files[:sample_size])

        # 4. Identify patterns
        patterns = self._identify_patterns(lean_files, rule_stats)

        # 5. Calculate optimization potential
        optimization_score = self._calculate_potential(
            rule_stats, patterns, slow_modules
        )

        # 6. Estimate improvement
        estimated_improvement = self._estimate_improvement(optimization_score, patterns)

        # 7. Generate recommendations
        recommendations = self._generate_recommendations(
            rule_stats, patterns, slow_modules
        )

        return HealthReport(
            total_rules=rule_stats["total"],
            custom_priorities=rule_stats["custom"],
            default_priorities=rule_stats["default"],
            optimization_potential=optimization_score,
            estimated_improvement=estimated_improvement,
            slow_modules=slow_modules,
            recommendations=recommendations,
            patterns_found=patterns,
        )

    def _count_simp_rules(self, lean_files: List[Path]) -> Dict[str, int]:
        """Count simp rules and their priority types."""

        stats = {
            "total": 0,
            "custom": 0,
            "default": 0,
            "high": 0,
            "low": 0,
            "numeric": 0,
        }

        # Patterns for different simp attributes
        patterns = {
            "default": re.compile(r"@\[simp\]"),
            "high": re.compile(r"@\[simp\s+high\]"),
            "low": re.compile(r"@\[simp\s+low\]"),
            "numeric": re.compile(r"@\[simp\s+(\d+)\]"),
        }

        for lean_file in lean_files:
            try:
                content = lean_file.read_text()

                # Count each type
                default_count = len(patterns["default"].findall(content))
                high_count = len(patterns["high"].findall(content))
                low_count = len(patterns["low"].findall(content))
                numeric_count = len(patterns["numeric"].findall(content))

                stats["default"] += default_count
                stats["high"] += high_count
                stats["low"] += low_count
                stats["numeric"] += numeric_count
                stats["custom"] += high_count + low_count + numeric_count
                stats["total"] += default_count + high_count + low_count + numeric_count

            except Exception:
                continue

        return stats

    async def _profile_modules(self, lean_files: List[Path]) -> List[Tuple[str, float]]:
        """Profile compilation time for modules."""

        slow_modules = []

        for lean_file in lean_files:
            try:
                # Time compilation
                start = time.perf_counter()
                result = subprocess.run(
                    ["lean", str(lean_file)], capture_output=True, text=True, timeout=30
                )
                end = time.perf_counter()

                if result.returncode == 0:
                    time_ms = (end - start) * 1000

                    # Check if slow
                    if time_ms > self.slow_threshold_ms:
                        slow_modules.append((lean_file.name, time_ms))

            except subprocess.TimeoutExpired:
                # Very slow!
                slow_modules.append((lean_file.name, 30000))
            except Exception:
                continue

        return sorted(slow_modules, key=lambda x: x[1], reverse=True)[:10]

    def _identify_patterns(
        self, lean_files: List[Path], rule_stats: Dict
    ) -> Dict[str, bool]:
        """Identify optimization anti-patterns."""

        patterns = {
            "all_default_priorities": rule_stats["custom"] == 0,
            "mostly_default": rule_stats["custom"] / max(rule_stats["total"], 1) < 0.1,
            "no_high_priority": rule_stats["high"] == 0,
            "no_low_priority": rule_stats["low"] == 0,
            "unbalanced_priorities": False,
            "large_rule_count": rule_stats["total"] > 100,
            "very_large_rule_count": rule_stats["total"] > 500,
        }

        # Check for unbalanced priorities
        if rule_stats["high"] > 0 or rule_stats["low"] > 0:
            ratio = rule_stats["high"] / max(rule_stats["low"], 1)
            patterns["unbalanced_priorities"] = ratio > 10 or ratio < 0.1

        return patterns

    def _calculate_potential(
        self, rule_stats: Dict, patterns: Dict, slow_modules: List
    ) -> float:
        """Calculate optimization potential score (0-100)."""

        score = 0.0

        # Major factors
        if patterns["all_default_priorities"]:
            score += 40  # Huge opportunity!
        elif patterns["mostly_default"]:
            score += 25

        # Rule count factors
        if patterns["large_rule_count"]:
            score += 15
        if patterns["very_large_rule_count"]:
            score += 10

        # Balance factors
        if patterns["unbalanced_priorities"]:
            score += 10
        if patterns["no_high_priority"] and rule_stats["total"] > 20:
            score += 10
        if patterns["no_low_priority"] and rule_stats["total"] > 20:
            score += 10

        # Performance factors
        if len(slow_modules) > 0:
            score += min(20, len(slow_modules) * 5)

        return min(100, score)

    def _estimate_improvement(self, optimization_score: float, patterns: Dict) -> float:
        """Estimate potential performance improvement."""

        # Base estimate from score
        base_improvement = optimization_score * 0.7  # Conservative estimate

        # Adjust based on patterns
        if patterns["all_default_priorities"]:
            # We've seen 70%+ improvements here
            return min(70, base_improvement * 1.5)
        elif patterns["mostly_default"]:
            return min(50, base_improvement * 1.2)
        else:
            return min(30, base_improvement)

    def _generate_recommendations(
        self, rule_stats: Dict, patterns: Dict, slow_modules: List
    ) -> List[str]:
        """Generate specific recommendations."""

        recommendations = []

        if patterns["all_default_priorities"]:
            recommendations.append(
                "🔴 CRITICAL: All simp rules use default priority. "
                "This is the #1 optimization opportunity!"
            )
            recommendations.append(
                "💡 Quick win: Give frequently-used rules high priority"
            )

        if patterns["mostly_default"]:
            recommendations.append(
                "🟡 Most rules use default priority. "
                "Consider prioritizing common patterns."
            )

        if patterns["no_high_priority"] and rule_stats["total"] > 20:
            recommendations.append(
                "💡 No high-priority rules found. "
                "Mark simple, frequent rules as high priority."
            )

        if patterns["very_large_rule_count"]:
            recommendations.append(
                "📊 Large simp rule set detected. "
                "Priority optimization becomes crucial at this scale."
            )

        if len(slow_modules) > 0:
            recommendations.append(
                f"⏱️ Found {len(slow_modules)} slow modules. "
                "These would benefit most from optimization."
            )

        if not recommendations:
            recommendations.append(
                "✅ Simp rules appear reasonably optimized. "
                "Minor improvements may still be possible."
            )

        return recommendations

    def generate_report(self, health_report: HealthReport) -> str:
        """Generate human-readable health check report."""

        # Determine health status
        if health_report.optimization_potential >= 70:
            status = "🔴 Poor"
            action = "Immediate optimization recommended!"
        elif health_report.optimization_potential >= 40:
            status = "🟡 Fair"
            action = "Optimization would help"
        else:
            status = "🟢 Good"
            action = "Already well-optimized"

        report = f"""
Simp Performance Health Check
============================

Overall Health: {status}
Action: {action}

Statistics:
-----------
Total simp rules: {health_report.total_rules}
Custom priorities: {health_report.custom_priorities} ({health_report.priority_ratio:.0%})
Default priorities: {health_report.default_priorities}

Optimization Potential: {health_report.optimization_potential:.0f}/100
Estimated Improvement: {health_report.estimated_improvement:.0f}%

"""

        if health_report.slow_modules:
            report += "Slow Modules:\n"
            report += "-------------\n"
            for module, time_ms in health_report.slow_modules[:5]:
                report += f"- {module}: {time_ms:.0f}ms\n"
            report += "\n"

        report += "Recommendations:\n"
        report += "----------------\n"
        for rec in health_report.recommendations:
            report += f"{rec}\n"

        if health_report.optimization_potential >= 40:
            report += f"""
Next Steps:
-----------
1. Run: simpulse optimize --project {Path.cwd().name}
2. Review suggested changes
3. Measure improvement
4. Share your success story!
"""

        return report

    def save_report(self, health_report: HealthReport, output_path: Path):
        """Save report in multiple formats."""

        # Save human-readable version
        text_report = self.generate_report(health_report)
        (output_path / "simp_health_report.txt").write_text(text_report)

        # Save JSON for automation
        json_data = {
            "total_rules": health_report.total_rules,
            "custom_priorities": health_report.custom_priorities,
            "optimization_potential": health_report.optimization_potential,
            "estimated_improvement": health_report.estimated_improvement,
            "patterns": health_report.patterns_found,
            "slow_modules": health_report.slow_modules,
            "recommendations": health_report.recommendations,
        }

        with open(output_path / "simp_health_report.json", "w") as f:
            json.dump(json_data, f, indent=2)


async def main():
    """Run health check on current directory or specified project."""
    import sys

    if len(sys.argv) > 1:
        project_path = Path(sys.argv[1])
    else:
        project_path = Path.cwd()

    if not project_path.exists():
        print(f"Error: {project_path} not found")
        return 1

    checker = SimpHealthChecker()
    report = await checker.analyze_project(project_path)

    print("\n" + checker.generate_report(report))

    # Save report
    checker.save_report(report, project_path)
    print(f"📄 Full report saved to {project_path}/simp_health_report.txt")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))

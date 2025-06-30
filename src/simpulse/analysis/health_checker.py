"""Health checker for Lean 4 projects."""

import re
from pathlib import Path
from typing import List

from pydantic import BaseModel


class HealthCheckResult(BaseModel):
    """Result of a simp rule health check analysis.

    Attributes:
        project_path: Path to the analyzed Lean project
        total_rules: Total number of simp rules found
        default_priority_rules: Number of rules using default priority (1000)
        default_priority_percentage: Percentage of rules with default priority
        score: Overall optimization potential score (0-100)
        estimated_improvement: Estimated performance improvement percentage
        recommendations: List of specific recommendations
    """

    project_path: Path
    total_rules: int
    default_priority_rules: int
    default_priority_percentage: float
    score: int
    estimated_improvement: int
    recommendations: List[str]


class HealthChecker:
    """Analyzes Lean 4 projects to identify simp rule optimization opportunities.

    The health checker examines all simp rules in a project and calculates
    metrics that indicate potential performance improvements through
    priority optimization.
    """

    def __init__(self):
        self.simp_pattern = re.compile(r"@\[simp(?:\s+(\d+))?\]")

    def check_project(self, project_path: Path) -> HealthCheckResult:
        """Analyze a Lean 4 project's simp rule health.

        Args:
            project_path: Path to the Lean 4 project directory

        Returns:
            HealthCheckResult with analysis metrics and recommendations
        """
        lean_files = list(project_path.glob("**/*.lean"))

        total_rules = 0
        default_priority = 0

        for lean_file in lean_files:
            if "lake-packages" in str(lean_file):
                continue

            try:
                content = lean_file.read_text()
                matches = self.simp_pattern.findall(content)

                for match in matches:
                    total_rules += 1
                    if not match or match == "1000":
                        default_priority += 1
            except Exception:
                pass

        if total_rules == 0:
            return HealthCheckResult(
                project_path=project_path,
                total_rules=0,
                default_priority_rules=0,
                default_priority_percentage=0,
                score=0,
                estimated_improvement=0,
                recommendations=["No simp rules found"],
            )

        default_percentage = (default_priority / total_rules) * 100

        # Calculate score
        score = 0
        if default_percentage > 90:
            score = 85
        elif default_percentage > 70:
            score = 70
        elif default_percentage > 50:
            score = 50
        else:
            score = 30

        # Estimate improvement
        estimated_improvement = int(default_percentage * 0.6)

        # Generate recommendations
        recommendations = []
        if score > 60:
            recommendations.append("High optimization potential detected")
            recommendations.append("Consider running simpulse optimize")

        return HealthCheckResult(
            project_path=project_path,
            total_rules=total_rules,
            default_priority_rules=default_priority,
            default_priority_percentage=default_percentage,
            score=score,
            estimated_improvement=estimated_improvement,
            recommendations=recommendations,
        )

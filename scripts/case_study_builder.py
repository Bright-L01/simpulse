#!/usr/bin/env python3
"""
Case Study Builder - Create compelling documentation of successful optimizations.
Success stories drive adoption!
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


@dataclass
class Metrics:
    """Performance metrics before/after optimization."""

    total_time: float  # seconds
    simp_time: float  # seconds
    total_rules: int
    custom_priorities: int
    slow_proof_count: int
    slowest_proof_time: float  # ms
    rules_modified: int = 0


@dataclass
class OptimizationChange:
    """A single rule priority change."""

    rule_name: str
    old_priority: str
    new_priority: str
    reason: str


class CaseStudyBuilder:
    """Create compelling case studies from successful optimizations."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("case_studies")
        self.output_dir.mkdir(exist_ok=True)

    async def build_case_study(
        self,
        project: str,
        before: Metrics,
        after: Metrics,
        changes: List[OptimizationChange],
        project_url: str = None,
    ) -> Path:
        """Generate a complete case study."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_dir = self.output_dir / f"{project}_{timestamp}"
        study_dir.mkdir(exist_ok=True)

        # Create visuals
        chart_path = self._create_performance_chart(before, after, study_dir)

        # Generate markdown report
        report = self._generate_markdown(
            project, before, after, changes, project_url, chart_path
        )

        # Save report
        report_path = study_dir / "README.md"
        report_path.write_text(report)

        # Save data for future analysis
        self._save_data(study_dir, project, before, after, changes)

        print(f"✅ Case study created: {report_path}")
        return report_path

    def _create_performance_chart(
        self, before: Metrics, after: Metrics, output_dir: Path
    ) -> Path:
        """Create visual performance comparison chart."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Time comparison
        categories = ["Total Build", "Simp Time", "Slowest Proof"]
        before_times = [
            before.total_time,
            before.simp_time,
            before.slowest_proof_time / 1000,
        ]
        after_times = [
            after.total_time,
            after.simp_time,
            after.slowest_proof_time / 1000,
        ]

        x = range(len(categories))
        width = 0.35

        ax1.bar(
            [i - width / 2 for i in x],
            before_times,
            width,
            label="Before",
            color="#ff6b6b",
        )
        ax1.bar(
            [i + width / 2 for i in x],
            after_times,
            width,
            label="After",
            color="#51cf66",
        )

        ax1.set_ylabel("Time (seconds)")
        ax1.set_title("Performance Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()

        # Improvement percentages
        improvements = [
            self._calculate_improvement(before.total_time, after.total_time),
            self._calculate_improvement(before.simp_time, after.simp_time),
            self._calculate_improvement(
                before.slowest_proof_time, after.slowest_proof_time
            ),
        ]

        colors = ["#51cf66" if imp > 0 else "#ff6b6b" for imp in improvements]
        ax2.bar(categories, improvements, color=colors)
        ax2.set_ylabel("Improvement (%)")
        ax2.set_title("Performance Gains")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        # Add value labels
        for i, v in enumerate(improvements):
            ax2.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom")

        plt.tight_layout()

        chart_path = output_dir / "performance_comparison.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()

        return chart_path

    def _calculate_improvement(self, before: float, after: float) -> float:
        """Calculate percentage improvement."""
        if before == 0:
            return 0
        return ((before - after) / before) * 100

    def _generate_markdown(
        self,
        project: str,
        before: Metrics,
        after: Metrics,
        changes: List[OptimizationChange],
        project_url: str,
        chart_path: Path,
    ) -> str:
        """Generate markdown case study report."""

        total_improvement = self._calculate_improvement(
            before.total_time, after.total_time
        )
        simp_improvement = self._calculate_improvement(
            before.simp_time, after.simp_time
        )

        report = f"""# Case Study: {project}

## Overview

**Project**: [{project}]({project_url or '#'})  
**Date**: {datetime.now().strftime('%B %d, %Y')}  
**Tool**: Simpulse v0.1.0

### Problem
- {before.slow_proof_count} slow proofs (>100ms)
- {before.total_rules} simp rules with only {before.custom_priorities} custom priorities
- Build time: {before.total_time:.1f}s

### Solution
- Optimized {after.rules_modified} rule priorities
- Applied data-driven priority assignments
- Focused on frequently-used rules

## Results

![Performance Comparison]({chart_path.name})

### Key Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total Build Time | {before.total_time:.1f}s | {after.total_time:.1f}s | **{total_improvement:.1f}%** |
| Simp Time | {before.simp_time:.1f}s | {after.simp_time:.1f}s | **{simp_improvement:.1f}%** |
| Slowest Proof | {before.slowest_proof_time:.0f}ms | {after.slowest_proof_time:.0f}ms | {self._calculate_improvement(before.slowest_proof_time, after.slowest_proof_time):.1f}% |
| Slow Proofs | {before.slow_proof_count} | {after.slow_proof_count} | -{before.slow_proof_count - after.slow_proof_count} |

## Technical Details

### Priority Changes Made

The following rule priority optimizations had the biggest impact:

"""

        # Add top 5 most impactful changes
        for i, change in enumerate(changes[:5], 1):
            report += f"""
{i}. **{change.rule_name}**
   - Changed from: `@[simp {change.old_priority}]` → `@[simp {change.new_priority}]`
   - Reason: {change.reason}
"""

        if len(changes) > 5:
            report += f"\n...and {len(changes) - 5} more optimizations\n"

        report += """
### Analysis

"""

        # Add analysis based on the changes
        if before.custom_priorities == 0:
            report += """This project had **zero custom priorities** before optimization, which is a common anti-pattern. 
By analyzing proof traces and rule usage patterns, Simpulse identified which rules were:
- Used most frequently (given high priority)
- Rarely used but expensive (given low priority)  
- Simple and fast (given high priority)

"""

        report += f"""The {simp_improvement:.1f}% improvement in simp performance translated directly to a {total_improvement:.1f}% 
improvement in overall build time, demonstrating that simp optimization can have significant real-world impact.

## How to Reproduce

1. Clone the project:
   ```bash
   git clone {project_url or 'PROJECT_URL'}
   cd {project}
   ```

2. Run Simpulse optimization:
   ```bash
   simpulse optimize .
   ```

3. Review and apply suggested changes

4. Measure the improvement:
   ```bash
   time lake build
   ```

## Lessons Learned

"""

        # Add specific lessons based on patterns
        if before.custom_priorities == 0:
            report += "- Projects with all default priorities have the highest optimization potential\n"
        if before.total_rules > 100:
            report += "- Large rule sets (>100 rules) benefit significantly from priority optimization\n"
        if simp_improvement > 50:
            report += "- Simple priority changes can yield dramatic performance improvements\n"

        report += """
## Conclusion

This case study demonstrates that Simpulse can deliver significant performance improvements with minimal effort. 
The {total_improvement:.1f}% build time reduction was achieved through automated analysis and simple priority adjustments.

### Impact

- **Developer Time Saved**: ~{self._calculate_time_saved(before.total_time, after.total_time)} per build
- **CI/CD Cost Reduction**: {total_improvement:.0f}% lower compute costs
- **Developer Experience**: Faster feedback loops and improved productivity

---

*Generated by Simpulse - [Learn more](https://github.com/Bright-L01/simpulse)*
"""

        return report

    def _calculate_time_saved(self, before_time: float, after_time: float) -> str:
        """Calculate human-readable time saved."""
        saved = before_time - after_time
        if saved < 1:
            return f"{saved*1000:.0f}ms"
        elif saved < 60:
            return f"{saved:.1f}s"
        else:
            return f"{saved/60:.1f}min"

    def _save_data(
        self,
        output_dir: Path,
        project: str,
        before: Metrics,
        after: Metrics,
        changes: List[OptimizationChange],
    ):
        """Save raw data for future analysis."""

        data = {
            "project": project,
            "timestamp": datetime.now().isoformat(),
            "before": {
                "total_time": before.total_time,
                "simp_time": before.simp_time,
                "total_rules": before.total_rules,
                "custom_priorities": before.custom_priorities,
                "slow_proof_count": before.slow_proof_count,
                "slowest_proof_time": before.slowest_proof_time,
            },
            "after": {
                "total_time": after.total_time,
                "simp_time": after.simp_time,
                "total_rules": after.total_rules,
                "custom_priorities": after.custom_priorities,
                "slow_proof_count": after.slow_proof_count,
                "slowest_proof_time": after.slowest_proof_time,
                "rules_modified": after.rules_modified,
            },
            "changes": [
                {
                    "rule": change.rule_name,
                    "old_priority": change.old_priority,
                    "new_priority": change.new_priority,
                    "reason": change.reason,
                }
                for change in changes
            ],
            "improvements": {
                "total_time": self._calculate_improvement(
                    before.total_time, after.total_time
                ),
                "simp_time": self._calculate_improvement(
                    before.simp_time, after.simp_time
                ),
                "slowest_proof": self._calculate_improvement(
                    before.slowest_proof_time, after.slowest_proof_time
                ),
            },
        }

        with open(output_dir / "data.json", "w") as f:
            json.dump(data, f, indent=2)

    def create_summary_report(self) -> str:
        """Create a summary of all case studies."""

        all_studies = []

        # Load all case study data
        for study_dir in self.output_dir.iterdir():
            if study_dir.is_dir():
                data_file = study_dir / "data.json"
                if data_file.exists():
                    with open(data_file) as f:
                        all_studies.append(json.load(f))

        if not all_studies:
            return "No case studies found."

        # Calculate aggregate statistics
        total_projects = len(all_studies)
        avg_improvement = (
            sum(s["improvements"]["total_time"] for s in all_studies) / total_projects
        )
        total_time_saved = sum(
            s["before"]["total_time"] - s["after"]["total_time"] for s in all_studies
        )

        report = f"""# Simpulse Case Studies Summary

## Impact Statistics

- **Projects Optimized**: {total_projects}
- **Average Improvement**: {avg_improvement:.1f}%
- **Total Time Saved**: {total_time_saved:.1f}s per build cycle
- **Best Result**: {max(s['improvements']['total_time'] for s in all_studies):.1f}% improvement

## Project Breakdown

| Project | Total Improvement | Simp Improvement | Rules Modified |
|---------|-------------------|------------------|----------------|
"""

        for study in sorted(
            all_studies, key=lambda x: x["improvements"]["total_time"], reverse=True
        ):
            report += f"| {study['project']} | {study['improvements']['total_time']:.1f}% | {study['improvements']['simp_time']:.1f}% | {study['after']['rules_modified']} |\n"

        return report


# Example usage
async def demo():
    """Demonstrate case study creation."""

    # Example metrics
    before = Metrics(
        total_time=45.2,
        simp_time=12.3,
        total_rules=156,
        custom_priorities=0,
        slow_proof_count=23,
        slowest_proof_time=850,
    )

    after = Metrics(
        total_time=28.7,
        simp_time=4.1,
        total_rules=156,
        custom_priorities=45,
        slow_proof_count=3,
        slowest_proof_time=120,
        rules_modified=45,
    )

    changes = [
        OptimizationChange(
            "list_append_nil", "default", "high", "Most frequently used rule"
        ),
        OptimizationChange("nat_add_zero", "default", "high", "Simple and common"),
        OptimizationChange(
            "complex_theorem_42", "default", "low", "Expensive and rarely used"
        ),
        OptimizationChange("mul_one", "default", "high", "Trivial and frequent"),
        OptimizationChange("div_self", "default", "medium", "Moderate usage"),
    ]

    builder = CaseStudyBuilder()
    await builder.build_case_study(
        "example-project", before, after, changes, "https://github.com/example/project"
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(demo())

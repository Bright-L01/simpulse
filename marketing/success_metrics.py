#!/usr/bin/env python3
"""
Success Metrics - Track Simpulse's real-world impact.
Success breeds success!
"""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ProjectSuccess:
    """Record of a successful optimization."""

    project_name: str
    github_url: str
    optimization_date: str
    time_saved_per_build: float  # seconds
    improvement_percent: float
    rules_optimized: int
    testimonial: Optional[str] = None
    case_study_url: Optional[str] = None


@dataclass
class ImpactMetrics:
    """Overall impact metrics."""

    projects_helped: int
    total_time_saved: float  # seconds per build cycle
    rules_optimized: int
    average_improvement: float
    best_improvement: float
    total_co2_saved: float  # grams
    user_testimonials: List[str]
    case_studies: List[str]
    github_stars: int
    community_contributors: int


class ImpactTracker:
    """Track Simpulse's real-world impact."""

    def __init__(self, db_path: str = "simpulse_impact.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for tracking."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS project_successes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL,
            github_url TEXT,
            optimization_date TEXT NOT NULL,
            time_saved_per_build REAL NOT NULL,
            improvement_percent REAL NOT NULL,
            rules_optimized INTEGER NOT NULL,
            testimonial TEXT,
            case_study_url TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS daily_metrics (
            date TEXT PRIMARY KEY,
            health_checks_run INTEGER DEFAULT 0,
            optimizations_performed INTEGER DEFAULT 0,
            github_stars INTEGER,
            contributors INTEGER
        )
        """
        )

        conn.commit()
        conn.close()

    def add_success(
        self,
        project: str,
        time_saved: float,
        improvement: float,
        rules_optimized: int,
        github_url: str = None,
        testimonial: str = None,
    ) -> int:
        """Record a successful optimization."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT INTO project_successes 
        (project_name, github_url, optimization_date, time_saved_per_build, 
         improvement_percent, rules_optimized, testimonial)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                project,
                github_url,
                datetime.now().isoformat(),
                time_saved,
                improvement,
                rules_optimized,
                testimonial,
            ),
        )

        success_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Calculate environmental impact
        co2_saved = self.calculate_co2_saved(time_saved)

        print(
            f"""
🎉 Success #{self._get_total_projects()}!
Project: {project}
Improvement: {improvement:.1f}%
Time saved: {time_saved:.1f}s per build
Rules optimized: {rules_optimized}
CO₂ saved: {co2_saved:.2f}g per build
        
Thank you for using Simpulse! 🚀
"""
        )

        return success_id

    def calculate_co2_saved(self, time_saved_seconds: float) -> float:
        """Calculate CO2 saved from reduced compute time."""

        # Average data center emissions: ~0.5g CO2 per CPU-second
        # Source: EPA estimates for cloud computing
        CO2_PER_CPU_SECOND = 0.5

        return time_saved_seconds * CO2_PER_CPU_SECOND

    def add_testimonial(self, project: str, testimonial: str):
        """Add a user testimonial."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        UPDATE project_successes 
        SET testimonial = ? 
        WHERE project_name = ? 
        ORDER BY optimization_date DESC 
        LIMIT 1
        """,
            (testimonial, project),
        )

        conn.commit()
        conn.close()

    def record_daily_activity(
        self,
        health_checks: int = 0,
        optimizations: int = 0,
        github_stars: int = None,
        contributors: int = None,
    ):
        """Record daily activity metrics."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now().date().isoformat()

        cursor.execute(
            """
        INSERT OR REPLACE INTO daily_metrics 
        (date, health_checks_run, optimizations_performed, github_stars, contributors)
        VALUES (?, ?, ?, ?, ?)
        """,
            (today, health_checks, optimizations, github_stars, contributors),
        )

        conn.commit()
        conn.close()

    def get_impact_summary(self) -> ImpactMetrics:
        """Get overall impact metrics."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get project metrics
        cursor.execute(
            """
        SELECT 
            COUNT(*) as projects,
            SUM(time_saved_per_build) as total_time,
            SUM(rules_optimized) as total_rules,
            AVG(improvement_percent) as avg_improvement,
            MAX(improvement_percent) as best_improvement
        FROM project_successes
        """
        )

        row = cursor.fetchone()
        projects, total_time, total_rules, avg_improvement, best_improvement = row

        # Get testimonials
        cursor.execute(
            """
        SELECT testimonial FROM project_successes 
        WHERE testimonial IS NOT NULL
        """
        )
        testimonials = [row[0] for row in cursor.fetchall()]

        # Get case studies
        cursor.execute(
            """
        SELECT case_study_url FROM project_successes 
        WHERE case_study_url IS NOT NULL
        """
        )
        case_studies = [row[0] for row in cursor.fetchall()]

        # Get latest GitHub metrics
        cursor.execute(
            """
        SELECT github_stars, contributors 
        FROM daily_metrics 
        ORDER BY date DESC 
        LIMIT 1
        """
        )

        github_row = cursor.fetchone()
        github_stars = github_row[0] if github_row else 0
        contributors = github_row[1] if github_row else 0

        conn.close()

        # Calculate total CO2 saved
        total_co2 = self.calculate_co2_saved(total_time or 0)

        return ImpactMetrics(
            projects_helped=projects or 0,
            total_time_saved=total_time or 0,
            rules_optimized=total_rules or 0,
            average_improvement=avg_improvement or 0,
            best_improvement=best_improvement or 0,
            total_co2_saved=total_co2,
            user_testimonials=testimonials,
            case_studies=case_studies,
            github_stars=github_stars,
            community_contributors=contributors,
        )

    def _get_total_projects(self) -> int:
        """Get total number of projects helped."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM project_successes")
        count = cursor.fetchone()[0]

        conn.close()
        return count

    def generate_impact_report(self) -> str:
        """Generate a comprehensive impact report."""

        metrics = self.get_impact_summary()

        # Format large numbers
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}min"
            elif seconds < 86400:
                return f"{seconds/3600:.1f}hrs"
            else:
                return f"{seconds/86400:.1f}days"

        report = f"""# Simpulse Impact Report

Generated: {datetime.now().strftime('%B %d, %Y')}

## 🎯 Overall Impact

- **Projects Optimized**: {metrics.projects_helped}
- **Average Improvement**: {metrics.average_improvement:.1f}%
- **Best Result**: {metrics.best_improvement:.1f}% faster
- **Total Rules Optimized**: {metrics.rules_optimized:,}

## ⏱️ Time Saved

- **Per Build Cycle**: {format_time(metrics.total_time_saved)}
- **Per Day** (10 builds): {format_time(metrics.total_time_saved * 10)}
- **Per Year** (2500 builds): {format_time(metrics.total_time_saved * 2500)}

## 🌍 Environmental Impact

- **CO₂ Saved Per Build**: {metrics.total_co2_saved:.1f}g
- **CO₂ Saved Per Year**: {metrics.total_co2_saved * 2500 / 1000:.1f}kg
- **Equivalent to**: {metrics.total_co2_saved * 2500 / 8887:.0f} gallons of gasoline not burned

## 📈 Community Growth

- **GitHub Stars**: {metrics.github_stars} ⭐
- **Contributors**: {metrics.community_contributors}
- **Testimonials**: {len(metrics.user_testimonials)}
- **Case Studies**: {len(metrics.case_studies)}

"""

        if metrics.user_testimonials:
            report += "## 💬 What Users Say\n\n"
            for testimonial in metrics.user_testimonials[:3]:
                report += f"> {testimonial}\n\n"

        report += """## 🚀 Join the Movement

Every optimization counts! If Simpulse helped your project:
1. Share your results
2. Star the repo
3. Tell others

Together, we're making Lean builds faster for everyone.

---

*Track your impact: `simpulse report --submit`*
"""

        return report

    def create_impact_visualization(self, output_path: Path = None):
        """Create visual impact charts."""

        if not output_path:
            output_path = Path("impact_charts.png")

        metrics = self.get_impact_summary()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Projects over time
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT DATE(optimization_date) as date, COUNT(*) as count
        FROM project_successes
        GROUP BY DATE(optimization_date)
        ORDER BY date
        """
        )

        dates = []
        counts = []
        cumulative = 0

        for date_str, count in cursor.fetchall():
            dates.append(datetime.fromisoformat(date_str))
            cumulative += count
            counts.append(cumulative)

        if dates:
            ax1.plot(dates, counts, "b-", linewidth=2)
            ax1.fill_between(dates, counts, alpha=0.3)
            ax1.set_title("Projects Optimized Over Time")
            ax1.set_ylabel("Total Projects")
            ax1.grid(True, alpha=0.3)

        # 2. Improvement distribution
        cursor.execute(
            """
        SELECT improvement_percent 
        FROM project_successes
        """
        )

        improvements = [row[0] for row in cursor.fetchall()]

        if improvements:
            ax2.hist(improvements, bins=10, color="green", alpha=0.7)
            ax2.axvline(
                np.mean(improvements),
                color="red",
                linestyle="--",
                label=f"Average: {np.mean(improvements):.1f}%",
            )
            ax2.set_title("Improvement Distribution")
            ax2.set_xlabel("Improvement %")
            ax2.set_ylabel("Projects")
            ax2.legend()

        # 3. Time saved accumulation
        cursor.execute(
            """
        SELECT optimization_date, time_saved_per_build
        FROM project_successes
        ORDER BY optimization_date
        """
        )

        dates = []
        time_saved = []
        cumulative_time = 0

        for date_str, time in cursor.fetchall():
            dates.append(datetime.fromisoformat(date_str))
            cumulative_time += time
            time_saved.append(cumulative_time / 60)  # Convert to minutes

        if dates:
            ax3.plot(dates, time_saved, "purple", linewidth=2)
            ax3.fill_between(dates, time_saved, alpha=0.3, color="purple")
            ax3.set_title("Cumulative Time Saved")
            ax3.set_ylabel("Total Minutes Saved Per Build")
            ax3.grid(True, alpha=0.3)

        # 4. Impact metrics
        categories = ["Projects", "Avg %", "Rules/100", "CO₂(kg)"]
        values = [
            metrics.projects_helped,
            metrics.average_improvement,
            metrics.rules_optimized / 100,
            metrics.total_co2_saved / 1000,
        ]

        bars = ax4.bar(categories, values, color=["blue", "green", "orange", "red"])
        ax4.set_title("Impact Summary")
        ax4.set_ylabel("Value")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.0f}",
                ha="center",
                va="bottom",
            )

        conn.close()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Impact visualization saved to {output_path}")

    def get_leaderboard(self, limit: int = 10) -> str:
        """Get top performing projects."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT project_name, improvement_percent, time_saved_per_build, rules_optimized
        FROM project_successes
        ORDER BY improvement_percent DESC
        LIMIT ?
        """,
            (limit,),
        )

        leaderboard = "## 🏆 Optimization Leaderboard\n\n"
        leaderboard += "| Rank | Project | Improvement | Time Saved | Rules |\n"
        leaderboard += "|------|---------|-------------|------------|-------|\n"

        for i, (project, improvement, time_saved, rules) in enumerate(
            cursor.fetchall(), 1
        ):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}"
            leaderboard += f"| {emoji} | {project} | {improvement:.1f}% | {time_saved:.1f}s | {rules} |\n"

        conn.close()
        return leaderboard

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics for external use."""

        metrics = self.get_impact_summary()

        if format == "json":
            return json.dumps(asdict(metrics), indent=2)
        elif format == "csv":
            return f"""metric,value
projects_helped,{metrics.projects_helped}
average_improvement,{metrics.average_improvement}
best_improvement,{metrics.best_improvement}
total_time_saved,{metrics.total_time_saved}
rules_optimized,{metrics.rules_optimized}
total_co2_saved,{metrics.total_co2_saved}
github_stars,{metrics.github_stars}
contributors,{metrics.community_contributors}
"""
        else:
            return str(metrics)


# Demo data for testing
def populate_demo_data(tracker: ImpactTracker):
    """Populate with demo data for testing."""

    demo_projects = [
        ("lean-crypto", 45.3, 52.1, 89, "Amazing tool! Cut our CI time in half."),
        (
            "formal-ml",
            23.7,
            38.9,
            156,
            "Simpulse found optimizations we never would have.",
        ),
        ("theorem-prover-lib", 89.2, 71.4, 234, None),
        ("category-theory", 12.5, 28.3, 67, "Simple to use, massive impact."),
        ("algebra-tactics", 34.1, 45.6, 123, None),
    ]

    for project, time_saved, improvement, rules, testimonial in demo_projects:
        tracker.add_success(
            project=project,
            time_saved=time_saved,
            improvement=improvement,
            rules_optimized=rules,
            testimonial=testimonial,
        )

    tracker.record_daily_activity(
        health_checks=47, optimizations=5, github_stars=127, contributors=8
    )


def main():
    """Demo the impact tracker."""

    print("🎯 Simpulse Impact Tracker Demo")
    print("=" * 50)

    tracker = ImpactTracker("demo_impact.db")

    # Add demo data
    populate_demo_data(tracker)

    # Generate reports
    print(tracker.generate_impact_report())
    print(tracker.get_leaderboard())

    # Create visualization
    tracker.create_impact_visualization(Path("demo_impact_charts.png"))

    # Export metrics
    metrics_json = tracker.export_metrics("json")
    Path("impact_metrics.json").write_text(metrics_json)
    print("\n📊 Metrics exported to impact_metrics.json")


if __name__ == "__main__":
    main()

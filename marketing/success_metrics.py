#!/usr/bin/env python3
"""
Success Metrics - Track Simpulse's real-world impact.
Measure our success and celebrate wins!
"""

import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class OptimizationRecord:
    """Record of a successful optimization."""
    project_name: str
    project_url: str
    timestamp: str
    improvement_percent: float
    time_saved_seconds: float
    rules_optimized: int
    total_rules: int
    build_time_before: float
    build_time_after: float
    user_name: Optional[str] = None
    testimonial: Optional[str] = None
    
    
@dataclass
class ImpactSummary:
    """Summary of total impact."""
    projects_helped: int
    total_time_saved: float  # seconds
    total_rules_optimized: int
    average_improvement: float
    co2_saved_grams: float
    developer_hours_saved: float
    testimonial_count: int
    

class ImpactTracker:
    """Track Simpulse's real-world impact."""
    
    def __init__(self, db_path: Path = Path("marketing/impact.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS optimizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL,
                project_url TEXT,
                timestamp TEXT NOT NULL,
                improvement_percent REAL NOT NULL,
                time_saved_seconds REAL NOT NULL,
                rules_optimized INTEGER NOT NULL,
                total_rules INTEGER NOT NULL,
                build_time_before REAL NOT NULL,
                build_time_after REAL NOT NULL,
                user_name TEXT,
                testimonial TEXT
            )
        """)
        conn.commit()
        conn.close()
        
    def add_success(self, 
                   project: str, 
                   improvement: float,
                   time_saved: float,
                   rules_optimized: int,
                   total_rules: int,
                   build_time_before: float,
                   build_time_after: float,
                   project_url: str = "",
                   user_name: str = "",
                   testimonial: str = "") -> int:
        """Record a successful optimization."""
        
        record = OptimizationRecord(
            project_name=project,
            project_url=project_url,
            timestamp=datetime.now().isoformat(),
            improvement_percent=improvement,
            time_saved_seconds=time_saved,
            rules_optimized=rules_optimized,
            total_rules=total_rules,
            build_time_before=build_time_before,
            build_time_after=build_time_after,
            user_name=user_name if user_name else None,
            testimonial=testimonial if testimonial else None
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO optimizations 
            (project_name, project_url, timestamp, improvement_percent, 
             time_saved_seconds, rules_optimized, total_rules,
             build_time_before, build_time_after, user_name, testimonial)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.project_name, record.project_url, record.timestamp,
            record.improvement_percent, record.time_saved_seconds,
            record.rules_optimized, record.total_rules,
            record.build_time_before, record.build_time_after,
            record.user_name, record.testimonial
        ))
        
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        
        # Calculate environmental impact
        co2_saved = self.calculate_co2_saved(time_saved)
        
        # Print celebration message
        total_impact = self.get_impact_summary()
        
        print(f"""
🎉 Success #{total_impact.projects_helped}!
════════════════════════════════════════════════════════════════
Project: {project}
Improvement: {improvement:.1f}%
Time saved: {time_saved:.1f}s per build
CO₂ saved: {co2_saved:.1f}g per build

Cumulative Impact:
- Projects helped: {total_impact.projects_helped}
- Total time saved: {total_impact.total_time_saved/3600:.1f} hours
- Developer hours saved: {total_impact.developer_hours_saved:.1f}
- CO₂ saved: {total_impact.co2_saved_grams/1000:.1f}kg
════════════════════════════════════════════════════════════════
""")
        
        return record_id
        
    def calculate_co2_saved(self, time_saved_seconds: float) -> float:
        """Calculate CO2 saved from reduced computation."""
        # Estimates based on average developer machine
        # ~200W power consumption, ~0.5kg CO2 per kWh
        
        kwh_saved = (time_saved_seconds / 3600) * 0.2  # 200W = 0.2kW
        co2_grams = kwh_saved * 500  # 500g CO2 per kWh (global average)
        
        return co2_grams
        
    def get_impact_summary(self) -> ImpactSummary:
        """Get total impact summary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get aggregate stats
        cursor.execute("""
            SELECT 
                COUNT(*) as projects,
                SUM(time_saved_seconds) as total_time,
                SUM(rules_optimized) as total_rules,
                AVG(improvement_percent) as avg_improvement,
                COUNT(testimonial) as testimonials
            FROM optimizations
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if not row or row[0] == 0:
            return ImpactSummary(0, 0, 0, 0, 0, 0, 0)
            
        projects, total_time, total_rules, avg_improvement, testimonials = row
        
        # Calculate derived metrics
        co2_total = sum(self.calculate_co2_saved(r[4]) 
                       for r in self.get_all_records())
        
        # Assume 100 builds per week per project
        builds_per_year = 100 * 52
        developer_hours = (total_time * builds_per_year * projects) / 3600
        
        return ImpactSummary(
            projects_helped=projects,
            total_time_saved=total_time or 0,
            total_rules_optimized=total_rules or 0,
            average_improvement=avg_improvement or 0,
            co2_saved_grams=co2_total,
            developer_hours_saved=developer_hours,
            testimonial_count=testimonials
        )
        
    def get_all_records(self) -> List[Tuple]:
        """Get all optimization records."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM optimizations ORDER BY timestamp DESC")
        records = cursor.fetchall()
        conn.close()
        return records
        
    def generate_impact_report(self) -> str:
        """Generate comprehensive impact report."""
        summary = self.get_impact_summary()
        
        report = f"""# Simpulse Impact Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Executive Summary

Simpulse has helped **{summary.projects_helped} projects** achieve an average performance improvement of **{summary.average_improvement:.1f}%**.

## Key Metrics

### Performance Impact
- **Total time saved**: {summary.total_time_saved:.1f} seconds per build
- **Developer hours saved annually**: {summary.developer_hours_saved:.0f} hours
- **Average improvement**: {summary.average_improvement:.1f}%
- **Rules optimized**: {summary.total_rules_optimized}

### Environmental Impact
- **CO₂ saved**: {summary.co2_saved_grams/1000:.1f}kg
- **Equivalent to**: {summary.co2_saved_grams/1000/8.887:.0f} gallons of gasoline not burned

### Community Impact
- **Projects helped**: {summary.projects_helped}
- **Testimonials**: {summary.testimonial_count}
- **GitHub stars**: (fetch from API)

## Success Stories
"""
        
        # Add recent successes
        records = self.get_all_records()[:5]  # Last 5
        
        for r in records:
            report += f"""
### {r[1]} - {r[4]:.0f}% Improvement
- **Before**: {r[8]:.1f}s build time
- **After**: {r[9]:.1f}s build time  
- **Rules optimized**: {r[6]}/{r[7]}
- **Date**: {r[3][:10]}
"""
            
            if r[11]:  # Testimonial
                report += f'> "{r[11]}"\n> - {r[10] or "Anonymous"}\n'
                
        # Add growth metrics
        report += self._generate_growth_section()
        
        return report
        
    def _generate_growth_section(self) -> str:
        """Generate growth metrics section."""
        records = self.get_all_records()
        if not records:
            return "\n## Growth: No data yet\n"
            
        # Group by month
        monthly = defaultdict(int)
        for r in records:
            month = r[3][:7]  # YYYY-MM
            monthly[month] += 1
            
        section = "\n## Growth Trajectory\n\n"
        section += "| Month | Projects Optimized | Cumulative |\n"
        section += "|-------|-------------------|------------|\n"
        
        cumulative = 0
        for month in sorted(monthly.keys()):
            count = monthly[month]
            cumulative += count
            section += f"| {month} | {count} | {cumulative} |\n"
            
        return section
        
    def create_impact_dashboard(self, output_dir: Path = Path("marketing/dashboard")):
        """Create visual impact dashboard."""
        output_dir.mkdir(exist_ok=True)
        
        summary = self.get_impact_summary()
        records = self.get_all_records()
        
        if not records:
            print("No data to visualize yet!")
            return
            
        # Create multiple visualizations
        self._create_improvement_histogram(records, output_dir)
        self._create_growth_chart(records, output_dir)
        self._create_impact_summary_graphic(summary, output_dir)
        
        # Create HTML dashboard
        self._create_html_dashboard(summary, output_dir)
        
        print(f"✅ Dashboard created in: {output_dir}/")
        
    def _create_improvement_histogram(self, records: List[Tuple], output_dir: Path):
        """Create histogram of improvements."""
        improvements = [r[4] for r in records]
        
        plt.figure(figsize=(10, 6))
        plt.hist(improvements, bins=10, color='#2ecc71', edgecolor='black')
        plt.xlabel('Improvement %')
        plt.ylabel('Number of Projects')
        plt.title('Distribution of Performance Improvements')
        plt.grid(axis='y', alpha=0.3)
        
        # Add average line
        avg = sum(improvements) / len(improvements)
        plt.axvline(avg, color='red', linestyle='--', linewidth=2, label=f'Average: {avg:.1f}%')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'improvement_distribution.png', dpi=300)
        plt.close()
        
    def _create_growth_chart(self, records: List[Tuple], output_dir: Path):
        """Create growth over time chart."""
        # Sort by date
        sorted_records = sorted(records, key=lambda x: x[3])
        
        dates = []
        cumulative = []
        
        for i, r in enumerate(sorted_records):
            dates.append(datetime.fromisoformat(r[3]))
            cumulative.append(i + 1)
            
        plt.figure(figsize=(10, 6))
        plt.plot(dates, cumulative, 'b-', linewidth=2, marker='o')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Projects Optimized')
        plt.title('Simpulse Adoption Growth')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(dates) > 1:
            # Simple linear projection
            days_elapsed = (dates[-1] - dates[0]).days
            if days_elapsed > 0:
                rate = len(cumulative) / days_elapsed
                future_date = dates[-1] + timedelta(days=90)
                future_count = len(cumulative) + (rate * 90)
                
                plt.plot([dates[-1], future_date], [cumulative[-1], future_count], 
                        'r--', alpha=0.5, label='90-day projection')
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'growth_chart.png', dpi=300)
        plt.close()
        
    def _create_impact_summary_graphic(self, summary: ImpactSummary, output_dir: Path):
        """Create impact summary infographic."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Projects helped
        ax1.text(0.5, 0.5, f'{summary.projects_helped}', 
                fontsize=72, ha='center', va='center', color='#2ecc71')
        ax1.text(0.5, 0.2, 'Projects\nOptimized', 
                fontsize=20, ha='center', va='center')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Average improvement
        ax2.text(0.5, 0.5, f'{summary.average_improvement:.0f}%', 
                fontsize=72, ha='center', va='center', color='#3498db')
        ax2.text(0.5, 0.2, 'Average\nImprovement', 
                fontsize=20, ha='center', va='center')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # Developer hours saved
        ax3.text(0.5, 0.5, f'{summary.developer_hours_saved:.0f}', 
                fontsize=48, ha='center', va='center', color='#e74c3c')
        ax3.text(0.5, 0.2, 'Developer\nHours Saved', 
                fontsize=20, ha='center', va='center')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # CO2 saved
        ax4.text(0.5, 0.5, f'{summary.co2_saved_grams/1000:.1f}kg', 
                fontsize=48, ha='center', va='center', color='#27ae60')
        ax4.text(0.5, 0.2, 'CO₂ Saved', 
                fontsize=20, ha='center', va='center')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.suptitle('Simpulse Impact Summary', fontsize=24, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / 'impact_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_html_dashboard(self, summary: ImpactSummary, output_dir: Path):
        """Create HTML dashboard."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Simpulse Impact Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .metric {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 48px;
            font-weight: bold;
            color: #2ecc71;
        }}
        .metric-label {{
            font-size: 18px;
            color: #666;
            margin-top: 10px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            margin: 20px 0;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .updated {{
            text-align: center;
            color: #666;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <h1>Simpulse Impact Dashboard</h1>
    <p class="updated">Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    
    <div class="grid">
        <div class="metric">
            <div class="metric-value">{summary.projects_helped}</div>
            <div class="metric-label">Projects Optimized</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.average_improvement:.0f}%</div>
            <div class="metric-label">Average Improvement</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.developer_hours_saved:.0f}</div>
            <div class="metric-label">Developer Hours Saved</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.co2_saved_grams/1000:.1f}kg</div>
            <div class="metric-label">CO₂ Saved</div>
        </div>
    </div>
    
    <h2>Performance Distribution</h2>
    <img src="improvement_distribution.png" alt="Improvement Distribution">
    
    <h2>Growth Over Time</h2>
    <img src="growth_chart.png" alt="Growth Chart">
    
    <h2>Impact Summary</h2>
    <img src="impact_summary.png" alt="Impact Summary">
    
    <p style="text-align: center; margin-top: 40px;">
        <a href="https://github.com/Bright-L01/simpulse">View on GitHub</a> | 
        <a href="mailto:contact@simpulse.dev">Contact</a>
    </p>
</body>
</html>"""
        
        (output_dir / 'index.html').write_text(html)
        
    def export_for_social(self) -> Dict[str, str]:
        """Export metrics formatted for social media."""
        summary = self.get_impact_summary()
        
        return {
            'twitter': f"""🚀 Simpulse Impact Update:

✅ {summary.projects_helped} projects optimized
📈 {summary.average_improvement:.0f}% average improvement  
⏱️ {summary.developer_hours_saved:.0f} developer hours saved
🌱 {summary.co2_saved_grams/1000:.1f}kg CO₂ saved

Make your #Lean4 proofs faster: github.com/Bright-L01/simpulse""",
            
            'linkedin': f"""Excited to share Simpulse's growing impact on the Lean 4 community!

Key metrics:
• {summary.projects_helped} projects optimized
• {summary.average_improvement:.0f}% average performance improvement
• {summary.developer_hours_saved:.0f} developer hours saved annually
• {summary.co2_saved_grams/1000:.1f}kg CO₂ emissions prevented

Simpulse helps Lean developers optimize their simp tactic performance through intelligent rule prioritization. The tool analyzes usage patterns and automatically reorders rules for optimal performance.

Interested in faster proof checking? Check out Simpulse: github.com/Bright-L01/simpulse

#Lean4 #PerformanceOptimization #OpenSource #Sustainability""",
            
            'milestone_10': f"""🎉 Milestone: 10 Projects Optimized!

Thanks to the amazing Lean community, Simpulse has now helped 10 projects achieve faster proof checking.

Cumulative impact:
• {summary.total_time_saved/60:.0f} minutes saved per build across all projects
• {summary.total_rules_optimized} simp rules optimized
• {summary.developer_hours_saved:.0f} total developer hours saved

Here's to the next 10! 🚀"""
        }


def create_demo_data():
    """Create demonstration data."""
    tracker = ImpactTracker(Path("marketing/demo_impact.db"))
    
    # Add some demo successes
    projects = [
        ("MathLib-Algebra", 71.4, 32.5, 89, 156, 45.2, 12.7, 
         "Amazing improvement! Our builds are so much faster now."),
        ("FormalML", 52.3, 18.7, 67, 134, 28.4, 13.5, 
         "Simpulse found optimizations we never would have thought of."),
        ("CategoryTheory", 64.8, 41.2, 112, 203, 63.5, 22.3, ""),
        ("LinearAlgebra", 38.9, 12.4, 45, 98, 31.9, 19.5, 
         "Solid improvement with zero effort required."),
        ("TopologyBase", 55.6, 27.8, 78, 167, 50.0, 22.2, ""),
    ]
    
    for i, (name, imp, saved, opt, total, before, after, test) in enumerate(projects):
        tracker.add_success(
            project=name,
            improvement=imp,
            time_saved=saved,
            rules_optimized=opt,
            total_rules=total,
            build_time_before=before,
            build_time_after=after,
            project_url=f"https://github.com/example/{name.lower()}",
            testimonial=test if test else ""
        )
        
    return tracker


def main():
    """Demo the impact tracking system."""
    print("Creating demo impact data...")
    
    tracker = create_demo_data()
    
    # Generate report
    report = tracker.generate_impact_report()
    report_path = Path("marketing/impact_report.md")
    report_path.write_text(report)
    print(f"\n✅ Impact report saved to: {report_path}")
    
    # Create dashboard
    tracker.create_impact_dashboard()
    
    # Export social media content
    social = tracker.export_for_social()
    social_path = Path("marketing/social_media_posts.json")
    social_path.write_text(json.dumps(social, indent=2))
    print(f"✅ Social media content saved to: {social_path}")
    
    # Show summary
    summary = tracker.get_impact_summary()
    print(f"\n📊 Current Impact:")
    print(f"   Projects helped: {summary.projects_helped}")
    print(f"   Average improvement: {summary.average_improvement:.1f}%")
    print(f"   Developer hours saved: {summary.developer_hours_saved:.0f}")
    
    print("\n🎯 Ready to track real successes!")


if __name__ == "__main__":
    # Note: matplotlib import will fail without the package
    try:
        main()
    except ImportError:
        print("Note: Install matplotlib for charts: pip install matplotlib")
        print("Impact tracking will still work without charts.")
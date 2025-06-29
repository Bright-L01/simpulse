#!/usr/bin/env python3
"""Track and manage community engagement efforts for Simpulse."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class CommunityEngagementTracker:
    def __init__(self):
        self.data_file = Path("community_engagement.json")
        self.load_data()

    def load_data(self):
        """Load existing engagement data."""
        if self.data_file.exists():
            with open(self.data_file) as f:
                self.data = json.load(f)
        else:
            self.data = {
                "platforms": {},
                "projects": {},
                "metrics": {
                    "total_outreach": 0,
                    "responses_received": 0,
                    "projects_optimized": 0,
                    "performance_improvements": [],
                },
            }

    def save_data(self):
        """Save engagement data."""
        with open(self.data_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def add_platform_post(self, platform: str, post_info: Dict):
        """Track a post on a platform (Zulip, GitHub, etc)."""
        if platform not in self.data["platforms"]:
            self.data["platforms"][platform] = []

        post_info["timestamp"] = datetime.now().isoformat()
        post_info["status"] = "posted"
        self.data["platforms"][platform].append(post_info)
        self.data["metrics"]["total_outreach"] += 1
        self.save_data()

    def update_project_status(
        self, project: str, status: str, details: Optional[Dict] = None
    ):
        """Update the status of a project engagement."""
        if project not in self.data["projects"]:
            self.data["projects"][project] = {
                "first_contact": datetime.now().isoformat(),
                "status": status,
                "history": [],
            }

        self.data["projects"][project]["status"] = status
        self.data["projects"][project]["last_update"] = datetime.now().isoformat()

        if details:
            self.data["projects"][project]["history"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "status": status,
                    "details": details,
                }
            )

        if status == "response_received":
            self.data["metrics"]["responses_received"] += 1
        elif status == "optimized":
            self.data["metrics"]["projects_optimized"] += 1
            if details and "improvement" in details:
                self.data["metrics"]["performance_improvements"].append(
                    details["improvement"]
                )

        self.save_data()

    def generate_summary(self):
        """Generate a summary of engagement efforts."""
        summary = []
        summary.append("# Community Engagement Summary\n")
        summary.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # Overall metrics
        summary.append("## Overall Metrics\n")
        summary.append(
            f"- Total Outreach Attempts: {self.data['metrics']['total_outreach']}"
        )
        summary.append(
            f"- Responses Received: {self.data['metrics']['responses_received']}"
        )
        summary.append(
            f"- Projects Optimized: {self.data['metrics']['projects_optimized']}"
        )

        if self.data["metrics"]["performance_improvements"]:
            avg_improvement = sum(
                self.data["metrics"]["performance_improvements"]
            ) / len(self.data["metrics"]["performance_improvements"])
            summary.append(
                f"- Average Performance Improvement: {avg_improvement:.1f}%\n"
            )

        # Platform breakdown
        summary.append("\n## Platform Activity\n")
        for platform, posts in self.data["platforms"].items():
            summary.append(f"\n### {platform}")
            summary.append(f"- Posts: {len(posts)}")
            for post in posts[-5:]:  # Last 5 posts
                summary.append(
                    f"  - {post.get('title', 'Untitled')} ({post.get('timestamp', 'Unknown')})"
                )

        # Project status
        summary.append("\n## Project Engagement Status\n")
        for project, info in sorted(
            self.data["projects"].items(),
            key=lambda x: x[1].get("last_update", ""),
            reverse=True,
        ):
            summary.append(f"\n### {project}")
            summary.append(f"- Status: {info['status']}")
            summary.append(f"- Last Update: {info.get('last_update', 'Unknown')}")

        return "\n".join(summary)

    def get_next_actions(self) -> List[str]:
        """Get recommended next actions based on current status."""
        actions = []

        # Check for projects awaiting response
        awaiting = [
            p
            for p, info in self.data["projects"].items()
            if info["status"] == "contacted"
        ]
        if awaiting:
            actions.append(f"Follow up with: {', '.join(awaiting[:3])}")

        # Check response rate
        if self.data["metrics"]["total_outreach"] > 0:
            response_rate = (
                self.data["metrics"]["responses_received"]
                / self.data["metrics"]["total_outreach"]
            )
            if response_rate < 0.2:
                actions.append(
                    "Consider refining outreach message or targeting different projects"
                )

        # Suggest new platforms
        if "Zulip" not in self.data["platforms"]:
            actions.append("Post on Lean Zulip to reach core community")
        if "Reddit" not in self.data["platforms"]:
            actions.append("Consider posting on r/lean or r/programming")

        return actions


def main():
    """Example usage and current status check."""
    tracker = CommunityEngagementTracker()

    # Example: Track Zulip post
    print("📊 Current Community Engagement Status\n")

    # Add planned Zulip post
    tracker.add_platform_post(
        "Zulip",
        {
            "title": "Simpulse - Automatic simp rule priority optimization",
            "url": "https://leanprover.zulipchat.com/#narrow/stream/general",
            "content_summary": "Introduced Simpulse tool, shared leansat case study",
        },
    )

    # Update leansat project status
    tracker.update_project_status(
        "leanprover/leansat",
        "pr_prepared",
        {
            "pr_branch": "simpulse-optimization",
            "rules_optimized": 37,
            "estimated_improvement": 63,
        },
    )

    # Mark other projects as contacted
    for project in [
        "AndrasKovacs/smalltt",
        "madvorak/duality",
        "lean-dojo/LeanCopilot",
    ]:
        tracker.update_project_status(
            project,
            "contacted",
            {
                "method": "GitHub issue",
                "message_file": f"outreach_messages/{project.split('/')[-1]}.txt",
            },
        )

    # Generate and print summary
    print(tracker.generate_summary())

    print("\n## 🎯 Recommended Next Actions\n")
    for action in tracker.get_next_actions():
        print(f"- {action}")

    # Save current engagement data
    print(f"\n✅ Engagement data saved to: {tracker.data_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Helper script for posting to community platforms."""

import webbrowser
from pathlib import Path

import pyperclip  # For clipboard support


class CommunityPoster:
    def __init__(self):
        self.posts_dir = Path("outreach_messages")

    def prepare_zulip_post(self):
        """Prepare and open Zulip for posting."""
        post_path = self.posts_dir / "lean_zulip_post.md"

        if not post_path.exists():
            print("❌ Zulip post not found!")
            return

        # Read the post content
        content = post_path.read_text()

        # Extract the shorter version for initial post
        if "Alternative shorter version" in content:
            short_version = content.split("Alternative shorter version")[1].strip()
        else:
            short_version = content[:500] + "..."

        print("📋 Zulip Post Instructions:\n")
        print("1. The post content has been copied to your clipboard")
        print("2. Go to: https://leanprover.zulipchat.com")
        print("3. Navigate to #general stream")
        print("4. Create new topic: 'performance'")
        print("5. Paste the content and post\n")

        print("Preview of short version:")
        print("-" * 50)
        print(short_version)
        print("-" * 50)

        try:
            pyperclip.copy(short_version)
            print("\n✅ Content copied to clipboard!")
        except:
            print("\n⚠️  Could not copy to clipboard. Please copy manually from above.")

        # Open Zulip in browser
        input("\nPress Enter to open Lean Zulip in your browser...")
        webbrowser.open(
            "https://leanprover.zulipchat.com/#narrow/stream/113488-general"
        )

    def prepare_github_pr(self, project: str = "leansat"):
        """Prepare GitHub PR submission."""
        pr_desc_path = Path(f"analyzed_repos/{project}/SIMPULSE_PR_DESCRIPTION.md")

        print(f"📋 GitHub PR Instructions for {project}:\n")

        if pr_desc_path.exists():
            print("1. PR description is ready at:")
            print(f"   {pr_desc_path}\n")

        print("2. Commands to push and create PR:")
        print(f"   cd analyzed_repos/{project}")
        print("   git push origin simpulse-optimization")
        print(
            "   gh pr create --title 'Optimize simp rule priorities' --body-file SIMPULSE_PR_DESCRIPTION.md\n"
        )

        print("3. Or create manually:")
        print(f"   - Go to: https://github.com/leanprover/{project}")
        print("   - Click 'Pull requests' → 'New pull request'")
        print("   - Select 'simpulse-optimization' branch")
        print("   - Use the PR description from the file\n")

        input("Press Enter to open the repository...")
        webbrowser.open(f"https://github.com/leanprover/{project}")

    def prepare_github_issue(self, project: str):
        """Prepare GitHub issue for a project."""
        owner, repo = project.split("/")
        message_file = self.posts_dir / f"{owner}_{repo}.txt"

        if not message_file.exists():
            print(f"❌ Message file not found for {project}")
            return

        content = message_file.read_text()

        print(f"📋 GitHub Issue Instructions for {project}:\n")
        print("1. Issue content has been prepared")
        print(
            "2. Suggested title: 'Optimize simp rule priorities for better performance'"
        )
        print("3. The message explains the optimization opportunity\n")

        print("Message preview:")
        print("-" * 50)
        print(content[:400] + "...")
        print("-" * 50)

        try:
            pyperclip.copy(content)
            print("\n✅ Content copied to clipboard!")
        except:
            print("\n⚠️  Could not copy to clipboard. Please copy manually.")

        input(f"\nPress Enter to open {project} issues page...")
        webbrowser.open(f"https://github.com/{project}/issues/new")

    def show_case_study(self):
        """Display case study location and key points."""
        case_study_path = Path("case_studies/leansat/README.md")

        print("📊 Leansat Case Study Summary:\n")
        print("Key Points:")
        print("- 134 simp rules analyzed")
        print("- ALL use default priority (1000)")
        print("- 63% estimated performance improvement")
        print("- 37 rules optimized in PR")
        print("- Zero semantic changes\n")

        print(f"Full case study: {case_study_path}")
        print("Visualizations: case_studies/leansat/*.png\n")

        print("Share links:")
        print(
            "- GitHub: https://github.com/Bright-L01/simpulse/tree/main/case_studies/leansat"
        )
        print(
            "- Direct link: https://github.com/Bright-L01/simpulse/blob/main/case_studies/leansat/README.md"
        )


def main():
    """Interactive menu for community posting."""
    poster = CommunityPoster()

    while True:
        print("\n🚀 Simpulse Community Outreach Tool\n")
        print("1. Post to Lean Zulip")
        print("2. Submit PR to leansat")
        print("3. Create issue for a project")
        print("4. View case study summary")
        print("5. Exit\n")

        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            poster.prepare_zulip_post()
        elif choice == "2":
            poster.prepare_github_pr("leansat")
        elif choice == "3":
            project = input("Enter project (e.g., AndrasKovacs/smalltt): ").strip()
            poster.prepare_github_issue(project)
        elif choice == "4":
            poster.show_case_study()
        elif choice == "5":
            print("👋 Good luck with your outreach!")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    # Check if pyperclip is available
    try:
        import pyperclip
    except ImportError:
        print("⚠️  Installing pyperclip for clipboard support...")
        import subprocess

        subprocess.run(["pip", "install", "pyperclip"])

    main()

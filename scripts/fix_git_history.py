#!/usr/bin/env python3
"""
Fix Git history to show single author.
WARNING: This will rewrite history!
"""

import os
import subprocess
import sys
from pathlib import Path


def clean_git_history():
    """Update Git history to show single author."""
    
    print("GIT HISTORY CLEANUP")
    print("="*70)
    print("WARNING: This will rewrite Git history!")
    print("You will need to force push after this operation.")
    print()
    
    # Get current branch
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True
    )
    current_branch = result.stdout.strip()
    print(f"Current branch: {current_branch}")
    
    # Get user input for author details
    print("\nEnter the author details to use for all commits:")
    author_name = input("Author name: ").strip()
    author_email = input("Author email: ").strip()
    
    if not author_name or not author_email:
        print("❌ Author name and email are required")
        return False
    
    print(f"\nThis will change all commits to:")
    print(f"  Name: {author_name}")
    print(f"  Email: {author_email}")
    print()
    
    confirm = input("Continue? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return False
    
    # Create backup branch
    print("\nCreating backup branch...")
    backup_branch = f"backup-before-history-rewrite-{int(subprocess.time.time())}"
    subprocess.run(["git", "branch", backup_branch])
    print(f"✓ Backup branch created: {backup_branch}")
    
    # The filter-branch command
    env_filter = f'''
export GIT_AUTHOR_NAME="{author_name}"
export GIT_AUTHOR_EMAIL="{author_email}"
export GIT_COMMITTER_NAME="{author_name}"
export GIT_COMMITTER_EMAIL="{author_email}"
'''
    
    print("\nRewriting history...")
    try:
        # Remove backup if exists
        subprocess.run(
            ["rm", "-rf", ".git/refs/original/"],
            capture_output=True
        )
        
        # Run filter-branch
        result = subprocess.run(
            ["git", "filter-branch", "-f", "--env-filter", env_filter, 
             "--tag-name-filter", "cat", "--", "--branches", "--tags"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ History rewritten successfully!")
            
            # Show new history
            print("\nNew commit history (last 5 commits):")
            log_result = subprocess.run(
                ["git", "log", "--oneline", "--pretty=format:%h %an <%ae> %s", "-5"],
                capture_output=True,
                text=True
            )
            print(log_result.stdout)
            
            print("\n" + "="*70)
            print("NEXT STEPS:")
            print("="*70)
            print("1. Review the changes:")
            print("   git log --oneline -10")
            print()
            print("2. If everything looks good, force push:")
            print(f"   git push --force origin {current_branch}")
            print()
            print("3. If something went wrong, restore from backup:")
            print(f"   git reset --hard {backup_branch}")
            print()
            print("⚠️  WARNING: Force pushing will overwrite remote history!")
            print("⚠️  Other contributors will need to re-clone or reset their local repos")
            
            return True
            
        else:
            print("❌ Error rewriting history:")
            print(result.stderr)
            print(f"\nYou can restore from backup: git checkout {backup_branch}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\nYou can restore from backup: git checkout {backup_branch}")
        return False


def alternative_approach():
    """Alternative approach using .mailmap file."""
    print("\nALTERNATIVE: Using .mailmap file")
    print("="*70)
    print("This approach doesn't rewrite history but changes how authors are displayed.")
    print()
    
    print("Enter the canonical author details:")
    canonical_name = input("Canonical name: ").strip()
    canonical_email = input("Canonical email: ").strip()
    
    if not canonical_name or not canonical_email:
        print("❌ Name and email are required")
        return
    
    # Get all commit authors
    result = subprocess.run(
        ["git", "log", "--format=%an <%ae>", "--all"],
        capture_output=True,
        text=True
    )
    
    authors = set(result.stdout.strip().split('\n'))
    
    print(f"\nFound {len(authors)} unique author entries")
    
    # Create .mailmap
    mailmap_path = Path(".mailmap")
    with open(mailmap_path, 'w') as f:
        for author in authors:
            if author and canonical_email not in author:
                f.write(f"{canonical_name} <{canonical_email}> {author}\n")
    
    print(f"✓ Created {mailmap_path}")
    print("\nNow commits will display as the canonical author.")
    print("This doesn't change history but changes how it's displayed.")
    print("\nDon't forget to commit the .mailmap file!")


def main():
    """Main entry point."""
    print("Git History Cleanup Options")
    print("="*70)
    print("1. Rewrite history (changes actual commits)")
    print("2. Use .mailmap (only changes display)")
    print("3. Exit")
    print()
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        clean_git_history()
    elif choice == "2":
        alternative_approach()
    elif choice == "3":
        print("Exiting.")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    # Extra safety check
    if "--force" not in sys.argv:
        print("⚠️  WARNING: This script can rewrite Git history!")
        print("⚠️  Make sure you have a backup of your repository")
        print()
        print("Run with --force to proceed")
        print("Example: python fix_git_history.py --force")
        sys.exit(1)
    
    main()
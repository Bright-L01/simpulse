#!/usr/bin/env python3
"""
Script to clean Claude references from git commit messages.
"""

import subprocess
import re

def clean_commit_message(message):
    """Clean Claude references from commit message."""
    lines = message.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip Claude-specific lines
        if '🤖 Generated with [Claude Code]' in line:
            continue
        if 'Co-Authored-By: Claude' in line:
            continue
        cleaned_lines.append(line)
    
    # Join back and clean up excessive newlines
    cleaned_message = '\n'.join(cleaned_lines)
    # Remove excessive blank lines
    cleaned_message = re.sub(r'\n\n\n+', '\n\n', cleaned_message)
    # Remove trailing whitespace
    cleaned_message = cleaned_message.strip()
    
    return cleaned_message

def get_commit_message(commit_hash):
    """Get the commit message for a specific commit."""
    result = subprocess.run(['git', 'show', '-s', '--format=%B', commit_hash], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return None

def rewrite_commit(commit_hash, new_message):
    """Rewrite a commit message."""
    # First, create a temporary commit with the new message
    cmd = ['git', 'commit', '--amend', '-m', new_message]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def main():
    # Commits that need cleaning (identified earlier)
    commits_to_clean = [
        'a5bad16',  # Final Simpulse 2.0 polish
        '6b714a5',  # Simpulse 2.0: Complete rewrite
    ]
    
    for commit_hash in commits_to_clean:
        print(f"Processing commit {commit_hash}...")
        
        # Get current message
        current_message = get_commit_message(commit_hash)
        if not current_message:
            print(f"Could not get message for {commit_hash}")
            continue
        
        # Clean the message
        cleaned_message = clean_commit_message(current_message)
        
        print(f"Original message length: {len(current_message)}")
        print(f"Cleaned message length: {len(cleaned_message)}")
        print(f"Cleaned message preview: {cleaned_message[:100]}...")
        
        # Show the diff
        print("\nCleaned message:")
        print(cleaned_message)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
#!/bin/bash
# Git history cleanup script

echo "🧹 Cleaning git history of Claude references..."

# Create backup branch
git checkout -b backup-before-cleanup

# Go back to main branch
git checkout main-combined

# Remove Claude references from all commit messages
git filter-branch -f --msg-filter '
    sed -e "s/🤖 Generated with \[Claude Code\].*//" \
        -e "/Co-Authored-By: Claude/d" \
        -e "s/Claude AI/AI Assistant/g" \
        -e "s/Claude/AI Assistant/g"
' -- --all

# Clean up refs
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

# Garbage collect
git gc --prune=now --aggressive

echo "✅ Git history cleaned!"
echo ""
echo "⚠️  WARNING: This rewrites history. To push:"
echo "   git push --force-with-lease origin main-combined:main"
echo ""
echo "To restore original history:"
echo "   git checkout backup-before-cleanup"

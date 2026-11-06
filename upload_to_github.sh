#!/usr/bin/env bash
# Helper script to link and push your e-Yantra project to GitHub

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "=========================================================="
echo "  🚀 Push e-Yantra StrataCobot (SC#4096) to GitHub"
echo "=========================================================="
echo ""
echo "Step 1: Go to https://github.com/new in your browser."
echo "Step 2: Create a new repository (e.g. named 'eyantra-stratacobot')."
echo "        (Do NOT check 'Initialize with README' or '.gitignore')"
echo "Step 3: Copy the repository URL (HTTPS or SSH)."
echo ""

read -rp "Paste your GitHub Repository URL (e.g. https://github.com/your-username/eyantra-stratacobot.git): " REMOTE_URL

if [ -z "$REMOTE_URL" ]; then
    echo "❌ Error: Repository URL cannot be empty."
    exit 1
fi

# Remove existing origin if any
git remote remove origin 2>/dev/null || true

# Add new origin
git remote add origin "$REMOTE_URL"
git branch -M main

echo ""
echo "Pushing code to $REMOTE_URL on branch main..."
echo "Note: If prompted for password, enter your GitHub Personal Access Token (classic)."
echo ""

git push -u origin main

echo ""
echo "🎉 Successfully uploaded to GitHub! Visit your repository to see the showcase!"

#!/bin/bash
# ============================================================================
# Quick Start Script for Hateful Meme Detection Project
# ============================================================================
# This script helps you set up Git and push to GitHub
# Run with: bash scripts/quick_start.sh
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  🚀 Hateful Meme Detection - Quick Start"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed!${NC}"
    echo "Please install Git first:"
    echo "  - Windows: https://git-scm.com/download/windows"
    echo "  - macOS: brew install git"
    echo "  - Linux: sudo apt install git"
    exit 1
fi

echo -e "${GREEN}✓ Git is installed${NC}"

# Check Git configuration
GIT_NAME=$(git config --global user.name || echo "")
GIT_EMAIL=$(git config --global user.email || echo "")

if [ -z "$GIT_NAME" ] || [ -z "$GIT_EMAIL" ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Git is not configured. Let's set it up:${NC}"
    
    read -p "Enter your name: " NAME
    read -p "Enter your email: " EMAIL
    
    git config --global user.name "$NAME"
    git config --global user.email "$EMAIL"
    
    echo -e "${GREEN}✓ Git configured${NC}"
else
    echo -e "${GREEN}✓ Git configured as: $GIT_NAME <$GIT_EMAIL>${NC}"
fi

# Check if already a git repo
if [ -d ".git" ]; then
    echo -e "${GREEN}✓ Already a Git repository${NC}"
else
    echo ""
    echo "Initializing Git repository..."
    git init
    echo -e "${GREEN}✓ Git repository initialized${NC}"
fi

# Check for LFS
if command -v git-lfs &> /dev/null; then
    echo -e "${GREEN}✓ Git LFS is installed${NC}"
    
    # Initialize LFS
    git lfs install
    
    # Track large files
    if [ ! -f ".gitattributes" ] || ! grep -q "*.pth" .gitattributes 2>/dev/null; then
        echo "Setting up LFS tracking for model files..."
        git lfs track "*.pth"
        git lfs track "*.pt"
        git lfs track "*.h5"
        git lfs track "*.pkl"
        git lfs track "*.bin"
        echo -e "${GREEN}✓ LFS tracking configured${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Git LFS not installed (optional, for large model files)${NC}"
    echo "   Install: https://git-lfs.github.io/"
fi

# Stage all files
echo ""
echo "Staging files..."
git add .
echo -e "${GREEN}✓ Files staged${NC}"

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo -e "${YELLOW}No new changes to commit${NC}"
else
    echo ""
    read -p "Enter commit message (or press Enter for 'Initial commit'): " COMMIT_MSG
    COMMIT_MSG=${COMMIT_MSG:-"Initial commit: Complete project structure"}
    
    git commit -m "$COMMIT_MSG"
    echo -e "${GREEN}✓ Changes committed${NC}"
fi

# Check for remote
REMOTE=$(git remote -v | grep origin || echo "")

if [ -z "$REMOTE" ]; then
    echo ""
    echo "=============================================="
    echo "  📡 Connect to GitHub"
    echo "=============================================="
    echo ""
    echo "1. Create a new repository on GitHub:"
    echo "   https://github.com/new"
    echo ""
    echo "2. Repository name: hateful-meme-detection"
    echo "   DON'T initialize with README, .gitignore, or license"
    echo ""
    read -p "Enter your GitHub username: " GITHUB_USER
    
    REPO_URL="https://github.com/$GITHUB_USER/hateful-meme-detection.git"
    
    echo ""
    echo "Adding remote: $REPO_URL"
    git remote add origin "$REPO_URL"
    echo -e "${GREEN}✓ Remote added${NC}"
fi

# Rename branch to main if needed
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    git branch -M main
    echo -e "${GREEN}✓ Branch renamed to main${NC}"
fi

# Push
echo ""
echo "=============================================="
echo "  📤 Push to GitHub"
echo "=============================================="
echo ""
echo "Ready to push to GitHub!"
echo ""
read -p "Push now? (y/n): " PUSH_NOW

if [ "$PUSH_NOW" = "y" ] || [ "$PUSH_NOW" = "Y" ]; then
    echo "Pushing to GitHub..."
    git push -u origin main
    echo ""
    echo -e "${GREEN}=============================================="
    echo "  ✅ SUCCESS!"
    echo "==============================================${NC}"
    echo ""
    echo "Your project is now on GitHub!"
    echo "Visit: https://github.com/$GITHUB_USER/hateful-meme-detection"
else
    echo ""
    echo "To push later, run:"
    echo "  git push -u origin main"
fi

echo ""
echo "=============================================="
echo "  📚 Next Steps"
echo "=============================================="
echo ""
echo "1. Add your trained model to models/best_model.pth"
echo "2. Update README.md with your GitHub username"
echo "3. Set up Discord bot with your token"
echo "4. Create a GitHub Release with model weights"
echo ""
echo "For help, see: docs/GITHUB_TUTORIAL.md"
echo ""

# 🎓 Complete GitHub Tutorial for Your ML Project

> A step-by-step guide to Git and GitHub, from zero to professional workflow.
> Tailored for the Hateful Meme Detection project.

---

## 📚 Table of Contents

1. [What is Git & GitHub?](#1-what-is-git--github)
2. [Installation & Setup](#2-installation--setup)
3. [Your First Repository](#3-your-first-repository)
4. [Essential Git Commands](#4-essential-git-commands)
5. [Pushing Your Project to GitHub](#5-pushing-your-project-to-github)
6. [Branching & Workflow](#6-branching--workflow)
7. [Pull Requests](#7-pull-requests)
8. [Handling Large Files (Git LFS)](#8-handling-large-files-git-lfs)
9. [GitHub Features](#9-github-features)
10. [Best Practices for ML Projects](#10-best-practices-for-ml-projects)
11. [Common Issues & Solutions](#11-common-issues--solutions)
12. [Quick Reference Cheatsheet](#12-quick-reference-cheatsheet)

---

## 1. What is Git & GitHub?

### Git
**Git** is a distributed version control system. Think of it as:
- 📸 A "save game" system for your code
- ⏰ A time machine to go back to any previous version
- 👥 A collaboration tool for teams

### GitHub
**GitHub** is a cloud platform that hosts Git repositories. It provides:
- ☁️ Remote storage for your code
- 🤝 Collaboration features (Pull Requests, Issues)
- 🚀 CI/CD automation (GitHub Actions)
- 📊 Project management tools

### Key Concepts

| Term | Description |
|------|-------------|
| **Repository (Repo)** | A project folder tracked by Git |
| **Commit** | A snapshot of your code at a point in time |
| **Branch** | A parallel version of your code |
| **Remote** | A copy of your repo on a server (GitHub) |
| **Clone** | Download a repo from GitHub to your computer |
| **Push** | Upload your commits to GitHub |
| **Pull** | Download changes from GitHub |
| **Merge** | Combine changes from different branches |

---

## 2. Installation & Setup

### Step 1: Install Git

**Windows:**
```bash
# Download from: https://git-scm.com/download/windows
# Or use winget:
winget install Git.Git
```

**macOS:**
```bash
# Using Homebrew
brew install git

# Or download from: https://git-scm.com/download/mac
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install git
```

### Step 2: Verify Installation
```bash
git --version
# Output: git version 2.40.0 (or similar)
```

### Step 3: Configure Your Identity

This is **required** - Git needs to know who you are:

```bash
# Set your name (use your real name)
git config --global user.name "Bhrigu Anilkumar"

# Set your email (use the same email as your GitHub account)
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

### Step 4: Create GitHub Account

1. Go to [github.com](https://github.com)
2. Click "Sign Up"
3. Choose a professional username
4. Verify your email

### Step 5: Set Up Authentication

GitHub requires authentication. Choose one method:

#### Option A: Personal Access Token (Recommended for beginners)

1. Go to GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name like "My Laptop"
4. Select scopes: `repo`, `workflow`
5. Generate and **copy the token immediately** (you won't see it again!)
6. When Git asks for password, paste this token instead

#### Option B: SSH Keys (Recommended for regular use)

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Press Enter for default location, then set a passphrase (optional)

# Start SSH agent
eval "$(ssh-agent -s)"

# Add your key
ssh-add ~/.ssh/id_ed25519

# Copy public key to clipboard
# Windows:
clip < ~/.ssh/id_ed25519.pub
# macOS:
pbcopy < ~/.ssh/id_ed25519.pub
# Linux:
cat ~/.ssh/id_ed25519.pub
# (then manually copy)
```

Then add to GitHub:
1. Go to GitHub → Settings → SSH and GPG Keys
2. Click "New SSH Key"
3. Paste your public key
4. Save

Test connection:
```bash
ssh -T git@github.com
# Should say: "Hi username! You've successfully authenticated..."
```

---

## 3. Your First Repository

### Understanding the Git Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR COMPUTER                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Working    │    │   Staging    │    │    Local     │       │
│  │  Directory   │───▶│    Area      │───▶│  Repository  │       │
│  │  (files)     │add │  (index)     │commit│   (.git)    │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │               │
└──────────────────────────────────────────────────│───────────────┘
                                                   │ push
                                                   ▼
                                          ┌──────────────┐
                                          │    GitHub    │
                                          │   (Remote)   │
                                          └──────────────┘
```

### Initialize a New Repository

```bash
# Navigate to your project folder
cd hateful-meme-detection

# Initialize Git (creates .git folder)
git init

# Check status
git status
```

Output:
```
On branch main
No commits yet
Untracked files:
  (use "git add <file>..." to include in what will be committed)
        README.md
        src/
        ...
```

---

## 4. Essential Git Commands

### The Big Four: add, commit, push, pull

#### 1. `git add` - Stage Changes

```bash
# Stage a specific file
git add README.md

# Stage multiple files
git add src/model.py src/train.py

# Stage all changes in current directory
git add .

# Stage all changes everywhere
git add -A

# Interactive staging (choose what to add)
git add -p
```

#### 2. `git commit` - Save Snapshot

```bash
# Commit with message
git commit -m "Add model architecture"

# Commit with detailed message (opens editor)
git commit

# Add and commit in one step (only for tracked files)
git commit -am "Fix bug in training loop"
```

**Good Commit Messages:**
```bash
# Format: <type>: <description>

git commit -m "Add: Cross-attention fusion module"
git commit -m "Fix: Memory leak in data loader"
git commit -m "Update: Increase batch size to 32"
git commit -m "Docs: Add API documentation"
git commit -m "Refactor: Simplify inference pipeline"
```

#### 3. `git push` - Upload to GitHub

```bash
# Push to remote (first time, set upstream)
git push -u origin main

# Regular push (after upstream is set)
git push

# Push a specific branch
git push origin feature-branch

# Force push (DANGEROUS - overwrites remote)
git push --force  # Avoid this!
```

#### 4. `git pull` - Download from GitHub

```bash
# Pull latest changes
git pull

# Pull from specific branch
git pull origin main

# Pull and rebase (cleaner history)
git pull --rebase
```

### Viewing Information

```bash
# Check current status
git status

# View commit history
git log

# Compact log (one line per commit)
git log --oneline

# Visual branch graph
git log --oneline --graph --all

# See what changed in a file
git diff README.md

# See staged changes
git diff --staged

# See who changed what (blame)
git blame src/model.py
```

### Undoing Things

```bash
# Unstage a file (keep changes)
git reset HEAD README.md

# Discard changes in working directory (DANGEROUS)
git checkout -- README.md

# Undo last commit (keep changes staged)
git reset --soft HEAD~1

# Undo last commit (keep changes unstaged)
git reset HEAD~1

# Completely undo last commit (DANGEROUS - loses changes)
git reset --hard HEAD~1

# Create a new commit that undoes a previous commit
git revert <commit-hash>
```

---

## 5. Pushing Your Project to GitHub

### Step-by-Step: Upload Hateful Meme Detection Project

#### Step 1: Create Repository on GitHub

1. Go to [github.com](https://github.com) and log in
2. Click the **+** icon → **New repository**
3. Fill in details:
   - **Repository name:** `hateful-meme-detection`
   - **Description:** `Multimodal Hateful Meme Detection using CLIP with Cross-Attention Fusion`
   - **Visibility:** Public (or Private)
   - **DO NOT** initialize with README, .gitignore, or license (we have these already)
4. Click **Create repository**

#### Step 2: Initialize Local Repository

```bash
# Navigate to your project
cd hateful-meme-detection

# Initialize Git
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Complete project structure"
```

#### Step 3: Connect to GitHub

```bash
# Add remote (replace with YOUR username)
git remote add origin https://github.com/YOUR_USERNAME/hateful-meme-detection.git

# Verify remote
git remote -v
```

#### Step 4: Push to GitHub

```bash
# Rename branch to main (if needed)
git branch -M main

# Push and set upstream
git push -u origin main
```

#### Step 5: Verify on GitHub

1. Go to your repository URL
2. You should see all your files!
3. Check that README renders correctly

### If Using SSH Instead of HTTPS

```bash
# Add remote with SSH URL
git remote add origin git@github.com:YOUR_USERNAME/hateful-meme-detection.git
```

---

## 6. Branching & Workflow

### Why Branches?

Branches let you:
- 🧪 Experiment without breaking main code
- 👥 Work on features independently
- 🔀 Merge changes when ready

### Branch Commands

```bash
# List all branches
git branch

# List all branches (including remote)
git branch -a

# Create new branch
git branch feature-name

# Switch to branch
git checkout feature-name

# Create AND switch (shortcut)
git checkout -b feature-name

# Modern way (Git 2.23+)
git switch feature-name
git switch -c new-feature  # create and switch

# Delete branch (local)
git branch -d feature-name

# Delete branch (force, even if not merged)
git branch -D feature-name

# Delete remote branch
git push origin --delete feature-name
```

### Recommended Workflow: Feature Branches

```
main (production-ready code)
  │
  ├── develop (integration branch)
  │     │
  │     ├── feature/add-attention ──────┐
  │     │                               │ merge
  │     ├── feature/improve-training ───┤
  │     │                               │
  │     ├── bugfix/fix-memory-leak ─────┘
  │     │
  │     └── (merge back to main when ready)
  │
  └── hotfix/critical-bug (emergency fixes)
```

### Practical Example

```bash
# You're on main, want to add a new feature
git checkout main
git pull  # Get latest changes

# Create feature branch
git checkout -b feature/add-visualization

# Make changes...
# Edit files, add new code

# Stage and commit
git add .
git commit -m "Add: Training visualization functions"

# Push feature branch to GitHub
git push -u origin feature/add-visualization

# When done, create Pull Request on GitHub (see next section)
```

### Merging Branches

```bash
# Switch to target branch
git checkout main

# Merge feature branch
git merge feature/add-visualization

# Delete merged branch
git branch -d feature/add-visualization
```

### Handling Merge Conflicts

When two branches change the same lines:

```bash
# Try to merge
git merge feature-branch

# Git says: CONFLICT!
# Open the conflicted file, you'll see:
<<<<<<< HEAD
your changes
=======
their changes
>>>>>>> feature-branch

# Manually edit to resolve, then:
git add resolved-file.py
git commit -m "Merge: Resolve conflicts in model.py"
```

---

## 7. Pull Requests

### What is a Pull Request (PR)?

A Pull Request is a GitHub feature that:
- 📝 Proposes changes from one branch to another
- 👀 Allows code review before merging
- 💬 Provides discussion space
- ✅ Can run automated tests

### Creating a Pull Request

#### Step 1: Push Your Branch
```bash
git push -u origin feature/your-feature
```

#### Step 2: Create PR on GitHub

1. Go to your repository on GitHub
2. You'll see a banner: "feature/your-feature had recent pushes"
3. Click **Compare & pull request**
4. Fill in:
   - **Title:** Clear description of changes
   - **Description:** What, why, and how
   - **Reviewers:** Add team members
   - **Labels:** bug, enhancement, documentation, etc.
5. Click **Create pull request**

#### Step 3: Code Review

- Reviewers comment on specific lines
- Discuss changes
- Request modifications if needed
- Approve when ready

#### Step 4: Merge

Options:
- **Merge commit:** Preserves all commits (creates merge commit)
- **Squash and merge:** Combines all commits into one
- **Rebase and merge:** Linear history, no merge commit

### PR Best Practices

1. **Keep PRs small** - Easier to review
2. **One feature per PR** - Don't mix unrelated changes
3. **Write good descriptions** - Help reviewers understand
4. **Respond to feedback** - Be collaborative
5. **Test before creating PR** - Don't waste reviewers' time

### PR Template (Optional)

Create `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
```

---

## 8. Handling Large Files (Git LFS)

### The Problem

Git isn't designed for large binary files:
- Model weights (`.pth`, `.pt`) can be 500MB+
- Datasets can be gigabytes
- Git stores entire file history, bloating repo

### Solution: Git Large File Storage (LFS)

Git LFS replaces large files with text pointers while storing the actual files on a separate server.

### Installing Git LFS

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# Windows (Git for Windows includes it, or):
# Download from https://git-lfs.github.io/
```

### Setting Up Git LFS

```bash
# Initialize LFS (once per machine)
git lfs install

# Navigate to your repo
cd hateful-meme-detection

# Track large file types
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.h5"
git lfs track "*.pkl"
git lfs track "*.bin"

# This creates/updates .gitattributes
cat .gitattributes
# Output:
# *.pth filter=lfs diff=lfs merge=lfs -text
# *.pt filter=lfs diff=lfs merge=lfs -text
# ...

# IMPORTANT: Add .gitattributes to Git
git add .gitattributes
git commit -m "Add: Git LFS tracking for model files"
```

### Using Git LFS

```bash
# Add large file normally
git add models/best_model.pth
git commit -m "Add: Trained model weights"
git push

# LFS handles everything automatically!
```

### Checking LFS Status

```bash
# List tracked patterns
git lfs track

# List LFS files in repo
git lfs ls-files

# Check LFS status
git lfs status
```

### GitHub LFS Limits

| Plan | Storage | Bandwidth/Month |
|------|---------|-----------------|
| Free | 1 GB | 1 GB |
| Pro | 2 GB | 2 GB |
| Data Packs | +50 GB | +50 GB ($5/month) |

### Alternative: External Storage

For very large files, consider:
- **Google Drive** - Link in README
- **Hugging Face Hub** - Great for ML models
- **AWS S3** - Scalable storage
- **GitHub Releases** - Attach files to releases

---

## 9. GitHub Features

### Issues

Track bugs, features, and tasks:

```markdown
# Creating an Issue

**Title:** Model crashes on images larger than 1024px

**Description:**
## Bug Description
The model throws an error when processing large images.

## Steps to Reproduce
1. Load image larger than 1024x1024
2. Call predictor.predict()
3. See error

## Expected Behavior
Should resize and process normally.

## Environment
- OS: Ubuntu 22.04
- Python: 3.10
- PyTorch: 2.0.1
```

### GitHub Actions (CI/CD)

Automated workflows. Your project already has `.github/workflows/ci.yml`!

```yaml
# Example: Run tests on every push
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -e .[dev]
      - run: pytest tests/
```

### GitHub Releases

Package your code with downloadable assets:

1. Go to repository → Releases → Create new release
2. Choose a tag (e.g., `v1.0.0`)
3. Write release notes
4. Attach binary files (model weights)
5. Publish

### GitHub Pages

Host documentation or demo:

1. Go to Settings → Pages
2. Select source branch (usually `main` or `gh-pages`)
3. Your site is at `https://username.github.io/repo-name`

### Useful Repository Settings

- **Branch protection:** Require PR reviews before merging
- **Secrets:** Store API keys securely
- **Webhooks:** Trigger external services on events
- **Collaborators:** Add team members

---

## 10. Best Practices for ML Projects

### Repository Structure

```
project/
├── README.md           # Project overview, setup, usage
├── LICENSE             # Legal stuff
├── requirements.txt    # Dependencies
├── setup.py           # Package installation
├── .gitignore         # What NOT to track
├── .gitattributes     # LFS configuration
│
├── src/               # Source code
├── notebooks/         # Jupyter notebooks
├── tests/             # Unit tests
├── configs/           # Configuration files
├── docs/              # Documentation
│
├── data/              # Data (usually gitignored)
│   └── README.md      # How to get data
│
└── models/            # Saved models (LFS or gitignored)
    └── README.md      # How to get models
```

### What to Commit

✅ **DO commit:**
- Source code
- Configuration files
- Documentation
- Requirements/dependencies
- Small data samples
- Notebooks (cleaned)

❌ **DON'T commit:**
- Large datasets
- Model weights (use LFS or external)
- Credentials/API keys
- Virtual environments
- Cache files
- IDE settings

### .gitignore for ML

Your project already has a comprehensive `.gitignore`. Key patterns:

```gitignore
# Data
data/
*.csv
*.json
!configs/*.json  # But keep config JSONs

# Models
*.pth
*.pt
*.h5
*.pkl

# Python
__pycache__/
*.pyc
venv/
.env

# Notebooks
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
```

### Notebook Best Practices

1. **Clear outputs before committing:**
   ```bash
   jupyter nbconvert --clear-output --inplace notebook.ipynb
   ```

2. **Or use nbstripout:**
   ```bash
   pip install nbstripout
   nbstripout --install  # Auto-strip on commit
   ```

### Versioning Models

Tag releases with model versions:

```bash
# Tag a release
git tag -a v1.0.0 -m "Model v1.0.0 - 64% accuracy"
git push origin v1.0.0

# List tags
git tag

# Checkout specific version
git checkout v1.0.0
```

### Experiment Tracking

Consider integrating:
- **Weights & Biases** (wandb)
- **MLflow**
- **TensorBoard**
- **DVC** (Data Version Control)

---

## 11. Common Issues & Solutions

### "Permission denied (publickey)"

```bash
# SSH key not set up correctly
# Solution: Add SSH key to GitHub (see Section 2)
ssh-add ~/.ssh/id_ed25519
```

### "Failed to push some refs"

```bash
# Remote has changes you don't have
# Solution: Pull first
git pull --rebase
git push
```

### "Your branch is behind"

```bash
# Your local branch is outdated
git pull origin main
```

### "Merge conflict"

```bash
# Edit conflicted files manually
# Look for <<<<<<< and >>>>>>>
# Then:
git add .
git commit -m "Resolve merge conflicts"
```

### "Detached HEAD"

```bash
# You're not on a branch
# Solution: Create or checkout a branch
git checkout main
# Or save your work:
git checkout -b temp-branch
```

### "File too large"

```bash
# File exceeds GitHub's 100MB limit
# Solution: Use Git LFS
git lfs track "*.pth"
git add .gitattributes
git add models/large_model.pth
git commit -m "Add model with LFS"
```

### Accidentally Committed Secrets

```bash
# IMMEDIATELY: Rotate the exposed credential!

# Remove from history (complex, use BFG Repo-Cleaner):
# https://rtyley.github.io/bfg-repo-cleaner/

# Or if recent:
git reset --hard HEAD~1
# Then recommit without the secret
```

### Undo a Push

```bash
# Revert the commit (creates new commit)
git revert HEAD
git push

# Or reset (DANGEROUS - rewrites history):
git reset --hard HEAD~1
git push --force  # Only if you're the only one using the repo!
```

---

## 12. Quick Reference Cheatsheet

### Setup
```bash
git config --global user.name "Name"
git config --global user.email "email@example.com"
git init
git clone <url>
```

### Daily Workflow
```bash
git status                  # Check status
git add .                   # Stage all changes
git commit -m "message"     # Commit
git push                    # Upload to GitHub
git pull                    # Download from GitHub
```

### Branching
```bash
git branch                  # List branches
git checkout -b feature     # Create & switch
git checkout main           # Switch to main
git merge feature           # Merge branch
git branch -d feature       # Delete branch
```

### Viewing History
```bash
git log --oneline          # Compact log
git diff                   # See changes
git show <commit>          # Show commit details
```

### Undoing
```bash
git reset HEAD <file>      # Unstage
git checkout -- <file>     # Discard changes
git reset --soft HEAD~1    # Undo commit (keep changes)
git revert <commit>        # Create undo commit
```

### Remote
```bash
git remote -v              # List remotes
git remote add origin <url>
git push -u origin main    # First push
git fetch                  # Download without merge
```

### LFS
```bash
git lfs install
git lfs track "*.pth"
git lfs ls-files
```

---

## 🚀 Your Action Plan

### Right Now (5 minutes)
1. [ ] Install Git if not installed
2. [ ] Configure name and email
3. [ ] Create GitHub account if needed

### Today (30 minutes)
1. [ ] Set up authentication (token or SSH)
2. [ ] Create repository on GitHub
3. [ ] Push your project

### This Week
1. [ ] Set up Git LFS for model files
2. [ ] Create your first branch
3. [ ] Make a commit from a branch
4. [ ] Create and merge a Pull Request

### Ongoing
1. [ ] Commit frequently with clear messages
2. [ ] Use branches for new features
3. [ ] Review team PRs
4. [ ] Tag releases when ready

---

## 📞 Quick Commands for Your Project

```bash
# === FIRST TIME SETUP ===
cd hateful-meme-detection
git init
git add .
git commit -m "Initial commit: Complete project structure"
git remote add origin https://github.com/YOUR_USERNAME/hateful-meme-detection.git
git branch -M main
git push -u origin main

# === SET UP LFS FOR MODELS ===
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add: Git LFS tracking"
git push

# === DAILY WORKFLOW ===
git pull                              # Get latest
git checkout -b feature/your-feature  # New branch
# ... make changes ...
git add .
git commit -m "Add: Your feature"
git push -u origin feature/your-feature
# Create PR on GitHub, get review, merge

# === AFTER PR IS MERGED ===
git checkout main
git pull
git branch -d feature/your-feature
```

---

**Congratulations!** 🎉 You now have all the knowledge needed to manage your ML project professionally with Git and GitHub.

Remember: The best way to learn Git is by using it. Don't be afraid to make mistakes - that's what branches are for!

---

*Created for the Hateful Meme Detection project by Bhrigu Anilkumar, Deepa Chandrasekar, and Arshpreet Kaur*

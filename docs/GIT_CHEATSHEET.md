# Git Quick Reference Card
## For Hateful Meme Detection Project

### 🚀 FIRST TIME SETUP (Run Once)

```bash
# 1. Navigate to project
cd hateful-meme-detection

# 2. Initialize Git
git init

# 3. Add all files
git add .

# 4. First commit
git commit -m "Initial commit: Complete project structure"

# 5. Connect to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/hateful-meme-detection.git

# 6. Rename branch to main
git branch -M main

# 7. Push to GitHub
git push -u origin main
```

### 📁 SET UP GIT LFS (For Model Files)

```bash
# Install LFS tracking
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add: Git LFS tracking"
git push
```

### 🔄 DAILY WORKFLOW

```bash
# Start of day: Get latest changes
git pull

# Create feature branch
git checkout -b feature/your-feature

# After making changes
git add .
git commit -m "Add: Description of changes"
git push -u origin feature/your-feature

# Then create Pull Request on GitHub!
```

### ✅ AFTER PR IS MERGED

```bash
git checkout main
git pull
git branch -d feature/your-feature
```

### 📝 COMMIT MESSAGE FORMAT

```
Add: New feature description
Fix: Bug description  
Update: What was changed
Docs: Documentation change
Refactor: Code improvement
Test: Test addition
```

### 🔧 COMMON COMMANDS

| Command | What it does |
|---------|--------------|
| `git status` | Check what's changed |
| `git log --oneline` | View commit history |
| `git diff` | See uncommitted changes |
| `git branch` | List branches |
| `git checkout main` | Switch to main |
| `git checkout -b name` | Create new branch |
| `git pull` | Download changes |
| `git push` | Upload changes |

### 🆘 UNDO COMMANDS

```bash
# Unstage a file
git reset HEAD filename

# Discard changes to a file
git checkout -- filename

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes) ⚠️
git reset --hard HEAD~1
```

### 🔗 USEFUL LINKS

- GitHub Repo: `https://github.com/YOUR_USERNAME/hateful-meme-detection`
- Full Tutorial: `docs/GITHUB_TUTORIAL.md`
- Git Documentation: `https://git-scm.com/doc`

---
*Team: Bhrigu Anilkumar, Deepa Chandrasekar, Arshpreet Kaur*

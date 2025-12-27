@echo off
REM ============================================================================
REM Quick Start Script for Hateful Meme Detection Project (Windows)
REM ============================================================================
REM Run this script to set up Git and push to GitHub
REM ============================================================================

echo ==============================================
echo   Hateful Meme Detection - Quick Start
echo ==============================================
echo.

REM Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed!
    echo Please install Git from: https://git-scm.com/download/windows
    pause
    exit /b 1
)
echo [OK] Git is installed

REM Check Git configuration
for /f "tokens=*" %%a in ('git config --global user.name 2^>nul') do set GIT_NAME=%%a
for /f "tokens=*" %%a in ('git config --global user.email 2^>nul') do set GIT_EMAIL=%%a

if "%GIT_NAME%"=="" (
    echo.
    echo Git is not configured. Let's set it up:
    set /p NAME="Enter your name: "
    set /p EMAIL="Enter your email: "
    git config --global user.name "%NAME%"
    git config --global user.email "%EMAIL%"
    echo [OK] Git configured
) else (
    echo [OK] Git configured as: %GIT_NAME%
)

REM Initialize Git if needed
if not exist ".git" (
    echo.
    echo Initializing Git repository...
    git init
    echo [OK] Git repository initialized
) else (
    echo [OK] Already a Git repository
)

REM Check for Git LFS
git lfs version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Git LFS not installed (optional, for large model files)
    echo    Install from: https://git-lfs.github.io/
) else (
    echo [OK] Git LFS is installed
    git lfs install
    
    REM Track large files
    git lfs track "*.pth"
    git lfs track "*.pt"
    git lfs track "*.h5"
    git lfs track "*.pkl"
    echo [OK] LFS tracking configured
)

REM Stage all files
echo.
echo Staging files...
git add .
echo [OK] Files staged

REM Commit
echo.
set /p COMMIT_MSG="Enter commit message (or press Enter for default): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Initial commit: Complete project structure
git commit -m "%COMMIT_MSG%"
echo [OK] Changes committed

REM Check for remote
git remote -v | findstr "origin" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ==============================================
    echo   Connect to GitHub
    echo ==============================================
    echo.
    echo 1. Create a new repository on GitHub:
    echo    https://github.com/new
    echo.
    echo 2. Repository name: hateful-meme-detection
    echo    DON'T initialize with README, .gitignore, or license
    echo.
    set /p GITHUB_USER="Enter your GitHub username: "
    
    git remote add origin "https://github.com/%GITHUB_USER%/hateful-meme-detection.git"
    echo [OK] Remote added
)

REM Rename branch to main
git branch -M main

REM Push
echo.
echo ==============================================
echo   Push to GitHub
echo ==============================================
echo.
set /p PUSH_NOW="Push now? (y/n): "
if /i "%PUSH_NOW%"=="y" (
    git push -u origin main
    echo.
    echo ==============================================
    echo   SUCCESS!
    echo ==============================================
    echo.
    echo Your project is now on GitHub!
) else (
    echo.
    echo To push later, run:
    echo   git push -u origin main
)

echo.
echo ==============================================
echo   Next Steps
echo ==============================================
echo.
echo 1. Add your trained model to models/best_model.pth
echo 2. Update README.md with your GitHub username
echo 3. Set up Discord bot with your token
echo 4. Create a GitHub Release with model weights
echo.
echo For help, see: docs/GITHUB_TUTORIAL.md
echo.
pause

@echo off
title SPARC Installer
echo.
echo  ========================================
echo   SPARC - One-Time Setup
echo  ========================================
echo.

:: Navigate to repo root
cd /d "%~dp0.."

:: Check Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python was not found on your PATH.
    echo  Install Python 3.10+ from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo  [1/3] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
) else (
    echo        .venv already exists — skipping.
)

echo  [2/3] Activating virtual environment...
call .venv\Scripts\activate.bat

echo  [3/3] Installing SPARC and all dependencies...
pip install -e ".[ui]" --quiet

if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] pip install failed. Check the output above.
    pause
    exit /b 1
)

echo.
echo  ========================================
echo   Setup complete!
echo  ========================================
echo.
echo  To start the UI:
echo    scripts\Start_SPARC.bat
echo.
echo  To use the CLI:
echo    .venv\Scripts\activate
echo    sparc --help
echo.
pause

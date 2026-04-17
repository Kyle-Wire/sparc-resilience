@echo off
title SPARC
echo.
echo  ========================================
echo   SPARC - Starting Application...
echo  ========================================
echo.

:: Navigate to the repo root (one level up from scripts/)
cd /d "%~dp0.."

:: Check that sparc package exists
if not exist "sparc\__init__.py" (
    echo  [ERROR] Cannot find sparc package.
    echo  Make sure you are running this from the SPARC repo.
    pause
    exit /b 1
)

:: Launch the desktop app (or fall back to server mode)
echo  Launching SPARC Desktop...
python -m sparc desktop

if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Failed to start. Make sure dependencies are installed.
    echo  Run:  scripts\Install_SPARC.bat
    echo.
    pause
)

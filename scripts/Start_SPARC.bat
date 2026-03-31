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
if not exist "sparc\ui\app.py" (
    echo  [ERROR] Cannot find sparc\ui\app.py
    echo  Make sure you are running this from the SPARC repo.
    pause
    exit /b 1
)

start "" http://localhost:8501
python -m streamlit run run_ui.py --server.headless true --server.port 8501

if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Failed to start. Make sure dependencies are installed.
    echo  Run:  scripts\Install_SPARC.bat
    echo.
    pause
)

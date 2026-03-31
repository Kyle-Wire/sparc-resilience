@echo off
title SPARC
echo.
echo  ========================================
echo   SPARC - Starting Application...
echo  ========================================
echo.

cd /d "%~dp0"

start "" http://localhost:8501
python -m streamlit run sparc/ui/app.py --server.headless true --server.port 8501

if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Failed to start. Make sure Python and Streamlit are installed.
    echo  Run:  pip install streamlit
    echo.
    pause
)

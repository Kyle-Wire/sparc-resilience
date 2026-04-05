@echo off
title SPARC Desktop - Dev Mode
echo.
echo  ========================================
echo   SPARC Desktop - Development Mode
echo  ========================================
echo.

:: Navigate to the repo root (one level up from scripts/)
cd /d "%~dp0.."

:: 1. Start Python FastAPI server in background
echo  Starting Python server on port 8008...
start "SPARC Server" /min cmd /c "python -m sparc server --port 8008"

:: Wait for server to start
echo  Waiting for server...
timeout /t 3 /nobreak >nul

:: 2. Start Tauri dev mode
echo  Starting Tauri dev server...
cd sparc-desktop
pnpm tauri dev

:: When Tauri exits, kill the server
echo  Shutting down server...
taskkill /FI "WINDOWTITLE eq SPARC Server" /F >nul 2>&1
echo  Done.

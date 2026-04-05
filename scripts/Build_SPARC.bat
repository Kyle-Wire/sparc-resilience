@echo off
title SPARC Desktop - Build
echo.
echo  ========================================
echo   SPARC Desktop - Build Release
echo  ========================================
echo.

cd /d "%~dp0.."

:: 1. Build the Python sidecar binary
echo  [1/3] Building Python sidecar...
python build-sidecar.py --target-dir sparc-desktop\src-tauri\binaries
if %errorlevel% neq 0 (
    echo  [ERROR] Sidecar build failed.
    pause
    exit /b 1
)

:: 2. Install frontend dependencies
echo  [2/3] Installing frontend dependencies...
cd sparc-desktop
pnpm install
if %errorlevel% neq 0 (
    echo  [ERROR] pnpm install failed.
    pause
    exit /b 1
)

:: 3. Build Tauri app
echo  [3/3] Building Tauri desktop app...
pnpm tauri build
if %errorlevel% neq 0 (
    echo  [ERROR] Tauri build failed.
    pause
    exit /b 1
)

echo.
echo  ========================================
echo   Build complete!
echo   Output: sparc-desktop\src-tauri\target\release\bundle\
echo  ========================================
echo.
pause

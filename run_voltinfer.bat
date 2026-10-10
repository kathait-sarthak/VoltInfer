@echo off
title VoltInfer ⚡ Energy-Aware Adaptive Runtime
cls
echo =====================================================================
echo    ⚡ VoltInfer: Energy-Aware Adaptive Runtime Proxy Launcher ⚡
echo =====================================================================
echo.

:: Check python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3 is not found in PATH!
    echo Please install Python 3.10+ from python.org and add it to PATH.
    pause
    exit /b 1
)

:: Launch VoltInfer Proxy & Web Dashboard
echo [*] Starting VoltInfer HTTP Proxy Server on http://127.0.0.1:8000 ...
python -m engine.cli start --host 127.0.0.1 --port 8000 --open-browser

pause

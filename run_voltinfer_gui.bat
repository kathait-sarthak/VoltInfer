@echo off
title VoltInfer ⚡ Desktop Control Center
cls

:: Check python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3 is not found in PATH!
    echo Please install Python 3.10+ from python.org and add it to PATH.
    pause
    exit /b 1
)

:: Launch VoltInfer Desktop GUI
echo [*] Launching VoltInfer Desktop Control Center...
start pythonw voltinfer_gui.py

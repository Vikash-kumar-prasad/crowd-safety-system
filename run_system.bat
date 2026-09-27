@echo off
title AI Crowd Safety & Threat Detection System
color 0A

echo =====================================================================
echo    AI-Based Smart Crowd Safety and Threat Detection System
echo =====================================================================
echo.

cd /d "%~dp0"

:: Check virtual environment
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in %~dp0venv
    echo Please create the virtual environment first using:
    echo   python -m venv venv
    echo   venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: Check model weights
if not exist "models\best.pt" (
    echo [WARNING] Head detection model models\best.pt not found!
)
if not exist "models\best_weapon_v4.pt" (
    echo [WARNING] Weapon detection model models\best_weapon_v4.pt not found!
)

echo [1/3] Activating Python virtual environment...
call venv\Scripts\activate.bat

echo [2/3] Launching web browser at http://127.0.0.1:5000 ...
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:5000"

echo [3/3] Starting backend detection engine & Flask server...
echo.
echo Press CTRL+C at any time in this window to stop the system.
echo =====================================================================
echo.

python backend/app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)

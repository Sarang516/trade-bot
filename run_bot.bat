@echo off
:: run_bot.bat — Windows startup script for Zerodha Trading Bot
::
:: Usage:
::   run_bot.bat              (paper trading, NIFTY)
::   run_bot.bat --symbol BANKNIFTY
::   run_bot.bat --paper --no-dashboard
::   run_bot.bat --help
::
:: First run: set your credentials in .env before starting.

setlocal

:: ── locate project root (directory containing this .bat file) ──────────────
set "ROOT=%~dp0"
cd /d "%ROOT%"

:: ── locate Python / venv ──────────────────────────────────────────────────
if exist "%ROOT%venv\Scripts\python.exe" (
    set "PYTHON=%ROOT%venv\Scripts\python.exe"
) else if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT%.venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found. Install Python 3.11+ or activate your venv first.
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

:: ── check .env exists ─────────────────────────────────────────────────────
if not exist "%ROOT%.env" (
    echo [WARNING] .env file not found.
    if exist "%ROOT%.env.example" (
        echo   Copy .env.example to .env and fill in your credentials.
    )
    echo   Running anyway — will use default dummy values.
    echo.
)

:: ── pre-flight checks (quick — skips broker connect) ─────────────────────
echo Running pre-flight checks...
"%PYTHON%" scripts\check_requirements.py
if errorlevel 1 (
    echo [ERROR] Dependency check failed. Run: pip install -r requirements.txt
    pause
    exit /b 1
)

:: ── pass all CLI args straight through to main.py ────────────────────────
echo.
echo Starting trading bot...
echo   Args: %*
echo.

"%PYTHON%" main.py %*

set EXIT_CODE=%errorlevel%

if %EXIT_CODE% neq 0 (
    echo.
    echo [ERROR] Bot exited with code %EXIT_CODE%. Check logs\ for details.
    pause
)

endlocal
exit /b %EXIT_CODE%

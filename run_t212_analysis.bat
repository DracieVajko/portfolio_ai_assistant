@echo off
title Trading212 - Portfolio Analysis
cd /d "%~dp0"

echo.
echo ==========================================
echo  Trading212 Portfolio Analysis
echo ==========================================
echo.

REM -- health check --
echo [1/2] Health check...
py -3.12 trading212_integration.py --health --config api.env
if errorlevel 1 (
    echo [ERROR] Trading212 connection failed.
    pause
    exit /b 1
)

echo.
echo [2/2] Running AI analysis (odysseus/Ollama)...
py -3.12 trading212_integration.py --analyze --config api.env
if errorlevel 1 (
    echo [ERROR] AI analysis failed.
    pause
    exit /b 1
)

echo.
pause

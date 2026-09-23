@echo off
title Portfolio Analysis - Full Report
cd /d "%~dp0"

echo Running full portfolio analysis...
py -3.12 portfolio_ai_assistant.py --config portfolio_config.json --investment-engine --generate-full-report

if errorlevel 1 (
    echo.
    echo [ERROR] Portfolio analysis failed. Check the messages above.
    pause
    exit /b 1
)

echo.
echo Report generation finished. See Python output above for the actual saved paths.
pause

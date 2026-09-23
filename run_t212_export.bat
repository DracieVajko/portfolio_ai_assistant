@echo off
title Trading212 - Export Portfolio
cd /d "%~dp0"

if not exist reports mkdir reports

echo Exporting Trading212 portfolio to JSON...
py -3.12 trading212_integration.py --export reports\t212_portfolio.json --config api.env

if errorlevel 1 (
    echo [ERROR] Export failed.
    pause
    exit /b 1
)

echo.
echo Exported: reports\t212_portfolio.json
pause

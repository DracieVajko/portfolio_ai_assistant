@echo off
cd /d "%~dp0"

REM Task Scheduler variant: no pause, and Python owns logs\latest_run.log.
REM Models are auto-routed (decision=deepseek-r1-finance-reasoning-14b,
REM summary=google/gemma-4-12b) and auto-loaded on demand by LM Studio.
py -3.12 portfolio_ai_assistant.py --config portfolio_config.json --investment-engine --generate-full-report
exit /b %ERRORLEVEL%

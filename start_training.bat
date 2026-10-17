@echo off
rem One-click start: double-click to train with the settings in tmai_settings.json.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_training.ps1"
pause

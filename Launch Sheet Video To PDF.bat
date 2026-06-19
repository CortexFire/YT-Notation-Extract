@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher "py" was not found.
  echo Install Python, then try again.
  echo.
  pause
  exit /b 1
)

py windows_app.py
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="9009" (
  echo.
  echo Python could not launch the app.
  echo.
  pause
)

exit /b %EXIT_CODE%

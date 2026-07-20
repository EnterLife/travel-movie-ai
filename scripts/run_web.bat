@echo off
setlocal

cd /d "%~dp0.."

set "PYTHON_EXE=.venv\Scripts\python.exe"
set "NEED_SETUP="

if not exist "%PYTHON_EXE%" (
  echo TravelMovieAI is not configured yet.
  set "NEED_SETUP=1"
)

if not defined NEED_SETUP (
  "%PYTHON_EXE%" -c "import travelmovieai, fastapi, uvicorn, cv2, scenedetect" >nul 2>&1
  if errorlevel 1 set "NEED_SETUP=1"
)

if defined NEED_SETUP (
  echo The base web and CPU video environment is incomplete. Running setup...
  call scripts\setup_windows.bat --base-only --non-interactive
  if errorlevel 1 goto :error
)

echo Starting TravelMovieAI...
"%PYTHON_EXE%" main.py %*
if errorlevel 1 goto :error

endlocal
exit /b 0

:error
echo.
echo TravelMovieAI failed to start.
echo Run scripts\setup_windows.bat and review its diagnostics.
pause
endlocal
exit /b 1

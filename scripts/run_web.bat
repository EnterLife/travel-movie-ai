@echo off
setlocal

cd /d "%~dp0.."

set "PYTHON_EXE=.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo TravelMovieAI is not configured yet.
  call scripts\setup_windows.bat --runtime-only
  if errorlevel 1 goto :error
)

"%PYTHON_EXE%" -c "import accelerate, travelmovieai, fastapi, uvicorn, cv2, scenedetect, torch, transformers" >nul 2>&1
if errorlevel 1 (
  echo The virtual environment is incomplete or lacks local AI support. Running setup...
  call scripts\setup_windows.bat --runtime-only
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

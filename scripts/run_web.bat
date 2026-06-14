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
  "%PYTHON_EXE%" -c "import accelerate, bitsandbytes, travelmovieai, fastapi, uvicorn, cv2, scenedetect, torch, transformers" >nul 2>&1
  if errorlevel 1 set "NEED_SETUP=1"
)

if not defined NEED_SETUP (
  where nvidia-smi >nul 2>&1
  if not errorlevel 1 (
    "%PYTHON_EXE%" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if errorlevel 1 set "NEED_SETUP=1"
  )
)

if defined NEED_SETUP (
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

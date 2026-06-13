@echo off
setlocal

cd /d "%~dp0.."

set "PYTHON_EXE=.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo Creating Python virtual environment...
  python -m venv .venv
  if errorlevel 1 goto :error
)

"%PYTHON_EXE%" -c "import travelmovieai, fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
  echo Installing TravelMovieAI dependencies...
  "%PYTHON_EXE%" -m pip install -e .
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
echo Check that Python 3.12 or newer is installed and available on PATH.
pause
endlocal
exit /b 1

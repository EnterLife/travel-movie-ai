@echo off
setlocal EnableExtensions

cd /d "%~dp0.."

if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help

set "RUNTIME_DIR=%~1"
if not defined RUNTIME_DIR set "RUNTIME_DIR=.cache\ace-step"
set "ACE_STEP_REVISION=dce621408bee8c31b4fcf4811682eb9359e1bc94"

if not exist ".venv\Scripts\python.exe" (
  echo TravelMovieAI virtual environment is missing.
  echo Run scripts\setup_windows.bat first.
  exit /b 1
)

where git >nul 2>&1
if errorlevel 1 (
  where winget >nul 2>&1
  if errorlevel 1 (
    echo Git is required to install ACE-Step.
    exit /b 1
  )
  echo Installing Git...
  winget install --id Git.Git --exact --accept-package-agreements --accept-source-agreements
  if errorlevel 1 exit /b 1
)

if not exist ".venv\Scripts\uv.exe" (
  echo Installing uv package manager...
  ".venv\Scripts\python.exe" -m pip install "uv>=0.7,<1"
  if errorlevel 1 exit /b 1
)

if not exist "%RUNTIME_DIR%\.git" (
  echo Downloading ACE-Step 1.5 runtime...
  git init "%RUNTIME_DIR%"
  if errorlevel 1 exit /b 1
  git -C "%RUNTIME_DIR%" remote add origin https://github.com/ACE-Step/ACE-Step-1.5.git
  if errorlevel 1 exit /b 1
  git -C "%RUNTIME_DIR%" fetch --depth 1 origin %ACE_STEP_REVISION%
  if errorlevel 1 exit /b 1
  git -C "%RUNTIME_DIR%" checkout --detach FETCH_HEAD
  if errorlevel 1 exit /b 1
)

echo Installing isolated ACE-Step dependencies...
".venv\Scripts\uv.exe" sync --project "%RUNTIME_DIR%"
if errorlevel 1 exit /b 1

echo ACE-Step runtime is ready: %RUNTIME_DIR%
endlocal
exit /b 0

:help
echo Usage:
echo   scripts\setup_music_ai.bat [runtime-directory]
echo.
echo Installs ACE-Step 1.5 into an isolated environment.
echo Model weights are downloaded automatically during the first generation.
endlocal
exit /b 0

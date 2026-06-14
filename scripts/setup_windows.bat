@echo off
setlocal EnableExtensions

cd /d "%~dp0.."

if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help

set "INSTALL_SPEC=.[all,dev]"
if /i "%~1"=="--runtime-only" set "INSTALL_SPEC=.[all]"

echo.
echo TravelMovieAI Windows setup
echo ===========================
echo.

call :find_python
if not defined SYSTEM_PYTHON (
  call :install_python
  if errorlevel 1 goto :error
  call :find_python
)

if not defined SYSTEM_PYTHON (
  echo Python 3.12 was installed but is not visible in this terminal.
  echo Close this window, open a new terminal, and run this script again.
  goto :error
)

echo Python: %SYSTEM_PYTHON%

call :ensure_ffmpeg
if errorlevel 1 goto :error

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo Creating virtual environment...
  "%SYSTEM_PYTHON%" -m venv .venv
  if errorlevel 1 goto :error
)

set "PYTHON_EXE=.venv\Scripts\python.exe"
"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
if errorlevel 1 (
  echo Existing .venv uses an unsupported Python version.
  echo Remove .venv and run this script again.
  goto :error
)

echo.
echo Updating packaging tools...
"%PYTHON_EXE%" -m pip install --upgrade pip wheel "setuptools<82"
if errorlevel 1 goto :error

call :ensure_pytorch_cuda
if errorlevel 1 goto :error

echo.
echo Installing TravelMovieAI dependencies: %INSTALL_SPEC%
"%PYTHON_EXE%" -m pip install -e "%INSTALL_SPEC%"
if errorlevel 1 goto :error

if not exist ".env" (
  copy /y ".env.example" ".env" >nul
  echo Created .env from .env.example.
) else (
  echo Existing .env was preserved.
)

echo.
echo Verifying Python dependencies...
"%PYTHON_EXE%" -c "import accelerate, cv2, fastapi, faster_whisper, faiss, huggingface_hub, safetensors, scenedetect, sentence_transformers, torch, transformers, travelmovieai, uvicorn"
if errorlevel 1 goto :error

echo.
echo Verifying external tools...
ffmpeg -version >nul 2>&1
if errorlevel 1 goto :ffmpeg_restart
ffprobe -version >nul 2>&1
if errorlevel 1 goto :ffmpeg_restart

echo.
echo Setup completed successfully.
echo Start the application with:
echo   scripts\run_web.bat
echo.
endlocal
exit /b 0

:ensure_pytorch_cuda
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo NVIDIA GPU was not detected. PyTorch will use the standard CPU package.
  exit /b 0
)

"%PYTHON_EXE%" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 (
  echo CUDA-enabled PyTorch is already available.
  exit /b 0
)

echo.
echo NVIDIA GPU detected. Installing CUDA-enabled PyTorch...
"%PYTHON_EXE%" -m pip uninstall --yes torch torchvision torchaudio
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -c "import torch; available = torch.cuda.is_available(); print('PyTorch', torch.__version__, '| CUDA runtime', torch.version.cuda, '| GPU', torch.cuda.get_device_name(0) if available else 'unavailable'); raise SystemExit(0 if available else 1)"
if errorlevel 1 (
  echo A CUDA PyTorch wheel was installed, but the GPU is still unavailable.
  echo Check the NVIDIA driver, restart Windows, and run this setup again.
  exit /b 1
)
exit /b 0

:find_python
set "SYSTEM_PYTHON="
for /f "usebackq delims=" %%I in (`py -3.12 -c "import sys; print(sys.executable)" 2^>nul`) do (
  set "SYSTEM_PYTHON=%%I"
)
if defined SYSTEM_PYTHON exit /b 0

for /f "usebackq delims=" %%I in (`python -c "import sys; print(sys.executable if sys.version_info >= (3, 12) else '')" 2^>nul`) do (
  if not "%%I"=="" set "SYSTEM_PYTHON=%%I"
)
exit /b 0

:install_python
where winget >nul 2>&1
if errorlevel 1 (
  echo Python 3.12 or newer was not found, and winget is unavailable.
  echo Install Python 3.12 manually from https://www.python.org/downloads/
  exit /b 1
)
echo Python 3.12 was not found. Installing it with winget...
winget install --id Python.Python.3.12 --exact --accept-package-agreements --accept-source-agreements
if errorlevel 1 exit /b 1
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
  set "SYSTEM_PYTHON=%LocalAppData%\Programs\Python\Python312\python.exe"
)
exit /b 0

:ensure_ffmpeg
where ffmpeg >nul 2>&1
if errorlevel 1 goto :install_ffmpeg
where ffprobe >nul 2>&1
if errorlevel 1 goto :install_ffmpeg
echo FFmpeg and FFprobe are available.
exit /b 0

:install_ffmpeg
where winget >nul 2>&1
if errorlevel 1 (
  echo FFmpeg or FFprobe was not found, and winget is unavailable.
  echo Install FFmpeg manually and add its bin directory to PATH.
  exit /b 1
)
echo FFmpeg was not found. Installing Gyan.FFmpeg with winget...
winget install --id Gyan.FFmpeg --exact --accept-package-agreements --accept-source-agreements
if errorlevel 1 exit /b 1

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$root = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'; Get-ChildItem $root -Directory -Filter 'Gyan.FFmpeg_*' -ErrorAction SilentlyContinue ^| Get-ChildItem -Directory -Recurse -Filter bin -ErrorAction SilentlyContinue ^| Where-Object { Test-Path (Join-Path $_.FullName 'ffmpeg.exe') } ^| Select-Object -First 1 -ExpandProperty FullName"`) do (
  set "PATH=%%I;%PATH%"
)
exit /b 0

:ffmpeg_restart
echo.
echo FFmpeg was installed but is not visible in this terminal.
echo Close this window, open a new terminal, and run this script again.
goto :error

:help
echo Usage:
echo   scripts\setup_windows.bat
echo   scripts\setup_windows.bat --runtime-only
echo.
echo Default mode installs all runtime, AI, media, and development dependencies.
echo --runtime-only skips pytest, Ruff, mypy, and other development tools.
endlocal
exit /b 0

:error
echo.
echo TravelMovieAI setup failed.
echo Review the message above, then run scripts\setup_windows.bat again.
echo.
endlocal
exit /b 1

@echo off
setlocal EnableExtensions

cd /d "%~dp0.."

if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help

set "INSTALL_SPEC=.[all,dev]"
set "INSTALL_MUSIC_AI=1"
set "MUSIC_AI_ONLY="
set "ACE_STEP_RUNTIME=.cache\ace-step"
set "ACE_STEP_REVISION=dce621408bee8c31b4fcf4811682eb9359e1bc94"

if /i "%~1"=="--runtime-only" set "INSTALL_SPEC=.[all]"
if /i "%~2"=="--runtime-only" set "INSTALL_SPEC=.[all]"
if /i "%~1"=="--skip-music-ai" set "INSTALL_MUSIC_AI="
if /i "%~2"=="--skip-music-ai" set "INSTALL_MUSIC_AI="
if /i "%~1"=="--music-ai-only" (
  set "MUSIC_AI_ONLY=1"
  if not "%~2"=="" set "ACE_STEP_RUNTIME=%~2"
)

echo.
echo TravelMovieAI Windows setup
echo ===========================
echo.

if defined MUSIC_AI_ONLY goto :music_ai_only

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

if defined INSTALL_MUSIC_AI (
  call :ensure_music_ai
  if errorlevel 1 goto :error
) else (
  echo.
  echo Skipping the ACE-Step music runtime.
)

if not exist "configs\settings.toml" (
  echo Required configuration file configs\settings.toml was not found.
  goto :error
)
echo Configuration: configs\settings.toml

echo.
echo Verifying Python dependencies...
"%PYTHON_EXE%" -c "import accelerate, bitsandbytes, cv2, fastapi, faster_whisper, faiss, huggingface_hub, json_repair, safetensors, scenedetect, sentence_transformers, torch, transformers, travelmovieai, uvicorn"
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
pause
exit /b 0

:music_ai_only
if not exist ".venv\Scripts\python.exe" (
  echo TravelMovieAI virtual environment is missing.
  echo Run scripts\setup_windows.bat first.
  goto :error
)
set "PYTHON_EXE=.venv\Scripts\python.exe"
call :ensure_music_ai
if errorlevel 1 goto :error
echo.
echo ACE-Step music runtime setup completed successfully.
endlocal
pause
exit /b 0

:ensure_music_ai
call :ensure_git
if errorlevel 1 exit /b 1

if not exist ".venv\Scripts\uv.exe" (
  echo.
  echo Installing uv package manager...
  "%PYTHON_EXE%" -m pip install "uv>=0.7,<1"
  if errorlevel 1 exit /b 1
)

set "ACE_STEP_CURRENT_REVISION="
if exist "%ACE_STEP_RUNTIME%\.git" (
  for /f "usebackq delims=" %%I in (`git -C "%ACE_STEP_RUNTIME%" rev-parse HEAD 2^>nul`) do (
    set "ACE_STEP_CURRENT_REVISION=%%I"
  )
)

if /i not "%ACE_STEP_CURRENT_REVISION%"=="%ACE_STEP_REVISION%" (
  echo.
  echo Downloading ACE-Step 1.5 runtime...
  if not exist "%ACE_STEP_RUNTIME%\.git" (
    git init "%ACE_STEP_RUNTIME%"
    if errorlevel 1 exit /b 1
    git -C "%ACE_STEP_RUNTIME%" remote add origin https://github.com/ACE-Step/ACE-Step-1.5.git
    if errorlevel 1 exit /b 1
  )
  git -C "%ACE_STEP_RUNTIME%" fetch --depth 1 origin %ACE_STEP_REVISION%
  if errorlevel 1 exit /b 1
  git -C "%ACE_STEP_RUNTIME%" checkout --detach FETCH_HEAD
  if errorlevel 1 exit /b 1
)

echo.
echo Installing isolated ACE-Step dependencies...
".venv\Scripts\uv.exe" sync --project "%ACE_STEP_RUNTIME%"
if errorlevel 1 exit /b 1
echo ACE-Step runtime is ready: %ACE_STEP_RUNTIME%
exit /b 0

:ensure_git
where git >nul 2>&1
if not errorlevel 1 exit /b 0

where winget >nul 2>&1
if errorlevel 1 (
  echo Git is required to install ACE-Step, and winget is unavailable.
  exit /b 1
)
echo.
echo Git was not found. Installing it with winget...
winget install --id Git.Git --exact --accept-package-agreements --accept-source-agreements
if errorlevel 1 exit /b 1

if exist "%ProgramFiles%\Git\cmd\git.exe" set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
if exist "%LocalAppData%\Programs\Git\cmd\git.exe" set "PATH=%LocalAppData%\Programs\Git\cmd;%PATH%"
where git >nul 2>&1
if errorlevel 1 (
  echo Git was installed but is not visible in this terminal.
  echo Close this window, open a new terminal, and run setup again.
  exit /b 1
)
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
echo   scripts\setup_windows.bat --skip-music-ai
echo   scripts\setup_windows.bat --music-ai-only [runtime-directory]
echo.
echo Default mode installs the application, local AI dependencies, and the
echo isolated ACE-Step music runtime.
echo --runtime-only skips pytest, Ruff, mypy, and other development tools.
echo --skip-music-ai skips the ACE-Step runtime.
echo --music-ai-only repairs or installs only the isolated ACE-Step runtime.
endlocal
pause
exit /b 0

:error
echo.
echo TravelMovieAI setup failed.
echo Review the message above, then run scripts\setup_windows.bat again.
echo.
endlocal
pause
exit /b 1

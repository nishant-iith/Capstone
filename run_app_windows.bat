@echo off
setlocal

cd /d "%~dp0"

if not exist models\v20_fixed_model.pth (
  echo Missing models\v20_fixed_model.pth
  echo Put the v20_fixed weights in models\v20_fixed_model.pth and run again.
  pause
  exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer, then run this file again.
  echo https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist .venv_app (
  echo Creating app environment...
  python -m venv .venv_app
)

call .venv_app\Scripts\activate.bat
if not exist .venv_app\.deps_installed (
  python -m pip install --upgrade pip
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  python -m pip install -r requirements_app.txt
  type nul > .venv_app\.deps_installed
)

python v20_stain_app.py %*

pause

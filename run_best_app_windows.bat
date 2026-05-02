@echo off
setlocal

cd /d "%~dp0"

set "MISSING=0"
if not exist models\v20_fixed_model.pth (
  echo Warning: missing models\v20_fixed_model.pth
  set "MISSING=1"
)
if not exist models\v22a_content_quality_top1000_ft_model.pth (
  echo Warning: missing models\v22a_content_quality_top1000_ft_model.pth
  set "MISSING=1"
)
if not exist models\v21b_hibou_content_quality_ft_model.pth (
  echo Warning: missing models\v21b_hibou_content_quality_ft_model.pth
  set "MISSING=1"
)
if "%MISSING%"=="1" (
  echo The balanced/best_ssim ensemble needs all three weights.
  echo v22_tta can still run with only models\v22a_content_quality_top1000_ft_model.pth.
)

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer, then run this file again.
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

python best_stain_app.py %*

pause

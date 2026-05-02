@echo off
setlocal

cd /d "%~dp0"

for %%F in (
  models\v20_fixed_model.pth
  models\v22a_content_quality_top1000_ft_model.pth
  models\v21b_hibou_content_quality_ft_model.pth
) do (
  if not exist %%F (
    echo Missing %%F
    pause
    exit /b 1
  )
)

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer, then run again.
  pause
  exit /b 1
)

if not exist .venv_build (
  python -m venv .venv_build
)

set "EXTRA_HIBOU="
if exist models\hibou-b set "EXTRA_HIBOU=--add-data models\hibou-b;models\hibou-b"

call .venv_build\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements_app.txt
python -m pip install pyinstaller

pyinstaller ^
  --noconfirm ^
  --clean ^
  --onedir ^
  --windowed ^
  --name VirtualHEBest ^
  --collect-all segmentation_models_pytorch ^
  --collect-all timm ^
  --collect-all torchvision ^
  --collect-all transformers ^
  --collect-all huggingface_hub ^
  --collect-all safetensors ^
  --collect-all tokenizers ^
  %EXTRA_HIBOU% ^
  --add-data "models\v20_fixed_model.pth;models" ^
  --add-data "models\v22a_content_quality_top1000_ft_model.pth;models" ^
  --add-data "models\v21b_hibou_content_quality_ft_model.pth;models" ^
  --add-data "best_model_manifest.json;." ^
  best_stain_app.py

copy APP_DISTRIBUTION.md dist\VirtualHEBest\README.md >nul

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "if (Test-Path dist\VirtualHEBest.zip) { Remove-Item dist\VirtualHEBest.zip }; Compress-Archive -Path dist\VirtualHEBest\* -DestinationPath dist\VirtualHEBest.zip"

echo.
echo Portable best app built:
echo   dist\VirtualHEBest\VirtualHEBest.exe
echo   dist\VirtualHEBest.zip
echo.
echo Note: best ensemble mode needs Hibou-B model code/config available to transformers.
echo If models\hibou-b exists, this script bundles it. Otherwise use v22A or v20 mode.
pause

@echo off
setlocal

cd /d "%~dp0"

if not exist models\v20_fixed_model.pth (
  echo Missing models\v20_fixed_model.pth
  echo Put the final v20_fixed weights there before building.
  pause
  exit /b 1
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
  --name VirtualHE ^
  --collect-all segmentation_models_pytorch ^
  --collect-all timm ^
  --collect-all torchvision ^
  --add-data "models\v20_fixed_model.pth;models" ^
  v20_stain_app.py

copy APP_DISTRIBUTION.md dist\VirtualHE\README.md >nul

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "if (Test-Path dist\VirtualHE.zip) { Remove-Item dist\VirtualHE.zip }; Compress-Archive -Path dist\VirtualHE\* -DestinationPath dist\VirtualHE.zip"

echo.
echo Portable app built:
echo   dist\VirtualHE\VirtualHE.exe
echo   dist\VirtualHE.zip
echo.
echo Share the ZIP. Users unzip it and double-click VirtualHE.exe.
pause

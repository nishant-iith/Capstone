@echo off
echo ============================================
echo  Virtual H^&E Stain Generator -- Build App
echo ============================================
echo.

REM Install PyInstaller if missing
pip install pyinstaller --quiet
if errorlevel 1 (
    echo ERROR: pip install pyinstaller failed. Make sure Python is in PATH.
    pause
    exit /b 1
)

REM Clean previous build
echo Cleaning previous build...
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

REM Build
echo Building executable (this takes 3-10 minutes)...
pyinstaller app.py ^
  --name VirtualHEStain ^
  --onedir ^
  --windowed ^
  --add-data "src;src" ^
  --add-data "app_inference.py;." ^
  --additional-hooks-dir "." ^
  --hidden-import torch ^
  --hidden-import torchvision ^
  --hidden-import torchvision.models ^
  --hidden-import torchvision.models.resnet ^
  --hidden-import pytorch_lightning ^
  --hidden-import PIL.ImageTk ^
  --hidden-import PIL.Image ^
  --hidden-import numpy ^
  --hidden-import src.models.gan ^
  --hidden-import src.training.lightning_module_v10 ^
  --hidden-import src.data.dataset ^
  --hidden-import src.validation.metrics ^
  --exclude-module pandas ^
  --exclude-module matplotlib ^
  --exclude-module scikit_image ^
  --exclude-module tensorboard ^
  --exclude-module notebook ^
  --exclude-module IPython ^
  --clean ^
  --noconfirm

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed. Check output above.
    pause
    exit /b 1
)

REM Create checkpoints folder inside dist
echo.
echo Creating checkpoints folder...
mkdir dist\VirtualHEStain\checkpoints 2>nul
echo Place your .ckpt file here > dist\VirtualHEStain\checkpoints\PUT_CHECKPOINT_FILE_HERE.txt

REM Zip the output
echo Zipping output folder...
powershell -Command "Compress-Archive -Path 'dist\VirtualHEStain' -DestinationPath 'dist\VirtualHEStain_v1.zip' -Force"

echo.
echo ============================================
echo  BUILD COMPLETE
echo ============================================
echo.
echo Output folder: dist\VirtualHEStain\
echo Zip file:      dist\VirtualHEStain_v1.zip
echo.
echo DISTRIBUTE:
echo   1. Share VirtualHEStain_v1.zip
echo   2. Share v7-epoch=022-val_ssim=0.2674.ckpt  (538 MB, separately)
echo.
echo USER INSTRUCTIONS:
echo   1. Extract VirtualHEStain_v1.zip
echo   2. Place .ckpt file into VirtualHEStain\checkpoints\
echo   3. Double-click VirtualHEStain.exe
echo.
pause

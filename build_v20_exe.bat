@echo off
setlocal

REM Build a Windows one-file executable for the v20 Virtual H&E app.
REM Run this from the repository root on the target Windows machine.
REM
REM Notes:
REM - The exe will be large because PyTorch and the model weights are bundled.
REM - If models\v20_fixed_model.pth exists, it is bundled first.
REM - Otherwise, copy your chosen checkpoint to models\v20_fixed_model.pth before building.

if not exist models\v20_fixed_model.pth (
  echo models\v20_fixed_model.pth not found.
  echo Copy the final fixed v20 checkpoint there before building:
  echo   copy checkpoints\v20_fixed\YOUR_BEST_CHECKPOINT.pth models\v20_fixed_model.pth
  exit /b 1
)

python -m pip install pyinstaller
pyinstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name VirtualHE-v20 ^
  --add-data "models\v20_fixed_model.pth;models" ^
  v20_stain_app.py

echo.
echo Build complete. Check dist\VirtualHE-v20.exe

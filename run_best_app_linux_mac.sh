#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

missing=0
for path in \
  models/v20_fixed_model.pth \
  models/v22a_content_quality_top1000_ft_model.pth \
  models/v21b_hibou_content_quality_ft_model.pth
do
  if [ ! -f "$path" ]; then
    echo "Warning: missing $path"
    missing=1
  fi
done

if [ "$missing" -eq 1 ]; then
  echo "The balanced/best_ssim ensemble needs all three weights."
  echo "v22_tta can still run with only models/v22a_content_quality_top1000_ft_model.pth."
fi

if [ ! -d .venv_app ]; then
  echo "Creating app environment..."
  python3 -m venv .venv_app
fi

source .venv_app/bin/activate
if [ ! -f .venv_app/.deps_installed ]; then
  python -m pip install --upgrade pip
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  python -m pip install -r requirements_app.txt
  touch .venv_app/.deps_installed
fi

python best_stain_app.py "$@"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f models/v20_fixed_model.pth ]; then
  echo "Missing models/v20_fixed_model.pth"
  echo "Put the v20_fixed weights in models/v20_fixed_model.pth and run again."
  exit 1
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

python v20_stain_app.py "$@"

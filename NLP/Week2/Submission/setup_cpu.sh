#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r task2/requirements.txt
.venv/bin/python -m ipykernel install --prefix tmp/jupyter --name neurova-lstm --display-name 'Python 3 (Neurova LSTM)'
printf '%s\n' 'Environment ready. Select .venv/bin/python in your notebook editor.'

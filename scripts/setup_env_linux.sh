#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${ROOT_DIR}/.venv"

python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio
python -m pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
python -m pip install datasets trl accelerate peft bitsandbytes sentencepiece protobuf huggingface_hub

echo "Environment ready at ${VENV_PATH}"

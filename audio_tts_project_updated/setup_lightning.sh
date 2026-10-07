#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=============================================="
echo " XTTS V2 - Lightning AI setup"
echo "=============================================="
python --version

# IMPORTANT:
# Lightning's current Studio image commonly uses Python 3.12.
# The old `TTS` package is limited to Python <3.12, so we use the
# maintained `coqui-tts` fork. It still exposes `from TTS.api import TTS`.
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade -r requirements-lightning-minimal.txt

echo ""
echo "Installing optional F5-TTS engine..."
if python -m pip install --upgrade "f5-tts==1.1.22"; then
  echo "✓ F5-TTS installed."
else
  echo "⚠ F5-TTS installation failed; XTTS remains usable."
fi

python - <<'PY'
import sys
import torch
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

from TTS.api import TTS
print("Coqui TTS import: OK")
PY

echo ""
echo "Setup completed successfully."
echo "Run: ./run_lightning.sh"

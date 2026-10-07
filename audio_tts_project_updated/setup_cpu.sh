#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=============================================="
echo " XTTS + SILMA — CPU Studio setup"
echo " 4 vCPU / 16 GB RAM"
echo "=============================================="
python --version

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade -r requirements-cpu.txt

# FFmpeg is required for converting uploaded audio formats.
if command -v ffmpeg >/dev/null 2>&1; then
  echo "✓ FFmpeg: $(command -v ffmpeg)"
else
  echo "⚠ FFmpeg is not installed. Audio uploads/conversion will not work until FFmpeg is available."
fi

python - <<'PY'
import sys, torch
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CPU threads:", torch.get_num_threads())
from TTS.api import TTS
print("Coqui TTS import: OK")
try:
    from silma_tts.api import SilmaTTS
    print("SILMA TTS import: OK")
except Exception as e:
    print("SILMA TTS import: unavailable:", e)
PY

echo ""
echo "Setup completed."
echo "Start with: ./run_cpu.sh"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export FORCE_CPU=1
export CPU_THREADS="${CPU_THREADS:-4}"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export PORT="${PORT:-8000}"

echo "=============================================="
echo " XTTS V2 Voice Studio — CPU MODE"
echo " CPU threads: $CPU_THREADS"
echo " Port: $PORT"
echo "=============================================="

python - <<'PY'
import torch
from TTS.api import TTS
torch.set_num_threads(4)
print("Python/Torch check: OK")
print("CUDA available:", torch.cuda.is_available())
print("Forced device: CPU")
print("Torch CPU threads:", torch.get_num_threads())
PY

exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1 --no-access-log

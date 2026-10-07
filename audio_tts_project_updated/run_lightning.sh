#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export FORCE_CPU="${FORCE_CPU:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PORT="${PORT:-8000}"

python - <<'PY'
try:
    import torch
    from TTS.api import TTS
except ModuleNotFoundError as e:
    print("\n[ERROR] XTTS dependencies are not installed.")
    print("Run this first: ./setup_lightning.sh\n")
    raise SystemExit(2)

print("Python/Torch check: OK")
print("CUDA:", torch.cuda.is_available())
print("FORCE_CPU:", __import__("os").getenv("FORCE_CPU", "1"))
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --no-access-log

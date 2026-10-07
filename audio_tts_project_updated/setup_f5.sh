#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "=========================================="
echo " تثبيت SILMA TTS v1 — عربي/إنجليزي — بدون RVC"
echo "=========================================="
python -m pip install -U pip setuptools wheel
python -m pip install -U "silma-tts>=1.0.0"
python - <<'PY'
from silma_tts.api import SilmaTTS
print("✓ تم تثبيت SILMA TTS بنجاح")
PY
echo "✓ XTTS + SILMA TTS جاهزان. أعد تشغيل السيرفر."

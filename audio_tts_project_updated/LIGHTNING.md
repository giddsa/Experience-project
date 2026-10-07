# تشغيل XTTS V2 على Lightning AI

## مهم جدًا
هذا المشروع يستخدم `coqui-tts` بدل الحزمة القديمة `TTS==0.22.0`.

السبب: بيئة Lightning Studio عندك تستخدم Python 3.12، بينما نسخة Coqui القديمة لا تدعم Python 3.12.

## أول تشغيل
من داخل المجلد الذي يحتوي `run_lightning.sh`:

```bash
chmod +x setup_lightning.sh run_lightning.sh
./setup_lightning.sh
```

بعد نجاح التثبيت:

```bash
./run_lightning.sh
```

## إذا أردت فحص GPU فقط

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

يجب أن تكون النتيجة الأولى `True` إذا كان Studio يعمل بــGPU.

## فتح الموقع
التطبيق يستمع على:

```text
0.0.0.0:$PORT
```

ويستخدم `8000` كقيمة افتراضية إذا لم تحدد Lightning متغير `PORT`.

في Lightning استخدم Ports/Port 8000 لفتح التطبيق.

## ملاحظة
أول تشغيل لـXTTS قد يستغرق وقتًا لأن النموذج يحتاج إلى تنزيله وتحميله إلى VRAM. بعد تحميله يبقى في الذاكرة ما دام الـStudio شغالًا.


## CPU Studio (4 vCPU / 16 GB)

استخدم:

```bash
chmod +x setup_cpu.sh run_cpu.sh
./setup_cpu.sh
./run_cpu.sh
```

`run_cpu.sh` يشغل Uvicorn بعملية واحدة ويضبط 4 خيوط CPU لتجنب استهلاك الذاكرة الزائد. المنفذ الافتراضي 8000.

import json
import re
import shutil
import subprocess
import time
import uuid
import threading
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from TTS.api import TTS

from app.audio_processing import postprocess_audio, preprocess_reference
from app.text_processing import normalize_text

# F5-TTS is optional. XTTS remains fully functional if F5-TTS is unavailable.
try:
    from silma_tts.api import SilmaTTS
    F5_AVAILABLE = True
except Exception as _f5_import_error:
    SilmaTTS = None
    F5_AVAILABLE = False
    _f5_import_error = str(_f5_import_error)

F5_ENGINE = None


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
VOICES = DATA / "voices"
OUTPUTS = DATA / "outputs"
STATIC = BASE / "app" / "static"
VOICES.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="XTTS + F5-TTS Voice Studio", version="5.1.0-audio-pipeline")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
PORT = int(os.getenv("PORT", "8000"))

# CPU-first mode for 4 vCPU / 16 GB RAM Studios.
# Set FORCE_CPU=1 (default) to prevent accidental GPU use.
FORCE_CPU = os.getenv("FORCE_CPU", "1").strip().lower() not in {"0", "false", "no", "off"}
CPU_THREADS = max(1, int(os.getenv("CPU_THREADS", os.getenv("OMP_NUM_THREADS", "4"))))
try:
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(max(1, min(2, CPU_THREADS)))
except Exception:
    pass

DEVICE = "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda"
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print("=" * 60)
print("XTTS V2 Voice Studio — Voice Only")
print(f"Device: {DEVICE}")
print(f"CPU threads: {CPU_THREADS}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
print("Loading XTTS V2 model...")
tts = TTS(MODEL).to(DEVICE)
print("XTTS V2 loaded successfully.")

XTTS_MODEL = getattr(tts.synthesizer, "tts_model", None)
if XTTS_MODEL is None:
    raise RuntimeError("لم أستطع الوصول إلى XTTS model الداخلي.")
print("XTTS internal model ready.")

GEN_LOCK = threading.Lock()
VOICE_CACHE = {}

LANGUAGES = {
    "ar": "العربية", "en": "English", "es": "Español", "fr": "Français",
    "de": "Deutsch", "it": "Italiano", "pt": "Português", "pl": "Polski",
    "tr": "Türkçe", "ru": "Русский", "nl": "Nederlands", "cs": "Čeština",
    "zh-cn": "中文", "ja": "日本語", "hu": "Magyar", "ko": "한국어", "hi": "हिन्दी",
}

# لا نقيد الرفع بقائمة صغيرة من الامتدادات. FFmpeg يتولى قراءة صيغ الصوت التي يدعمها.
MAX_MB = int(os.getenv("MAX_AUDIO_MB", "100"))
MAX_TEXT = int(os.getenv("MAX_TEXT", "3000"))
CHUNK_LIMIT = 350
OUTPUT_MAX_AGE = 24 * 3600
ID_RE = re.compile(r"[a-f0-9]{32}")

# صيغ التنزيل الشائعة والعملية للمستخدم.
OUTPUT_FORMATS = {
    "wav": {"ext": "wav", "mime": "audio/wav", "args": ["-c:a", "pcm_s16le", "-ar", "24000"]},
    "mp3": {"ext": "mp3", "mime": "audio/mpeg", "args": ["-c:a", "libmp3lame", "-b:a", "320k"]},
    "m4a": {"ext": "m4a", "mime": "audio/mp4", "args": ["-c:a", "aac", "-b:a", "256k"]},
    "flac": {"ext": "flac", "mime": "audio/flac", "args": ["-c:a", "flac"]},
    "ogg": {"ext": "ogg", "mime": "audio/ogg", "args": ["-c:a", "libvorbis", "-q:a", "8"]},
    "opus": {"ext": "opus", "mime": "audio/ogg", "args": ["-c:a", "libopus", "-b:a", "160k"]},
}

LENGTH_PENALTY = 1.0
TOP_K = 50
TOP_P = 0.80
SPEED = 1.0

# أوضاع مطابقة الصوت: "high" يستخدم حتى 30 ثانية من العينة وإعدادات أكثر ثباتًا.
FIDELITY_MODES = {
    "fast": {"label": "سريع", "gpt_cond_len": 6, "gpt_cond_chunk_len": 6,
             "temperature": 0.65, "repetition_penalty": 5.0},
    "high": {"label": "تطابق عالٍ", "gpt_cond_len": 30, "gpt_cond_chunk_len": 6,
             "temperature": 0.45, "repetition_penalty": 3.0},
}
DEFAULT_FIDELITY = "high"

FFMPEG_PATH = shutil.which("ffmpeg")
HAS_FFMPEG = FFMPEG_PATH is not None


def get_f5_engine():
    """Lazy-load F5-TTS so XTTS can start even when F5 dependencies/model are unavailable."""
    global F5_ENGINE
    if not F5_AVAILABLE:
        raise RuntimeError(f"F5-TTS غير متاح: {_f5_import_error}")
    if F5_ENGINE is None:
        print("[F5] Loading SILMA TTS v1 (Arabic/English, F5 architecture)...")
        F5_ENGINE = SilmaTTS()
        print("[F5] SILMA TTS loaded successfully.")
    return F5_ENGINE

def generate_f5(reference_path, reference_text, text, output_path):
    engine = get_f5_engine()
    # F5-TTS works best with a short, clean reference. The official docs recommend <12s.
    engine.infer(
        ref_file=str(reference_path),
        ref_text=reference_text or None,
        gen_text=text,
        file_wave=str(output_path),
        seed=None,
        speed=float(os.getenv("F5_SPEED", "1.0")),
    )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("F5-TTS لم ينشئ ملف WAV صالح.")

def valid_id(value):
    return bool(value) and ID_RE.fullmatch(value) is not None


def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_\-\u0600-\u06FF ]+", "", name or "").strip()[:60] or "Voice"


def meta(voice_id):
    if not valid_id(voice_id):
        return None
    path = VOICES / voice_id / "meta.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def cleanup_outputs():
    now = time.time()
    for file in OUTPUTS.iterdir():
        if not file.is_file():
            continue
        try:
            if now - file.stat().st_mtime > OUTPUT_MAX_AGE:
                file.unlink(missing_ok=True)
        except OSError:
            pass


def split_text(text, limit=CHUNK_LIMIT):
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?؟۔،,;؛:\n])\s+", text)
    pieces = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        while len(part) > limit:
            cut = part.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            pieces.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            pieces.append(part)
    chunks, current = [], ""
    for part in pieces:
        if not current:
            current = part
        elif len(current) + 1 + len(part) <= limit:
            current += " " + part
        else:
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks


def ffmpeg_run(args):
    if not FFMPEG_PATH:
        raise RuntimeError("FFmpeg غير مثبت أو غير موجود في PATH.")
    result = subprocess.run(
        [FFMPEG_PATH, "-y", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-3500:])
    return result


def convert_to_wav(input_path, output_path):
    ffmpeg_run([
        "-i", str(input_path), "-vn", "-map", "0:a:0",
        "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le", str(output_path)
    ])
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg لم ينشئ ملف WAV صالح.")


FFPROBE_PATH = shutil.which("ffprobe")
HAS_FFPROBE = FFPROBE_PATH is not None


def has_audio_stream(path):
    """يفحص الملف باستخدام ffprobe. يعيد True/False، أو None إذا لم يكن ffprobe متاحًا."""
    if not FFPROBE_PATH:
        return None
    result = subprocess.run(
        [FFPROBE_PATH, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "default=nw=1:nk=1", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore"
    )
    return result.returncode == 0 and "audio" in result.stdout.lower()


def clear_voice_cache(voice_id):
    for key in [k for k in VOICE_CACHE if k == voice_id or k.startswith(f"{voice_id}:")]:
        VOICE_CACHE.pop(key, None)


def load_voice_conditioning(voice_id, speaker_path, fidelity="high"):
    mode = FIDELITY_MODES.get(fidelity, FIDELITY_MODES["high"])
    cache_key = f"{voice_id}:{fidelity}"
    try:
        mtime = speaker_path.stat().st_mtime
    except OSError:
        raise RuntimeError("ملف الصوت المرجعي غير موجود.")

    cached = VOICE_CACHE.get(cache_key)
    if cached and cached["mtime"] == mtime:
        return cached["gpt_cond_latent"], cached["speaker_embedding"]

    print(f"[VOICE] Computing conditioning: {cache_key} (gpt_cond_len={mode['gpt_cond_len']}s)")
    start = time.time()
    with torch.inference_mode():
        gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path=str(speaker_path),
            gpt_cond_len=mode["gpt_cond_len"],
            gpt_cond_chunk_len=mode["gpt_cond_chunk_len"],
            max_ref_length=30,
            sound_norm_refs=False,
        )
    VOICE_CACHE[cache_key] = {
        "mtime": mtime,
        "gpt_cond_latent": gpt_cond_latent,
        "speaker_embedding": speaker_embedding,
    }
    print(f"[VOICE] Conditioning ready in {time.time() - start:.2f}s")
    return gpt_cond_latent, speaker_embedding


def output_base(output_id):
    path = OUTPUTS / f"{output_id}.wav"
    if not path.exists():
        raise HTTPException(404, "الصوت الناتج غير موجود.")
    return path

cleanup_outputs()


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {
        "ok": True, "device": DEVICE, "force_cpu": FORCE_CPU, "cpu_threads": CPU_THREADS, "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ffmpeg": HAS_FFMPEG, "ffprobe": HAS_FFPROBE, "model": MODEL, "languages": len(LANGUAGES),
        "engines": {"xtts": True, "f5_tts": F5_AVAILABLE, "f5_loaded": F5_ENGINE is not None},
        "cached_voices": len(VOICE_CACHE), "output_formats": list(OUTPUT_FORMATS.keys()),
        "fidelity_modes": FIDELITY_MODES,
        "audio_pipeline": {"reference_target_lufs": -23, "output_target_lufs": -18, "text_normalization": True},
    }


@app.get("/api/languages")
def languages():
    return LANGUAGES


@app.get("/api/voices")
def list_voices():
    result = []
    for directory in VOICES.iterdir():
        if not directory.is_dir():
            continue
        data = meta(directory.name)
        if data:
            data["audio_url"] = f"/api/voices/{data['id']}/audio"
            result.append(data)
    return result


@app.post("/api/voices")
async def create_voice(name: str = Form(...), audio: UploadFile = File(...)):
    if not HAS_FFMPEG:
        raise HTTPException(500, "FFmpeg غير مثبت على السيرفر.")

    name = safe_name(name)
    original_name = audio.filename or "voice"
    suffix = Path(original_name).suffix.lower() or ".audio"
    voice_id = uuid.uuid4().hex
    folder = VOICES / voice_id
    folder.mkdir(parents=True, exist_ok=True)
    original_path = folder / f"original{suffix}"
    reference_path = folder / "reference.wav"
    total = 0

    try:
        with original_path.open("wb") as out:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_MB * 1024 * 1024:
                    raise HTTPException(413, f"ملف الصوت أكبر من {MAX_MB}MB.")
                out.write(chunk)

        if total == 0:
            raise HTTPException(400, "ملف الصوت فارغ.")

        # ffprobe يفحص الملف فعليًا بدل الاعتماد على الامتداد فقط.
        # إن لم يكن ffprobe متاحًا (None) نتخطى الفحص المسبق ونعتمد على نتيجة التحويل.
        if has_audio_stream(original_path) is False:
            raise HTTPException(400, "الملف لا يحتوي على مسار صوت صالح. ارفع ملفًا صوتيًا.")

        preprocess_info = preprocess_reference(original_path, reference_path)
        original_path.unlink(missing_ok=True)

        metadata = {
            "id": voice_id,
            "name": name,
            "filename": "reference.wav",
            "original_filename": original_name,
            "size_bytes": total,
            "created_at": time.time(),
            "preprocessing": preprocess_info,
            "audio_url": f"/api/voices/{voice_id}/audio",
        }
        (folder / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        # تجهيز الـ conditioning مباشرة (بالوضع العالي) حتى يكون التوليد التالي أسرع.
        load_voice_conditioning(voice_id, reference_path, DEFAULT_FIDELITY)
        return metadata

    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(500, f"فشل تجهيز الصوت: {e}")


@app.get("/api/voices/{vid}/audio")
def voice_audio(vid: str):
    data = meta(vid)
    if not data:
        raise HTTPException(404, "الصوت غير موجود.")
    path = VOICES / vid / data["filename"]
    if not path.exists():
        raise HTTPException(404, "ملف الصوت غير موجود.")
    return FileResponse(path, media_type="audio/wav", filename=f"{safe_name(data['name'])}.wav")


@app.delete("/api/voices/{vid}")
def delete_voice(vid: str):
    if not meta(vid):
        raise HTTPException(404, "الصوت غير موجود.")
    clear_voice_cache(vid)
    shutil.rmtree(VOICES / vid, ignore_errors=True)
    return {"ok": True}


@app.post("/api/generate")
def generate(voice_id: str = Form(...), text: str = Form(...), language: str = Form("ar"),
             fidelity: str = Form(DEFAULT_FIDELITY), engine: str = Form("xtts"),
             reference_text: str = Form("")):
    start_total = time.time()
    text = normalize_text(text, language)
    if not text:
        raise HTTPException(400, "اكتب النص أولًا.")
    if len(text) > MAX_TEXT:
        raise HTTPException(400, f"الحد الأقصى للنص {MAX_TEXT} حرف.")
    if language not in LANGUAGES:
        raise HTTPException(400, "اللغة غير مدعومة.")
    if fidelity not in FIDELITY_MODES:
        raise HTTPException(400, "وضع المطابقة غير معروف.")
    engine = engine.strip().lower()
    if engine not in {"xtts", "f5", "auto", "ensemble"}:
        raise HTTPException(400, "محرك الصوت غير معروف.")
    if engine == "f5" and not F5_AVAILABLE:
        raise HTTPException(503, "SILMA TTS غير مثبت. ثبّت المتطلبات ثم أعد التشغيل.")
    mode = FIDELITY_MODES[fidelity]

    voice_meta = meta(voice_id)
    if not voice_meta:
        raise HTTPException(404, "الصوت غير موجود.")
    speaker = VOICES / voice_id / voice_meta["filename"]
    if not speaker.exists():
        raise HTTPException(404, "ملف الصوت المرجعي غير موجود.")

    output_id = uuid.uuid4().hex
    output_path = OUTPUTS / f"{output_id}.wav"
    cleanup_outputs()

    extra_output = None
    try:
        with GEN_LOCK:
            selected_engine = engine
            # SILMA TTS v1 is a bilingual Arabic/English model built on F5-TTS.
            # Keep XTTS for the wider multilingual set.
            f5_language_ok = language in {"ar", "en"}
            if selected_engine == "auto":
                selected_engine = "ensemble" if (F5_AVAILABLE and reference_text.strip() and f5_language_ok) else "xtts"

            print(f"[GENERATE] Voice={voice_meta['name']} Language={language} Fidelity={fidelity} Engine={selected_engine}")

            if selected_engine == "ensemble":
                if not F5_AVAILABLE:
                    raise HTTPException(503, "SILMA TTS غير مثبت، لذلك لا يمكن تشغيل الوضع المزدوج.")
                if not reference_text.strip():
                    raise HTTPException(400, "الوضع المزدوج يحتاج نص العينة المرجعية لـ F5-TTS.")
                if not f5_language_ok:
                    raise HTTPException(400, "محرك F5 العربي المستخدم هنا هو SILMA TTS v1 ويدعم العربية والإنجليزية. للغات الأخرى استخدم XTTS.")

                # IMPORTANT: XTTS is not a voice-conversion stage. We therefore
                # generate two candidates from the same reference instead of
                # pretending that XTTS can re-voice an arbitrary F5 waveform.
                f5_path = OUTPUTS / f"{output_id}_f5.wav"
                generate_f5(speaker, normalize_text(reference_text, language), text, f5_path)

                gpt_cond_latent, speaker_embedding = load_voice_conditioning(voice_id, speaker, fidelity)
                chunks_list = split_text(text, CHUNK_LIMIT)
                sample_rate = 24000
                all_audio = []
                silence = torch.zeros(int(sample_rate * 0.12))
                for index, chunk in enumerate(chunks_list):
                    with torch.inference_mode():
                        result = XTTS_MODEL.inference(
                            text=chunk, language=language,
                            gpt_cond_latent=gpt_cond_latent,
                            speaker_embedding=speaker_embedding,
                            temperature=mode["temperature"],
                            length_penalty=LENGTH_PENALTY,
                            repetition_penalty=mode["repetition_penalty"],
                            top_k=TOP_K, top_p=TOP_P, do_sample=True,
                            num_beams=1, speed=SPEED, enable_text_splitting=False,
                        )
                    audio_tensor = torch.tensor(result["wav"], dtype=torch.float32)
                    if index > 0:
                        all_audio.append(silence)
                    all_audio.append(audio_tensor)
                final_audio = torch.cat(all_audio)
                import torchaudio
                torchaudio.save(str(output_path), final_audio.unsqueeze(0), sample_rate)
                chunks = len(chunks_list)
                # XTTS is returned as the primary candidate because it is the
                # engine whose multilingual voice conditioning this project uses.
                selected_engine = "ensemble"
                extra_output = {
                    "engine": "f5",
                    "url": f"/api/audio/{output_id}_f5",
                    "download": f"/api/download/{output_id}_f5?format=wav",
                    "filename": "generated_f5.wav",
                }
            elif selected_engine == "f5":
                if not reference_text.strip():
                    raise HTTPException(400, "محرك SILMA TTS يحتاج نص العينة المرجعية. اكتب ما قيل في العينة.")
                generate_f5(speaker, normalize_text(reference_text, language), text, output_path)
                chunks = 1
            else:
                gpt_cond_latent, speaker_embedding = load_voice_conditioning(voice_id, speaker, fidelity)
                chunks_list = split_text(text, CHUNK_LIMIT)
                sample_rate = 24000
                all_audio = []
                silence = torch.zeros(int(sample_rate * 0.12))

                for index, chunk in enumerate(chunks_list):
                    print(f"[XTTS] Chunk {index + 1}/{len(chunks_list)}: {chunk[:100]}")
                    with torch.inference_mode():
                        result = XTTS_MODEL.inference(
                            text=chunk, language=language,
                            gpt_cond_latent=gpt_cond_latent,
                            speaker_embedding=speaker_embedding,
                            temperature=mode["temperature"],
                            length_penalty=LENGTH_PENALTY,
                            repetition_penalty=mode["repetition_penalty"],
                            top_k=TOP_K, top_p=TOP_P, do_sample=True,
                            num_beams=1, speed=SPEED,
                            enable_text_splitting=False,
                        )
                    audio_tensor = torch.tensor(result["wav"], dtype=torch.float32)
                    if index > 0:
                        all_audio.append(silence)
                    all_audio.append(audio_tensor)

                final_audio = torch.cat(all_audio)
                import torchaudio
                torchaudio.save(str(output_path), final_audio.unsqueeze(0), sample_rate)
                chunks = len(chunks_list)

        # Postprocess every produced candidate to a consistent speech level.
        for candidate in (output_path, OUTPUTS / f"{output_id}_f5.wav"):
            if candidate.exists():
                processed = candidate.with_name(candidate.stem + "_processed.wav")
                postprocess_audio(candidate, processed, sample_rate=24000)
                processed.replace(candidate)

        elapsed = round(time.time() - start_total, 2)
        return {
            "id": output_id,
            "url": f"/api/audio/{output_id}",
            "download": f"/api/download/{output_id}?format=wav",
            "filename": "generated.wav",
            "device": DEVICE,
            "language": language,
            "fidelity": fidelity,
            "engine": selected_engine,
            "voice": voice_meta["name"],
            "chunks": chunks,
            "generation_time": elapsed,
            "formats": {fmt: f"/api/download/{output_id}?format={fmt}" for fmt in OUTPUT_FORMATS},
            "alternate": extra_output,
        }
    except HTTPException:
        output_path.unlink(missing_ok=True)
        (OUTPUTS / f"{output_id}_f5.wav").unlink(missing_ok=True)
        raise
    except Exception as e:
        output_path.unlink(missing_ok=True)
        (OUTPUTS / f"{output_id}_f5.wav").unlink(missing_ok=True)
        print(f"[ERROR] {repr(e)}")
        raise HTTPException(500, f"فشل توليد الصوت: {e}")


@app.get("/api/audio/{output_id}_f5")
def alternate_audio(output_id: str):
    if not valid_id(output_id):
        raise HTTPException(404, "الصوت غير موجود.")
    path = OUTPUTS / f"{output_id}_f5.wav"
    if not path.exists():
        raise HTTPException(404, "الصوت البديل غير موجود.")
    return FileResponse(path, media_type="audio/wav", filename="generated_f5.wav")

@app.get("/api/audio/{output_id}")
def audio(output_id: str):
    if not valid_id(output_id):
        raise HTTPException(400, "معرّف غير صالح.")
    path = OUTPUTS / f"{output_id}.wav"
    if not path.exists():
        raise HTTPException(404, "الصوت الناتج غير موجود.")
    return FileResponse(path, media_type="audio/wav", filename="generated.wav")


@app.get("/api/download/{output_id}_f5")
def alternate_download(output_id: str, format: str = "wav"):
    if not valid_id(output_id):
        raise HTTPException(400, "معرّف غير صالح.")
    format = format.lower().lstrip(".")
    if format not in OUTPUT_FORMATS:
        raise HTTPException(400, "صيغة التنزيل غير مدعومة.")
    source = OUTPUTS / f"{output_id}_f5.wav"
    if not source.exists():
        raise HTTPException(404, "الصوت البديل غير موجود.")
    if format == "wav":
        return FileResponse(source, media_type="audio/wav", filename="generated_f5.wav")
    target = OUTPUTS / f"{output_id}_f5.{OUTPUT_FORMATS[format]['ext']}"
    if not target.exists():
        cmd = [FFMPEG_PATH, "-y", "-i", str(source)] + OUTPUT_FORMATS[format]["args"] + [str(target)]
        if not FFMPEG_PATH:
            raise HTTPException(500, "FFmpeg غير مثبت على السيرفر.")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    spec = OUTPUT_FORMATS[format]
    return FileResponse(target, media_type=spec["mime"], filename=f"generated_f5.{spec['ext']}")

@app.get("/api/download/{output_id}")
def download(output_id: str, format: str = "wav"):
    if not valid_id(output_id):
        raise HTTPException(400, "معرّف غير صالح.")
    format = format.lower().lstrip(".")
    if format not in OUTPUT_FORMATS:
        raise HTTPException(400, "صيغة التنزيل غير مدعومة.")
    source = output_base(output_id)
    spec = OUTPUT_FORMATS[format]
    if format == "wav":
        return FileResponse(source, media_type=spec["mime"], filename="generated.wav")

    target = OUTPUTS / f"{output_id}.{spec['ext']}"
    try:
        if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
            ffmpeg_run(["-i", str(source), *spec["args"], str(target)])
    except Exception as e:
        raise HTTPException(500, f"فشل تحويل الصوت إلى {format.upper()}: {e}")
    return FileResponse(target, media_type=spec["mime"], filename=f"generated.{spec['ext']}")

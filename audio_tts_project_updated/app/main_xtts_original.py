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


# =========================================================
# PATHS
# =========================================================

BASE = Path(__file__).resolve().parent.parent

DATA = BASE / "data"
VOICES = DATA / "voices"
OUTPUTS = DATA / "outputs"
STATIC = BASE / "app" / "static"

VOICES.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="XTTS V2 Voice Studio",
    version="3.1.0-lightning"
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC),
    name="static"
)

# Lightning exposes the process through the port configured in the Studio.
PORT = int(os.getenv("PORT", "8000"))



# =========================================================
# DEVICE
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lightning AI: optimize CUDA execution when a GPU is attached.
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print("=" * 60)
print("XTTS V2 Voice Studio")
print("=" * 60)
print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU غير متوفر - سيتم استخدام CPU")
    print("التوليد على CPU سيكون أبطأ من مواقع XTTS التي تستخدم GPU.")

print("=" * 60)


# =========================================================
# MODEL
# =========================================================

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

print("Loading XTTS V2 model...")

tts = TTS(MODEL).to(DEVICE)

print("XTTS V2 loaded successfully.")
if DEVICE == "cuda":
    try:
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    except Exception:
        pass


# =========================================================
# XTTS INTERNAL MODEL
# =========================================================

# في Coqui XTTS يكون الموديل الحقيقي هنا.
XTTS_MODEL = getattr(
    tts.synthesizer,
    "tts_model",
    None
)

if XTTS_MODEL is None:
    raise RuntimeError(
        "لم أستطع الوصول إلى XTTS model الداخلي."
    )

print("XTTS internal model ready.")


# =========================================================
# LOCK
# =========================================================

GEN_LOCK = threading.Lock()


# =========================================================
# VOICE LATENT CACHE
# =========================================================

# نخزن معلومات الصوت في RAM.
#
# الشكل:
#
# {
#     voice_id: {
#         "mtime": ...,
#         "gpt_cond_latent": ...,
#         "speaker_embedding": ...
#     }
# }

VOICE_CACHE = {}


# =========================================================
# SETTINGS
# =========================================================

LANGUAGES = {
    "ar": "العربية",
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "pt": "Português",
    "pl": "Polski",
    "tr": "Türkçe",
    "ru": "Русский",
    "nl": "Nederlands",
    "cs": "Čeština",
    "zh-cn": "中文",
    "ja": "日本語",
    "hu": "Magyar",
    "ko": "한국어",
    "hi": "हिन्दी",
}


ALLOWED = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
}


NEEDS_FFMPEG = {
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
}


FFMPEG_PATH = shutil.which("ffmpeg")

HAS_FFMPEG = FFMPEG_PATH is not None


MAX_MB = 30
MAX_TEXT = 3000

# XTTS لا يستطيع استقبال نصوص ضخمة جدًا دفعة واحدة.
# نقسم النص بشكل ذكي.
CHUNK_LIMIT = 350

OUTPUT_MAX_AGE = 24 * 3600

ID_RE = re.compile(
    r"[a-f0-9]{32}"
)


# =========================================================
# XTTS QUALITY SETTINGS
# =========================================================

TEMPERATURE = 0.65

LENGTH_PENALTY = 1.0

REPETITION_PENALTY = 5.0

TOP_K = 50

TOP_P = 0.80

SPEED = 1.0


# =========================================================
# HELPERS
# =========================================================

def valid_id(value):
    return bool(value) and ID_RE.fullmatch(value) is not None


def safe_name(name):
    return (
        re.sub(
            r"[^a-zA-Z0-9_\-\u0600-\u06FF ]+",
            "",
            name
        ).strip()[:60]
        or "Voice"
    )


def meta(voice_id):
    if not valid_id(voice_id):
        return None

    path = VOICES / voice_id / "meta.json"

    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None


# =========================================================
# OUTPUT CLEANUP
# =========================================================

def cleanup_outputs():
    now = time.time()

    for file in OUTPUTS.glob("*.wav"):

        try:

            if now - file.stat().st_mtime > OUTPUT_MAX_AGE:
                file.unlink(
                    missing_ok=True
                )

        except OSError:
            pass


# =========================================================
# TEXT SPLITTER
# =========================================================

def split_text(text, limit=CHUNK_LIMIT):

    text = text.strip()

    if not text:
        return []

    # نفصل عند نهاية الجمل.
    parts = re.split(
        r"(?<=[.!?؟۔،,;؛:\n])\s+",
        text
    )

    pieces = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        while len(part) > limit:

            cut = part.rfind(
                " ",
                0,
                limit
            )

            if cut <= 0:
                cut = limit

            pieces.append(
                part[:cut].strip()
            )

            part = part[cut:].strip()

        if part:
            pieces.append(part)

    chunks = []
    current = ""

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


# =========================================================
# FFMPEG CONVERSION
# =========================================================

def convert_to_wav(input_path, output_path):

    if not FFMPEG_PATH:
        raise RuntimeError(
            "FFmpeg غير مثبت."
        )

    command = [
        FFMPEG_PATH,

        "-y",

        "-i",
        str(input_path),

        "-vn",

        # Mono
        "-ac",
        "1",

        # XTTS reference sample rate
        "-ar",
        "22050",

        # PCM WAV
        "-c:a",
        "pcm_s16le",

        str(output_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    if result.returncode != 0:

        raise RuntimeError(
            "فشل تحويل الصوت إلى WAV:\n"
            + result.stderr[-2000:]
        )

    if not output_path.exists():

        raise RuntimeError(
            "FFmpeg لم ينشئ ملف WAV."
        )


# =========================================================
# VOICE CACHE
# =========================================================

def clear_voice_cache(voice_id):

    VOICE_CACHE.pop(
        voice_id,
        None
    )


def load_voice_conditioning(
    voice_id,
    speaker_path
):

    """
    يحسب conditioning للصوت مرة واحدة فقط.

    هذه أهم نقطة لتسريع التوليد.
    """

    try:

        mtime = speaker_path.stat().st_mtime

    except OSError:

        raise RuntimeError(
            "ملف الصوت المرجعي غير موجود."
        )

    cached = VOICE_CACHE.get(
        voice_id
    )

    # إذا الصوت موجود ولم يتغير الملف
    if cached:

        if cached["mtime"] == mtime:

            print(
                f"[CACHE] Using cached voice: {voice_id}"
            )

            return (
                cached["gpt_cond_latent"],
                cached["speaker_embedding"]
            )

    print(
        f"[VOICE] Computing voice conditioning: {voice_id}"
    )

    start = time.time()

    with torch.inference_mode():

        gpt_cond_latent, speaker_embedding = (
            XTTS_MODEL.get_conditioning_latents(
                audio_path=str(speaker_path),

                # أفضل نطاق للصوت المرجعي
                gpt_cond_len=6,

                gpt_cond_chunk_len=6,

                max_ref_length=30,

                sound_norm_refs=False
            )
        )

    elapsed = time.time() - start

    print(
        f"[VOICE] Conditioning ready in {elapsed:.2f}s"
    )

    VOICE_CACHE[voice_id] = {
        "mtime": mtime,
        "gpt_cond_latent": gpt_cond_latent,
        "speaker_embedding": speaker_embedding,
    }

    return (
        gpt_cond_latent,
        speaker_embedding
    )


# =========================================================
# STARTUP CLEANUP
# =========================================================

cleanup_outputs()


# =========================================================
# HOME
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def index():

    return (
        STATIC / "index.html"
    ).read_text(
        encoding="utf-8"
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "ok": True,

        "device": DEVICE,

        "cuda": torch.cuda.is_available(),

        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),

        "ffmpeg": HAS_FFMPEG,

        "ffmpeg_path": FFMPEG_PATH,

        "model": MODEL,

        "languages": len(LANGUAGES),

        "cached_voices": len(VOICE_CACHE),
    }


# =========================================================
# LANGUAGES
# =========================================================

@app.get("/api/languages")
def languages():

    return LANGUAGES


# =========================================================
# LIST VOICES
# =========================================================

@app.get("/api/voices")
def list_voices():

    result = []

    for directory in VOICES.iterdir():

        if not directory.is_dir():
            continue

        data = meta(
            directory.name
        )

        if data:
            result.append(data)

    return result


# =========================================================
# CREATE VOICE
# =========================================================

@app.post("/api/voices")
async def create_voice(
    name: str = Form(...),
    audio: UploadFile = File(...)
):

    name = safe_name(name)

    original_name = (
        audio.filename or ""
    )

    suffix = Path(
        original_name
    ).suffix.lower()

    if suffix not in ALLOWED:

        raise HTTPException(
            400,
            "صيغة الصوت غير مدعومة."
        )

    if (
        suffix in NEEDS_FFMPEG
        and not HAS_FFMPEG
    ):

        raise HTTPException(
            400,
            "هذي الصيغة تحتاج FFmpeg."
        )

    voice_id = uuid.uuid4().hex

    folder = VOICES / voice_id

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    original_path = (
        folder / f"original{suffix}"
    )

    total = 0

    try:

        with original_path.open("wb") as out:

            while True:

                chunk = await audio.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total += len(chunk)

                if total > MAX_MB * 1024 * 1024:

                    raise HTTPException(
                        413,
                        "ملف الصوت أكبر من 30MB."
                    )

                out.write(chunk)

        # =================================================
        # WAV
        # =================================================

        if suffix == ".wav":

            reference_path = (
                folder / "reference.wav"
            )

            # نعيد تحويل WAV أيضًا للتأكد
            # من mono + 22050Hz
            convert_to_wav(
                original_path,
                reference_path
            )

            if original_path != reference_path:
                original_path.unlink(
                    missing_ok=True
                )

        else:

            reference_path = (
                folder / "reference.wav"
            )

            convert_to_wav(
                original_path,
                reference_path
            )

            original_path.unlink(
                missing_ok=True
            )

        # =================================================
        # META
        # =================================================

        metadata = {
            "id": voice_id,
            "name": name,
            "filename": "reference.wav",
            "original_filename": original_name,
        }

        (
            folder / "meta.json"
        ).write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        # =================================================
        # PRE-CACHE VOICE
        # =================================================

        print(
            f"[VOICE] Preparing voice: {name}"
        )

        load_voice_conditioning(
            voice_id,
            reference_path
        )

        print(
            f"[VOICE] Voice ready: {name}"
        )

        return metadata

    except HTTPException:

        shutil.rmtree(
            folder,
            ignore_errors=True
        )

        raise

    except Exception as e:

        shutil.rmtree(
            folder,
            ignore_errors=True
        )

        raise HTTPException(
            500,
            f"فشل تجهيز الصوت: {e}"
        )


# =========================================================
# DELETE VOICE
# =========================================================

@app.delete("/api/voices/{vid}")
def delete_voice(vid: str):

    if not meta(vid):

        raise HTTPException(
            404,
            "الصوت غير موجود."
        )

    # امسح الـ cache أولًا
    clear_voice_cache(vid)

    shutil.rmtree(
        VOICES / vid,
        ignore_errors=True
    )

    return {
        "ok": True
    }


# =========================================================
# GENERATE
# =========================================================

@app.post("/api/generate")
def generate(
    voice_id: str = Form(...),
    text: str = Form(...),
    language: str = Form("ar")
):

    start_total = time.time()

    text = text.strip()

    # =====================================================
    # VALIDATION
    # =====================================================

    if not text:

        raise HTTPException(
            400,
            "اكتب نصًا أولًا."
        )

    if len(text) > MAX_TEXT:

        raise HTTPException(
            400,
            f"الحد الأقصى للنص {MAX_TEXT} حرف."
        )

    if language not in LANGUAGES:

        raise HTTPException(
            400,
            "اللغة غير مدعومة."
        )

    voice_meta = meta(
        voice_id
    )

    if not voice_meta:

        raise HTTPException(
            404,
            "الصوت غير موجود."
        )

    speaker = (
        VOICES
        / voice_id
        / voice_meta["filename"]
    )

    if not speaker.exists():

        raise HTTPException(
            404,
            "ملف الصوت المرجعي غير موجود."
        )

    # =====================================================
    # OUTPUT
    # =====================================================

    output_id = uuid.uuid4().hex

    output_path = (
        OUTPUTS / f"{output_id}.wav"
    )

    cleanup_outputs()

    # =====================================================
    # GENERATION
    # =====================================================

    try:

        with GEN_LOCK:

            print("=" * 60)

            print(
                f"[GENERATE] Voice: {voice_meta['name']}"
            )

            print(
                f"[GENERATE] Language: {language}"
            )

            print(
                f"[GENERATE] Text length: {len(text)}"
            )

            # ---------------------------------------------
            # CACHE
            # ---------------------------------------------

            gpt_cond_latent, speaker_embedding = (
                load_voice_conditioning(
                    voice_id,
                    speaker
                )
            )

            # ---------------------------------------------
            # SPLIT
            # ---------------------------------------------

            chunks = split_text(
                text,
                CHUNK_LIMIT
            )

            print(
                f"[GENERATE] Chunks: {len(chunks)}"
            )

            # ---------------------------------------------
            # SAMPLE RATE
            # ---------------------------------------------

            sample_rate = 24000

            # XTTS outputs float waveform.
            all_audio = []

            # 0.12 ثانية صمت بين الجمل
            silence = torch.zeros(
                int(sample_rate * 0.12)
            )

            # ---------------------------------------------
            # GENERATE EACH CHUNK
            # ---------------------------------------------

            for index, chunk in enumerate(chunks):

                print(
                    f"[GENERATE] Chunk "
                    f"{index + 1}/{len(chunks)}"
                )

                print(
                    f"[TEXT] {chunk[:100]}"
                )

                chunk_start = time.time()

                with torch.inference_mode():

                    result = XTTS_MODEL.inference(

                        text=chunk,

                        language=language,

                        gpt_cond_latent=gpt_cond_latent,

                        speaker_embedding=speaker_embedding,

                        # جودة / ثبات الصوت
                        temperature=TEMPERATURE,

                        length_penalty=LENGTH_PENALTY,

                        repetition_penalty=REPETITION_PENALTY,

                        top_k=TOP_K,

                        top_p=TOP_P,

                        do_sample=True,

                        num_beams=1,

                        speed=SPEED,

                        # نحن نقسم النص بأنفسنا
                        enable_text_splitting=False
                    )

                audio_tensor = torch.tensor(
                    result["wav"],
                    dtype=torch.float32
                )

                if index > 0:

                    all_audio.append(
                        silence
                    )

                all_audio.append(
                    audio_tensor
                )

                chunk_time = (
                    time.time()
                    - chunk_start
                )

                print(
                    f"[GENERATE] Chunk time: "
                    f"{chunk_time:.2f}s"
                )

            # ---------------------------------------------
            # JOIN
            # ---------------------------------------------

            final_audio = torch.cat(
                all_audio
            )

            # ---------------------------------------------
            # SAVE
            # ---------------------------------------------

            import torchaudio

            torchaudio.save(
                str(output_path),

                final_audio.unsqueeze(0),

                sample_rate
            )

            total_time = (
                time.time()
                - start_total
            )

            print(
                f"[GENERATE] Finished in "
                f"{total_time:.2f}s"
            )

            print(
                f"[GENERATE] Output: "
                f"{output_path}"
            )

            print("=" * 60)

    except Exception as e:

        output_path.unlink(
            missing_ok=True
        )

        print(
            f"[ERROR] {repr(e)}"
        )

        raise HTTPException(
            500,
            f"فشل توليد الصوت: {e}"
        )

    return {
        "id": output_id,

        "url":
            f"/api/audio/{output_id}",

        "download":
            f"/api/download/{output_id}",

        "filename":
            "generated.wav",

        "device":
            DEVICE,

        "language":
            language,

        "voice":
            voice_meta["name"],

        "chunks":
            len(chunks),

        "generation_time":
            round(
                time.time() - start_total,
                2
            ),
    }


# =========================================================
# AUDIO
# =========================================================

@app.get("/api/audio/{oid}")
def audio(oid: str):

    if not valid_id(oid):

        raise HTTPException(
            404,
            "الملف غير موجود."
        )

    path = (
        OUTPUTS / f"{oid}.wav"
    )

    if not path.exists():

        raise HTTPException(
            404,
            "الملف غير موجود."
        )

    return FileResponse(
        path,

        media_type="audio/wav",

        filename="generated.wav"
    )


# =========================================================
# DOWNLOAD
# =========================================================

@app.get("/api/download/{oid}")
def download(oid: str):

    if not valid_id(oid):

        raise HTTPException(
            404,
            "الملف غير موجود."
        )

    path = (
        OUTPUTS / f"{oid}.wav"
    )

    if not path.exists():

        raise HTTPException(
            404,
            "الملف غير موجود."
        )

    return FileResponse(
        path,

        media_type="audio/wav",

        filename="generated.wav"
    )
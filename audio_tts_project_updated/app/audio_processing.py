"""FFmpeg-based reference and generated-audio processing.

Uses widely available FFmpeg filters to avoid forcing heavyweight ML audio
packages onto CPU-only deployments. DeepFilterNet can be added as an optional
front-end later; it is not silently assumed to remove room reverberation.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

FFMPEG = shutil.which("ffmpeg")


def _run(source: Path, target: Path, filters: str, sample_rate: int = 22050) -> None:
    if not FFMPEG:
        raise RuntimeError("FFmpeg غير مثبت أو غير موجود في PATH.")
    command = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-vn", "-map", "0:a:0", "-af", filters, "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode or not target.exists() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise RuntimeError((result.stderr or "تعذرت معالجة الصوت.")[-2500:])


def preprocess_reference(source: Path, target: Path) -> dict:
    """Denoise, trim long silence and normalize reference to -23 LUFS.

    Afftdn reduces stationary background noise; FFmpeg does not provide robust
    general-purpose dereverberation. If the denoiser is unavailable, retry with
    conservative filtering and normalization rather than rejecting the upload.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        "highpass=f=70,lowpass=f=11000,afftdn=nf=-25," 
        "silenceremove=start_periods=1:start_duration=0.20:start_threshold=-42dB:"
        "stop_periods=-1:stop_duration=1.0:stop_threshold=-42dB," 
        "loudnorm=I=-23:TP=-1.5:LRA=11"
    )
    try:
        _run(source, target, filters, 22050)
        return {"target_lufs": -23, "denoise": "afftdn", "silence_trim": True}
    except RuntimeError:
        target.unlink(missing_ok=True)
        fallback = (
            "highpass=f=70,lowpass=f=11000," 
            "silenceremove=start_periods=1:start_duration=0.20:start_threshold=-42dB:"
            "stop_periods=-1:stop_duration=1.0:stop_threshold=-42dB," 
            "loudnorm=I=-23:TP=-1.5:LRA=11"
        )
        _run(source, target, fallback, 22050)
        return {"target_lufs": -23, "denoise": "unavailable-fallback", "silence_trim": True}


def postprocess_audio(source: Path, target: Path, sample_rate: int = 24000) -> None:
    """Apply gentle EQ and EBU loudness normalization to generated speech."""
    filters = (
        "highpass=f=65,lowpass=f=15000,equalizer=f=250:t=q:w=1:g=-1.0," 
        "equalizer=f=3500:t=q:w=1:g=1.0,loudnorm=I=-18:TP=-1.0:LRA=11"
    )
    _run(source, target, filters, sample_rate)

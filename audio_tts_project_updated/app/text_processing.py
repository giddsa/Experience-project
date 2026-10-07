"""Lightweight text cleanup and speech-friendly number expansion."""
from __future__ import annotations

import re
import unicodedata

try:
    from num2words import num2words
except ImportError:  # Keep startup resilient if an optional deployment omitted it.
    num2words = None

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w])([+-]?\d+(?:[.,]\d+)?)(?!\w)")
_EMOJI_OR_SYMBOL = re.compile(
    "[" "\\U0001F000-\\U0001FAFF\\U00002600-\\U000027BF\\U0001F1E6-\\U0001F1FF" "]+"
)
_CURRENCIES = {
    "$": "دولار", "€": "يورو", "£": "جنيه إسترليني", "د.إ": "درهم إماراتي",
    "ر.س": "ريال سعودي", "ر.ق": "ريال قطري", "ر.ك": "دينار كويتي",
}


def _number_words(value: str, language: str) -> str:
    if not num2words:
        return value
    normalized = value.replace("٫", ".").replace(",", ".")
    try:
        if "." in normalized:
            whole, fraction = normalized.split(".", 1)
            left = num2words(int(whole), lang=language)
            right = " ".join(num2words(int(d), lang=language) for d in fraction if d.isdigit())
            return f"{left} فاصلة {right}" if right else left
        return num2words(int(normalized), lang=language)
    except (ValueError, TypeError, NotImplementedError):
        return value


def normalize_text(text: str, language: str = "ar") -> str:
    """Remove non-speech clutter, expand numbers where supported, and tidy punctuation.

    Existing diacritics are preserved. Automatic Arabic diacritization is not
    attempted: adding guessed harakat can change a word's meaning/pronunciation.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _URL.sub(" ", text)
    text = _EMOJI_OR_SYMBOL.sub(" ", text)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))
    # Expand a currency sign immediately following an amount.
    for symbol, word in _CURRENCIES.items():
        text = re.sub(rf"(?<=\d)\s*{re.escape(symbol)}", f" {word}", text)
    number_lang = {"ar": "ar", "en": "en", "fr": "fr", "es": "es", "de": "de", "it": "it", "pt": "pt_BR", "ru": "ru"}.get(language, "en")
    text = _NUMBER.sub(lambda m: _number_words(m.group(1), number_lang), text)
    # Common TTS-friendly punctuation normalization; keep Arabic punctuation.
    text = text.replace("…", "، ").replace("—", "، ").replace("–", "، ")
    text = re.sub(r"([،؛:,.!?؟])\1+", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    text = text.strip(" \t\n،؛,:.")
    if text and text[-1] not in ".!?؟؛":
        text += "."
    return text

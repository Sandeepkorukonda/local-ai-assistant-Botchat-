"""Speech to text with faster-whisper."""
import io
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def load_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(config.WHISPER_SIZE, device="cpu", compute_type="int8")


def transcribe(audio_bytes: bytes) -> str:
    segments, _ = load_whisper().transcribe(io.BytesIO(audio_bytes))  # language auto-detected
    return " ".join(s.text.strip() for s in segments).strip()
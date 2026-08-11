"""Shared transcriber interface and PCM helpers."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .. import SAMPLE_RATE


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None
    #: Mean token log-probability where the backend reports one; higher is better.
    confidence: float | None = None


class Transcriber(Protocol):
    #: Shown in the TUI status bar, e.g. "whisper small/cpu int8".
    describe: str

    def transcribe(self, pcm: bytes) -> Transcript: ...

    def close(self) -> None: ...


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def pcm_to_wav(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return buffer.getvalue()

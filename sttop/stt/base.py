"""Shared transcriber interface and PCM helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


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


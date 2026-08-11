"""Local transcription with faster-whisper (CTranslate2)."""

from __future__ import annotations

import threading

from ..config import SttConfig
from .base import Transcript, pcm_to_float32


def detect_device(requested: str = "auto") -> str:
    """Resolve "auto" to the fastest device CTranslate2 can actually use.

    CTranslate2 ships CUDA and CPU backends only - there is no ROCm build - so
    an AMD GPU cannot accelerate this path no matter what torch reports.
    """
    if requested != "auto":
        return requested

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:  # pragma: no cover - ctranslate2 always present in practice
        pass
    return "cpu"


def detect_compute_type(device: str, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if device == "cuda":
        return "float16"
    try:
        import ctranslate2

        supported = ctranslate2.get_supported_compute_types("cpu")
        if "int8" in supported:
            return "int8"
    except Exception:  # pragma: no cover
        pass
    return "float32"


class LocalTranscriber:
    def __init__(self, config: SttConfig) -> None:
        from faster_whisper import WhisperModel

        self.config = config
        self.device = detect_device(config.device)
        self.compute_type = detect_compute_type(self.device, config.compute_type)
        self.describe = f"whisper {config.model}/{self.device} {self.compute_type}"

        self._model = WhisperModel(
            config.model, device=self.device, compute_type=self.compute_type
        )
        # One model, one decode at a time - CTranslate2 is already internally
        # parallel and concurrent calls just thrash the CPU.
        self._lock = threading.Lock()

    def transcribe(self, pcm: bytes) -> Transcript:
        audio = pcm_to_float32(pcm)
        with self._lock:
            segments, info = self._model.transcribe(
                audio,
                language=self.config.language,
                beam_size=self.config.beam_size,
                # We already ran webrtcvad upstream; re-running it here would
                # only re-trim audio that is known to be speech.
                vad_filter=False,
                # Each segment is an independent utterance, so carrying decoder
                # context across them mostly imports hallucinations.
                condition_on_previous_text=False,
            )
            collected = list(segments)

        text = " ".join(seg.text.strip() for seg in collected).strip()
        confidence = None
        if collected:
            confidence = sum(seg.avg_logprob for seg in collected) / len(collected)
        return Transcript(
            text=text,
            language=getattr(info, "language", None),
            confidence=confidence,
        )

    def close(self) -> None:
        self._model = None

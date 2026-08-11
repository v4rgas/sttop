"""Speech-to-text backends."""

from __future__ import annotations

from .base import Transcriber, Transcript, pcm_to_float32, pcm_to_wav

__all__ = [
    "BACKENDS",
    "DEFAULT_MODELS",
    "Transcriber",
    "Transcript",
    "build",
    "pcm_to_float32",
    "pcm_to_wav",
]

BACKENDS = ("parakeet", "whisper", "cloud")

#: Used when stt.model is left blank, so switching backend does not require
#: also remembering to switch model.
DEFAULT_MODELS = {
    "parakeet": "nemo-parakeet-tdt-0.6b-v3",
    "whisper": "small",
    "cloud": "whisper-large-v3-turbo",
}


def build(config) -> Transcriber:
    """Instantiate the transcriber named by `config.stt.backend`."""
    backend = config.stt.backend.lower()
    if backend not in BACKENDS:
        options = ", ".join(BACKENDS)
        raise ValueError(f"unknown stt backend {config.stt.backend!r} - pick {options}")

    settings = config.stt
    if not settings.model:
        settings.model = DEFAULT_MODELS[backend]

    if backend == "parakeet":
        from .parakeet import ParakeetTranscriber

        return ParakeetTranscriber(settings)
    if backend == "whisper":
        from .local import LocalTranscriber

        return LocalTranscriber(settings)

    from .cloud import CloudTranscriber

    return CloudTranscriber(config.cloud, settings)

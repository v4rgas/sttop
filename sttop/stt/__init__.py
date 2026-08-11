"""Speech-to-text backends."""

from __future__ import annotations

from dataclasses import replace

from ..config import SttConfig
from .base import Transcriber, Transcript, pcm_to_float32

__all__ = [
    "BACKENDS",
    "DEFAULT_MODELS",
    "Transcriber",
    "Transcript",
    "build",
    "pcm_to_float32",
]

BACKENDS = ("parakeet", "whisper")

#: Used when stt.model is left blank, so switching backend does not require
#: also remembering to switch model.
DEFAULT_MODELS = {
    "parakeet": "nemo-parakeet-tdt-0.6b-v3",
    "whisper": "small",
}


def build(config: SttConfig) -> Transcriber:
    """Instantiate the transcriber named by `config.backend`.

    Takes the stt section rather than the whole Config, and resolves the
    default model into a copy: writing it back would pin the user's config to
    one concrete model, so a later change of backend would keep the old
    backend's model.
    """
    backend = config.backend.lower()
    if backend not in BACKENDS:
        options = ", ".join(BACKENDS)
        raise ValueError(f"unknown stt backend {config.backend!r} - pick {options}")

    # Blank means "unset" for both of these: a config file cannot write None,
    # so `model = ""` and `language = ""` are how a user says "you decide".
    settings = replace(
        config,
        model=config.model or DEFAULT_MODELS[backend],
        language=config.language or None,
    )

    if backend == "parakeet":
        from .parakeet import ParakeetTranscriber

        return ParakeetTranscriber(settings)

    if backend == "whisper":
        from .local import LocalTranscriber

        return LocalTranscriber(settings)

    raise AssertionError(f"{backend!r} is in BACKENDS but nothing constructs it")

"""Local speech-to-text with Parakeet."""

from __future__ import annotations

import logging
import platform
from dataclasses import replace

from ..config import SttConfig
from .base import Transcriber, Transcript, pcm_to_float32

__all__ = ["DEFAULT_MODEL", "Transcriber", "Transcript", "build", "pcm_to_float32"]

DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"
MLX_MODELS = {
    f"nemo-parakeet-tdt-0.6b-{version}": f"mlx-community/parakeet-tdt-0.6b-{version}"
    for version in ("v2", "v3")
}


def apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def build(config: SttConfig) -> Transcriber:
    """Resolve defaults without changing the caller's configuration."""
    from .parakeet import ParakeetTranscriber

    if config.backend not in ("auto", "onnx", "mlx"):
        raise ValueError("stt.backend must be auto, onnx, or mlx")
    resolved = replace(
        config, model=config.model or DEFAULT_MODEL, language=config.language or None
    )
    wants_mlx = config.backend == "mlx" or (
        config.backend == "auto" and (
            resolved.model.startswith("mlx-community/")
            or (apple_silicon() and resolved.model in MLX_MODELS)
        )
    )
    if wants_mlx:
        if not apple_silicon():
            raise ValueError("The MLX backend requires an Apple Silicon Mac")
        from .parakeet_mlx import ParakeetMLXTranscriber

        try:
            return ParakeetMLXTranscriber(replace(
                resolved, backend="mlx",
                model=MLX_MODELS.get(resolved.model, resolved.model),
            ))
        except ImportError:
            # Only the known ONNX aliases have a safe CPU fallback. An explicit
            # MLX request or custom model must not silently select other weights.
            if config.backend == "mlx" or resolved.model not in MLX_MODELS:
                raise
            logging.getLogger(__name__).warning(
                "MLX dependencies unavailable; using Parakeet on CPU."
            )
    return ParakeetTranscriber(replace(resolved, backend="onnx"))

"""Parakeet on Apple Silicon, consuming the capture's PCM directly."""

from __future__ import annotations

from .. import SAMPLE_RATE
from ..config import SttConfig
from .base import Transcript, pcm_to_float32
from .sources import MLX_FILES, MLX_SOURCES, snapshot


class ParakeetMLXTranscriber:
    def __init__(self, config: SttConfig) -> None:
        import mlx.core as mx
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel

        if not mx.metal.is_available():
            raise RuntimeError("MLX needs Metal GPU support; use --backend onnx")
        self.config = config
        self._mx = mx
        self._get_logmel = get_logmel
        model_path = config.model
        if revision := MLX_SOURCES.get(config.model):
            model_path = snapshot(config.model, revision, MLX_FILES)
        self._model = from_pretrained(model_path)
        if self._model.preprocessor_config.sample_rate != SAMPLE_RATE:
            raise ValueError("The MLX model must accept 16 kHz audio")
        # Finish lazy weight loading during preparation, not the first utterance.
        mx.eval(self._model.parameters())
        self.describe = f"{config.model.rsplit('/', 1)[-1]}/gpu mlx"

    def transcribe(self, pcm: bytes) -> Transcript:
        if not pcm:
            return Transcript(text="", language=self.config.language)
        audio = self._mx.array(pcm_to_float32(pcm))
        mel = self._get_logmel(audio, self._model.preprocessor_config)
        result = self._model.generate(mel)[0]
        return Transcript(text=result.text.strip(), language=self.config.language)

    def close(self) -> None:
        self._model = None
        self._mx.clear_cache()

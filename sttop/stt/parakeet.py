"""Local transcription with NVIDIA Parakeet via onnxruntime.

Parakeet TDT 0.6b v3 is multilingual (25 European languages, autodetected) and
runs comfortably faster than real time on CPU. onnx-asr runs the exported ONNX
graph directly, so this backend needs neither torch nor the NeMo toolkit.
"""

from __future__ import annotations

from .. import SAMPLE_RATE
from ..config import SttConfig
from .base import Transcript, pcm_to_float32


class ParakeetTranscriber:
    def __init__(self, config: SttConfig) -> None:
        import onnx_asr

        from ..nativelog import quiet_onnxruntime

        # onnxruntime warns about every operator its CoreML provider cannot
        # take, straight to fd 2 - which is the terminal the UI is drawing on.
        quiet_onnxruntime()

        self.config = config
        self._model = onnx_asr.load_model(config.model)
        short = config.model.removeprefix("nemo-").removesuffix("-0.6b-v3")
        self.describe = f"{short}/cpu onnx"

    def transcribe(self, pcm: bytes) -> Transcript:
        text = self._model.recognize(pcm_to_float32(pcm), sample_rate=SAMPLE_RATE)
        return Transcript(text=(text or "").strip(), language=self.config.language)

    def close(self) -> None:
        self._model = None

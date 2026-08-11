"""Speaker identification.

Two sources of truth, in order of reliability:

1. Which stream the audio came from. Anything on the microphone is you; that
   needs no model and is never wrong.
2. Voice embeddings, for splitting the *system* stream into the individual
   remote participants. ECAPA-TDNN embeddings are clustered online: each new
   utterance is matched against running centroids by cosine similarity, and
   opens a new speaker when nothing is close enough.

Online clustering means labels are assigned as audio arrives and are never
revised. That is the price of real time - an offline pass would be more
accurate but could not label a segment until the meeting ended.
"""

from __future__ import annotations

import threading

import numpy as np

from .audio.segmenter import Segment
from .config import DiarizeConfig
from .stt.base import pcm_to_float32

SELF_LABEL = "you"
UNKNOWN_LABEL = "spk?"


class NullDiarizer:
    """Used when diarization is off or its dependencies are missing."""

    speaker_count = 0

    def __init__(self, reason: str = "disabled") -> None:
        self.describe = f"diarize off ({reason})"

    def label(self, segment: Segment, is_mic: bool) -> str:
        return SELF_LABEL if is_mic else "them"

    def close(self) -> None:
        pass


class Diarizer:
    def __init__(self, config: DiarizeConfig) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        from .config import DATA_DIR

        self.config = config
        self._lock = threading.Lock()
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []
        self._last_label: str | None = None
        self._last_end: float = 0.0

        self._encoder = EncoderClassifier.from_hparams(
            source=config.model,
            savedir=str(DATA_DIR / "models" / config.model.replace("/", "--")),
            run_opts={"device": "cpu"},
        )
        self.describe = f"ecapa @{config.threshold:.2f}"

    @property
    def speaker_count(self) -> int:
        return len(self._centroids)

    def label(self, segment: Segment, is_mic: bool) -> str:
        if is_mic:
            return SELF_LABEL

        if segment.duration < self.config.min_speech_s:
            # Too short to embed reliably. Assume the current speaker is still
            # talking if this lands right after their last utterance.
            if self._last_label and segment.start - self._last_end < 5.0:
                return self._last_label
            return UNKNOWN_LABEL

        try:
            embedding = self._embed(segment.pcm)
        except Exception:
            return UNKNOWN_LABEL

        with self._lock:
            label = self._assign(embedding)
        self._last_label = label
        self._last_end = segment.end
        return label

    def _embed(self, pcm: bytes) -> np.ndarray:
        import torch

        audio = torch.from_numpy(pcm_to_float32(pcm).copy()).unsqueeze(0)
        with torch.no_grad():
            embedding = self._encoder.encode_batch(audio).squeeze().cpu().numpy()
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm else embedding

    def _assign(self, embedding: np.ndarray) -> str:
        best_index, best_score = -1, -1.0
        for index, centroid in enumerate(self._centroids):
            score = float(np.dot(embedding, centroid))
            if score > best_score:
                best_index, best_score = index, score

        if best_index >= 0 and best_score >= self.config.threshold:
            count = self._counts[best_index]
            # Running mean, so a centroid firms up as a speaker talks more.
            merged = (self._centroids[best_index] * count + embedding) / (count + 1)
            norm = np.linalg.norm(merged)
            self._centroids[best_index] = merged / norm if norm else merged
            self._counts[best_index] = count + 1
            return f"spk{best_index + 1}"

        # Hysteresis. Without it, one noisy embedding from an established
        # speaker mints a whole new one, and a real meeting ends up with a
        # dozen phantom participants. In the grey zone we take the nearest
        # match but leave its centroid alone, so a bad frame cannot poison it.
        if best_index >= 0 and best_score >= self.config.threshold - self.config.margin:
            return f"spk{best_index + 1}"

        self._centroids.append(embedding)
        self._counts.append(1)
        return f"spk{len(self._centroids)}"

    def close(self) -> None:
        self._encoder = None


def build(config: DiarizeConfig):
    """Return a diarizer, degrading to NullDiarizer rather than failing the run."""
    if not config.enabled:
        return NullDiarizer("disabled")
    try:
        return Diarizer(config)
    except ImportError:
        return NullDiarizer("install sttop[diarize]")
    except Exception as exc:  # model download failure, corrupt cache, ...
        return NullDiarizer(str(exc).splitlines()[0][:40])

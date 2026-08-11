"""Speaker identification.

Two sources of truth, in order of reliability:

1. Which stream the audio came from. Anything on the microphone is you; that
   needs no model and is never wrong.
2. Voice embeddings, for splitting the *system* stream into the individual
   remote participants. ECAPA-TDNN embeddings are clustered online: each new
   utterance is matched against running centroids by cosine similarity, and
   opens a new speaker when nothing is close enough.

Online clustering assigns labels as audio arrives, which is the only way to
show a name next to a line the moment it is transcribed. The cost is that the
first decision about a voice is made on the least evidence we will ever have,
and greedy assignment alone freezes that mistake forever - one person ends up
scattered across spk2, spk5 and spk7.

So the clustering here is greedy but not final. Speakers accumulate their
embeddings, centroids firm up as people talk, and once two settled speakers
look like the same person they are merged and the earlier label is *rewritten*
in the transcript. That mirrors what incremental-clustering diarizers do
offline: decide now, revise when the evidence arrives.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from . import SAMPLE_RATE
from .audio.segmenter import Segment
from .config import DiarizeConfig
from .stt.base import pcm_to_float32

SELF_LABEL = "you"
OTHER_LABEL = "them"
UNKNOWN_LABEL = "spk?"

#: Cap on windows embedded per utterance. A long monologue gains nothing from
#: averaging fifty windows, and this runs on the transcription thread.
MAX_WINDOWS = 4

#: A rename this diarizer decided on: (old label, label it is now part of).
Merge = tuple[str, str]


def speaker_name(index: int) -> str:
    """The label for the nth clustered voice, counting from zero."""
    return f"spk{index + 1}"


class SpeakerLabeler(Protocol):
    """What the engine needs from a diarizer - see `Transcriber` for the twin."""

    #: Shown in the TUI status bar, e.g. "ecapa @0.30".
    describe: str

    @property
    def speaker_count(self) -> int: ...

    def label(self, segment: Segment, is_mic: bool) -> str: ...

    def take_merges(self) -> list[Merge]: ...

    def close(self) -> None: ...


class NullDiarizer:
    """Used when diarization is off or its dependencies are missing."""

    speaker_count = 0

    def __init__(self, reason: str = "disabled") -> None:
        self.describe = f"diarize off ({reason})"

    def label(self, segment: Segment, is_mic: bool) -> str:
        return SELF_LABEL if is_mic else OTHER_LABEL

    def take_merges(self) -> list[Merge]:
        return []

    def close(self) -> None:
        pass


@dataclass
class _Speaker:
    """One clustered voice.

    `id` is fixed at creation and owns the label, so merging a speaker away
    does not renumber the ones that outlive it - a transcript whose speakers
    shuffle every time two voices join would be unreadable.
    """

    id: int
    #: Every accepted embedding, so the centroid is a true mean. A running mean
    #: weights the first, noisiest embedding as heavily as the hundredth.
    embeddings: list[np.ndarray] = field(default_factory=list)
    centroid: np.ndarray | None = None

    @property
    def name(self) -> str:
        return speaker_name(self.id)

    def add(self, embedding: np.ndarray) -> None:
        self.embeddings.append(embedding)
        self.centroid = _normalize(np.mean(self.embeddings, axis=0))


class Clusters:
    """The clustering half of the diarizer, with no model attached.

    Split out from `Diarizer` because the decisions worth testing - when a
    voice is new, when two labels are one person - are decisions about
    vectors, and pinning them down should not require downloading ECAPA.
    """

    def __init__(self, config: DiarizeConfig) -> None:
        self.config = config
        self._speakers: list[_Speaker] = []
        self._next_id = 0
        self._merges: list[Merge] = []

    @property
    def count(self) -> int:
        return len(self._speakers)

    def take_merges(self) -> list[Merge]:
        merges, self._merges = self._merges, []
        return merges

    def _settled(self, speaker: _Speaker) -> bool:
        return len(speaker.embeddings) >= self.config.warmup

    def assign(self, embedding: np.ndarray) -> str:
        """Label this voice, then fold together anyone it just revealed to be
        the same person. Returns the label; read `take_merges` for the rest."""
        speaker = self._nearest_or_new(embedding)
        renamed = {}
        for old, new in self._collapse_duplicates():
            renamed[old] = new
            self._merges.append((old, new))
        return renamed.get(speaker.name, speaker.name)

    def _nearest_or_new(self, embedding: np.ndarray) -> _Speaker:
        best, best_score = None, -1.0
        for speaker in self._speakers:
            score = float(np.dot(embedding, speaker.centroid))
            if score > best_score:
                best, best_score = speaker, score

        if best is not None:
            floor = self.config.threshold - self.config.margin
            if best_score >= self.config.threshold:
                best.add(embedding)
                return best
            if best_score >= floor:
                # The grey zone. An unsettled speaker takes the embedding: its
                # centroid is still one or two noisy vectors and this is how it
                # firms up. A settled one takes the *label* but not the
                # embedding, so a bad frame cannot poison a good centroid.
                if not self._settled(best):
                    best.add(embedding)
                return best

        speaker = _Speaker(id=self._next_id)
        speaker.add(embedding)
        self._next_id += 1
        self._speakers.append(speaker)
        return speaker

    def _collapse_duplicates(self) -> list[Merge]:
        """Fold together speakers that turned out to be one person.

        Only settled speakers are eligible: merging on two utterances' worth of
        evidence would undo a split that was right. The survivor is the older
        speaker, so the label the transcript has used longest is the one that
        stays. Runs to a fixed point - absorbing one speaker can pull a third
        within reach.
        """
        merges: list[Merge] = []
        while (merge := self._collapse_once()) is not None:
            merges.append(merge)
        return merges

    def _collapse_once(self) -> Merge | None:
        for index, keep in enumerate(self._speakers):
            for drop in self._speakers[index + 1 :]:
                if not (self._settled(keep) and self._settled(drop)):
                    continue
                score = float(np.dot(keep.centroid, drop.centroid))
                if score < self.config.merge_threshold:
                    continue
                for embedding in drop.embeddings:
                    keep.add(embedding)
                self._speakers.remove(drop)
                return (drop.name, keep.name)
        return None


class Diarizer:
    def __init__(self, config: DiarizeConfig) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        from .config import DATA_DIR

        self.config = config
        self._lock = threading.Lock()
        self._clusters = Clusters(config)
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
        return self._clusters.count

    def label(self, segment: Segment, is_mic: bool) -> str:
        if is_mic:
            return SELF_LABEL

        if segment.duration < self.config.min_speech_s:
            # Too short to embed reliably. Assume the current speaker is still
            # talking if this lands right after their last utterance.
            with self._lock:
                if self._last_label and segment.start - self._last_end < 5.0:
                    return self._last_label
            return UNKNOWN_LABEL

        try:
            embedding = self._embed(segment.pcm)
        except Exception:
            return UNKNOWN_LABEL

        # `_last_label` and `_last_end` describe the same utterance, so they are
        # updated under the lock that the reader above holds: a torn pair would
        # carry one utterance's speaker into another's timing.
        with self._lock:
            label = self._clusters.assign(embedding)
            self._last_label = label
            self._last_end = segment.end
        return label

    def take_merges(self) -> list[Merge]:
        """Renames decided since the last call, oldest first.

        The engine drains these on the event loop and rewrites the journal, so
        the diarizer never touches the transcript itself.
        """
        with self._lock:
            merges = self._clusters.take_merges()
            for old, new in merges:
                # The short-segment fallback hands out `_last_label` without
                # consulting the clusters, so it has to follow a merge itself.
                if self._last_label == old:
                    self._last_label = new
        return merges

    # -- embedding ---------------------------------------------------------

    def _embed(self, pcm: bytes) -> np.ndarray:
        audio = pcm_to_float32(pcm)
        windows = _windows(audio, self.config.window_s)
        return _normalize(np.mean([self._encode(w) for w in windows], axis=0))

    def _encode(self, audio: np.ndarray) -> np.ndarray:
        import torch

        tensor = torch.from_numpy(audio.copy()).unsqueeze(0)
        with torch.no_grad():
            embedding = self._encoder.encode_batch(tensor).squeeze().cpu().numpy()
        return _normalize(embedding)

    def close(self) -> None:
        self._encoder = None


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _windows(audio: np.ndarray, window_s: float) -> list[np.ndarray]:
    """Split into overlapping windows, or one window if it is short enough.

    Half-window hops, so every moment of speech is seen twice and a window
    boundary landing mid-phoneme costs nothing.
    """
    size = int(window_s * SAMPLE_RATE)
    if len(audio) <= size * 1.5:
        return [audio]
    hop = size // 2
    starts = range(0, len(audio) - size + 1, hop)
    return [audio[s : s + size] for s in list(starts)[:MAX_WINDOWS]]


def build(config: DiarizeConfig) -> SpeakerLabeler:
    """Return a diarizer, degrading to NullDiarizer rather than failing the run."""
    if not config.enabled:
        return NullDiarizer("disabled")
    try:
        return Diarizer(config)
    except ImportError:
        return NullDiarizer("reinstall sttop")
    except Exception as exc:  # model download failure, corrupt cache, ...
        return NullDiarizer(str(exc).splitlines()[0][:40])

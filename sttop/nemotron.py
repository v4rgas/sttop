"""NVIDIA Nemotron 3 Diarization, run on CPU through a community ONNX export.

Nemotron 3 is an end-to-end streaming diarizer (a Sortformer): it reads audio
in chunks and emits, every 10 ms, the probability that each of up to eight
speakers is talking. Speaker identity lives in a speaker cache of past encoder
frames that the model attends to on every step, and outputs are ordered by
first arrival, so channel 0 is whoever spoke first for the whole session. No
cosine thresholds, no centroids, no merges.

NVIDIA's streaming recipes call the model every 0.3-1 s, which costs about one
CPU core in real time. sttop only needs a speaker once per utterance, after the
segmenter has already heard 0.7 s of silence, so this runs one step per
utterance instead: the chunk ends where the utterance ends, the lookahead is
silence we already know about, and the cost drops by an order of magnitude.

The speaker-cache update below mirrors `Nemotron3DiarizationSpeakerCache` in
transformers; the constants it needs ship with the export in `constants.npz`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE

#: Pinned like the Parakeet sources: updating it is an explicit source change.
REPO = "joosthel/Nemotron-3-Diarization-ONNX"
REVISION = "4a911fc3ca821b76a99fffd5ce5135bd1efca540"
FILES = ("preprocessor_core.onnx", "model.int8.onnx", "constants.npz")

HOP = 160  # 10 ms mel frames
N_FFT = 512
PREEMPHASIS = 0.97
#: Encoder frames (80 ms each) of lookahead per step. The model card's
#: low-latency configs use 1-4; the segmenter closes an utterance only after
#: 0.7 s of silence, so 4 frames of it are always already in hand.
RIGHT_CONTEXT = 4
#: FIFO of recent encoder frames kept beside the speaker cache. Small, as in
#: the card's offline config, because steps here are whole utterances rather
#: than 0.3 s slivers, and the FIFO is attended to on every step.
FIFO_LENGTH = 40
UPDATE_PERIOD = 300
#: Longest single step, in encoder frames - the card's offline chunk (27 s).
MAX_CHUNK = 340
#: Silence inserted between utterances. The model only sees what the segmenter
#: kept, so gaps are rebuilt as zeros - capped, because ten minutes of nobody
#: talking carries no speaker information and would cost a full step each.
MIN_GAP_S = 0.4
MAX_GAP_S = 2.0


def fetch() -> Path:
    from .stt.sources import snapshot

    return Path(snapshot(REPO, REVISION, FILES))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest along axis 1, ties to the lower index."""
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


class SpeakerCache:
    """Who the model has met so far: encoder frames it attends to every step.

    A FIFO of the most recent frames, plus a fixed-size cache that the FIFO
    overflows into; when the cache is full, it is compressed to the frames
    that are most confidently one speaker, with a quota per speaker so a quiet
    participant is not forgotten.
    """

    def __init__(self, c: dict[str, np.ndarray], fifo_length: int, update_period: int):
        self.hidden = int(c["hidden_size"])
        self.speakers = int(c["num_speakers"])
        self.subsampling = int(c["subsampling_factor"])
        self.length = int(c["speaker_cache_length"])
        self.silence_frames = int(c["speaker_cache_silence_frames_per_speaker"])
        self.score_floor = float(c["prediction_score_threshold"])
        self.latest_boost = float(c["latest_frames_score_boost"])
        self.silence = c["silence_embeds"].astype(np.float32)
        self.fifo_length = fifo_length
        self.update_period = update_period
        budget = self.length // self.speakers - self.silence_frames
        self.min_positive = math.floor(budget * float(c["min_positive_scores_rate"]))
        self.strong_boosted = math.floor(budget * float(c["strong_boost_rate"]))
        self.weak_boosted = math.floor(budget * float(c["weak_boost_rate"]))

        self.embeds = np.zeros((1, self.length, self.hidden), np.float32)
        self.probs = np.zeros((1, self.length, self.speakers), np.float32)
        self.fifo = np.zeros((1, fifo_length, self.hidden), np.float32)
        self.cached = 0
        self.queued = 0
        self.compressed = False

    def context(self) -> np.ndarray:
        return np.concatenate(
            [self.embeds[:, : self.cached], self.fifo[:, : self.queued]], axis=1
        )

    def update(self, step_embeds: np.ndarray, logits: np.ndarray, frames: int) -> None:
        cached, queued = self.cached, self.queued
        probs = _sigmoid(logits)
        batch, steps, speakers = probs.shape
        pooled = (batch, steps // self.subsampling, self.subsampling, speakers)
        probs = probs.reshape(pooled).mean(axis=2).astype(np.float32)

        start = cached + queued
        fifo = np.concatenate(
            [self.fifo[:, :queued], step_embeds[:, start : start + frames]], axis=1
        )
        total = fifo.shape[1]
        popped = 0
        if total > self.fifo_length:
            popped = min(max(self.update_period, total - self.fifo_length), total)
        if popped:
            fifo_probs = probs[:, cached : cached + total]
            stored = self.probs[:, :cached] if self.compressed else probs[:, :cached]
            embeds = np.concatenate([self.embeds[:, :cached], fifo[:, :popped]], axis=1)
            kept = np.concatenate([stored, fifo_probs[:, :popped]], axis=1)
            fifo = fifo[:, popped:]
            if embeds.shape[1] > self.length:
                embeds, kept = self._compress(embeds, kept)
                self.compressed = True
            self.cached = embeds.shape[1]
            self.embeds[:, : self.cached] = embeds
            self.probs[:, : self.cached] = kept
        self.queued = fifo.shape[1]
        self.fifo[:, : self.queued] = fifo

    def _scores(self, probs: np.ndarray) -> np.ndarray:
        log_p = np.log(np.clip(probs, self.score_floor, None))
        log_q = np.log(np.clip(1.0 - probs, self.score_floor, None))
        scores = log_p - log_q + log_q.sum(axis=-1, keepdims=True) - math.log(0.5)
        speech = probs > 0.5
        scores = np.where(speech, scores, -np.inf)
        positive = scores > 0
        enough = positive.sum(axis=1, keepdims=True) >= self.min_positive
        return np.where(~positive & speech & enough, -np.inf, scores)

    @staticmethod
    def _boost(scores: np.ndarray, count: int, boost: float) -> np.ndarray:
        if count <= 0:
            return scores
        index = _topk(scores, count)
        scores = scores.copy()
        np.put_along_axis(scores, index, np.take_along_axis(scores, index, 1) + boost, 1)
        return scores

    def _compress(self, embeds: np.ndarray, probs: np.ndarray):
        batch, frames, speakers = probs.shape
        scores = self._scores(probs).copy()
        scores[:, self.length :] += self.latest_boost
        scores = self._boost(scores, self.strong_boosted, -2.0 * math.log(0.5))
        scores = self._boost(scores, self.weak_boosted, -math.log(0.5))
        scores = np.pad(
            scores, ((0, 0), (0, self.silence_frames), (0, 0)), constant_values=np.inf
        )
        silence = np.broadcast_to(
            self.silence.reshape(1, 1, -1), (batch, 1, embeds.shape[-1])
        )
        embeds = np.concatenate([embeds, silence], axis=1)
        probs = np.pad(probs, ((0, 0), (0, 1), (0, 0)))

        scored = frames + self.silence_frames
        sentinel = scored * speakers
        flat = scores.transpose(0, 2, 1).reshape(batch, -1)
        index = _topk(flat, self.length)
        index = np.where(np.take_along_axis(flat, index, 1) == -np.inf, sentinel, index)
        index = np.sort(index, axis=1)
        chosen = np.where(index == sentinel, frames, np.minimum(index % scored, frames))
        rows = np.arange(batch)[:, None]
        return embeds[rows, chosen], probs[rows, chosen]


class NemotronStream:
    """One session's worth of system audio, fed an utterance at a time.

    `push` appends an utterance (after rebuilding the silence before it) and
    returns the per-10 ms speaker probabilities over exactly that utterance.
    Audio is kept only as far back as the next step's mel window reaches.
    """

    def __init__(self, mel, model, constants: dict[str, np.ndarray]) -> None:
        """`mel` and `model` are onnxruntime sessions (or anything with their
        `run`); `load` builds the real ones."""
        self._mel = mel
        self._model = model
        self._sub = int(constants["subsampling_factor"])
        self.speakers = int(constants["num_speakers"])
        self._cache = SpeakerCache(constants, FIFO_LENGTH, UPDATE_PERIOD)

        self._audio = np.zeros(0, np.float32)
        self._origin = 0  # absolute sample index of self._audio[0]
        self._total = 0  # absolute samples pushed, gaps included
        self._done = 0  # encoder frames already run through the model
        self._started = False

    @classmethod
    def load(cls, model_dir: Path, threads: int = 1) -> NemotronStream:
        import onnxruntime as ort

        from .onnx import cpu_options

        options = cpu_options(threads)
        providers = ["CPUExecutionProvider"]

        def session(name: str):
            path = str(model_dir / name)
            return ort.InferenceSession(path, options, providers=providers)

        with np.load(model_dir / "constants.npz") as data:
            constants = {key: data[key] for key in data.files}
        mel, model = session("preprocessor_core.onnx"), session("model.int8.onnx")
        return cls(mel, model, constants)

    def push(self, audio: np.ndarray, gap_s: float) -> np.ndarray:
        """Feed one utterance; return its probabilities, shape (frames, speakers)."""
        if self._started:
            gap = int(min(max(gap_s, MIN_GAP_S), MAX_GAP_S) * SAMPLE_RATE)
            self._append(np.zeros(gap, np.float32))
        self._started = True
        begin = self._total
        self._append(np.asarray(audio, np.float32))

        mel_end = self._total // HOP
        target = -(-mel_end // self._sub)  # encoder frames covering the utterance
        logits = []
        first = self._done
        while self._done < target:
            stop = min(self._done + MAX_CHUNK, target)
            logits.append(self._step(self._done, stop))
            self._done = stop
        self._trim()
        if not logits:
            return np.zeros((0, self.speakers), np.float32)
        probs = _sigmoid(np.concatenate(logits, axis=0))
        lo = begin // HOP - first * self._sub
        hi = mel_end - first * self._sub
        return probs[max(lo, 0) : max(hi, lo + 1)]

    def _append(self, samples: np.ndarray) -> None:
        self._audio = np.concatenate([self._audio, samples])
        self._total += len(samples)

    def _trim(self) -> None:
        # The next step's first mel frame needs N_FFT/2 samples (+1 for
        # pre-emphasis) before it.
        keep_from = self._done * self._sub * HOP - N_FFT // 2 - 1
        drop = keep_from - self._origin
        if drop > 0:
            self._audio = self._audio[drop:]
            self._origin += drop

    def _samples(self, start: int, stop: int) -> np.ndarray:
        """Absolute sample range, zeros outside what we hold - before the session
        and the lookahead past its end, which the segmenter heard as silence."""
        out = np.zeros(stop - start, np.float32)
        lo = max(start, self._origin)
        hi = min(stop, self._origin + len(self._audio))
        if hi > lo:
            held = self._audio[lo - self._origin : hi - self._origin]
            out[lo - start : hi - start] = held
        return out

    def _log_mel(self, first: int, stop: int) -> np.ndarray:
        """Mel frames [first, stop): centered STFT windows, pre-emphasis first."""
        raw_start = first * HOP - N_FFT // 2
        raw_stop = (stop - 1) * HOP - N_FFT // 2 + N_FFT
        # One sample of lookback for pre-emphasis. Before the session it reads
        # as zero, which keeps the first real sample as-is, as the reference does.
        raw = self._samples(raw_start - 1, raw_stop)
        emphasized = raw[1:] - PREEMPHASIS * raw[:-1]
        (mel,) = self._mel.run(None, {"preemphasized": emphasized[None, :]})
        return mel[:, : stop - first]

    def _step(self, start: int, stop: int) -> np.ndarray:
        """Run encoder frames [start, stop) with lookahead; return their logits."""
        mel_first = start * self._sub
        mel_stop = (stop + RIGHT_CONTEXT) * self._sub
        context = self._cache.context()
        logits, embeds = self._model.run(
            None,
            {
                "chunk_mel": self._log_mel(mel_first, mel_stop),
                "chunk_mel_length": np.array(mel_stop - mel_first, np.int64),
                "context_embeds": context,
                "context_length": np.array(context.shape[1], np.int64),
            },
        )
        frames = stop - start
        self._cache.update(embeds, logits, frames)
        offset = context.shape[1] * self._sub
        return logits[0, offset : offset + frames * self._sub]

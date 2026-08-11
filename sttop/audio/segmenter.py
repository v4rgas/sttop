"""Turn a continuous PCM frame stream into utterance-sized speech segments."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import webrtcvad

from .. import FRAME_MS, SAMPLE_RATE
from ..config import VadConfig

FRAME_S = FRAME_MS / 1000.0

#: Fraction of the pre-roll window that must be speech before a segment opens.
#: Measured against the *full* window, not the frames seen so far, so a single
#: stray voiced frame at stream start cannot trigger.
TRIGGER_RATIO = 0.6


@dataclass(frozen=True)
class Segment:
    """A contiguous run of speech from one source."""

    source: str
    pcm: bytes
    start: float  # seconds since session start
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class Segmenter:
    """Voice-activity state machine over 20 ms frames.

    Collects frames while speech is present, and emits a Segment once the
    speaker has been quiet for `silence_ms` (or the segment hits its ceiling).
    A rolling pre-roll buffer is prepended so the first syllable isn't clipped.
    """

    def __init__(
        self,
        source: str,
        config: VadConfig,
        on_segment: Callable[[Segment], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source = source
        self._on_segment = on_segment
        self._clock = clock

        self._vad = webrtcvad.Vad(config.aggressiveness)
        self._pad_frames = max(1, config.pad_ms // FRAME_MS)
        self._silence_frames = max(1, config.silence_ms // FRAME_MS)
        self._max_frames = max(1, int(config.max_segment_s / FRAME_S))
        self._min_frames = max(1, config.min_segment_ms // FRAME_MS)

        self._preroll: deque[bytes] = deque(maxlen=self._pad_frames)
        self._voiced: deque[bool] = deque(maxlen=self._pad_frames)
        self._buffer: list[bytes] = []
        self._triggered = False
        self._silence_run = 0
        self._segment_start_index = 0
        self._voiced_count = 0  # speech frames in the open segment

        self._index = 0  # frames consumed since the stream opened
        self._origin: float | None = None  # session time of frame 0
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        """Pausing closes any open utterance, so resuming never splices audio
        from either side of the gap into a single segment."""
        if value and not self._paused:
            self.close()
        self._paused = value

    def feed(self, frame: bytes) -> None:
        if self._origin is None:
            self._origin = self._clock()
        index = self._index
        self._index += 1

        if self._paused:
            return

        speech = self._vad.is_speech(frame, SAMPLE_RATE)

        if not self._triggered:
            self._preroll.append(frame)
            self._voiced.append(speech)
            # Enough of the recent window is speech - open a segment.
            if sum(self._voiced) >= TRIGGER_RATIO * self._voiced.maxlen:
                self._triggered = True
                self._segment_start_index = index - len(self._preroll) + 1
                self._buffer = list(self._preroll)
                self._voiced_count = sum(self._voiced)
                self._silence_run = 0
                self._preroll.clear()
                self._voiced.clear()
            return

        self._buffer.append(frame)
        self._voiced_count += speech
        self._silence_run = 0 if speech else self._silence_run + 1

        if self._silence_run >= self._silence_frames:
            self._flush(trailing_silence=self._silence_run)
        elif len(self._buffer) >= self._max_frames:
            self._flush(trailing_silence=0)
            # Long monologue: stay open so the next chunk continues immediately.
            self._triggered = True
            self._segment_start_index = index + 1
            self._buffer = []
            self._voiced_count = 0
            self._silence_run = 0

    def _flush(self, trailing_silence: int) -> None:
        frames = self._buffer
        # Drop the silence we used as an end-of-utterance signal, keeping one
        # frame so the last word has a little air after it.
        if trailing_silence:
            keep = max(0, len(frames) - trailing_silence + 1)
            frames = frames[:keep]

        voiced_count = self._voiced_count
        self._triggered = False
        self._buffer = []
        self._voiced_count = 0
        self._silence_run = 0
        self._preroll.clear()
        self._voiced.clear()

        # Measure against actual speech, not padded length - otherwise the
        # pre-roll alone can push a blip over the minimum.
        if voiced_count < self._min_frames:
            return

        origin = self._origin or 0.0
        start = origin + self._segment_start_index * FRAME_S
        self._on_segment(
            Segment(
                source=self.source,
                pcm=b"".join(frames),
                start=start,
                end=start + len(frames) * FRAME_S,
            )
        )

    def close(self) -> None:
        """Emit whatever is still buffered - call when the stream ends."""
        if self._triggered and self._buffer:
            self._flush(trailing_silence=0)

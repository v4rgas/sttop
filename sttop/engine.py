"""Wires capture -> VAD -> transcription -> diarization -> Markdown journal.

Concurrency model - one thread boundary, and it is the model:

  capture tasks (2)   asyncio: read ffmpeg stdout, run VAD inline (microseconds)
        |
   asyncio.Queue
        |
  consume task        asyncio: awaits the executor, appends, notifies the UI
        |
  ThreadPoolExecutor(max_workers=1)   the only thread: transcribe + embed

Everything except the model runs on the event loop, so the UI needs no
cross-thread marshalling and shutdown is ordinary task cancellation. The
executor is deliberately single-threaded: transcription is CPU-bound and
already internally parallel, so a second worker would only thrash the cache -
and serialising it keeps utterances in the order they were spoken.

When transcription falls behind, the queue absorbs the lag and `backlog`
reports it rather than dropping audio.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import diarize as diarize_mod
from . import stt
from .audio import devices
from .audio.capture import SourceCapture
from .audio.segmenter import Segment, Segmenter
from .config import Config
from .journal import Journal, Utterance

MIC = "mic"
SYSTEM = "system"

UtteranceCallback = Callable[[Utterance], None]
ErrorCallback = Callable[[str], None]


@dataclass
class EngineStatus:
    """Everything the UI needs to draw one frame of the status bar."""

    elapsed: float = 0.0
    backlog: int = 0
    levels: dict[str, float] = field(default_factory=dict)
    paused: bool = False
    running: bool = False
    utterances: int = 0
    speakers: int = 0
    backend: str = "no backend"
    diarizer: str = "diarize off"


class Engine:
    def __init__(
        self,
        config: Config,
        on_utterance: UtteranceCallback,
        on_error: ErrorCallback | None = None,
    ) -> None:
        self.config = config
        self._on_utterance = on_utterance
        self._on_error = on_error or (lambda message: None)

        self.transcriber: stt.Transcriber | None = None
        self.diarizer: diarize_mod.SpeakerLabeler | None = None
        self.journal: Journal | None = None
        #: Resolved by prepare(); empty until then.
        self.mic_source: str = ""
        self.sys_source: str = ""

        self._captures: list[SourceCapture] = []
        self._segmenters: dict[str, Segmenter] = {}
        self._queue: asyncio.Queue[Segment | None] = asyncio.Queue()
        self._consumer: asyncio.Task | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._t0: float = 0.0
        self._paused = False
        self._running = False

    # -- lifecycle ---------------------------------------------------------

    async def prepare(self) -> None:
        """Resolve devices and load models - seconds of blocking work, so all
        of it goes to the executor and the caller stays responsive."""
        loop = asyncio.get_running_loop()
        self.mic_source, self.sys_source = await loop.run_in_executor(
            self._executor, self._resolve_sources
        )
        self.transcriber = await loop.run_in_executor(
            self._executor, stt.build, self.config.stt
        )
        self.diarizer = await loop.run_in_executor(
            self._executor, diarize_mod.build, self.config.diarize
        )

    def _resolve_sources(self) -> tuple[str, str]:
        return (
            devices.resolve(self.config.audio.mic_source, monitor=False),
            devices.resolve(self.config.audio.system_source, monitor=True),
        )

    async def start(self, title: str | None = None) -> Path:
        if self._running:
            raise RuntimeError("engine already running")
        if self.transcriber is None:
            await self.prepare()

        self.journal = Journal.create(
            Path(self.config.sessions_dir),
            title,
            mic_source=self.mic_source,
            sys_source=self.sys_source,
            backend=self.transcriber.describe,
        )
        self._t0 = time.monotonic()
        self._running = True

        for label, source in ((MIC, self.mic_source), (SYSTEM, self.sys_source)):
            segmenter = Segmenter(
                label, self.config.vad, self._queue.put_nowait, clock=self._session_clock
            )
            self._segmenters[label] = segmenter
            capture = SourceCapture(
                label,
                source,
                on_frame=segmenter.feed,
                on_error=self._on_error,
                wav_path=self._wav_path(label),
            )
            self._captures.append(capture)
            await capture.start()

        self._consumer = asyncio.create_task(self._consume(), name="transcribe")
        return self.journal.path

    async def stop(self, drain_timeout: float = 30.0) -> Path | None:
        """Stop capture, then finish transcribing whatever is still queued.

        Safe to call at any point in the lifecycle, and always releases: a run
        that failed *during* start() has still loaded the models and started
        the executor thread, and those must go back whether or not any audio
        was ever captured.
        """
        if self._running:
            self._running = False
            await self._drain(drain_timeout)
        return self._release()

    async def _drain(self, timeout: float) -> None:
        for capture in self._captures:
            await capture.stop()
        for segmenter in self._segmenters.values():
            segmenter.close()  # flush any utterance still open

        self._queue.put_nowait(None)  # sentinel: drain, then finish
        if self._consumer is not None:
            try:
                await asyncio.wait_for(self._consumer, timeout=timeout)
            except TimeoutError:
                self._on_error(f"[stt] gave up draining after {timeout:.0f}s")
                self._consumer.cancel()
            self._consumer = None

    def _release(self) -> Path | None:
        """Close the journal and hand back the models, executor and captures.

        Leaves the engine prepared-from-scratch rather than half-alive: a
        transcriber whose model has been closed must not look loaded to
        start(), or the next run transcribes against nothing.
        """
        path = None
        if self.journal is not None:
            path = self.journal.close(self._session_clock())
        for resource in (self.transcriber, self.diarizer):
            if resource is not None:
                resource.close()
        self.transcriber = None
        self.diarizer = None
        self._executor.shutdown(wait=True)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._captures.clear()
        self._segmenters.clear()
        return path

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        for segmenter in self._segmenters.values():
            segmenter.paused = self._paused
        return self._paused

    def rename_speaker(self, old: str, new: str) -> int:
        return self.journal.rename_speaker(old, new) if self.journal else 0

    # -- status ------------------------------------------------------------

    def status(self) -> EngineStatus:
        """A snapshot of the run. Valid at any point in the lifecycle -
        before prepare(), mid-session, or after stop()."""
        return EngineStatus(
            elapsed=self._session_clock() if self._t0 else 0.0,
            backlog=self._queue.qsize(),
            levels={c.label: c.level for c in self._captures},
            paused=self._paused,
            running=self._running,
            utterances=self.journal.count if self.journal else 0,
            speakers=self.diarizer.speaker_count if self.diarizer else 0,
            backend=self.transcriber.describe if self.transcriber else "no backend",
            diarizer=self.diarizer.describe if self.diarizer else "diarize off",
        )

    # -- internals ---------------------------------------------------------

    def _session_clock(self) -> float:
        return time.monotonic() - self._t0

    def _wav_path(self, label: str) -> Path | None:
        if not self.config.audio.save_wav or self.journal is None:
            return None
        return Path(self.config.audio_dir) / f"{self.journal.path.stem}-{label}.wav"

    async def _consume(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            segment = await self._queue.get()
            if segment is None:
                return
            try:
                utterance = await loop.run_in_executor(
                    self._executor, self._transcribe, segment
                )
            except Exception as exc:  # one bad segment must not end the run
                self._on_error(f"[stt] {type(exc).__name__}: {exc}")
                continue
            if utterance is None:
                continue
            # Back on the loop: writing and notifying are cheap and ordered.
            self.journal.append(utterance)
            self._on_utterance(utterance)

    def _transcribe(self, segment: Segment) -> Utterance | None:
        """The only code that runs off the event loop."""
        assert self.transcriber is not None

        transcript = self.transcriber.transcribe(segment.pcm)
        if not transcript.text:
            return None  # VAD fired on noise, not speech

        return Utterance(
            source=segment.source,
            speaker=self.diarizer.label(segment, is_mic=segment.source == MIC),
            start=segment.start,
            end=segment.end,
            text=transcript.text,
            language=transcript.language,
            confidence=transcript.confidence,
        )
